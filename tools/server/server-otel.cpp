#ifdef LLAMA_OTEL

#include "server-otel.h"

#include <opentelemetry/sdk/trace/tracer_provider_factory.h>
#include <opentelemetry/sdk/trace/batch_span_processor_factory.h>
#include <opentelemetry/sdk/resource/resource.h>
#include <opentelemetry/exporters/otlp/otlp_http_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_http_exporter_options.h>
#include <opentelemetry/trace/provider.h>
#include <opentelemetry/trace/tracer.h>
#include <opentelemetry/trace/span.h>
#include <opentelemetry/trace/scope.h>
#include <opentelemetry/trace/span_startoptions.h>
#include <opentelemetry/trace/propagation/http_trace_context.h>
#include <opentelemetry/context/propagation/global_propagator.h>
#include <opentelemetry/context/propagation/text_map_propagator.h>

#include <cstdlib>
#include <string>

namespace trace_api  = opentelemetry::trace;
namespace trace_sdk  = opentelemetry::sdk::trace;
namespace resource   = opentelemetry::sdk::resource;
namespace otlp       = opentelemetry::exporter::otlp;
namespace context    = opentelemetry::context;
namespace propagation = opentelemetry::context::propagation;

// TextMapCarrier adapter for std::map<string,string> headers
class HeaderCarrier : public propagation::TextMapCarrier {
public:
    explicit HeaderCarrier(const std::map<std::string, std::string> & headers)
        : headers_(headers) {}

    opentelemetry::nostd::string_view Get(opentelemetry::nostd::string_view key) const noexcept override {
        auto it = headers_.find(std::string(key));
        if (it != headers_.end()) {
            return it->second;
        }
        return "";
    }

    void Set(opentelemetry::nostd::string_view /* key */,
             opentelemetry::nostd::string_view /* value */) noexcept override {
        // not needed for extraction
    }

private:
    const std::map<std::string, std::string> & headers_;
};

static const char * TRACER_NAME = "llama.cpp.server";

// Helper to get env var with a default
static std::string get_env(const char * name, const char * default_val) {
    const char * val = std::getenv(name);
    return (val && val[0]) ? std::string(val) : std::string(default_val);
}

otel_span::~otel_span() {
    // Ensure span is ended if not already
    if (span) {
        span->End();
    }
}

void otel_init() {
    // Configure the OTLP HTTP exporter
    otlp::OtlpHttpExporterOptions exporter_opts;
    // The SDK reads OTEL_EXPORTER_OTLP_ENDPOINT automatically via the options,
    // but we set it explicitly for clarity
    std::string endpoint = get_env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318");
    exporter_opts.url = endpoint + "/v1/traces";

    auto exporter = otlp::OtlpHttpExporterFactory::Create(exporter_opts);

    // BatchSpanProcessor for async export
    trace_sdk::BatchSpanProcessorOptions processor_opts;
    auto processor = trace_sdk::BatchSpanProcessorFactory::Create(std::move(exporter), processor_opts);

    // Resource with service name
    std::string service_name = get_env("OTEL_SERVICE_NAME", "llama-cpp-server");
    auto resource_attrs = resource::ResourceAttributes{
        {"service.name", service_name},
    };
    auto otel_resource = resource::Resource::Create(resource_attrs);

    // Create TracerProvider
    auto provider = trace_sdk::TracerProviderFactory::Create(std::move(processor), otel_resource);
    trace_api::Provider::SetTracerProvider(std::move(provider));

    // Set up W3C TraceContext propagator
    propagation::GlobalTextMapPropagator::SetGlobalPropagator(
        opentelemetry::nostd::shared_ptr<propagation::TextMapPropagator>(
            new trace_api::propagation::HttpTraceContext()));
}

void otel_shutdown() {
    auto provider = trace_api::Provider::GetTracerProvider();
    if (provider) {
        auto * sdk_provider = dynamic_cast<trace_sdk::TracerProvider *>(provider.get());
        if (sdk_provider) {
            sdk_provider->ForceFlush();
            sdk_provider->Shutdown();
        }
    }
}

std::unique_ptr<otel_span> otel_start_span(
        const std::string & span_name,
        const std::map<std::string, std::string> & headers) {
    auto provider = trace_api::Provider::GetTracerProvider();
    auto tracer = provider->GetTracer(TRACER_NAME);

    // Extract parent context from incoming headers (W3C traceparent)
    HeaderCarrier carrier(headers);
    auto propagator = propagation::GlobalTextMapPropagator::GetGlobalPropagator();
    auto parent_ctx = propagator->Extract(carrier, context::RuntimeContext::GetCurrent());

    // Start a SERVER span as a child of the extracted context
    trace_api::StartSpanOptions opts;
    opts.kind = trace_api::SpanKind::kServer;
    opts.parent = parent_ctx;

    auto span = tracer->StartSpan(span_name, {}, opts);
    auto scope = std::make_unique<trace_api::Scope>(span);

    auto result = std::make_unique<otel_span>();
    result->span = std::move(span);
    result->scope = std::move(scope);
    return result;
}

void otel_end_span(
        otel_span * span,
        const std::string & model,
        int32_t input_tokens,
        int32_t output_tokens,
        double prompt_ms,
        double predicted_ms,
        double prompt_per_second,
        double predicted_per_second,
        int32_t cache_tokens,
        const std::string & finish_reason,
        bool is_error,
        const std::string & error_message) {
    if (!span || !span->span) {
        return;
    }

    auto & s = span->span;

    // GenAI semantic convention attributes
    s->SetAttribute("gen_ai.operation.name", "chat");
    s->SetAttribute("gen_ai.request.model", model);
    s->SetAttribute("gen_ai.response.model", model);

    if (input_tokens >= 0) {
        s->SetAttribute("gen_ai.usage.input_tokens", static_cast<int64_t>(input_tokens));
    }
    if (output_tokens >= 0) {
        s->SetAttribute("gen_ai.usage.output_tokens", static_cast<int64_t>(output_tokens));
    }

    // llama.cpp specific timing attributes
    s->SetAttribute("llama.prompt_ms", prompt_ms);
    s->SetAttribute("llama.predicted_ms", predicted_ms);
    s->SetAttribute("llama.prompt_per_second", prompt_per_second);
    s->SetAttribute("llama.predicted_per_second", predicted_per_second);

    if (cache_tokens >= 0) {
        s->SetAttribute("llama.cache_tokens", static_cast<int64_t>(cache_tokens));
    }

    if (!finish_reason.empty()) {
        s->SetAttribute("gen_ai.response.finish_reasons", finish_reason);
    }

    if (is_error) {
        s->SetStatus(trace_api::StatusCode::kError, error_message);
    }

    s->End();
    // Prevent double-end in destructor
    span->span = nullptr;
}

#endif // LLAMA_OTEL

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
#include <opentelemetry/trace/span_startoptions.h>
#include <opentelemetry/trace/propagation/http_trace_context.h>
#include <opentelemetry/context/propagation/global_propagator.h>
#include <opentelemetry/context/propagation/text_map_propagator.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

namespace trace_api  = opentelemetry::trace;
namespace trace_sdk  = opentelemetry::sdk::trace;
namespace resource   = opentelemetry::sdk::resource;
namespace otlp       = opentelemetry::exporter::otlp;
namespace context    = opentelemetry::context;
namespace propagation = opentelemetry::context::propagation;

// Fix #1: Case-insensitive header lookup.
// HTTP headers are case-insensitive (RFC 7230). Store lowercased keys.
class HeaderCarrier : public propagation::TextMapCarrier {
public:
    explicit HeaderCarrier(const std::map<std::string, std::string> & headers) {
        for (const auto & h : headers) {
            std::string lower_key = h.first;
            std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            headers_[lower_key] = h.second;
        }
    }

    opentelemetry::nostd::string_view Get(opentelemetry::nostd::string_view key) const noexcept override {
        std::string lower_key(key);
        std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        auto it = headers_.find(lower_key);
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
    std::map<std::string, std::string> headers_;
};

static const char * TRACER_NAME = "llama.cpp.server";

// Helper to get env var with a default
static std::string get_env(const char * name, const char * default_val) {
    const char * val = std::getenv(name);
    return (val && val[0]) ? std::string(val) : std::string(default_val);
}

// Fix #3: Mark orphaned spans as errors in destructor
otel_span::~otel_span() {
    if (span) {
        span->SetStatus(trace_api::StatusCode::kError, "span ended without completion");
        span->End();
    }
}

// Fix #5: Let the SDK handle OTEL_EXPORTER_OTLP_ENDPOINT automatically.
// OtlpHttpExporterOptions reads the env var and appends /v1/traces itself.
// We only manually handle OTEL_SERVICE_NAME since the SDK uses Resource for that.
void otel_init() {
    otlp::OtlpHttpExporterOptions exporter_opts;
    // SDK reads OTEL_EXPORTER_OTLP_ENDPOINT from environment automatically

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

// Fix #10: No Scope created — it would attach to the creating thread's context
// but the span may be used from a different thread (streaming lambda).
// Since no child spans are created, Scope is unnecessary.
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

    auto result = std::make_unique<otel_span>();
    result->span = std::move(span);
    return result;
}

// Fix #6: struct-based attrs, #2: finish_reasons array, #7: gen_ai.system,
// #8: span name "chat {model}", #9: operation name from attrs
void otel_end_span(otel_span * span, const otel_span_attrs & attrs) {
    if (!span || !span->span) {
        return;
    }

    auto & s = span->span;

    // Fix #8: Update span name to "chat {model}" (or "{operation} {model}") per semconv
    s->UpdateName(attrs.operation_name + " " + attrs.model);

    // gen_ai.provider.name (renamed from gen_ai.system in semconv)
    s->SetAttribute("gen_ai.provider.name", "llama_cpp");

    // Fix #9: operation name from attrs (chat or infill)
    s->SetAttribute("gen_ai.operation.name", attrs.operation_name);
    s->SetAttribute("gen_ai.request.model", attrs.model);
    s->SetAttribute("gen_ai.response.model", attrs.model);

    if (attrs.input_tokens >= 0) {
        s->SetAttribute("gen_ai.usage.input_tokens", static_cast<int64_t>(attrs.input_tokens));
    }
    if (attrs.output_tokens >= 0) {
        s->SetAttribute("gen_ai.usage.output_tokens", static_cast<int64_t>(attrs.output_tokens));
    }

    // llama.cpp specific timing attributes
    s->SetAttribute("llama.prompt_ms", attrs.prompt_ms);
    s->SetAttribute("llama.predicted_ms", attrs.predicted_ms);
    s->SetAttribute("llama.prompt_per_second", attrs.prompt_per_second);
    s->SetAttribute("llama.predicted_per_second", attrs.predicted_per_second);

    if (attrs.cache_tokens >= 0) {
        s->SetAttribute("gen_ai.usage.cache_read.input_tokens", static_cast<int64_t>(attrs.cache_tokens));
    }

    // Fix #2: finish_reasons must be an array per semconv
    if (!attrs.finish_reason.empty()) {
        s->SetAttribute("gen_ai.response.finish_reasons",
                         std::vector<std::string>{attrs.finish_reason});
    }

    if (attrs.is_error) {
        s->SetStatus(trace_api::StatusCode::kError, attrs.error_message);
        if (!attrs.error_message.empty()) {
            s->SetAttribute("error.type", attrs.error_message);
        }
    }

    s->End();
    // Prevent double-end in destructor
    span->span = nullptr;
}

#endif // LLAMA_OTEL

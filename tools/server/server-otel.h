#pragma once

#ifdef LLAMA_OTEL

#include <string>
#include <map>
#include <memory>

// Forward declare OTel types to avoid header pollution
namespace opentelemetry {
namespace trace {
    class Span;
    class Scope;
}
}

// Opaque handle for an active OTel span + scope
struct otel_span {
    std::shared_ptr<opentelemetry::trace::Span> span;
    std::unique_ptr<opentelemetry::trace::Scope> scope;

    ~otel_span();
};

// Initialize the OTel TracerProvider + OTLP HTTP exporter.
// Reads standard env vars: OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME, etc.
// Call once at startup.
void otel_init();

// Shutdown the OTel SDK, flush remaining spans.
// Call once before exit.
void otel_shutdown();

// Start a server span, extracting W3C TraceContext from the request headers.
// Returns nullptr if OTel is not initialized.
std::unique_ptr<otel_span> otel_start_span(
    const std::string & span_name,
    const std::map<std::string, std::string> & headers);

// Add GenAI attributes to the span and end it.
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
    bool is_error = false,
    const std::string & error_message = "");

#endif // LLAMA_OTEL

#pragma once

#ifdef LLAMA_OTEL

#include <string>
#include <map>
#include <memory>

// Forward declare OTel types to avoid header pollution
namespace opentelemetry {
namespace trace {
    class Span;
}
}

// Attributes for ending a GenAI span
struct otel_span_attrs {
    std::string model;
    int32_t input_tokens  = -1;
    int32_t output_tokens = -1;
    double prompt_ms          = 0;
    double predicted_ms       = 0;
    double prompt_per_second  = 0;
    double predicted_per_second = 0;
    int32_t cache_tokens  = -1;
    std::string finish_reason;
    std::string operation_name = "chat";
    bool is_error = false;
    std::string error_message;
};

// Opaque handle for an active OTel span
struct otel_span {
    std::shared_ptr<opentelemetry::trace::Span> span;

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
void otel_end_span(otel_span * span, const otel_span_attrs & attrs);

#endif // LLAMA_OTEL

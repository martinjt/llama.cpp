# Building llama.cpp with OpenTelemetry support

Follow the [official opentelemetry-cpp getting started guide](https://opentelemetry.io/docs/languages/cpp/getting-started/)
for the recommended build approach: build opentelemetry-cpp from source, install to a
local prefix, and point llama.cpp's CMake at it.

## 1. Prerequisites

```bash
apt-get update
apt-get install -y cmake g++ git libcurl4-openssl-dev libprotobuf-dev protobuf-compiler
```

## 2. Build and install opentelemetry-cpp to a local prefix

```bash
cd /path/to/llama.cpp
git clone --depth 1 --branch v1.20.0 \
    https://github.com/open-telemetry/opentelemetry-cpp.git otel-cpp-src

cd otel-cpp-src
mkdir build && cd build
cmake -DBUILD_TESTING=OFF ..
cmake --build . --parallel $(nproc)
cmake --install . --prefix ../../otel-cpp
cd ../..
```

This installs opentelemetry-cpp to `otel-cpp/` alongside the llama.cpp source tree.

## 3. Build llama.cpp with OTel enabled

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_OTEL=ON \
    -Dopentelemetry-cpp_DIR=otel-cpp/lib/cmake/opentelemetry-cpp

cmake --build build --target llama-server --parallel $(nproc)
```

For ROCm/HIP (e.g. Strix Halo), add the GPU flags:

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_OTEL=ON \
    -Dopentelemetry-cpp_DIR=otel-cpp/lib/cmake/opentelemetry-cpp \
    -DGGML_HIP=ON \
    -DGGML_HIP_NO_VMM=ON

cmake --build build --target llama-server --parallel $(nproc)
```

## 4. Run with OTel enabled

Set the standard OpenTelemetry environment variables:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://your-collector:4318"
export OTEL_SERVICE_NAME="llama-server"

./build/bin/llama-server \
    -m /path/to/model.gguf \
    --host 0.0.0.0 --port 8080
```

You should see `OpenTelemetry tracing enabled` in the server log.

## 5. Verify

Send a request with a traceparent header and check your collector:

```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "traceparent: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01" \
    -d '{"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
```

## Disabling OTel

When built with `-DLLAMA_OTEL=OFF` (the default), all OTel code compiles to no-ops.
There is zero runtime overhead and no dependency on opentelemetry-cpp.

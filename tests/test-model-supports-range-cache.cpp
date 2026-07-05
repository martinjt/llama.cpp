#include "server-memory-detect.h"

#include "llama.h"
#include <cstdio>

// Verifies the allowlist dynamic_cast in model_supports_range_cache() against a real loaded
// model. The CI fixture (stories15M, a plain llama-arch model) only exercises the "eligible"
// side; see server-memory-detect.h for why a hybrid-model fixture is a documented gap rather
// than something faked here.
int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <plain-attention-model.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (model == nullptr) {
        fprintf(stderr, "FAIL: failed to load model %s\n", argv[1]);
        return 1;
    }
    llama_context_params cparams = llama_context_default_params();
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "FAIL: failed to create context\n");
        return 1;
    }

    if (!model_supports_range_cache(ctx)) {
        fprintf(stderr, "FAIL: expected a plain-attention model to be allowlisted for the range-chunk cache\n");
        return 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("OK: memory-type detection correctly allowlists a plain-attention model\n");
    return 0;
}

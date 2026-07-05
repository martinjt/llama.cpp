#include "server-memory-detect.h"

#include "llama.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

bool model_supports_range_cache(llama_context * ctx) {
    llama_memory_t mem = llama_get_memory(ctx);

    return dynamic_cast<llama_kv_cache *>(mem) != nullptr
        || dynamic_cast<llama_kv_cache_iswa *>(mem) != nullptr;
}

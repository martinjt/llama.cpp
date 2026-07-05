#include "server-chunk-hash.h"

#include <algorithm>

static uint64_t fnv1a_64_update(uint64_t hash, const uint8_t * data, size_t len) {
    constexpr uint64_t prime = 1099511628211ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= prime;
    }
    return hash;
}

static uint64_t fnv1a_64_update_u64(uint64_t hash, uint64_t value) {
    return fnv1a_64_update(hash, reinterpret_cast<const uint8_t *>(&value), sizeof(value));
}

std::vector<chunk_key_range> compute_chunk_chain(
    const std::vector<llama_token> & tokens,
    const std::string & weights_fingerprint,
    size_t chunk_size) {

    constexpr uint64_t fnv_offset_basis = 14695981039346656037ULL;

    std::vector<chunk_key_range> chain;
    uint64_t parent_hash = fnv1a_64_update(fnv_offset_basis,
        reinterpret_cast<const uint8_t *>(weights_fingerprint.data()), weights_fingerprint.size());

    for (size_t start = 0; start < tokens.size(); start += chunk_size) {
        const size_t n = std::min(chunk_size, tokens.size() - start);

        uint64_t h = fnv1a_64_update_u64(fnv_offset_basis, parent_hash);
        h = fnv1a_64_update(h,
            reinterpret_cast<const uint8_t *>(tokens.data() + start), n * sizeof(llama_token));

        chain.push_back(chunk_key_range{ weights_fingerprint, h, chain.size(), n });
        parent_hash = h;
    }

    return chain;
}

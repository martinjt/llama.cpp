#include "server-snapshot-hash.h"
#include <algorithm>

static uint64_t fnv1a_64_update(uint64_t hash, const uint8_t * data, size_t len) {
    constexpr uint64_t prime = 1099511628211ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= prime;
    }
    return hash;
}

uint64_t compute_snapshot_hash(const std::vector<llama_token> & tokens, size_t n, const std::string & weights_fingerprint) {
    constexpr uint64_t fnv_offset_basis = 14695981039346656037ULL;

    uint64_t hash = fnv1a_64_update(fnv_offset_basis,
        reinterpret_cast<const uint8_t *>(weights_fingerprint.data()), weights_fingerprint.size());

    const size_t clamped_n = std::min(n, tokens.size());
    hash = fnv1a_64_update(hash,
        reinterpret_cast<const uint8_t *>(tokens.data()), clamped_n * sizeof(llama_token));

    return hash;
}

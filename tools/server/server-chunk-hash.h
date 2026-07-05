#pragma once
#include "llama.h"
#include <cstdint>
#include <string>
#include <vector>

// One 256-token (by default) segment of the secondary range-chunk cache path's chunk chain.
// `chunk_hash` is chained from the previous chunk's hash (or the fingerprint, for chunk 0), so a
// hit at chunk N implies every ancestor chunk 0..N-1 also matched -- content-addressing the whole
// prefix without re-hashing it on every lookup.
struct chunk_key_range {
    std::string weights_fingerprint;
    uint64_t    chunk_hash;
    size_t      chunk_index;
    size_t      n_tokens;
};

std::vector<chunk_key_range> compute_chunk_chain(
    const std::vector<llama_token> & tokens,
    const std::string & weights_fingerprint,
    size_t chunk_size = 256);

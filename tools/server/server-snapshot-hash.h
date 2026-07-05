#pragma once
#include "llama.h"
#include <cstdint>
#include <string>
#include <vector>

// Hashes the first `n` tokens of `tokens`, folded with `weights_fingerprint`, into a stable
// 64-bit content hash. Two token sequences that share the same first n tokens and the same
// model always hash identically at that n -- this is the key a full-state snapshot (Task 12)
// is stored/looked-up under, distinct from the chained per-256-token-chunk hash the secondary
// range-chunk path (Task 13) uses.
uint64_t compute_snapshot_hash(const std::vector<llama_token> & tokens, size_t n, const std::string & weights_fingerprint);

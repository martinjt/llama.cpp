#include "server-snapshot-hash.h"
#include <cstdio>
#include <vector>

int main() {
    std::vector<llama_token> tokens_a;
    for (int i = 0; i < 100; ++i) tokens_a.push_back(i % 1000);

    const uint64_t h_a_50 = compute_snapshot_hash(tokens_a, 50, "fp1");
    const uint64_t h_a_50_again = compute_snapshot_hash(tokens_a, 50, "fp1");
    if (h_a_50 != h_a_50_again) {
        fprintf(stderr, "FAIL: h_a_50 != h_a_50_again (expected deterministic hash)\n");
        return 1;
    }

    const uint64_t h_a_60 = compute_snapshot_hash(tokens_a, 60, "fp1");
    if (h_a_50 == h_a_60) {
        fprintf(stderr, "FAIL: h_a_50 == h_a_60 (expected different prefix length to produce different hash)\n");
        return 1;
    }

    const uint64_t h_a_50_fp2 = compute_snapshot_hash(tokens_a, 50, "fp2");
    if (h_a_50 == h_a_50_fp2) {
        fprintf(stderr, "FAIL: h_a_50 == h_a_50_fp2 (expected different model fingerprint to produce different hash, same tokens)\n");
        return 1;
    }

    // A second sequence sharing the first 50 tokens with tokens_a must hash identically at n=50.
    std::vector<llama_token> tokens_b = tokens_a;
    tokens_b.resize(50);
    tokens_b.push_back(99999); // diverges only after position 50
    const uint64_t h_b_50 = compute_snapshot_hash(tokens_b, 50, "fp1");
    if (h_a_50 != h_b_50) {
        fprintf(stderr, "FAIL: h_a_50 != h_b_50 (expected shared 50-token prefix to hash identically)\n");
        return 1;
    }

    printf("OK: snapshot hashing is deterministic, prefix-length-sensitive, and model-sensitive\n");
    return 0;
}

#include "server-chunk-hash.h"
#include <cstdio>
#include <vector>

int main() {
    std::vector<llama_token> tokens;
    for (int i = 0; i < 600; ++i) tokens.push_back(i % 1000);

    // 600 tokens at chunk_size=256 -> chunks of 256, 256, 88
    const auto chain = compute_chunk_chain(tokens, "fp1", 256);
    if (chain.size() != 3) {
        fprintf(stderr, "FAIL: expected 3 chunks, got %zu\n", chain.size());
        return 1;
    }
    if (chain[0].n_tokens != 256 || chain[1].n_tokens != 256 || chain[2].n_tokens != 88) {
        fprintf(stderr, "FAIL: unexpected chunk sizes: %zu, %zu, %zu\n",
                chain[0].n_tokens, chain[1].n_tokens, chain[2].n_tokens);
        return 1;
    }
    if (chain[0].chunk_index != 0 || chain[1].chunk_index != 1 || chain[2].chunk_index != 2) {
        fprintf(stderr, "FAIL: unexpected chunk indices\n");
        return 1;
    }

    // deterministic: recomputing the same chain must produce identical hashes
    const auto chain_again = compute_chunk_chain(tokens, "fp1", 256);
    if (chain[0].chunk_hash != chain_again[0].chunk_hash || chain[1].chunk_hash != chain_again[1].chunk_hash) {
        fprintf(stderr, "FAIL: compute_chunk_chain is not deterministic\n");
        return 1;
    }

    // a second sequence sharing the first 256 tokens must produce the same chunk-0 hash, but a
    // different chunk-1 hash once it diverges (the chain must be sensitive to the CHAIN, not just
    // the chunk's own content, since chunk 1's hash folds in chunk 0's hash as its parent)
    std::vector<llama_token> tokens_b(tokens.begin(), tokens.begin() + 256);
    tokens_b.push_back(99999); // diverges only after position 256
    for (int i = 0; i < 255; ++i) tokens_b.push_back(i);
    const auto chain_b = compute_chunk_chain(tokens_b, "fp1", 256);
    if (chain_b[0].chunk_hash != chain[0].chunk_hash) {
        fprintf(stderr, "FAIL: shared first chunk must hash identically across sequences\n");
        return 1;
    }
    if (chain_b.size() > 1 && chain[1].n_tokens == chain_b[1].n_tokens && chain_b[1].chunk_hash == chain[1].chunk_hash) {
        fprintf(stderr, "FAIL: diverging chunk 1 content must produce a different hash\n");
        return 1;
    }

    // different model fingerprint must change chunk 0's hash even for identical tokens
    const auto chain_fp2 = compute_chunk_chain(tokens, "fp2", 256);
    if (chain_fp2[0].chunk_hash == chain[0].chunk_hash) {
        fprintf(stderr, "FAIL: different weights_fingerprint must produce a different chunk-0 hash\n");
        return 1;
    }

    // an unrelated chunk 0 (different content) chained forward must NOT collide with a chain that
    // happens to have identical chunk-1 raw tokens but a different parent -- this proves the hash
    // is chained (folds in parent_hash) rather than being an independent per-chunk hash.
    std::vector<llama_token> tokens_c;
    for (int i = 0; i < 256; ++i) tokens_c.push_back(i + 1); // different chunk 0 content
    for (int i = 256; i < 512; ++i) tokens_c.push_back(tokens[i]); // identical chunk 1 content to `tokens`
    const auto chain_c = compute_chunk_chain(tokens_c, "fp1", 256);
    if (chain_c[1].chunk_hash == chain[1].chunk_hash) {
        fprintf(stderr, "FAIL: identical chunk-1 content under a different parent chunk must hash differently (chain not actually chained)\n");
        return 1;
    }

    // empty token list -> empty chain
    if (!compute_chunk_chain({}, "fp1", 256).empty()) {
        fprintf(stderr, "FAIL: expected empty chain for empty token list\n");
        return 1;
    }

    printf("OK: chunk-chain hashing is deterministic, chained, prefix-sharing, and model-sensitive\n");
    return 0;
}

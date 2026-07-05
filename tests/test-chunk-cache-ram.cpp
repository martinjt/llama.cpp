#include "server-chunk-cache.h"
#include <cassert>
#include <cstdio>

int main() {
    auto backend = make_ram_chunk_cache_backend(/*limit_mib=*/1);

    chunk_key k1{ "fp1", 111 };
    std::vector<uint8_t> data_out;

    assert(!backend->get(k1, data_out)); // miss on empty cache

    std::vector<uint8_t> data_in(1024, 0x42);
    backend->put(k1, data_in);

    assert(backend->get(k1, data_out));
    assert(data_out == data_in);

    // Different content_hash -> different key -> miss.
    chunk_key k2{ "fp1", 222 };
    assert(!backend->get(k2, data_out));

    // Eviction: push enough data past the 1 MiB limit, confirm the oldest entry is evicted (LRU).
    std::vector<uint8_t> big(600 * 1024, 0x01);
    for (int i = 0; i < 5; ++i) {
        chunk_key ki{ "fp1", (uint64_t) (1000 + i) };
        backend->put(ki, big);
    }
    // k1 (the first thing inserted) should have been evicted by now.
    assert(!backend->get(k1, data_out));

    printf("OK: RAM chunk cache backend get/put/evict behave correctly\n");
    return 0;
}

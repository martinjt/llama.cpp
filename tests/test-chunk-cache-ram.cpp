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

    // --- Distinguish true LRU from plain FIFO ---
    //
    // The eviction test above only proves "the oldest *inserted* entry gets evicted first",
    // which a pure FIFO cache (no recency tracking) would also satisfy. To prove real LRU
    // semantics, we need a case where an entry is `get()`-refreshed *after* insertion but
    // *before* another, later-inserted entry -- and confirm the refreshed one survives while
    // the never-refreshed one gets evicted, even though the refreshed one was inserted first.
    {
        auto lru_backend = make_ram_chunk_cache_backend(/*limit_mib=*/1);

        // 3 entries of this size = 1200 KiB > 1 MiB limit, but any 2 fit comfortably (800 KiB).
        const size_t chunk_size = 400 * 1024;
        std::vector<uint8_t> blob_a(chunk_size, 0xAA);
        std::vector<uint8_t> blob_b(chunk_size, 0xBB);
        std::vector<uint8_t> blob_c(chunk_size, 0xCC);

        chunk_key ka{ "fp2", 1 };
        chunk_key kb{ "fp2", 2 };
        chunk_key kc{ "fp2", 3 };

        lru_backend->put(ka, blob_a); // total = 400 KiB
        lru_backend->put(kb, blob_b); // total = 800 KiB -- still under the 1 MiB limit, no eviction yet

        // Refresh `a`'s recency by reading it. Under real LRU this splices `a` to the
        // most-recently-used end, ahead of `b`, even though `a` was inserted first.
        std::vector<uint8_t> refreshed;
        assert(lru_backend->get(ka, refreshed));
        assert(refreshed == blob_a);

        // Insert a third entry: 800 KiB + 400 KiB = 1200 KiB > 1 MiB limit, forcing exactly one
        // eviction (removing one 400 KiB entry brings total back to 800 KiB, under the limit).
        lru_backend->put(kc, blob_c);

        std::vector<uint8_t> out;
        // Real LRU: `b` was never refreshed, so it's now the least-recently-used entry and gets
        // evicted first -- despite being inserted *after* `a`.
        assert(!lru_backend->get(kb, out));
        // `a` was refreshed via get() above, so it must survive this eviction even though it was
        // inserted before `b`. A plain FIFO cache (no splice-on-hit) would evict `a` here instead,
        // since `a` was the oldest *inserted* entry -- this pair of assertions is what actually
        // catches that regression.
        assert(lru_backend->get(ka, out));
        assert(out == blob_a);
        // `c`, the newest entry, must also still be present.
        assert(lru_backend->get(kc, out));
        assert(out == blob_c);

        printf("OK: RAM chunk cache backend distinguishes true LRU from FIFO eviction\n");
    }

    return 0;
}

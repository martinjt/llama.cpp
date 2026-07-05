#include "server-chunk-cache.h"
#include <cstdio>
#include <cstdlib>

int main() {
    const char * host = getenv("LMCACHE_TEST_HOST") ? getenv("LMCACHE_TEST_HOST") : "127.0.0.1";
    const int port = getenv("LMCACHE_TEST_PORT") ? atoi(getenv("LMCACHE_TEST_PORT")) : 65432;

    auto backend = make_lmcache_chunk_cache_backend(host, port);

    // A key that shouldn't already exist on the server (chosen distinct from the round-trip
    // key below to avoid any cross-run collision on a shared/persistent server instance).
    chunk_key k1{ "fp1_miss", 111 };
    std::vector<uint8_t> data_out;
    if (backend->get(k1, data_out)) {
        fprintf(stderr, "FAIL: backend->get(k1, ...) hit on a key that was never put() (expected a miss)\n");
        return 1;
    }

    chunk_key k2{ "fp1", 222 };
    std::vector<uint8_t> data_in(1024, 0x42);
    backend->put(k2, data_in);

    if (!backend->get(k2, data_out)) {
        fprintf(stderr, "FAIL: backend->get(k2, ...) missed right after put (expected a hit)\n");
        return 1;
    }
    if (data_out != data_in) {
        fprintf(stderr, "FAIL: data_out != data_in (expected get() to return the exact bytes that were put())\n");
        return 1;
    }

    printf("OK: LMCache TCP client backend round-trips real payloads through a live lmcache server\n");
    return 0;
}

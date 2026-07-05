#include "server-chunk-cache.h"
#include <cstdio>
#include <cstdlib>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

// Standalone connectivity probe: opens a raw TCP socket to host:port and immediately
// closes it. Used only to distinguish "no lmcache_server reachable at all" (skip) from
// "server reachable but protocol/logic is broken" (hard fail) -- deliberately does not
// reuse chunk_cache_backend::get()/put(), whose internal failure paths conflate both
// cases into a single `false`/no-op return.
static bool lmcache_server_reachable(const char * host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return false;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t) port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        close(fd);
        return false;
    }
    bool ok = connect(fd, (sockaddr *) &addr, sizeof(addr)) == 0;
    close(fd);
    return ok;
}

int main() {
    const char * host = getenv("LMCACHE_TEST_HOST") ? getenv("LMCACHE_TEST_HOST") : "127.0.0.1";
    const int port = getenv("LMCACHE_TEST_PORT") ? atoi(getenv("LMCACHE_TEST_PORT")) : 65432;

    if (!lmcache_server_reachable(host, port)) {
        printf("SKIP: no lmcache_server reachable at %s:%d (set LMCACHE_TEST_HOST/LMCACHE_TEST_PORT, "
               "or start `lmcache_server %s %d` to run this test for real)\n", host, port, host, port);
        exit(EXIT_SUCCESS);
    }

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

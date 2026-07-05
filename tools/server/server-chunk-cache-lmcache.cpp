#include "server-chunk-cache.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <sstream>

// Wire protocol verified directly against LMCache's own source
// (lmcache/v1/protocol.py, lmcache/v1/server/__main__.py, lmcache/utils.py at
// https://github.com/LMCache/LMCache, commit current as of this writing) -- see
// tools/server/tests/... task report for the verification transcript against a live
// `lmcache_server` process. Request and response are two DIFFERENT struct shapes, not
// one shared struct: the request carries a 150-byte key and no `location` trailer, the
// response has no key at all but does have a trailing `location` field.

namespace {

// ClientMetaMessage: struct.pack("iiiiiiiii150s", command, length, fmt, dtype, location,
// shape0, shape1, shape2, shape3, key) -- 186 bytes total.
constexpr int k_max_key_length = 150;

// packed: Python's struct.pack has no trailing struct-alignment padding, but a plain
// C++ struct would add 2 bytes after the 150-byte char[] to round the whole struct up
// to a multiple of alignof(int32_t) -- that would silently desync the wire format.
struct __attribute__((packed)) lmcache_request {
    int32_t command;
    int32_t length;
    int32_t fmt;
    int32_t dtype;
    int32_t location;
    int32_t shape[4];
    char    key[k_max_key_length];
};
static_assert(sizeof(lmcache_request) == 4 * 9 + k_max_key_length, "lmcache_request must match ClientMetaMessage's packed layout");

// ServerMetaMessage: struct.pack("iiiiiiiii", code, length, fmt, dtype, shape0..3, location)
// -- 36 bytes, no key field.
struct lmcache_response {
    int32_t code;
    int32_t length;
    int32_t fmt;
    int32_t dtype;
    int32_t shape[4];
    int32_t location;
};
static_assert(sizeof(lmcache_response) == 4 * 9, "lmcache_response must match ServerMetaMessage's packed layout");

// lmcache.v1.protocol.ClientCommand (IntEnum, auto() starting at 1)
constexpr int32_t k_cmd_put    = 1;
constexpr int32_t k_cmd_get    = 2;

// lmcache.v1.protocol.ServerReturnCode
constexpr int32_t k_code_success = 200;

// lmcache.v1.memory_management.MemoryFormat.BINARY_BUFFER (UNDEFINED=0, KV_2LTD=1,
// KV_T2D=2, KV_2TD=3, BINARY=4, BINARY_BUFFER=5)
constexpr int32_t k_fmt_binary_buffer = 5;

// lmcache.v1.protocol.DTYPE_TO_INT[torch.uint8]
constexpr int32_t k_dtype_uint8 = 6;

// lmcache.v1.protocol.LOCATION_TO_INT[None]
constexpr int32_t k_location_none = 0;

int connect_to(const std::string & host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t) port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        close(fd);
        return -1;
    }
    if (connect(fd, (sockaddr *) &addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

bool send_all(int fd, const void * buf, size_t len) {
    const uint8_t * p = (const uint8_t *) buf;
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, p + sent, len - sent, 0);
        if (n <= 0) {
            return false;
        }
        sent += (size_t) n;
    }
    return true;
}

bool recv_all(int fd, void * buf, size_t len) {
    uint8_t * p = (uint8_t *) buf;
    size_t got = 0;
    while (got < len) {
        ssize_t n = recv(fd, p + got, len - got, 0);
        if (n <= 0) {
            return false;
        }
        got += (size_t) n;
    }
    return true;
}

// Mirrors LMCache's own CacheEngineKey.to_string(): "model_name@world_size@worker_id@
// <hex chunk_hash, unpadded>@<dtype str>". Using this exact format (rather than an
// arbitrary bespoke one) keeps entries inspectable/interoperable with real LMCache
// tooling pointed at the same server. world_size/worker_id are always 0 here -- this
// client always speaks as a single, non-sharded worker.
std::string key_to_lmcache_key(const chunk_key & key) {
    std::ostringstream oss;
    oss << key.weights_fingerprint << "@0@0@" << std::hex << key.content_hash << "@uint8";
    return oss.str();
}

lmcache_request make_request(int32_t command, const chunk_key & key, int32_t length) {
    lmcache_request req{};
    req.command  = command;
    req.length   = length;
    req.fmt      = k_fmt_binary_buffer;
    req.dtype    = k_dtype_uint8;
    req.location = k_location_none;
    req.shape[0] = length;
    req.shape[1] = 0;
    req.shape[2] = 0;
    req.shape[3] = 0;
    const std::string k = key_to_lmcache_key(key);
    // Space-padded, not NUL-padded: the real server does `key.decode().strip()`, which
    // strips whitespace but leaves embedded NUL bytes in the string -- a NUL-padded key
    // (e.g. from zero-initializing the struct) fails to match on the server side with a
    // KeyError, since the trailing NULs never get stripped. Mirrors the real Python
    // client's `key_str.encode().ljust(MAX_KEY_LENGTH)`, whose fillchar is a space.
    memset(req.key, ' ', sizeof(req.key));
    if (k.size() >= sizeof(req.key)) {
        fprintf(stderr, "lmcache_chunk_cache_backend: key too long for wire format (%zu >= %zu)\n", k.size(), sizeof(req.key));
    } else {
        memcpy(req.key, k.data(), k.size());
    }
    return req;
}

class lmcache_chunk_cache_backend : public chunk_cache_backend {
public:
    lmcache_chunk_cache_backend(std::string host, int port) : host(std::move(host)), port(port) {}

    bool get(const chunk_key & key, std::vector<uint8_t> & data) override {
        int fd = connect_to(host, port);
        if (fd < 0) {
            return false;
        }

        lmcache_request req = make_request(k_cmd_get, key, 0);
        if (!send_all(fd, &req, sizeof(req))) {
            close(fd);
            return false;
        }

        lmcache_response resp{};
        if (!recv_all(fd, &resp, sizeof(resp)) || resp.code != k_code_success || resp.length <= 0) {
            close(fd);
            return false;
        }

        data.resize((size_t) resp.length);
        bool ok = recv_all(fd, data.data(), (size_t) resp.length);
        close(fd);
        if (!ok) {
            data.clear();
            return false;
        }
        return true;
    }

    void put(const chunk_key & key, std::vector<uint8_t> data) override {
        int fd = connect_to(host, port);
        if (fd < 0) {
            return;
        }

        lmcache_request req = make_request(k_cmd_put, key, (int32_t) data.size());
        if (!send_all(fd, &req, sizeof(req))) {
            close(fd);
            return;
        }
        // No server reply follows a PUT (see lmcache/v1/server/__main__.py's
        // handle_client -- the PUT case never calls sendall back to the client).
        if (!data.empty()) {
            send_all(fd, data.data(), data.size());
        }
        close(fd);
    }

private:
    std::string host;
    int port;
};

} // namespace

std::unique_ptr<chunk_cache_backend> make_lmcache_chunk_cache_backend(const std::string & host, int port) {
    return std::make_unique<lmcache_chunk_cache_backend>(host, port);
}

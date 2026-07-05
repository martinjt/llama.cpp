#include "weights-fingerprint.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <sstream>
#include <iomanip>

static uint64_t fnv1a_64(const uint8_t * data, size_t len, uint64_t hash) {
    constexpr uint64_t prime = 1099511628211ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= prime;
    }
    return hash;
}

std::string common_weights_fingerprint(const std::string & model_path) {
    constexpr uint64_t fnv_offset_basis = 14695981039346656037ULL;

    FILE * f = fopen(model_path.c_str(), "rb");
    if (!f) {
        return "";
    }

    uint64_t hash = fnv_offset_basis;
    std::vector<uint8_t> buf(1 << 20); // 1 MiB chunks
    size_t nread;
    while ((nread = fread(buf.data(), 1, buf.size(), f)) > 0) {
        hash = fnv1a_64(buf.data(), nread, hash);
    }
    fclose(f);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return oss.str();
}

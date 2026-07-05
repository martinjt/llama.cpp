#include "weights-fingerprint.h"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <cstdlib>

int main() {
    // Write two small temp files: identical content, and one differing by a single byte.
    const char * path_a = "/tmp/fp_test_a.bin";
    const char * path_b = "/tmp/fp_test_b.bin";
    const char * path_c = "/tmp/fp_test_c.bin"; // same content as a

    {
        std::ofstream fa(path_a, std::ios::binary);
        std::ofstream fb(path_b, std::ios::binary);
        std::ofstream fc(path_c, std::ios::binary);
        for (int i = 0; i < 10000; ++i) {
            fa.put((char) (i % 256));
            fc.put((char) (i % 256));
            fb.put((char) ((i == 5000 ? i + 1 : i) % 256));
        }
    }

    const std::string fp_a = common_weights_fingerprint(path_a);
    const std::string fp_b = common_weights_fingerprint(path_b);
    const std::string fp_c = common_weights_fingerprint(path_c);

    assert(!fp_a.empty());
    assert(fp_a == fp_c);   // identical content -> identical fingerprint
    assert(fp_a != fp_b);   // one differing byte -> different fingerprint

    std::remove(path_a);
    std::remove(path_b);
    std::remove(path_c);

    printf("OK: weights fingerprint is stable and content-sensitive\n");
    return 0;
}

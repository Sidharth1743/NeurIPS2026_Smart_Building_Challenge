/* Versioned, checksummed full-training snapshot stream. Host I/O only; CUDA
 * allocations are copied in bounded chunks after all streams have quiesced. */
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace sbfull {
constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
constexpr uint64_t FNV_PRIME = 1099511628211ULL;
static uint64_t hash_add(uint64_t h, const void* ptr, size_t n) {
    const unsigned char* p = (const unsigned char*)ptr;
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * FNV_PRIME;
    return h;
}
struct Record {
    char name[64];
    uint64_t bytes;
    uint64_t hash;
};
static void fail(const char* why, const char* name = "") {
    fprintf(stderr, "full checkpoint: %s %s\n", why, name);
    abort();
}
static void ck(cudaError_t e) {
    if (e != cudaSuccess) {
        fprintf(stderr, "full checkpoint CUDA error: %s\n", cudaGetErrorString(e));
        abort();
    }
}
struct Stream {
    FILE* file;
    bool reading;
    std::vector<unsigned char> chunk;
    std::vector<unsigned char> readback;
    Stream(FILE* f, bool r): file(f), reading(r), chunk(4 * 1024 * 1024), readback(r ? 4 * 1024 * 1024 : 0) {}
    void block(const char* name, void* ptr, uint64_t bytes, bool device) {
        if (!name || strlen(name) >= sizeof(Record::name) || (!ptr && bytes)) fail("bad section", name);
        Record rec{};
        long pos = ftell(file);
        if (pos < 0) fail("ftell failed", name);
        if (reading) {
            if (fread(&rec, sizeof(rec), 1, file) != 1 ||
                memchr(rec.name, 0, sizeof(rec.name)) == nullptr ||
                strcmp(rec.name, name) || rec.bytes != bytes) fail("section mismatch", name);
        } else {
            snprintf(rec.name, sizeof(rec.name), "%s", name);
            rec.bytes = bytes;
            if (fwrite(&rec, sizeof(rec), 1, file) != 1) fail("write header failed", name);
        }
        uint64_t hash = FNV_OFFSET;
        for (uint64_t off = 0; off < bytes;) {
            size_t n = (size_t)std::min<uint64_t>(chunk.size(), bytes - off);
            if (reading) {
                if (fread(chunk.data(), 1, n, file) != n) fail("truncated section", name);
                hash = hash_add(hash, chunk.data(), n);
                if (device) {
                    ck(cudaMemcpy((char*)ptr + off, chunk.data(), n, cudaMemcpyHostToDevice));
                    ck(cudaMemcpy(readback.data(), (char*)ptr + off, n, cudaMemcpyDeviceToHost));
                    if (memcmp(chunk.data(),readback.data(),n)) fail("device restore verification failed",name);
                }
                else memcpy((char*)ptr + off, chunk.data(), n);
            } else {
                if (device) ck(cudaMemcpy(chunk.data(), (char*)ptr + off, n, cudaMemcpyDeviceToHost));
                else memcpy(chunk.data(), (char*)ptr + off, n);
                hash = hash_add(hash, chunk.data(), n);
                if (fwrite(chunk.data(), 1, n, file) != n) fail("write failed", name);
            }
            off += n;
        }
        if (reading) {
            if (rec.hash != hash) fail("checksum mismatch", name);
        } else {
            long end = ftell(file);
            if (end < 0 || fseek(file, pos, SEEK_SET)) fail("seek failed", name);
            rec.hash = hash;
            if (fwrite(&rec, sizeof(rec), 1, file) != 1 || fseek(file, end, SEEK_SET))
                fail("record finalization failed", name);
        }
    }
    template<typename T> void host(const char* name, T* p, size_t count = 1) {
        block(name, p, (uint64_t)count * sizeof(T), false);
    }
    template<typename T> void device(const char* name, T* p, size_t count) {
        block(name, p, (uint64_t)count * sizeof(T), true);
    }
};
/* Check every record and the complete-file terminator before restoring any
 * device state. The typed reader subsequently also checks schema and sizes. */
static void preflight(FILE* f, long start) {
    if (fseek(f, start, SEEK_SET)) fail("preflight seek failed");
    std::vector<unsigned char> buf(4 * 1024 * 1024);
    while (true) {
        Record rec{};
        if (fread(&rec, sizeof(rec), 1, f) != 1 ||
            memchr(rec.name, 0, sizeof(rec.name)) == nullptr) fail("missing record or terminator");
        if (!strcmp(rec.name, "END")) {
            if (rec.bytes || rec.hash != FNV_OFFSET || fgetc(f) != EOF) fail("invalid terminator");
            break;
        }
        uint64_t hash = FNV_OFFSET;
        for (uint64_t left = rec.bytes; left;) {
            size_t n = (size_t)std::min<uint64_t>(buf.size(), left);
            if (fread(buf.data(), 1, n, f) != n) fail("truncated checkpoint", rec.name);
            hash = hash_add(hash, buf.data(), n);
            left -= n;
        }
        if (hash != rec.hash) fail("checksum mismatch", rec.name);
    }
    if (fseek(f, start, SEEK_SET)) fail("preflight rewind failed");
}
}

/* SBB1 blob loader (mirrors pysim/sbb.py). mmap-backed so a big blob costs no
 * resident heap. Header-only-friendly: declarations here, impl in sb_baked.c. */
#ifndef SB_BAKED_H
#define SB_BAKED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SB_MAGIC "SBB1"
#define SB_VERSION 1u

enum {
    SB_F64 = 0, SB_F32 = 1, SB_I64 = 2, SB_I32 = 3,
    SB_I8 = 4, SB_U8 = 5, SB_U32 = 6, SB_STR = 7
};

typedef struct {
    const char* name;      /* points into the names blob (NUL-terminated) */
    uint32_t dtype;
    uint32_t ndim;
    uint64_t shape[4];
    uint64_t count;        /* product of shape */
    uint64_t data_off;     /* byte offset into the mapping */
} SbSection;

typedef struct {
    void* map;             /* mmap base */
    size_t size;
    const char* names;     /* names blob base */
    uint64_t names_len;
    int n_sec;
    SbSection* sec;        /* parsed directory */
} SbBlob;

int   sb_open(SbBlob* b, const char* path);   /* 0 ok, -1 error */
void  sb_close(SbBlob* b);

const SbSection* sb_find(const SbBlob* b, const char* name);
const void*      sb_data(const SbBlob* b, const SbSection* s);

/* Typed getters: assert-free, return NULL if missing/typed wrong. */
const double*  sb_f64(const SbBlob* b, const char* name);
const float*   sb_f32(const SbBlob* b, const char* name);
const int64_t* sb_i64(const SbBlob* b, const char* name);
const int32_t* sb_i32(const SbBlob* b, const char* name);
const uint8_t* sb_u8(const SbBlob* b, const char* name);
const uint32_t* sb_u32(const SbBlob* b, const char* name);
/* str sections: raw bytes, not NUL-terminated; returns base and sets *len. */
const char*    sb_str(const SbBlob* b, const char* name, uint64_t* len);

/* Scalar convenience (first element of a f64/i64 section). */
double  sb_f64_1(const SbBlob* b, const char* name, double dflt);
int64_t sb_i64_1(const SbBlob* b, const char* name, int64_t dflt);

/* Splits a str section into `out` C strings IN PLACE (mutates names copy).
 * Returns count. `buf` must be writable and hold a NUL-terminated copy. */
int sb_split_lines(char* buf, size_t n, char** out, int max_out);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* SB_BAKED_H */

#include "sb_baked.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* Little-endian readers (portable via memcpy). */
static uint32_t rd_u32(const uint8_t* p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}
static uint64_t rd_u64(const uint8_t* p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

int sb_open(SbBlob* b, const char* path) {
    memset(b, 0, sizeof(*b));
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "sb_open: cannot open %s\n", path);
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }
    size_t size = (size_t)st.st_size;
    void* map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        fprintf(stderr, "sb_open: mmap failed for %s\n", path);
        return -1;
    }
    const uint8_t* m = (const uint8_t*)map;
    if (size < 52 || memcmp(m, SB_MAGIC, 4) != 0) {
        fprintf(stderr, "sb_open: bad magic in %s\n", path);
        munmap(map, size);
        return -1;
    }
    uint32_t version = rd_u32(m + 4);
    if (version != SB_VERSION) {
        fprintf(stderr, "sb_open: version %u != %u\n", version, SB_VERSION);
        munmap(map, size);
        return -1;
    }
    uint32_t n_sec = rd_u32(m + 8);
    uint64_t dir_off = rd_u64(m + 12);
    uint64_t names_off = rd_u64(m + 20);
    uint64_t names_len = rd_u64(m + 28);

    b->map = map;
    b->size = size;
    b->names = (const char*)(m + names_off);
    b->names_len = names_len;
    b->n_sec = (int)n_sec;
    b->sec = (SbSection*)calloc(n_sec ? n_sec : 1, sizeof(SbSection));

    for (uint32_t i = 0; i < n_sec; i++) {
        const uint8_t* s = m + dir_off + (size_t)i * 56;
        uint32_t name_off = rd_u32(s + 0);
        SbSection* sec = &b->sec[i];
        sec->dtype = rd_u32(s + 4);
        sec->ndim = rd_u32(s + 8);
        for (int k = 0; k < 4; k++) sec->shape[k] = rd_u64(s + 16 + 8 * k);
        sec->data_off = rd_u64(s + 48);
        sec->count = 1;
        for (uint32_t k = 0; k < sec->ndim; k++) sec->count *= sec->shape[k];
        if (sec->ndim == 0) sec->count = 1;
        sec->name = b->names + name_off;
    }
    return 0;
}

void sb_close(SbBlob* b) {
    if (b->sec) free(b->sec);
    if (b->map && b->map != MAP_FAILED) munmap(b->map, b->size);
    memset(b, 0, sizeof(*b));
}

const SbSection* sb_find(const SbBlob* b, const char* name) {
    for (int i = 0; i < b->n_sec; i++)
        if (strcmp(b->sec[i].name, name) == 0) return &b->sec[i];
    return NULL;
}

const void* sb_data(const SbBlob* b, const SbSection* s) {
    return (const uint8_t*)b->map + s->data_off;
}

static const void* get(const SbBlob* b, const char* name, uint32_t dt) {
    const SbSection* s = sb_find(b, name);
    if (!s || s->dtype != dt) return NULL;
    return sb_data(b, s);
}

const double*   sb_f64(const SbBlob* b, const char* n) { return (const double*)get(b, n, SB_F64); }
const float*    sb_f32(const SbBlob* b, const char* n) { return (const float*)get(b, n, SB_F32); }
const int64_t*  sb_i64(const SbBlob* b, const char* n) { return (const int64_t*)get(b, n, SB_I64); }
const int32_t*  sb_i32(const SbBlob* b, const char* n) { return (const int32_t*)get(b, n, SB_I32); }
const uint8_t*  sb_u8(const SbBlob* b, const char* n) { return (const uint8_t*)get(b, n, SB_U8); }
const uint32_t* sb_u32(const SbBlob* b, const char* n) { return (const uint32_t*)get(b, n, SB_U32); }

const char* sb_str(const SbBlob* b, const char* name, uint64_t* len) {
    const SbSection* s = sb_find(b, name);
    if (!s || s->dtype != SB_STR) { if (len) *len = 0; return NULL; }
    if (len) *len = s->count;
    return (const char*)sb_data(b, s);
}

double sb_f64_1(const SbBlob* b, const char* name, double dflt) {
    const double* p = sb_f64(b, name);
    return p ? p[0] : dflt;
}
int64_t sb_i64_1(const SbBlob* b, const char* name, int64_t dflt) {
    const int64_t* p = sb_i64(b, name);
    return p ? p[0] : dflt;
}

int sb_split_lines(char* buf, size_t n, char** out, int max_out) {
    int cnt = 0;
    if (n == 0) return 0;
    char* p = buf;
    char* end = buf + n;
    /* names blob is newline-separated with no trailing newline */
    out[cnt++] = p;
    while (p < end && cnt < max_out) {
        if (*p == '\n') { *p = '\0'; if (p + 1 < end) out[cnt++] = p + 1; }
        p++;
    }
    return cnt;
}

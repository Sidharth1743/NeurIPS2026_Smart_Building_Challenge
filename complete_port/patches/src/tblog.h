/* Minimal TensorBoard event writer: TFRecord + Event/Summary/SimpleValue
 * protos, dependency-free (no protobuf, no network). One scalar per
 * record; tags mirror the trainer log-dict keys ("env/perf",
 * "loss/policy", ...). View with: tensorboard --logdir <log_dir>/tb
 *
 * Layout per record: [u64 len][u32 masked-crc32c(len)][Event bytes]
 * [u32 masked-crc32c(Event)]. Event{1: wall_time fixed64, 2: step
 * varint, 5: Summary len}; Summary{1: repeated Value len}; Value{1: tag
 * string len, 2: simple_value fixed32}. (Field numbers from TB 2.21's
 * compiled Event descriptor -- verified programmatically, not by hand:
 * Event.summary = 5 (9 is meta_graph_def), Value.tag = 1,
 * Value.simple_value = 2, Value.tensor = 8.)
 */
#pragma once

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    FILE* fp;
} Tblog;

static uint32_t tblog_crc_table[256];
static int tblog_crc_ready = 0;

static void tblog_crc_init(void) {
    if (tblog_crc_ready)
        return;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++)
            c = (c & 1) ? (0x82F63B78u ^ (c >> 1)) : (c >> 1);
        tblog_crc_table[i] = c;
    }
    tblog_crc_ready = 1;
}

static uint32_t tblog_crc32c(const unsigned char* p, size_t n) {
    tblog_crc_init();
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++)
        c = tblog_crc_table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

static uint32_t tblog_masked(uint32_t c) {
    return ((c >> 15) | (c << 17)) + 0xA282EAD8u;
}

static unsigned char* tblog_varint(unsigned char* o, uint64_t v) {
    while (v >= 0x80) {
        *o++ = (unsigned char)(0x80 | (v & 0x7F));
        v >>= 7;
    }
    *o++ = (unsigned char)v;
    return o;
}

static void tblog_u64le(unsigned char* o, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        o[i] = (unsigned char)(v & 0xFF);
        v >>= 8;
    }
}

static void tblog_u32le(unsigned char* o, uint32_t v) {
    for (int i = 0; i < 4; i++) {
        o[i] = (unsigned char)(v & 0xFF);
        v >>= 8;
    }
}

static void tblog_f64le(unsigned char* o, double v) {
    uint64_t u;
    memcpy(&u, &v, 8);
    tblog_u64le(o, u);
}

static void tblog_f32le(unsigned char* o, float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    tblog_u32le(o, u);
}

/* Open <log_dir>/tb/events.out.tfevents.<sec>.<pid> (creates tb/). */
static int tblog_open(Tblog* tb, const char* log_dir) {
    char dir[4096], path[4096];
    snprintf(dir, sizeof(dir), "%s/tb", log_dir);
    if (mkdir(dir, 0755) != 0 && errno != EEXIST)
        return -1;
    snprintf(path, sizeof(path), "%s/events.out.tfevents.%lld.%d", dir,
             (long long)time(NULL), (int)getpid());
    tb->fp = fopen(path, "wb");
    if (!tb->fp)
        return -1;
    /* File-version first event (like TF writers do). */
    unsigned char ev[32], *o = ev;
    *o++ = 0x09; /* field 1, fixed64: wall_time */
    tblog_f64le(o, (double)time(NULL));
    o += 8;
    *o++ = 0x10; /* field 2, varint: step */
    *o++ = 0x00;
    *o++ = 0x1A; /* field 3, len-delim: file_version */
    const char* ver = "brain.Event:2";
    o = tblog_varint(o, 13);
    memcpy(o, ver, 13);
    o += 13;
    size_t n = (size_t)(o - ev);
    unsigned char hdr[12];
    tblog_u64le(hdr, (uint64_t)n);
    tblog_u32le(hdr + 8, tblog_masked(tblog_crc32c(hdr, 8)));
    unsigned char ftr[4];
    tblog_u32le(ftr, tblog_masked(tblog_crc32c(ev, n)));
    fwrite(hdr, 1, 12, tb->fp);
    fwrite(ev, 1, n, tb->fp);
    fwrite(ftr, 1, 4, tb->fp);
    return 0;
}

static void tblog_scalar(Tblog* tb, const char* tag, int64_t step,
        double wall_time, float value) {
    if (!tb->fp || !isfinite(value))
        return;
    size_t tagn = strlen(tag);
    if (tagn == 0 || tagn > 120)
        return;
    /* Value{1: tag, 2: simple_value} */
    unsigned char val[160], *o = val;
    *o++ = 0x0A;
    o = tblog_varint(o, (uint64_t)tagn);
    memcpy(o, tag, tagn);
    o += tagn;
    *o++ = 0x15;
    tblog_f32le(o, value);
    o += 4;
    size_t vn = (size_t)(o - val);
    /* Summary{1: value} */
    unsigned char sum[170];
    unsigned char* s = sum;
    *s++ = 0x0A;
    s = tblog_varint(s, (uint64_t)vn);
    memcpy(s, val, vn);
    s += vn;
    size_t sn = (size_t)(s - sum);
    /* Event{1: wall_time, 2: step, 5: summary} */
    unsigned char ev[200];
    unsigned char* e = ev;
    *e++ = 0x09;
    tblog_f64le(e, wall_time);
    e += 8;
    *e++ = 0x10;
    e = tblog_varint(e, (uint64_t)step);
    *e++ = 0x2A;
    e = tblog_varint(e, (uint64_t)sn);
    memcpy(e, sum, sn);
    e += sn;
    size_t n = (size_t)(e - ev);
    unsigned char hdr[12];
    tblog_u64le(hdr, (uint64_t)n);
    tblog_u32le(hdr + 8, tblog_masked(tblog_crc32c(hdr, 8)));
    unsigned char ftr[4];
    tblog_u32le(ftr, tblog_masked(tblog_crc32c(ev, n)));
    fwrite(hdr, 1, 12, tb->fp);
    fwrite(ev, 1, n, tb->fp);
    fwrite(ftr, 1, 4, tb->fp);
}

static void tblog_flush(Tblog* tb) {
    if (tb->fp)
        fflush(tb->fp);
}

static void tblog_close(Tblog* tb) {
    if (tb->fp) {
        fclose(tb->fp);
        tb->fp = NULL;
    }
}

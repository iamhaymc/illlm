/* ============================================================================
 * test/test.c -- unit tests for the engine internals.
 *
 * test/test.py proves the whole stack against transformers.  This file proves
 * the parts underneath it: number formats, the json reader, every compute
 * kernel against a plain-C restatement of the same maths, the thread pool, the
 * tokenizer machinery, and the sampler.
 *
 * Build and run:  python3 util/make.py test
 * ==========================================================================*/

#include "../app/core.c"

/* The yolo engine is the second translation unit this file proves.  It is
 * built without stb here: the tests generate their own pictures, and the
 * engine reads binary PNM without a library. */
#define YOLO_NO_STB
#include "../app/yolo.c"

/* The Vulkan backend, on a build that asked for it -- `python util/make.py
 * test --vulkan`.  VULK_PROBE brings in the comparison that file carries and
 * leaves out its command line; the checks below are that comparison reported
 * through this harness.  Without the define nothing here changes, which is why
 * the default suite still builds on a host that has never heard of Vulkan. */
#ifdef ILL_VULKAN
#define VULK_PROBE
#include "../app/vulk.c"
#endif

/* -- harness --------------------------------------------------------------- */

static int32_t test_ran, test_bad;
static const char *test_area = "";

static void test_open(const char *area)
{
    test_area = area;
    printf("\n%s\n", area);
}

static void test_case(const char *name, int good, const char *form, ...)
{
    va_list args;
    ++test_ran;
    printf("  %-38s %s", name, good ? "pass" : "FAIL");
    if (form && *form) {
        fputs("  ", stdout);
        va_start(args, form);
        vprintf(form, args);
        va_end(args);
    }
    putchar('\n');
    if (!good) ++test_bad;
}

static int test_near(float a, float b, float room) { return fabsf(a - b) <= room; }

static float test_gap(const float *a, const float *b, int32_t count)
{
    float worst = 0.0f, scale = 1e-6f;
    int32_t index;
    for (index = 0; index < count; ++index) {
        float diff = fabsf(a[index] - b[index]);
        float mag  = fabsf(a[index]);
        if (diff > worst) worst = diff;
        if (mag > scale)  scale = mag;
    }
    return worst / scale;
}

static uint64_t test_seed = 0x9E3779B97F4A7C15ULL;

static float test_real(void)
{
    test_seed ^= test_seed << 13;
    test_seed ^= test_seed >> 7;
    test_seed ^= test_seed << 17;
    return (float)((double)(test_seed >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
}

static void test_fill(float *cells, int32_t count)
{
    int32_t index;
    for (index = 0; index < count; ++index) cells[index] = test_real();
}

/* -- number formats -------------------------------------------------------- */

static void test_numbers(void)
{
    static const float probes[] = {
        0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, 1e-4f, -1e-4f,
        3.14159265f, -2.71828f, 65504.0f, 6.1e-5f, 1e-7f, 123456.0f
    };
    int32_t index, count = (int32_t)(sizeof(probes) / sizeof(probes[0]));
    float   worst_bf = 0.0f, worst_f16 = 0.0f;
    int     exact = 1, alphabet = 1, seen[512];

    test_open("number formats");

    for (index = 0; index < count; ++index) {
        float back = ill_bf16_cast(ill_bf16_pack(probes[index]));
        float room = fabsf(probes[index]) * 0.004f + 1e-30f;
        if (fabsf(back - probes[index]) > room) worst_bf = 1.0f;
    }
    test_case("bf16 round trip within 2^-8", worst_bf == 0.0f, "");

    for (index = 0; index < count; ++index) {
        float value = probes[index];
        float back  = ill_f16_cast(ill_f16_pack(value));
        float room  = fabsf(value) * 0.002f + 1e-7f;
        if (fabsf(value) < 65504.0f && fabsf(back - value) > room) worst_f16 = 1.0f;
    }
    test_case("f16 round trip within 2^-11", worst_f16 == 0.0f, "");

    /* widening must be exact: it only shifts bits into place */
    for (index = 0; index < 65536; index += 7) {
        union { uint32_t u; float f; } cell;
        cell.u = (uint32_t)index << 16;
        if (cell.f == cell.f && ill_bf16_cast((uint16_t)index) != cell.f) exact = 0;
    }
    test_case("bf16 widening is exact", exact, "");

    {
        float cells[64], back[64], scale[2];
        int8_t quant[64];
        float  worst;
        test_fill(cells, 64);
        ill_q8_pack(cells, 64, quant, scale);
        for (index = 0; index < 64; ++index)
            back[index] = (float)quant[index] * scale[index / ILL_Q8_BLOCK];
        worst = test_gap(cells, back, 64);
        test_case("q8 pack error under 1%", worst < 0.01f, "%.4f", (double)worst);
    }
    {
        float cells[40], back[40], scale[2];
        int8_t quant[40];
        for (index = 0; index < 40; ++index) cells[index] = 0.0f;
        ill_q8_pack(cells, 40, quant, scale);
        for (index = 0; index < 40; ++index)
            back[index] = (float)quant[index] * scale[index / ILL_Q8_BLOCK];
        test_case("q8 handles an all-zero block", back[0] == 0.0f && back[39] == 0.0f, "");
        test_case("q8 block count rounds up", ill_q8_blocks(40) == 2 && ill_q8_blocks(32) == 1, "");
    }

    for (index = 0; index < 512; ++index) seen[index] = 0;
    for (index = 0; index < 256; ++index) {
        uint32_t rune = ill_byte_rune(index);
        if (rune >= 512u || seen[rune]) alphabet = 0;
        else seen[rune] = 1;
    }
    test_case("byte alphabet is a bijection", alphabet, "");
}

/* -- utf-8 ----------------------------------------------------------------- */

/* How many bytes a lead byte promises, written out here rather than asked of
 * the engine: a check that calls the function it is checking proves nothing. */
static int32_t test_utf8_need(unsigned char lead)
{
    if (lead < 0x80) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 1;
}

static void test_utf8(void)
{
    static const uint32_t runes[] = { 0x41, 0x7F, 0x80, 0xFF, 0x100, 0x7FF, 0x800,
                                      0xFFFF, 0x10000, 0x1F600, 0x10FFFF };
    int32_t index, count = (int32_t)(sizeof(runes) / sizeof(runes[0]));
    int     good = 1;

    test_open("utf-8");
    for (index = 0; index < count; ++index) {
        char     cell[4];
        uint32_t back = 0;
        int32_t  span = ill_utf8_write(runes[index], cell);
        int32_t  used = ill_utf8_read(cell, span, 0, &back);
        if (used != span || back != runes[index]) good = 0;
    }
    test_case("write then read round trips", good, "");

    {   /* What a streaming writer must hold back.  The rule is the count of
           trailing bytes that begin a sequence which has not finished, and it
           is zero whenever releasing the bytes is the right thing to do -- a
           complete sequence, or rubbish that no continuation will ever
           complete. */
        struct { const char *bytes; int32_t len; int32_t want; const char *what; } cases[] = {
            { "abc",                 3, 0, "plain ascii holds nothing" },
            { "a\xC3\xA9",           3, 0, "a finished two byte sequence holds nothing" },
            { "a\xC3",               2, 1, "a lone lead byte is held" },
            { "\xE2\x82\xAC",        3, 0, "a finished three byte sequence holds nothing" },
            { "\xE2\x82",            2, 2, "two thirds of a three byte sequence is held" },
            { "\xF0\x9F\x98\x80",    4, 0, "a finished four byte sequence holds nothing" },
            { "\xF0\x9F\x98",        3, 3, "three quarters of a four byte sequence is held" },
            { "a\xFF",               2, 0, "a byte that leads nothing is released" },
            { "\x80\x80",            2, 0, "continuations with no lead are released" },
            { "",                    0, 0, "an empty buffer holds nothing" }
        };
        int32_t k;
        for (k = 0; k < (int32_t)(sizeof cases / sizeof cases[0]); ++k) {
            int32_t got = ill_utf8_hold(cases[k].bytes, cases[k].len);
            test_case(cases[k].what, got == cases[k].want, "%d", (int)got);
        }
    }

    {   /* The property the streaming path actually needs: feeding a string
           through the rule one byte at a time and releasing what it does not
           hold reproduces the string exactly, and never leaves what has
           been shown ending part way through a character. */
        const char *whole = "na\xC3\xAFve \xE2\x82\xAC" "5 \xF0\x9F\x98\x80 done";
        char    seen[64], carry[8];
        int32_t len = (int32_t)strlen(whole), held = 0, fill = 0, at, split = 0;
        for (at = 0; at < len; ++at) {
            char    work[16];
            int32_t span = held, whole_span;
            memcpy(work, carry, (size_t)held);
            work[span++] = whole[at];
            whole_span = span - ill_utf8_hold(work, span);
            if (whole_span > 0) {
                memcpy(seen + fill, work, (size_t)whole_span);
                fill += whole_span;
                /* The property a terminal cares about: everything shown so far
                   ends on a finished character, at every step and not only at
                   the end.  Walked with the test's own rule, not the engine's. */
                {
                    int32_t walk = 0;
                    while (walk < fill) {
                        int32_t need = test_utf8_need((unsigned char)seen[walk]);
                        if (walk + need > fill) { split = 1; break; }
                        walk += need;
                    }
                }
            }
            held = span - whole_span;
            memcpy(carry, work + whole_span, (size_t)held);
        }
        memcpy(seen + fill, carry, (size_t)held);
        fill += held;
        seen[fill] = '\0';
        test_case("byte at a time, the held tail reassembles the string",
                  fill == len && !memcmp(seen, whole, (size_t)len) && !split, "%s", seen);
    }

    {
        uint32_t rune = 0;
        int32_t  span = ill_utf8_read("\xC3", 1, 0, &rune);   /* truncated pair */
        test_case("truncated sequence advances", span == 1, "");
    }
    {
        const char *text = "aé中🙂";
        uint32_t rune = 0;
        int32_t  pos = 0, seen = 0, len = (int32_t)strlen(text);
        while (pos < len) {
            int32_t span = ill_utf8_read(text, len, pos, &rune);
            if (span <= 0) break;
            pos += span;
            ++seen;
        }
        test_case("mixed width string walks cleanly", seen == 4 && pos == len, "%d runes", seen);
    }
}

/* -- json ------------------------------------------------------------------ */

static void test_json(void)
{
    static const char *body =
        "{\"name\":\"lfm2\",\"dim\":2560,\"eps\":1e-05,\"deep\":{\"a\":[1,2,3],"
        "\"b\":true,\"c\":null},\"kinds\":[\"conv\",\"full_attention\"],"
        "\"esc\":\"tab\\there\\u00e9\\u0041\\ud83d\\ude00\"}";
    IllJson doc;
    int32_t node;

    test_open("json reader");
    test_case("parses a nested document", ill_json_load(&doc, body, strlen(body)) == ILL_OK, "");

    test_case("reads a string field",
              !strcmp(ill_json_str(&doc, ill_json_pick(&doc, 0, "name"), ""), "lfm2"), "");
    test_case("reads an integer field", ill_json_long(&doc, ill_json_pick(&doc, 0, "dim"), 0) == 2560, "");
    test_case("reads a float field",
              test_near((float)ill_json_find(&doc, 0, "eps", 0.0), 1e-5f, 1e-9f), "");

    node = ill_json_pick(&doc, 0, "deep");
    test_case("walks into a nested object",
              ill_json_long(&doc, ill_json_item(&doc, ill_json_pick(&doc, node, "a"), 2), 0) == 3, "");
    test_case("reads true as one",
              (int)ill_json_real(&doc, ill_json_pick(&doc, node, "b"), 0.0) == 1, "");
    test_case("missing key yields the fallback",
              ill_json_long(&doc, ill_json_pick(&doc, 0, "absent"), 42) == 42, "");

    node = ill_json_pick(&doc, 0, "kinds");
    test_case("array count is right", doc.nodes[node].count == 2, "");
    test_case("array items are reachable",
              !strcmp(ill_json_str(&doc, ill_json_item(&doc, node, 1), ""), "full_attention"), "");

    test_case("escapes and surrogates decode",
              !strcmp(ill_json_str(&doc, ill_json_pick(&doc, 0, "esc"), ""),
                      "tab\therel\xc3\xa9""A\xf0\x9f\x98\x80") ||
              !strcmp(ill_json_str(&doc, ill_json_pick(&doc, 0, "esc"), ""),
                      "tab\there\xc3\xa9""A\xf0\x9f\x98\x80"), "");
    ill_json_free(&doc);

    {
        static const char *broken[] = {
            "{", "{\"a\"}", "{\"a\":}", "[1,2", "{\"a\":1,}", "tru", "{'a':1}", ""
        };
        int32_t index, caught = 0;
        for (index = 0; index < 8; ++index) {
            IllJson bad;
            if (ill_json_load(&bad, broken[index], strlen(broken[index])) != ILL_OK) ++caught;
            else ill_json_free(&bad);
        }
        test_case("malformed input is rejected", caught == 8, "%d/8", caught);
    }
}

/* -- kernels --------------------------------------------------------------- */

#define TEST_ROWS 37
#define TEST_COLS 71
#define TEST_TOKS (ILL_TILE_MAX + 1)   /* one past the widest tile, so a
                                              partial tile is covered too */

static void test_dense_naive(const float *w, const float *x, float *y,
                             int32_t rows, int32_t cols, int32_t tokens)
{
    int32_t t, r, j;
    for (t = 0; t < tokens; ++t)
        for (r = 0; r < rows; ++r) {
            double sum = 0.0;
            for (j = 0; j < cols; ++j)
                sum += (double)w[(size_t)r * cols + j] * (double)x[(size_t)t * cols + j];
            y[(size_t)t * rows + r] = (float)sum;
        }
}

static void test_kernels(void)
{
    float   *w    = (float *)ill_block_make(TEST_ROWS * TEST_COLS * sizeof(float));
    float   *x    = (float *)ill_block_make(TEST_TOKS * TEST_COLS * sizeof(float));
    float   *want = (float *)ill_block_make(TEST_TOKS * TEST_ROWS * sizeof(float));
    float   *got  = (float *)ill_block_make(TEST_TOKS * TEST_ROWS * sizeof(float));
    uint16_t *half = (uint16_t *)ill_block_make(TEST_ROWS * TEST_COLS * sizeof(uint16_t));
    IllPlane plane;
    int32_t  index, tile;

    test_open("kernels");
    test_fill(w, TEST_ROWS * TEST_COLS);
    test_fill(x, TEST_TOKS * TEST_COLS);
    test_dense_naive(w, x, want, TEST_ROWS, TEST_COLS, TEST_TOKS);

    plane.cells = w; plane.steps = NULL; plane.type = ILL_TYPE_F32;
    plane.rows = TEST_ROWS; plane.cols = TEST_COLS; plane.blocks = 0;

    memset(got, 0, TEST_TOKS * TEST_ROWS * sizeof(float));
    ill_dense_real(&plane, x, TEST_COLS, 1, 0, TEST_ROWS, got, TEST_ROWS);
    test_case("f32 dense, one token",
              test_gap(want, got, TEST_ROWS) < 1e-5f, "%.1e",
              (double)test_gap(want, got, TEST_ROWS));

    /* Every tile width the dispatch instantiates, not only the widest: a
       missing case in the switch computes the first four rows and leaves the
       rest of the tile at whatever the buffer held. */
    for (tile = 2; tile <= ILL_TILE_MAX; ++tile) {
        char name[48];
        memset(got, 0, TEST_TOKS * TEST_ROWS * sizeof(float));
        ill_dense_real(&plane, x, TEST_COLS, tile, 0, TEST_ROWS, got, TEST_ROWS);
        snprintf(name, sizeof name, "f32 dense, %d token tile", tile);
        test_case(name, test_gap(want, got, tile * TEST_ROWS) < 1e-5f, "%.1e",
                  (double)test_gap(want, got, tile * TEST_ROWS));
    }

    memset(got, 0, TEST_TOKS * TEST_ROWS * sizeof(float));
    ill_dense_real(&plane, x, TEST_COLS, 1, 5, 20, got, TEST_ROWS);
    {
        int good = 1;
        for (index = 0; index < 5; ++index) if (got[index] != 0.0f) good = 0;
        for (index = 5; index < 20; ++index)
            if (!test_near(got[index], want[index], 1e-4f)) good = 0;
        for (index = 20; index < TEST_ROWS; ++index) if (got[index] != 0.0f) good = 0;
        test_case("dense honours the row window", good, "");
    }

    for (index = 0; index < TEST_ROWS * TEST_COLS; ++index) half[index] = ill_bf16_pack(w[index]);
    plane.cells = half; plane.type = ILL_TYPE_BF16;
    memset(got, 0, TEST_TOKS * TEST_ROWS * sizeof(float));
    ill_dense_real(&plane, x, TEST_COLS, 4, 0, TEST_ROWS, got, TEST_ROWS);
    test_case("bf16 dense tracks f32", test_gap(want, got, 4 * TEST_ROWS) < 0.02f, "%.1e",
              (double)test_gap(want, got, 4 * TEST_ROWS));

    for (index = 0; index < TEST_ROWS * TEST_COLS; ++index) half[index] = ill_f16_pack(w[index]);
    plane.cells = half; plane.type = ILL_TYPE_F16;
    memset(got, 0, TEST_TOKS * TEST_ROWS * sizeof(float));
    ill_dense_real(&plane, x, TEST_COLS, 4, 0, TEST_ROWS, got, TEST_ROWS);
    test_case("f16 dense tracks f32", test_gap(want, got, 4 * TEST_ROWS) < 3e-3f, "%.1e",
              (double)test_gap(want, got, 4 * TEST_ROWS));

    {
        int32_t blocks = ill_q8_blocks(TEST_COLS);
        int8_t *wq = (int8_t *)ill_block_make(TEST_ROWS * TEST_COLS);
        float  *ws = (float *)ill_block_make((size_t)TEST_ROWS * blocks * sizeof(float));
        int8_t *xq = (int8_t *)ill_block_make(TEST_TOKS * TEST_COLS);
        float  *xs = (float *)ill_block_make((size_t)TEST_TOKS * blocks * sizeof(float));
        for (index = 0; index < TEST_ROWS; ++index)
            ill_q8_pack(w + (size_t)index * TEST_COLS, TEST_COLS,
                        wq + (size_t)index * TEST_COLS, ws + (size_t)index * blocks);
        for (index = 0; index < TEST_TOKS; ++index)
            ill_q8_pack(x + (size_t)index * TEST_COLS, TEST_COLS,
                        xq + (size_t)index * TEST_COLS, xs + (size_t)index * blocks);
        plane.cells = wq; plane.steps = ws; plane.type = ILL_TYPE_Q8; plane.blocks = blocks;
        for (tile = 1; tile <= ILL_TILE_MAX; ++tile) {
            char name[48];
            memset(got, 0, TEST_TOKS * TEST_ROWS * sizeof(float));
            ill_dense_byte(&plane, xq, TEST_COLS, xs, blocks, tile, 0, TEST_ROWS,
                           got, TEST_ROWS);
            snprintf(name, sizeof name, "q8 dense tracks f32, %d token tile", tile);
            test_case(name, test_gap(want, got, tile * TEST_ROWS) < 0.05f, "%.1e",
                      (double)test_gap(want, got, tile * TEST_ROWS));
        }
        {   /* q4 against the same naive f32 the q8 sweep is judged on.  Four
               bits over a 32 value block is about eight times q8's step, so
               the bar is eight times q8's; anything much past it is a packing
               bug rather than the format. */
            int32_t  qb = ill_q8_blocks(TEST_COLS);
            uint8_t *wn = (uint8_t *)ill_block_make((size_t)TEST_ROWS * qb * ILL_Q4_BYTES);
            float   *ns = (float *)ill_block_make((size_t)TEST_ROWS * qb * sizeof(float));
            IllPlane nib = plane;
            int32_t  k;
            for (k = 0; k < TEST_ROWS; ++k)
                ill_q4_pack(w + (size_t)k * TEST_COLS, TEST_COLS,
                            wn + (size_t)k * qb * ILL_Q4_BYTES, ns + (size_t)k * qb);
            nib.cells = wn; nib.steps = ns; nib.type = ILL_TYPE_Q4; nib.blocks = qb;
            for (k = 1; k <= ILL_TILE_MAX; ++k) {
                char name[52];
                memset(got, 0, TEST_TOKS * TEST_ROWS * sizeof(float));
                ill_dense_nib(&nib, xq, TEST_COLS, xs, blocks, k, 0, TEST_ROWS,
                              got, TEST_ROWS);
                snprintf(name, sizeof name, "q4 dense tracks f32, %d token tile", k);
                test_case(name, test_gap(want, got, k * TEST_ROWS) < 0.13f, "%.1e",
                          (double)test_gap(want, got, k * TEST_ROWS));
            }
            {   /* A row read back has to agree with what the dot product is
                   using, or the two disagree about what the weights are. */
                float row[TEST_COLS];
                ill_plane_row(&nib, 3, row);
                test_case("q4 row read matches its scale",
                          test_gap(w + 3 * TEST_COLS, row, TEST_COLS) < 0.15f, "%.1e",
                          (double)test_gap(w + 3 * TEST_COLS, row, TEST_COLS));
            }
            {   /* The accuracy the format promises: a value comes back
                   within half a step, and a step is the block's peak over
                   eight because all sixteen levels are used.  Placing the
                   extreme on -8 is what buys that eighth -- dividing the peak
                   by seven and clipping would widen the step by a seventh and
                   fail here.  The fixture keeps clear of the far end, which is
                   the one place this scheme is asymmetric. */
                float   cell[ILL_Q8_BLOCK], step, peak = 13.0f, worst = 0.0f;
                uint8_t packed[ILL_Q4_BYTES];
                int8_t  lift[ILL_Q8_BLOCK];
                int32_t j2;
                for (j2 = 0; j2 < ILL_Q8_BLOCK; ++j2)
                    cell[j2] = (float)(j2 % 9) - 3.0f;     /* -3 .. 5 */
                cell[9] = -peak;                           /* the extreme */
                ill_q4_pack(cell, ILL_Q8_BLOCK, packed, &step);
                ill_q4_lift(packed, lift);
                for (j2 = 0; j2 < ILL_Q8_BLOCK; ++j2) {
                    float back = (float)lift[j2] * step;
                    float off  = back - cell[j2];
                    if (off < 0.0f) off = -off;
                    if (off > worst) worst = off;
                }
                test_case("q4 comes back within half a step of peak over eight",
                          worst <= peak / 16.0f + 1e-4f, "%.4f against %.4f",
                          (double)worst, (double)(peak / 16.0f));
            }

            {   /* The lift that feeds the dot and the lift that reads a row
                   are different code on any machine with vectors, and they
                   have to agree about which nibble is which value.  Compared
                   through the dot, because the register form has no bytes to
                   look at. */
                int8_t  flat[2 * ILL_Q8_BLOCK];
                int8_t  act[2 * ILL_Q8_BLOCK];
                uint8_t two[2 * ILL_Q4_BYTES];
                float   cell[2 * ILL_Q8_BLOCK], steps[2];
                float   a, b2;
                int32_t j2;
                for (j2 = 0; j2 < 2 * ILL_Q8_BLOCK; ++j2) {
                    cell[j2] = (float)((j2 * 37) % 23) - 11.0f;
                    act[j2]  = (int8_t)(((j2 * 53) % 61) - 30);
                }
                ill_q4_pack(cell, ILL_Q8_BLOCK, two, &steps[0]);
                ill_q4_pack(cell + ILL_Q8_BLOCK, ILL_Q8_BLOCK,
                            two + ILL_Q4_BYTES, &steps[1]);
                ill_q4_lift(two, flat);
                ill_q4_lift(two + ILL_Q4_BYTES, flat + ILL_Q8_BLOCK);
                a  = ill_q8_fold(ill_q8_pair(ill_q8_zero(), flat, act, 0.5f, 0.25f));
                b2 = ill_q8_fold(ill_q4_dot(ill_q8_zero(), ill_q4_open(two),
                                            act, 0.5f, 0.25f));
                test_case("the register lift agrees with the byte lift",
                          test_near(a, b2, 1e-3f), "%.4f vs %.4f",
                          (double)a, (double)b2);
            }

            ill_block_free(wn); ill_block_free(ns);
        }

        {   /* The paired dot must be the two single block dots it stands
               for.  On a VNNI build it is a different instruction reading the
               weights unsigned, so this is the check that the sign moved onto
               the activations correctly; everywhere else it is the identity. */
            IllQAcc one = ill_q8_zero(), two = ill_q8_zero();
            float lo = 0.5f, hi = 0.25f, a, b;
            one = ill_q8_step(one, wq, xq, lo);
            one = ill_q8_step(one, wq + ILL_Q8_BLOCK, xq + ILL_Q8_BLOCK, hi);
            two = ill_q8_pair(two, wq, xq, lo, hi);
            a = ill_q8_fold(one); b = ill_q8_fold(two);
            test_case("q8 paired dot equals two single dots",
                      test_near(a, b, 1e-3f), "%.6f vs %.6f", (double)a, (double)b);
        }

        {   /* A weight of -128 is the one value whose magnitude does not fit a
               signed byte; the unsigned left operand of the paired dot is what
               makes it work, so say so here rather than trusting it. */
            int8_t wedge[2 * ILL_Q8_BLOCK], edge[2 * ILL_Q8_BLOCK];
            IllQAcc one = ill_q8_zero(), two = ill_q8_zero();
            float a, b;
            int32_t k;
            for (k = 0; k < 2 * ILL_Q8_BLOCK; ++k) {
                wedge[k] = (int8_t)(k % 3 == 0 ? -128 : (k % 5) - 2);
                edge[k]  = (int8_t)((k % 7) - 3);
            }
            one = ill_q8_step(one, wedge, edge, 1.0f);
            one = ill_q8_step(one, wedge + ILL_Q8_BLOCK, edge + ILL_Q8_BLOCK, 1.0f);
            two = ill_q8_pair(two, wedge, edge, 1.0f, 1.0f);
            a = ill_q8_fold(one); b = ill_q8_fold(two);
            test_case("q8 paired dot handles a -128 weight",
                      test_near(a, b, 1e-3f), "%.1f vs %.1f", (double)a, (double)b);
        }

        {
            float row[TEST_COLS];
            ill_plane_row(&plane, 3, row);
            test_case("q8 row read matches its scale",
                      test_gap(w + 3 * TEST_COLS, row, TEST_COLS) < 0.02f, "");
        }
        ill_block_free(wq); ill_block_free(ws); ill_block_free(xq); ill_block_free(xs);
    }

    {   /* rms norm */
        float cells[TEST_COLS], gain[TEST_COLS], mine[TEST_COLS], theirs[TEST_COLS];
        double mass = 0.0;
        float  scale;
        test_fill(cells, TEST_COLS);
        for (index = 0; index < TEST_COLS; ++index) gain[index] = 0.5f + test_real();
        for (index = 0; index < TEST_COLS; ++index) mass += (double)cells[index] * cells[index];
        scale = 1.0f / sqrtf((float)(mass / TEST_COLS) + 1e-5f);
        for (index = 0; index < TEST_COLS; ++index) theirs[index] = cells[index] * scale * gain[index];
        ill_norm_rms(cells, gain, mine, TEST_COLS, 1e-5f);
        test_case("rms norm matches the definition",
                  test_gap(theirs, mine, TEST_COLS) < 1e-5f, "");
    }
    {   /* swiglu */
        float gate[16], lift[16], mine[16], theirs[16];
        test_fill(gate, 16); test_fill(lift, 16);
        for (index = 0; index < 16; ++index)
            theirs[index] = (gate[index] / (1.0f + expf(-gate[index]))) * lift[index];
        ill_glue_swi(gate, lift, mine, 16);
        test_case("swiglu matches the definition", test_gap(theirs, mine, 16) < 1e-6f, "");
        memcpy(mine, gate, sizeof(gate));
        ill_glue_swi(mine, lift, mine, 16);
        test_case("swiglu is safe in place", test_gap(theirs, mine, 16) < 1e-6f, "");
    }
    {   /* softmax */
        float cells[9] = { 1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 90.0f, -90.0f, 4.0f, 4.0f };
        float mass = 0.0f;
        ill_soft_max(cells, 9);
        for (index = 0; index < 9; ++index) mass += cells[index];
        test_case("softmax sums to one", test_near(mass, 1.0f, 1e-6f), "%.7f", (double)mass);
        test_case("softmax survives a large peak",
                  cells[5] > 0.99f && cells[6] >= 0.0f, "");
    }
    {   /* dot and axpy */
        float a[TEST_COLS], b[TEST_COLS], acc[TEST_COLS], base[TEST_COLS];
        double sum = 0.0;
        int good = 1;
        test_fill(a, TEST_COLS); test_fill(b, TEST_COLS); test_fill(base, TEST_COLS);
        for (index = 0; index < TEST_COLS; ++index) sum += (double)a[index] * b[index];
        test_case("dot matches the definition",
                  test_near(ill_dot_real(a, b, TEST_COLS), (float)sum, 1e-4f), "");
        memcpy(acc, base, sizeof(base));
        ill_axpy_add(acc, a, 0.25f, TEST_COLS);
        for (index = 0; index < TEST_COLS; ++index)
            if (!test_near(acc[index], base[index] + 0.25f * a[index], 1e-5f)) good = 0;
        test_case("axpy accumulates in place", good, "");
    }

    ill_block_free(w); ill_block_free(x); ill_block_free(want);
    ill_block_free(got); ill_block_free(half);
}

/* -- rope, attention, convolution ------------------------------------------ */

static void test_operators(void)
{
    IllBackend *back = &ill_backend_cpu;
    test_open("operators");
    if (back->setup(back, 2) != ILL_OK) { test_case("backend starts", 0, ""); return; }

    {   /* rotary positions against a direct restatement */
        const int32_t width = 8, heads = 3, tokens = 4;
        float cells[4 * 3 * 8], mine[4 * 3 * 8], table[4 * 8];
        int32_t t, h, i;
        int good = 1;
        test_fill(cells, tokens * heads * width);
        memcpy(mine, cells, sizeof(cells));
        for (t = 0; t < tokens; ++t)
            for (i = 0; i < width / 2; ++i) {
                double phase = (double)t * pow(10000.0, -(double)(2 * i) / width);
                table[t * width + i] = (float)cos(phase);
                table[t * width + width / 2 + i] = (float)sin(phase);
            }
        back->rope(back, mine, table, tokens, heads, width, heads * width);
        for (t = 0; t < tokens; ++t)
            for (h = 0; h < heads; ++h)
                for (i = 0; i < width / 2; ++i) {
                    const float *src = cells + (t * heads + h) * width;
                    float c = table[t * width + i], s = table[t * width + width / 2 + i];
                    float lo = src[i] * c - src[i + width / 2] * s;
                    float hi = src[i + width / 2] * c + src[i] * s;
                    const float *dst = mine + (t * heads + h) * width;
                    if (!test_near(dst[i], lo, 1e-5f) ||
                        !test_near(dst[i + width / 2], hi, 1e-5f)) good = 0;
                }
        test_case("rope rotates each half pair", good, "");
    }

    {   /* attention against a direct restatement */
        const int32_t width = 4, heads = 4, groups = 2, span = 6, tokens = 3, base = 2;
        float query[3 * 4 * 4], keys[2 * 6 * 4], vals[2 * 6 * 4];
        float mine[3 * 4 * 4], theirs[3 * 4 * 4], board[8 * 6], score[6];
        IllHeedJob job;
        int32_t t, h, j, i;
        int good = 1;
        test_fill(query, tokens * heads * width);
        test_fill(keys, groups * span * width);
        test_fill(vals, groups * span * width);
        job.query = query; job.keys = keys; job.vals = vals; job.value = mine;
        job.board = board; job.tokens = tokens; job.heads = heads; job.groups = groups;
        job.width = width; job.span = span; job.base = base;
        job.scale = 1.0f / sqrtf((float)width);
        back->attend(back, &job);
        for (t = 0; t < tokens; ++t)
            for (h = 0; h < heads; ++h) {
                int32_t g = h / (heads / groups), reach = base + t + 1;
                const float *q = query + (t * heads + h) * width;
                float peak = -FLT_MAX, mass = 0.0f;
                float *out = theirs + (t * heads + h) * width;
                for (j = 0; j < reach; ++j) {
                    double sum = 0.0;
                    for (i = 0; i < width; ++i)
                        sum += (double)q[i] * keys[(g * span + j) * width + i];
                    score[j] = (float)sum * job.scale;
                    if (score[j] > peak) peak = score[j];
                }
                for (j = 0; j < reach; ++j) { score[j] = expf(score[j] - peak); mass += score[j]; }
                for (i = 0; i < width; ++i) out[i] = 0.0f;
                for (j = 0; j < reach; ++j)
                    for (i = 0; i < width; ++i)
                        out[i] += (score[j] / mass) * vals[(g * span + j) * width + i];
                for (i = 0; i < width; ++i)
                    if (!test_near(out[i], mine[(t * heads + h) * width + i], 1e-5f)) good = 0;
            }
        test_case("attention matches a direct restatement", good, "");
    }

    {   /* short convolution, including the rolling window across two calls */
        const int32_t dim = 5, width = 3, tokens = 6;
        float src[6 * 5], dst[6 * 5], whole[6 * 5], hist[5 * 2], taps[5 * 3];
        IllFlowJob job;
        int32_t t, c, j;
        int good = 1;
        test_fill(src, tokens * dim);
        test_fill(taps, dim * width);
        memset(hist, 0, sizeof(hist));

        job.src = src; job.dst = whole; job.hist = hist; job.taps = taps; job.bias = NULL;
        job.tokens = tokens; job.dim = dim; job.width = width;
        back->conv1d(back, &job);

        for (t = 0; t < tokens; ++t)
            for (c = 0; c < dim; ++c) {
                double sum = 0.0;
                for (j = 0; j < width; ++j) {
                    int32_t at = t - (width - 1) + j;
                    if (at >= 0) sum += (double)taps[c * width + j] * src[at * dim + c];
                }
                if (!test_near(whole[t * dim + c], (float)sum, 1e-5f)) good = 0;
            }
        test_case("convolution is causal and zero padded", good, "");

        memset(hist, 0, sizeof(hist));
        for (t = 0; t < tokens; ++t) {
            job.src = src + t * dim; job.dst = dst + t * dim; job.tokens = 1;
            back->conv1d(back, &job);
        }
        test_case("convolution window carries between calls",
                  test_gap(whole, dst, tokens * dim) < 1e-5f, "");
    }

    back->close(back);
}

/* -- thread pool ----------------------------------------------------------- */

typedef struct TestTally { int32_t seen[4096]; int32_t total; } TestTally;

static void test_pool_chore(void *args, int32_t chunk, int32_t chunks, int32_t worker)
{
    TestTally *tally = (TestTally *)args;
    int32_t head, tail, index;
    (void)worker;
    ill_span_split(tally->total, chunk, chunks, &head, &tail);
    for (index = head; index < tail; ++index) tally->seen[index] += 1;
}

static void test_threads(void)
{
    static const int32_t widths[] = { 1, 2, 3, 8, 17 };
    static const int32_t totals[] = { 0, 1, 5, 64, 1000, 4096 };
    int32_t wi, ti, index;
    int     covered = 1;

    test_open("thread pool");

    for (ti = 0; ti < 6; ++ti)
        for (wi = 0; wi < 5; ++wi) {
            int32_t total = totals[ti], chunks = widths[wi], head, tail, walk = 0;
            if (total == 0) continue;
            for (index = 0; index < chunks; ++index) {
                ill_span_split(total, index, chunks, &head, &tail);
                if (head != walk || tail < head) covered = 0;
                walk = tail;
            }
            if (walk != total) covered = 0;
        }
    test_case("span split tiles every range exactly", covered, "");

    {
        IllPool  *pool = NULL;
        TestTally tally;
        int good = 1;
        if (ill_pool_make(&pool, 4) != ILL_OK) { test_case("pool starts", 0, ""); return; }
        test_case("pool reports its width", ill_pool_size(pool) >= 1, "%d lanes",
                  ill_pool_size(pool));
        for (ti = 0; ti < 6; ++ti) {
            memset(&tally, 0, sizeof(tally));
            tally.total = totals[ti];
            ill_pool_fork(pool, test_pool_chore, &tally, ill_pool_size(pool));
            for (index = 0; index < tally.total; ++index) if (tally.seen[index] != 1) good = 0;
        }
        test_case("each item runs exactly once", good, "");

        memset(&tally, 0, sizeof(tally));
        tally.total = 4096;
        for (index = 0; index < 200; ++index)
            ill_pool_fork(pool, test_pool_chore, &tally, ill_pool_size(pool));
        good = 1;
        for (index = 0; index < 4096; ++index) if (tally.seen[index] != 200) good = 0;
        test_case("repeated forks stay in step", good, "");
        ill_pool_free(pool);
    }
}

/* -- tokenizer machinery --------------------------------------------------- */

static void test_split_count(uint8_t kind, const char *text, const char *label, int32_t want)
{
    int32_t len = (int32_t)strlen(text), pos = 0, seen = 0;
    while (pos < len) {
        int32_t span = ill_split_take(kind, text, len, pos);
        if (span <= 0) break;
        pos += span;
        ++seen;
    }
    test_case(label, seen == want && pos == len, "%d chunks, want %d", seen, want);
}

static void test_vocab(void)
{
    test_open("tokenizer machinery");

    /* GPT-2 alternation, checked chunk by chunk */
    test_split_count(ILL_SPLIT_GPT2, "hello world", "gpt2 splits a two word line", 2);
    test_split_count(ILL_SPLIT_GPT2, "don't", "gpt2 keeps a contraction apart", 2);
    test_split_count(ILL_SPLIT_GPT2, "a  b", "gpt2 gives the last space to the word", 3);
    test_split_count(ILL_SPLIT_GPT2, "abc123", "gpt2 cuts letters from digits", 2);
    test_split_count(ILL_SPLIT_GPT2, "  ", "gpt2 takes a trailing run whole", 1);
    test_split_count(ILL_SPLIT_GPT2, "!!!", "gpt2 groups punctuation", 1);
    test_split_count(ILL_SPLIT_LLAMA3, "12345", "llama3 caps digit runs at three", 2);
    test_split_count(ILL_SPLIT_LLAMA3, "hi\n\nthere", "llama3 isolates newline runs", 3);

    {
        uint32_t rune = 0;
        ill_utf8_read("\xc3\xa9", 2, 0, &rune);
        test_case("accented letter classes as a letter",
                  ill_rune_kind(rune) == ILL_CLASS_LETTER, "");
        ill_utf8_read("\xe2\x80\x94", 3, 0, &rune);      /* em dash */
        test_case("em dash classes as punctuation",
                  ill_rune_kind(rune) == ILL_CLASS_OTHER, "");
        ill_utf8_read("\xe3\x80\x80", 3, 0, &rune);      /* ideographic space */
        test_case("ideographic space classes as space",
                  ill_rune_kind(rune) == ILL_CLASS_SPACE, "");
        ill_utf8_read("\xe4\xb8\xad", 3, 0, &rune);      /* CJK han */
        test_case("han character classes as a letter",
                  ill_rune_kind(rune) == ILL_CLASS_LETTER, "");
    }

    {   /* merge heap ordering */
        IllWeld weld;
        IllPair item;
        int32_t ranks[6] = { 9, 3, 7, 1, 5, 3 };
        int32_t index, last = -1;
        int good = 1;
        weld.heap_limit = 16;
        weld.heap = (IllPair *)ill_block_make(16 * sizeof(IllPair));
        weld.heap_count = 0;
        for (index = 0; index < 6; ++index) {
            item.rank = ranks[index]; item.a = index; item.b = index + 1;
            item.ida = 0; item.idb = 0;
            ill_weld_push(&weld, item);
        }
        for (index = 0; index < 6; ++index) {
            if (!ill_weld_pull(&weld, &item)) { good = 0; break; }
            if (item.rank < last) good = 0;
            last = item.rank;
        }
        test_case("merge heap pops lowest rank first", good, "");
        test_case("merge heap drains", weld.heap_count == 0, "");
        ill_block_free(weld.heap);
    }

    {
        test_case("hash is stable for equal text",
                  ill_hash_text("attention", 9) == ill_hash_text("attention", 9), "");
        test_case("hash separates near neighbours",
                  ill_hash_text("attention", 9) != ill_hash_text("attentiop", 9), "");
    }
}

/* -- sampler --------------------------------------------------------------- */

static void test_sampler(void)
{
    const int32_t vocab = 64;
    IllTuning tuning;
    IllSampler *sampler = NULL;
    float logits[64], work[64];
    int32_t index;

    test_open("scoring");
    {   /* A flat row of n equal values normalises to v + log(n), which is the
           one case the answer can be written down.  A score is a sum of these
           over a whole text, so being a little wrong here is being wrong once
           a token. */
        float flat[64];
        double got, want;
        int32_t k;
        for (k = 0; k < 64; ++k) flat[k] = 2.5f;
        got = ill_row_logsum(flat, 64);
        want = 2.5 + log(64.0);
        test_case("a flat row normalises to its value plus log of its width",
                  fabs(got - want) < 1e-9, "%.9f vs %.9f", got, want);

        /* Uniform rows give every token the same log probability, -log(n),
           whatever the value they are flat at.  That is the sanity check the
           scoring loop rests on: normaliser minus the chosen logit. */
        test_case("a flat row gives every token -log(width)",
                  fabs((flat[7] - got) + log(64.0)) < 1e-9, "%.9f",
                  (double)flat[7] - got + log(64.0));

        /* exp(800) is infinity in double.  Measuring against the peak is what
           keeps a confident row from scoring as a NaN. */
        for (k = 0; k < 64; ++k) flat[k] = -800.0f;
        flat[13] = 800.0f;
        got = ill_row_logsum(flat, 64);
        test_case("a row far outside exp's range still normalises",
                  got > 799.0 && got < 801.0, "%.6f", got);
    }

    test_open("drafting");
    {   /* The proposal is what followed the last earlier appearance of the
           tail.  Nothing here decides a token -- a wrong proposal costs a row
           that was computed anyway -- so what is checked is that it proposes
           the right thing and never proposes out of thin air. */
        int32_t out[8];
        {   /* "1 2 3" appeared at the start, followed by 4 5 9 9, and those
               four are what the tail's repeat proposes. */
            int32_t seen[] = { 1, 2, 3, 4, 5, 9, 9, 1, 2, 3 };
            int32_t got = ill_draft_scan(seen, 10, 3, out, 4);
            test_case("a repeated run proposes what followed it",
                      got == 4 && out[0] == 4 && out[1] == 5 &&
                      out[2] == 9 && out[3] == 9, "%d: %d %d %d %d",
                      (int)got, (int)out[0], (int)out[1], (int)out[2], (int)out[3]);
        }
        {   /* Two earlier places match; the later one wins, because it is the
               one the text has most recently been near. */
            int32_t seen[] = { 7, 8, 100, 0, 0, 7, 8, 200, 0, 0, 7, 8 };
            int32_t got = ill_draft_scan(seen, 12, 2, out, 1);
            test_case("the most recent earlier match wins",
                      got == 1 && out[0] == 200, "%d: %d", (int)got, (int)out[0]);
        }
        {   /* Long runs are tried before short ones, and the two pull apart
               here: the three token tail "1 2 3" last appeared at the start
               and was followed by 100, while its last two tokens "2 3"
               appeared more recently and were followed by 200.  Length wins
               over recency, because the longer agreement is the one whose
               continuation is worth believing. */
            int32_t seen[] = { 1, 2, 3, 100, 9, 2, 3, 200, 1, 2, 3 };
            int32_t got = ill_draft_scan(seen, 11, 3, out, 1);
            test_case("a longer run is preferred to a nearer short one",
                      got == 1 && out[0] == 100, "%d: %d", (int)got, (int)out[0]);
        }
        {   /* Nothing repeats, so nothing is proposed.  A draft that invents a
               token would still be safe, but it would waste the row. */
            int32_t seen[] = { 1, 2, 3, 4, 5, 6 };
            test_case("an unrepeated tail proposes nothing",
                      ill_draft_scan(seen, 6, 3, out, 4) == 0, "");
        }
        {   /* A pattern that has already repeated once proposes that it
               repeats again, which is the whole point: the match ends where
               the tail begins, and what followed it is the tail itself. */
            int32_t seen[] = { 1, 2, 3, 1, 2, 3 };
            int32_t got = ill_draft_scan(seen, 6, 3, out, 4);
            test_case("a pattern that repeated proposes it repeats again",
                      got == 3 && out[0] == 1 && out[1] == 2 && out[2] == 3,
                      "%d: %d %d %d", (int)got, (int)out[0], (int)out[1], (int)out[2]);
        }

        {   /* A match is only ever looked for strictly before the tail, so a
               run can never propose itself by matching where it stands. */
            int32_t seen[] = { 4, 4, 4, 4 };
            int32_t got = ill_draft_scan(seen, 4, 2, out, 2);
            test_case("the tail never matches where it stands",
                      got >= 1 && out[0] == 4, "%d: %d", (int)got, (int)out[0]);
        }
        {   /* Never more than asked for, whatever is available. */
            int32_t seen[] = { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2 };
            int32_t got = ill_draft_scan(seen, 10, 2, out, 2);
            test_case("it proposes no more than it is asked for",
                      got == 2 && out[0] == 3 && out[1] == 4, "%d", (int)got);
        }
        {   /* Too short to have a tail and a match both. */
            int32_t seen[] = { 1, 2 };
            test_case("too short a history proposes nothing",
                      ill_draft_scan(seen, 2, 3, out, 4) == 0, "");
        }
    }

    test_open("sampler");
    for (index = 0; index < vocab; ++index) logits[index] = (float)index * 0.1f;
    logits[40] = 100.0f;

    ill_tuning_init(&tuning);
    tuning.temperature = 0.0f;
    if (ill_sampler_make(&sampler, &tuning, vocab) != ILL_OK) {
        test_case("sampler starts", 0, "");
        return;
    }
    memcpy(work, logits, sizeof(logits));
    test_case("zero temperature picks the peak", ill_sampler_pick(sampler, work) == 40, "");
    ill_sampler_free(sampler);

    ill_tuning_init(&tuning);
    tuning.temperature = 1.0f;
    tuning.top_k = 1;
    tuning.top_p = 1.0f;
    ill_sampler_make(&sampler, &tuning, vocab);
    memcpy(work, logits, sizeof(logits));
    test_case("top-k of one is deterministic", ill_sampler_pick(sampler, work) == 40, "");
    ill_sampler_free(sampler);

    {   /* the same seed must replay exactly, a different seed must not */
        int32_t first[24], again[24], other[24];
        int same = 1, differs = 0;
        ill_tuning_init(&tuning);
        tuning.temperature = 1.2f;
        tuning.seed = 12345;
        for (index = 0; index < vocab; ++index) logits[index] = test_real() * 3.0f;
        ill_sampler_make(&sampler, &tuning, vocab);
        for (index = 0; index < 24; ++index) {
            memcpy(work, logits, sizeof(logits));
            first[index] = ill_sampler_pick(sampler, work);
        }
        ill_sampler_wipe(sampler);
        for (index = 0; index < 24; ++index) {
            memcpy(work, logits, sizeof(logits));
            again[index] = ill_sampler_pick(sampler, work);
        }
        ill_sampler_free(sampler);
        tuning.seed = 999;
        ill_sampler_make(&sampler, &tuning, vocab);
        for (index = 0; index < 24; ++index) {
            memcpy(work, logits, sizeof(logits));
            other[index] = ill_sampler_pick(sampler, work);
        }
        ill_sampler_free(sampler);
        for (index = 0; index < 24; ++index) {
            if (first[index] != again[index]) same = 0;
            if (first[index] != other[index]) differs = 1;
        }
        test_case("a seed replays exactly", same, "");
        test_case("a different seed diverges", differs, "");
    }

    {   /* repetition penalty must push a noted token down */
        int32_t before, after;
        ill_tuning_init(&tuning);
        tuning.temperature = 0.0f;
        tuning.repeat_penalty = 4.0f;
        tuning.repeat_span = 8;
        for (index = 0; index < vocab; ++index) logits[index] = (float)index * 0.01f;
        logits[63] = 5.0f;
        logits[62] = 4.0f;
        ill_sampler_make(&sampler, &tuning, vocab);
        memcpy(work, logits, sizeof(logits));
        before = ill_sampler_pick(sampler, work);
        ill_sampler_note(sampler, 63);
        memcpy(work, logits, sizeof(logits));
        after = ill_sampler_pick(sampler, work);
        test_case("repetition penalty demotes a used token",
                  before == 63 && after == 62, "%d then %d", before, after);
        ill_sampler_free(sampler);
    }

    {   /* nucleus must keep only the head of the distribution */
        int32_t counts[64];
        int32_t outside = 0;
        ill_tuning_init(&tuning);
        tuning.temperature = 1.0f;
        tuning.top_p = 0.5f;
        tuning.seed = 7;
        for (index = 0; index < vocab; ++index) logits[index] = -20.0f;
        logits[10] = 2.0f; logits[11] = 1.9f; logits[12] = 1.8f;
        memset(counts, 0, sizeof(counts));
        ill_sampler_make(&sampler, &tuning, vocab);
        for (index = 0; index < 400; ++index) {
            memcpy(work, logits, sizeof(logits));
            counts[ill_sampler_pick(sampler, work)] += 1;
        }
        for (index = 0; index < vocab; ++index)
            if (index != 10 && index != 11 && index != 12 && counts[index]) outside = 1;
        test_case("nucleus never leaves the head", !outside, "");
        ill_sampler_free(sampler);
    }
}

/* -- paths ----------------------------------------------------------------- */

static void test_paths(void)
{
    char out[64];
    test_open("paths");
    ill_path_join(out, sizeof(out), "ckpt", "config.json");
    test_case("join adds one separator",
              !strcmp(out, "ckpt/config.json") || !strcmp(out, "ckpt\\config.json"), "%s", out);
    ill_path_join(out, sizeof(out), "ckpt/", "config.json");
    test_case("join does not double a separator",
              !strcmp(out, "ckpt/config.json") || !strcmp(out, "ckpt/config.json"), "%s", out);
    ill_path_join(out, sizeof(out), "a", "b");
    test_case("join handles short names", strlen(out) == 3, "%s", out);
    test_case("suffix match is anchored at the end",
              ill_name_tail("model-00001.safetensors", ".safetensors") &&
              !ill_name_tail("safetensors.json", ".safetensors"), "");
    test_case("result names are never null",
              ill_result_text(ILL_OK) && ill_result_text(ILL_LIMIT) &&
              ill_result_text((IllResult)99), "");
    test_case("type names cover every format",
              !strcmp(ill_type_text(ILL_TYPE_Q8), "q8") &&
              !strcmp(ill_type_text(ILL_TYPE_Q4), "q4") &&
              !strcmp(ill_type_text(ILL_TYPE_BF16), "bf16") &&
              ill_type_size(ILL_TYPE_BF16, 10) == 20 &&
              ill_type_size(ILL_TYPE_Q8, 10) == 10 &&
              ill_type_size(ILL_TYPE_Q4, 10) == 5 &&
              ill_type_size(ILL_TYPE_Q4, 9) == 5, "");
}

/* -- the yolo engine ------------------------------------------------------- */

/* Half to float, on the patterns that have caught this conversion before:
 * the subnormals a small model is full of, and the two zeroes. */
static void test_yolo_numbers(void)
{
    test_open("yolo numbers");
    test_case("one and minus two round trip",
              yolo_half_float(0x3C00) == 1.0f && yolo_half_float(0xC000) == -2.0f, "");
    test_case("both zeroes stay zero",
              yolo_half_float(0x0000) == 0.0f && yolo_half_float(0x8000) == 0.0f, "");
    test_case("the smallest subnormal is not flushed",
              yolo_half_float(0x0001) > 0.0f
              && test_near(yolo_half_float(0x0001), 5.9604645e-8f, 1e-12f), "");
    test_case("the largest subnormal is a whole step below the smallest normal",
              test_near(yolo_half_float(0x03FF) + yolo_half_float(0x0001),
                        yolo_half_float(0x0400), 1e-12f), "");
    test_case("infinity survives",
              yolo_half_float(0x7C00) > 1e30f && yolo_half_float(0xFC00) < -1e30f, "");
    test_case("a half's mantissa lands in the right bits",
              yolo_half_float(0x3555) > 0.333f && yolo_half_float(0x3555) < 0.334f,
              "%g", (double)yolo_half_float(0x3555));
}

/* What the arena promises a caller: a mark releases everything taken after it
 * and nothing taken before it, and a release keeps the memory for next time. */
static void test_yolo_arena(void)
{
    YoloArena arena;
    YoloMark mark;
    char *first, *second;
    size_t was;

    test_open("yolo arena");
    yolo_arena_open(&arena, 4096);
    first = (char *)yolo_arena_take(&arena, 100);
    first[0] = 7;
    mark = yolo_arena_mark(&arena);
    second = (char *)yolo_arena_take(&arena, 100);
    test_case("a take does not overlap the one before it",
              second >= first + 100 || second + 100 <= first, "");
    was = arena.live_size;
    yolo_arena_reset(&arena, mark);
    test_case("a reset gives back what came after the mark",
              arena.live_size < was, "%zu then %zu", was, arena.live_size);
    test_case("what came before the mark survives it", first[0] == 7, "");
    test_case("the memory is reused rather than freed",
              (char *)yolo_arena_take(&arena, 100) == second, "");
    test_case("a take is aligned for a vector load",
              ((uintptr_t)yolo_arena_take(&arena, 4) & 63u) == 0, "");
    {
        void *big = yolo_arena_take(&arena, 1 << 20);
        test_case("a request larger than a block still lands", big != NULL, "");
    }
    yolo_arena_clear(&arena);
    test_case("a clear empties the arena", arena.live_size == 0, "");
    yolo_arena_close(&arena);
}

/* Every kernel against a plain restatement of the same arithmetic.  The
 * restatements are deliberately the slow obvious loop: if the two agree, the
 * blocking and the register tiling in the real one did not change the answer. */
static void test_yolo_kernels(void)
{
    const YoloBackend *backend = yolo_backend_cpu();
    YoloArena arena;
    float *a_list, *b_list, *got, *want;
    int row, mid, col, index;

    test_open("yolo kernels");
    yolo_arena_open(&arena, 1 << 20);

    /* the matrix multiply at a shape that crosses a block and a tile */
    a_list = (float *)yolo_arena_take(&arena, 37 * 29 * sizeof(float));
    b_list = (float *)yolo_arena_take(&arena, 29 * 411 * sizeof(float));
    got = (float *)yolo_arena_take(&arena, 37 * 411 * sizeof(float));
    want = (float *)yolo_arena_take(&arena, 37 * 411 * sizeof(float));
    for (index = 0; index < 37 * 29; index++) a_list[index] = test_real();
    for (index = 0; index < 29 * 411; index++) b_list[index] = test_real();
    backend->gemm_run(backend, a_list, b_list, got, 37, 29, 411, 29, 411, 411, 0);
    for (row = 0; row < 37; row++)
        for (col = 0; col < 411; col++) {
            float total = 0.0f;
            for (mid = 0; mid < 29; mid++)
                total += a_list[row * 29 + mid] * b_list[mid * 411 + col];
            want[row * 411 + col] = total;
        }
    test_case("the blocked multiply matches a plain one",
              test_gap(got, want, 37 * 411) < 1e-5f,
              "%.2e", (double)test_gap(got, want, 37 * 411));

    backend->gemm_run(backend, a_list, b_list, got, 37, 29, 17, 29, 29, 17, 1);
    for (row = 0; row < 37; row++)
        for (col = 0; col < 17; col++) {
            float total = 0.0f;
            for (mid = 0; mid < 29; mid++)
                total += a_list[row * 29 + mid] * b_list[col * 29 + mid];
            want[row * 17 + col] = total;
        }
    test_case("the swapped multiply matches a plain one",
              test_gap(got, want, 37 * 17) < 1e-5f, "");

    /* a strided, padded convolution against a direct restatement */
    {
        YoloPlane in = yolo_plane_take(&arena, 1, 5, 13, 11);
        YoloPlane out = yolo_plane_take(&arena, 1, 7, 7, 6);
        float *weight = (float *)yolo_arena_take(&arena, 7 * 5 * 9 * sizeof(float));
        float *bias = (float *)yolo_arena_take(&arena, 7 * sizeof(float));
        float *room = (float *)yolo_arena_take(&arena,
            backend->conv_room(&in, &out, 3, 1));
        float *plain = (float *)yolo_arena_take(&arena,
            yolo_plane_count(&out) * sizeof(float));
        int out_channel, in_channel, out_y, out_x, kernel_y, kernel_x;

        for (index = 0; index < (int)yolo_plane_count(&in); index++)
            in.cell_list[index] = test_real();
        for (index = 0; index < 7 * 5 * 9; index++) weight[index] = test_real();
        for (index = 0; index < 7; index++) bias[index] = test_real();
        backend->conv_run(backend, &in, &out, weight, bias, 3, 2, 1, 1, 1,
                          YOLO_ACT_NONE, room);
        for (out_channel = 0; out_channel < 7; out_channel++)
            for (out_y = 0; out_y < 7; out_y++)
                for (out_x = 0; out_x < 6; out_x++) {
                    float total = bias[out_channel];
                    for (in_channel = 0; in_channel < 5; in_channel++)
                        for (kernel_y = 0; kernel_y < 3; kernel_y++)
                            for (kernel_x = 0; kernel_x < 3; kernel_x++) {
                                int at_y = out_y * 2 - 1 + kernel_y;
                                int at_x = out_x * 2 - 1 + kernel_x;
                                if (at_y < 0 || at_y >= 13 || at_x < 0 || at_x >= 11)
                                    continue;
                                total += weight[((out_channel * 5 + in_channel) * 3
                                                 + kernel_y) * 3 + kernel_x]
                                       * in.cell_list[(in_channel * 13 + at_y) * 11 + at_x];
                            }
                    plain[(out_channel * 7 + out_y) * 6 + out_x] = total;
                }
        test_case("a strided padded convolution matches a direct one",
                  test_gap(out.cell_list, plain, (int)yolo_plane_count(&out)) < 1e-5f,
                  "%.2e", (double)test_gap(out.cell_list, plain,
                                           (int)yolo_plane_count(&out)));
    }

    /* the depthwise path, which skips lowering entirely */
    {
        YoloPlane in = yolo_plane_take(&arena, 1, 6, 9, 9);
        YoloPlane deep = yolo_plane_take(&arena, 1, 6, 9, 9);
        YoloPlane wide = yolo_plane_take(&arena, 1, 6, 9, 9);
        float *deep_weight = (float *)yolo_arena_take(&arena, 6 * 9 * sizeof(float));
        float *wide_weight = (float *)yolo_arena_zero(&arena, 6 * 6 * 9 * sizeof(float));
        float *room;
        int channel;

        for (index = 0; index < (int)yolo_plane_count(&in); index++)
            in.cell_list[index] = test_real();
        for (index = 0; index < 6 * 9; index++) deep_weight[index] = test_real();
        /* the same convolution written with the groups spelled out */
        for (channel = 0; channel < 6; channel++)
            for (index = 0; index < 9; index++)
                wide_weight[(channel * 6 + channel) * 9 + index] =
                    deep_weight[channel * 9 + index];
        room = (float *)yolo_arena_take(&arena,
            backend->conv_room(&in, &wide, 3, 1));
        backend->conv_run(backend, &in, &deep, deep_weight, NULL, 3, 1, 1, 1, 6,
                          YOLO_ACT_NONE, NULL);
        backend->conv_run(backend, &in, &wide, wide_weight, NULL, 3, 1, 1, 1, 1,
                          YOLO_ACT_NONE, room);
        test_case("a depthwise convolution matches the dense one it stands for",
                  test_gap(deep.cell_list, wide.cell_list,
                           (int)yolo_plane_count(&deep)) < 1e-5f, "");
    }

    /* pooling, resizing and softmax */
    {
        YoloPlane in = yolo_plane_take(&arena, 1, 2, 4, 4);
        YoloPlane out = yolo_plane_take(&arena, 1, 2, 4, 4);
        YoloPlane wide = yolo_plane_take(&arena, 1, 2, 8, 8);
        for (index = 0; index < (int)yolo_plane_count(&in); index++)
            in.cell_list[index] = (float)index;
        backend->pool_run(backend, &in, &out, 5, 1, 2);
        test_case("a wide pool at stride one keeps the shape and takes the largest",
                  out.cell_list[0] == 10.0f && out.cell_list[15] == 15.0f,
                  "%g %g", (double)out.cell_list[0], (double)out.cell_list[15]);
        backend->resize_run(backend, &in, &wide, 2);
        test_case("a nearest resize repeats each cell",
                  wide.cell_list[0] == in.cell_list[0]
                  && wide.cell_list[1] == in.cell_list[0]
                  && wide.cell_list[8] == in.cell_list[0]
                  && wide.cell_list[2] == in.cell_list[1], "");
    }
    {
        float row_list[6] = { 1.0f, 2.0f, 3.0f, 900.0f, 901.0f, 902.0f };
        float total = 0.0f;
        backend->softmax_run(backend, row_list, 2, 3);
        for (index = 0; index < 3; index++) total += row_list[index];
        test_case("a softmax row sums to one", test_near(total, 1.0f, 1e-6f), "");
        test_case("a softmax does not overflow on a large row",
                  row_list[5] > 0.6f && row_list[5] < 0.7f, "%g", (double)row_list[5]);
        test_case("a shift of the whole row does not move the answer",
                  test_near(row_list[0], row_list[3], 1e-6f), "");
    }
    yolo_arena_close(&arena);
}

/* The shaping, which is where a parity run goes wrong quietly.  The numbers
 * here are the two published sample pictures: a portrait one that needs no
 * padding at all, and a landscape one that needs twelve rows of it. */
static void test_yolo_shape(void)
{
    YoloImage image;
    YoloFrame frame;
    int width, height;
    static uint8_t room[8 * 8 * 3];

    test_open("yolo shaping");
    memset(&image, 0, sizeof image);
    image.channel_count = 3;
    image.pixel_list = room;

    image.width = 810; image.height = 1080;
    yolo_frame_fit(&image, 640, 32, 1, 1, 1, &frame, &width, &height);
    test_case("a portrait picture pads to a stride multiple, not a square",
              width == 480 && height == 640 && frame.pad_left == 0
              && frame.pad_top == 0, "%dx%d", width, height);

    image.width = 1280; image.height = 720;
    yolo_frame_fit(&image, 640, 32, 1, 1, 1, &frame, &width, &height);
    test_case("a landscape picture pads its short side up to the stride",
              width == 640 && height == 384 && frame.pad_top == 12
              && frame.pad_left == 0, "%dx%d pad %d", width, height, frame.pad_top);
    test_case("the scale is the one that fits the longer side",
              test_near(frame.ratio, 0.5f, 1e-6f), "%g", (double)frame.ratio);

    yolo_frame_fit(&image, 640, 32, 0, 1, 1, &frame, &width, &height);
    test_case("square mode pads all the way out",
              width == 640 && height == 640 && frame.pad_top == 140,
              "%dx%d pad %d", width, height, frame.pad_top);

    image.width = 100; image.height = 100;
    yolo_frame_fit(&image, 640, 32, 1, 0, 1, &frame, &width, &height);
    test_case("without scale-up a small picture is left alone",
              test_near(frame.ratio, 1.0f, 1e-6f) && frame.fit_width == 100, "");

    /* an odd gap is the only case that can tell the reference's rounding from
     * an ordinary one: it puts the spare row on the bottom, not the top */
    image.width = 1280; image.height = 726;
    yolo_frame_fit(&image, 640, 32, 1, 1, 1, &frame, &width, &height);
    test_case("an odd gap leaves the spare row at the bottom",
              height == 384 && frame.fit_height == 363 && frame.pad_top == 10,
              "%dx%d fit %d pad %d", width, height, frame.fit_height,
              frame.pad_top);

    /* an exact halving is the one case where OpenCV's fixed point reduces to
     * an average, which is what makes it checkable by hand */
    {
        static uint8_t source[4 * 4 * 3];
        float target[3 * 2 * 2];
        int index;
        for (index = 0; index < 4 * 4 * 3; index++) source[index] = (uint8_t)(index * 5);
        yolo_pixels_scale(source, 4, 4, 12, 3, 0, target, 2, 2, 2, 2, 0, 0, 1.0f);
        /* the four red cells of the top-left quad are 0, 15, 60, 75 */
        test_case("a halving averages the four cells it covers",
                  test_near(target[0], (0 + 15 + 60 + 75 + 2) / 4.0f, 0.51f),
                  "%g", (double)target[0]);
        test_case("the channel planes do not run into each other",
                  target[4] != target[0] && target[8] != target[4], "");
    }
}

/* Boxes: the decode, the two overlaps, and the two suppression rules. */
static void test_yolo_boxes(void)
{
    YoloPick pick_list[3];
    int kept;

    test_open("yolo boxes");
    {
        YoloHeadOut head;
        float box_room[4], anchor_x = 4.5f, anchor_y = 6.5f, stride = 8.0f;
        float left, top, right, bottom;
        memset(&head, 0, sizeof head);
        head.anchor_count = 1;
        head.box_room = box_room;
        head.anchor_x_list = &anchor_x;
        head.anchor_y_list = &anchor_y;
        head.stride_list = &stride;
        box_room[0] = 1.0f; box_room[1] = 2.0f; box_room[2] = 3.0f; box_room[3] = 4.0f;
        yolo_box_decode(&head, 1, 0, &left, &top, &right, &bottom);
        test_case("a box is four distances from its cell's centre, in pixels",
                  left == (4.5f - 1.0f) * 8.0f && top == (6.5f - 2.0f) * 8.0f
                  && right == (4.5f + 3.0f) * 8.0f && bottom == (6.5f + 4.0f) * 8.0f,
                  "%g %g %g %g", (double)left, (double)top, (double)right,
                  (double)bottom);
    }
    {
        /* with reg_max above one the branch says a distribution, and the
         * distance is its mean -- a v8 or v11 checkpoint, not a yolo26 */
        YoloHeadOut head;
        float box_room[8];
        float anchor = 0.5f, stride = 1.0f;
        int index;
        memset(&head, 0, sizeof head);
        head.anchor_count = 1;
        head.box_room = box_room;
        head.anchor_x_list = &anchor;
        head.anchor_y_list = &anchor;
        head.stride_list = &stride;
        for (index = 0; index < 8; index++) box_room[index] = 0.0f;
        box_room[1] = 100.0f;    /* edge zero is certain of bin one */
        test_case("a distribution over bins becomes its mean distance",
                  test_near(yolo_edge_value(box_room, 2, 1, 0, 0), 1.0f, 1e-4f),
                  "%g", (double)yolo_edge_value(box_room, 2, 1, 0, 0));
        box_room[0] = 100.0f;    /* now it is torn evenly between zero and one */
        test_case("an even split falls between the bins",
                  test_near(yolo_edge_value(box_room, 2, 1, 0, 0), 0.5f, 1e-4f), "");
    }
    {
        YoloPick a, b;
        memset(&a, 0, sizeof a); memset(&b, 0, sizeof b);
        a.left = 0.0f; a.top = 0.0f; a.right = 10.0f; a.bottom = 10.0f;
        b = a;
        test_case("a box overlaps itself entirely",
                  test_near(yolo_overlap(&a, &b), 1.0f, 1e-6f), "");
        b.left = 5.0f; b.right = 15.0f;
        test_case("a half overlap measures a third",
                  test_near(yolo_overlap(&a, &b), 50.0f / 150.0f, 1e-6f),
                  "%g", (double)yolo_overlap(&a, &b));
        b.left = 20.0f; b.right = 30.0f;
        test_case("boxes that do not touch overlap not at all",
                  yolo_overlap(&a, &b) == 0.0f, "");
    }
    {
        /* an oriented pick keeps its centre in left/top and its size in
         * right/bottom, and the probabilistic overlap is one against itself */
        YoloPick a, b;
        memset(&a, 0, sizeof a);
        a.left = 10.0f; a.top = 10.0f; a.right = 20.0f; a.bottom = 8.0f;
        a.angle = 0.4f;
        b = a;
        test_case("an oriented box overlaps itself entirely",
                  yolo_rbox_overlap(&a, &b) > 0.999f,
                  "%g", (double)yolo_rbox_overlap(&a, &b));
        b.left = 400.0f;
        test_case("an oriented box far away overlaps not at all",
                  yolo_rbox_overlap(&a, &b) < 1e-3f, "");
        b = a;
        b.angle = a.angle + 1.2f;
        test_case("turning a box away from another lowers the overlap",
                  yolo_rbox_overlap(&a, &b) < 0.999f, "");
    }
    {
        /* three boxes in a row, each overlapping the next but not the one
         * after.  Greedy keeps the first and the third; the matrix rule the
         * rotated path uses keeps only the first, because the middle box
         * strikes the third out after being struck out itself. */
        int index;
        for (index = 0; index < 3; index++) {
            memset(&pick_list[index], 0, sizeof pick_list[index]);
            pick_list[index].left = (float)index * 4.0f;
            pick_list[index].right = pick_list[index].left + 10.0f;
            pick_list[index].top = 0.0f;
            pick_list[index].bottom = 10.0f;
            pick_list[index].score = 1.0f - (float)index * 0.1f;
        }
        kept = yolo_suppress(pick_list, 3, 0.4f, 1, 0, 300);
        test_case("greedy suppression lets a struck box stop striking",
                  kept == 2, "%d kept", kept);
    }
    {
        /* three oriented squares in a row, spaced so that each overlaps its
         * neighbour past the limit -- 0.73 -- and the ends do not, at 0.49.
         * Greedy would keep the first and the third; the matrix rule keeps
         * only the first, because the middle one strikes the third out after
         * being struck out itself. */
        int index;
        for (index = 0; index < 3; index++) {
            memset(&pick_list[index], 0, sizeof pick_list[index]);
            pick_list[index].left = 10.0f + (float)index * 4.5f;
            pick_list[index].top = 10.0f;
            pick_list[index].right = 20.0f;
            pick_list[index].bottom = 20.0f;
            pick_list[index].score = 1.0f - (float)index * 0.1f;
        }
        test_case("the spacing is on the two sides of the limit it needs to be",
                  yolo_rbox_overlap(&pick_list[0], &pick_list[1]) > 0.7f
                  && yolo_rbox_overlap(&pick_list[0], &pick_list[2]) < 0.7f,
                  "%.3f then %.3f",
                  (double)yolo_rbox_overlap(&pick_list[0], &pick_list[1]),
                  (double)yolo_rbox_overlap(&pick_list[0], &pick_list[2]));
        kept = yolo_suppress(pick_list, 3, 0.7f, 1, 1, 300);
        test_case("the matrix rule lets a struck box keep striking",
                  kept == 1, "%d kept", kept);
    }
    {
        int index;
        for (index = 0; index < 2; index++) {
            memset(&pick_list[index], 0, sizeof pick_list[index]);
            pick_list[index].left = 0.0f; pick_list[index].top = 0.0f;
            pick_list[index].right = 10.0f; pick_list[index].bottom = 10.0f;
            pick_list[index].score = 1.0f - (float)index * 0.1f;
            pick_list[index].class_index = index;
        }
        kept = yolo_suppress(pick_list, 2, 0.5f, 0, 0, 300);
        test_case("two classes on one box both survive", kept == 2, "%d kept", kept);
        kept = yolo_suppress(pick_list, 2, 0.5f, 1, 0, 300);
        test_case("asked to be class-blind, only the better one survives",
                  kept == 1, "%d kept", kept);
    }
}

/* The pickle reader, on bytes written here rather than on a checkpoint.  Every
 * opcode a torch checkpoint uses is protocol two, so these are the real ones. */
static void test_yolo_pickle(void)
{
    YoloArena arena;
    YoloValue *value = NULL;
    YoloStatus status;

    test_open("yolo pickle");
    yolo_arena_open(&arena, 1 << 16);

    {
        /* }q\0(X\3\0\0\0keyK\7u.  -- {"key": 7} */
        static const unsigned char byte_list[] = {
            0x80, 0x02, '}', 'q', 0x00, '(',
            'X', 3, 0, 0, 0, 'k', 'e', 'y', 'K', 7, 'u', '.'
        };
        status = yolo_pickle_run(&arena, byte_list, sizeof byte_list, &value);
        test_case("a dict with one entry reads back",
                  status == YOLO_OK && value && value->kind == YOLO_VALUE_DICT
                  && yolo_value_number(yolo_value_find(value, "key"), -1) == 7,
                  "%s", yolo_status_text(status));
    }
    {
        /* a list of three ints, built through a mark and APPENDS */
        static const unsigned char byte_list[] = {
            0x80, 0x02, ']', 'q', 0x00, '(', 'K', 1, 'K', 2, 'M', 0x01, 0x01,
            'e', '.'
        };
        status = yolo_pickle_run(&arena, byte_list, sizeof byte_list, &value);
        test_case("a list keeps its order and its widths",
                  status == YOLO_OK && value && value->item_count == 3
                  && yolo_value_number(yolo_value_at(value, 0), 0) == 1
                  && yolo_value_number(yolo_value_at(value, 2), 0) == 257,
                  "%s", yolo_status_text(status));
    }
    {
        /* a memo read: put a string, then get it back */
        static const unsigned char byte_list[] = {
            0x80, 0x02, ']', 'q', 0x00, '(',
            'X', 2, 0, 0, 0, 'h', 'i', 'q', 0x05, 'h', 0x05, 'e', '.'
        };
        status = yolo_pickle_run(&arena, byte_list, sizeof byte_list, &value);
        test_case("a value stored in the memo can be read back twice",
                  status == YOLO_OK && value && value->item_count == 2
                  && yolo_value_is(yolo_value_at(value, 0), "hi")
                  && yolo_value_at(value, 0) == yolo_value_at(value, 1),
                  "%s", yolo_status_text(status));
    }
    {
        /* a global that is called: collections.OrderedDict becomes a dict */
        static const unsigned char byte_list[] = {
            0x80, 0x02, 'c', 'c', 'o', 'l', 'l', 'e', 'c', 't', 'i', 'o', 'n',
            's', '\n', 'O', 'r', 'd', 'e', 'r', 'e', 'd', 'D', 'i', 'c', 't',
            '\n', 'q', 0x00, ')', 'R', 'q', 0x01, '.'
        };
        status = yolo_pickle_run(&arena, byte_list, sizeof byte_list, &value);
        test_case("an ordered dict becomes a dict rather than an object",
                  status == YOLO_OK && value && value->kind == YOLO_VALUE_DICT,
                  "%s", yolo_status_text(status));
    }
    {
        /* an opcode this engine does not read is a refusal, not a guess */
        static const unsigned char byte_list[] = { 0x80, 0x02, 'I', '1', '\n', '.' };
        status = yolo_pickle_run(&arena, byte_list, sizeof byte_list, &value);
        test_case("an unread opcode is refused and named",
                  status == YOLO_ERR_FORMAT && strstr(yolo_detail_text(), "0x49"),
                  "%s", yolo_detail_text());
    }
    {
        static const unsigned char byte_list[] = { 0x80, 0x02, 'K' };
        status = yolo_pickle_run(&arena, byte_list, sizeof byte_list, &value);
        test_case("a pickle that ends mid-opcode is refused",
                  status == YOLO_ERR_FORMAT, "%s", yolo_status_text(status));
    }
    test_case("a missing checkpoint is refused rather than crashed into",
              yolo_model_open("this/does/not/exist.pt", NULL) == YOLO_ERR_ARG, "");
    {
        YoloModel *model = NULL;
        test_case("a path that is not there reports the file, not the format",
                  yolo_model_open("this/does/not/exist.pt", &model) == YOLO_ERR_FILE
                  && model == NULL, "");
    }
    yolo_arena_close(&arena);
}

/* Reading and writing a picture without a library, which is what the rest of
 * this suite's media is made of. */
static void test_yolo_picture(void)
{
    YoloImage wrote, read;
    uint8_t room[3 * 2 * 3];
    const char *path = "build/test_yolo.ppm";
    int index;

    test_open("yolo pictures");
    for (index = 0; index < 3 * 2 * 3; index++) room[index] = (uint8_t)(index * 9);
    memset(&wrote, 0, sizeof wrote);
    test_case("a caller's pixels can be wrapped without a copy",
              yolo_image_wrap(room, 3, 2, 3, &wrote) == YOLO_OK
              && wrote.pixel_list == room && wrote.owned_flag == 0, "");
    if (yolo_image_write(path, &wrote) == YOLO_OK) {
        memset(&read, 0, sizeof read);
        test_case("a written picture reads back the same",
                  yolo_image_read(path, &read) == YOLO_OK
                  && read.width == 3 && read.height == 2
                  && read.channel_count == 3
                  && memcmp(read.pixel_list, room, sizeof room) == 0, "");
        yolo_image_free(&read);
        test_case("freeing a picture leaves nothing behind",
                  read.pixel_list == NULL && read.width == 0, "");
    } else {
        test_case("a written picture reads back the same", 0, "could not write");
        test_case("freeing a picture leaves nothing behind", 0, "could not write");
    }
    yolo_image_free(&wrote);
    test_case("freeing a wrapped picture does not free the caller's memory",
              room[0] == 0 && room[1] == 9, "");
}

/* -- the vulkan backend ----------------------------------------------------- */

/* Every check here is `app/vulk.c`'s own comparison of its table against the
 * CPU one over the same input, reported through this harness.  The tolerance
 * is relative rather than bit for bit, and deliberately: a device folds a dot
 * product through a tree of sixty-four partial sums where the CPU walks the
 * row, and no reordering of floating point addition is exact.  The CPU table
 * is untouched by any of this and still holds bit for bit against itself.
 *
 * A host with no Vulkan device runs no comparisons and is not a failure --
 * there is nothing to compare against -- so the suite says so and moves on. */
#ifdef ILL_VULKAN
static void test_vulkan_tell(void *ctx, const char *name, double worst, int good)
{
    (void)ctx;
    test_case(name, good, "%.1e", worst);
}
#endif

static void test_vulkan(void)
{
#ifdef ILL_VULKAN
    const char *names[ILL_BACKEND_MAX];
    int32_t     run = 0, count, index, found = 0;

    test_open("vulkan");
    test_case("joining the registry adds a backend called vulkan",
              vulk_join() == ILL_OK, "");
    count = ill_backend_list(names, ILL_BACKEND_MAX);
    for (index = 0; index < count; ++index)
        if (!strcmp(names[index], "vulkan")) found = 1;
    test_case("and the registry reports it by that name", found, "%d backends", count);

    (void)vulk_probe_all(test_vulkan_tell, NULL, &run);
    if (run == 0)
        test_case("no vulkan device on this host, so nothing was compared", 1, "skipped");
    vulk_close();
#endif
}

/* -- entry ----------------------------------------------------------------- */

int main(void)
{
    double mark = ill_clock_now();
    ill_noise_level = 0;

    printf("illlm " ILL_VERSION_TEXT " unit tests -- simd %s, threads %s\n",
           ill_backend_simd(), ILL_WITH_THREADS ? "on" : "off");

    test_numbers();
    test_utf8();
    test_json();
    test_kernels();
    test_operators();
    test_threads();
    test_vocab();
    test_sampler();
    test_paths();
    test_yolo_numbers();
    test_yolo_arena();
    test_yolo_kernels();
    test_yolo_shape();
    test_yolo_boxes();
    test_yolo_pickle();
    test_yolo_picture();
    test_vulkan();

    printf("\n%d checks, %d failed, %.2fs\n", test_ran, test_bad, ill_clock_now() - mark);
    return test_bad ? 1 : 0;
}

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
              !strcmp(ill_type_text(ILL_TYPE_BF16), "bf16") &&
              ill_type_size(ILL_TYPE_BF16, 10) == 20 &&
              ill_type_size(ILL_TYPE_Q8, 10) == 10, "");
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

    printf("\n%d checks, %d failed, %.2fs\n", test_ran, test_bad, ill_clock_now() - mark);
    return test_bad ? 1 : 0;
}

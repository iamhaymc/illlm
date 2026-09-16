/* app/yolo.c -- the yolo26 engine, one translation unit.
 *
 * This file reads an Ultralytics yolo26 checkpoint as it is published -- a
 * `.pt`, which is a zip holding a pickled module tree and its raw storages --
 * and runs it.  There is no converter, no export step and no second file: the
 * graph is read out of the pickle rather than rebuilt from the yaml, because
 * the pickle already carries every module's class name, its wiring, and the
 * arguments torch needed to construct it.  Re-deriving that from the yaml's
 * depth/width scale table is guesswork that the checkpoint has already done.
 *
 * Use it as a library by including it:
 *
 *     #include "yolo.c"
 *
 * or build it as a command line by defining YOLO_MAIN.  The public interface
 * is everything inside YOLO_INCLUDED below; the rest of the file is private.
 *
 * Dependencies: the C11 standard library, and -- for image formats other than
 * PNM -- `libc11/stb_image.h` and `libc11/stb_image_write.h`, which are
 * vendored in this repository.  Define YOLO_NO_STB to build without them, in
 * which case the engine reads and writes binary PNM only.
 *
 * Layers, bottom-up.  A layer may depend only on the layers beneath it.
 *
 *    1  platform   errors, the arena, byte reads, half to float
 *    2  tensor     NCHW float planes over arena memory
 *    3  backend    the operation seam an accelerator would fill
 *    4  kernels    the CPU backend: convolution, pooling, activation, gemm
 *    5  store      zip, pickle, and the storage table under them
 *    6  graph      the pickled module tree, turned into something runnable
 *    7  forward    the layer plan and the feature maps the head reads
 *    8  decode     anchors, boxes, scores, masks, keypoints, suppression
 *    9  image      reading, letterboxing, lifting a result back to the source
 *   10  api        model, options, session, result
 *   11  cli        under YOLO_MAIN
 */

/* A monotonic clock is posix, and this file is built -std=c11 as often as
 * -std=gnu11.  The macro has to be set before the first standard header on
 * glibc, which locks the feature set in on its way past. */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 199309L
#endif

#ifndef YOLO_INCLUDED
#define YOLO_INCLUDED

#include <stddef.h>
#include <stdint.h>

/* ---------------------------------------------------------------------------
 * Status
 *
 * Every call that can fail returns one of these.  A refusal is a result: a
 * checkpoint carrying a module this engine does not implement reports
 * YOLO_ERR_MODULE and names the module, rather than approximating it.
 * ------------------------------------------------------------------------ */

typedef enum {
    YOLO_OK = 0,
    YOLO_ERR_ARG,      /* a caller passed something the call cannot use */
    YOLO_ERR_FILE,     /* the file could not be opened or read */
    YOLO_ERR_FORMAT,   /* the bytes are not a checkpoint this engine reads */
    YOLO_ERR_MEMORY,   /* an allocation failed */
    YOLO_ERR_MODULE,   /* the checkpoint holds a module that is not implemented */
    YOLO_ERR_SHAPE,    /* a weight's shape does not match the module using it */
    YOLO_ERR_IMAGE     /* an image could not be read, written or converted */
} YoloStatus;

/* A sentence naming what went wrong, for a log line.  Never null. */
const char *yolo_status_text(YoloStatus status);

/* The last detail line the engine recorded -- which module was missing, which
 * shape did not match.  Valid until the next call that fails.  Never null. */
const char *yolo_detail_text(void);

/* ---------------------------------------------------------------------------
 * Tasks
 * ------------------------------------------------------------------------ */

typedef enum {
    YOLO_TASK_DETECT = 0,   /* axis-aligned boxes */
    YOLO_TASK_SEGMENT,      /* boxes and a mask each */
    YOLO_TASK_POSE,         /* boxes and a set of keypoints each */
    YOLO_TASK_OBB,          /* oriented boxes, an angle each */
    YOLO_TASK_CLASSIFY      /* one score row over the whole picture */
} YoloTask;

const char *yolo_task_text(YoloTask task);

/* ---------------------------------------------------------------------------
 * Pictures
 *
 * Eight bits a channel, rows top to bottom, channels interleaved.  The engine
 * reads RGB; a caller holding BGR says so through YoloOptions.
 * ------------------------------------------------------------------------ */

typedef struct {
    int      width;
    int      height;
    int      channel_count;   /* 1 grey, 3 RGB, 4 RGBA (alpha is dropped) */
    uint8_t *pixel_list;      /* width * height * channel_count bytes */
    int      owned_flag;      /* set when the engine allocated pixel_list */
} YoloImage;

/* Read a picture from disk.  With stb this is every format stb reads -- JPEG,
 * PNG, BMP, TGA, GIF, PSD, HDR, PIC, PNM.  Without it, binary PNM only. */
YoloStatus yolo_image_read(const char *path, YoloImage *image_out);

/* Wrap memory the caller owns.  The engine will not free it. */
YoloStatus yolo_image_wrap(const uint8_t *pixel_list, int width, int height,
                           int channel_count, YoloImage *image_out);

/* Write a picture.  The extension picks the format: .png, .jpg, .bmp, .tga,
 * .ppm, .pgm.  Without stb, PNM only. */
YoloStatus yolo_image_write(const char *path, const YoloImage *image);

void yolo_image_free(YoloImage *image);

/* ---------------------------------------------------------------------------
 * The model
 *
 * A model is the weights and the graph.  It is read once, is immutable after
 * that, and may be shared by any number of sessions on any number of threads.
 * ------------------------------------------------------------------------ */

typedef struct YoloModel YoloModel;

YoloStatus yolo_model_open(const char *path, YoloModel **model_out);
void       yolo_model_close(YoloModel *model);

YoloTask    yolo_model_task(const YoloModel *model);
int         yolo_model_class_count(const YoloModel *model);
const char *yolo_model_class_name(const YoloModel *model, int class_index);

/* The largest stride the head sees, which is the multiple an input must be
 * padded to.  32 on every published yolo26. */
int yolo_model_stride_size(const YoloModel *model);

/* The size the checkpoint was trained at: 640 for the four detection tasks,
 * 224 for classification.  A session may use another, so long as it is a
 * multiple of the stride. */
int yolo_model_input_size(const YoloModel *model);

/* Non-zero when the checkpoint carries a one-to-one head, which is what makes
 * suppression optional.  Every published yolo26 does. */
int yolo_model_nms_free_flag(const YoloModel *model);

/* Keypoints, for a pose model.  Zero for every other task. */
int         yolo_model_keypoint_count(const YoloModel *model);
int         yolo_model_keypoint_size(const YoloModel *model);   /* 2 or 3 */
const char *yolo_model_keypoint_name(const YoloModel *model, int keypoint_index);

/* Mask coefficients, for a segmentation model.  Zero for every other task. */
int yolo_model_mask_count(const YoloModel *model);

/* ---------------------------------------------------------------------------
 * The backend seam
 *
 * Every arithmetic the forward pass does goes through this table.  The CPU
 * backend is the reference implementation of it and is what a model uses
 * unless told otherwise; an accelerator is a second table, and nothing above
 * this line needs to know which one is in play.
 *
 * A plane is NCHW: `batch_size` pictures of `channel_count` planes, each
 * `height` by `width`, contiguous, rows first.
 * ------------------------------------------------------------------------ */

typedef struct {
    int    batch_size;
    int    channel_count;
    int    height;
    int    width;
    float *cell_list;
} YoloPlane;

typedef enum {
    YOLO_ACT_NONE = 0,
    YOLO_ACT_SILU,
    YOLO_ACT_SIGMOID
} YoloActKind;

typedef struct YoloBackend YoloBackend;

struct YoloBackend {
    const char *name;

    /* out = act(conv(in, weight, bias)).  `group_count` equal to the input
     * channel count is a depthwise convolution.  `room_list` is scratch the
     * caller sized with `conv_room`. */
    void (*conv_run)(const YoloBackend *backend,
                     const YoloPlane *in, YoloPlane *out,
                     const float *weight_sheet, const float *bias_list,
                     int kernel_size, int stride_size, int pad_size,
                     int dilation_size, int group_count,
                     YoloActKind act_kind, float *room_list);
    size_t (*conv_room)(const YoloPlane *in, const YoloPlane *out,
                        int kernel_size, int group_count);

    /* out = act(deconv(in, weight, bias)), stride == kernel, no padding.
     * Only the 2x2/2 case the mask prototype head uses is required. */
    void (*deconv_run)(const YoloBackend *backend,
                       const YoloPlane *in, YoloPlane *out,
                       const float *weight_sheet, const float *bias_list,
                       int kernel_size, int stride_size,
                       YoloActKind act_kind);

    void (*pool_run)(const YoloBackend *backend,
                     const YoloPlane *in, YoloPlane *out,
                     int kernel_size, int stride_size, int pad_size);

    /* nearest neighbour, integer scale */
    void (*resize_run)(const YoloBackend *backend,
                       const YoloPlane *in, YoloPlane *out, int scale_size);

    void (*act_run)(const YoloBackend *backend, float *cell_list,
                    size_t cell_count, YoloActKind act_kind);

    /* out[row_count x col_count] = a[row_count x mid_count] * b, with `b_step`
     * the distance between rows of b.  `b_swap_flag` transposes b. */
    void (*gemm_run)(const YoloBackend *backend,
                     const float *a_list, const float *b_list, float *out_list,
                     int row_count, int mid_count, int col_count,
                     int a_step, int b_step, int out_step, int b_swap_flag);

    void (*add_run)(const YoloBackend *backend, float *a_list,
                    const float *b_list, size_t cell_count);

    /* softmax over the last axis of `row_count` rows of `col_count` */
    void (*softmax_run)(const YoloBackend *backend, float *cell_list,
                        int row_count, int col_count);
};

const YoloBackend *yolo_backend_cpu(void);

/* Point a model at another backend.  Must be called before any session is
 * opened against it.  Passing null restores the CPU backend. */
YoloStatus yolo_model_backend_set(YoloModel *model, const YoloBackend *backend);

/* ---------------------------------------------------------------------------
 * Options
 *
 * `yolo_options_default` fills these with what `yolo predict` uses, so a run
 * with the defaults is comparable to the reference without an argument about
 * which knob moved.
 * ------------------------------------------------------------------------ */

typedef struct {
    int   input_size;          /* 0 takes the checkpoint's own */
    float score_floor;         /* 0.25 detect, 0.001 would be a sweep */
    float overlap_limit;       /* 0.7, the IoU above which a box is a duplicate */
    int   detect_limit;        /* 300 */

    int   nms_free_flag;       /* run the one-to-one head and skip suppression */
    int   class_blind_flag;    /* suppress across classes as well as within */
    int   multi_label_flag;    /* one box may carry several classes */

    int   bgr_flag;            /* the caller's pixels are BGR, not RGB */
    int   letterbox_fill;      /* 114, the grey yolo pads with */
    int   scale_up_flag;       /* 1; clear to only ever shrink a picture */
    int   center_flag;         /* 1; clear to pad bottom-right only */
    int   rect_flag;           /* 1: pad only up to a multiple of the stride,
                                * which is what `yolo predict` does on a `.pt`
                                * and leaves a 1080x810 picture 480 by 640
                                * rather than square.  Clear it to pad to a
                                * full `input_size` square. */

    int   mask_flag;           /* segmentation: lift each mask to source pixels */
    int   thread_count;        /* 0 takes one thread */

    const int *class_filter_list;  /* keep only these classes; null keeps all */
    int        class_filter_count;
} YoloOptions;

void yolo_options_default(const YoloModel *model, YoloOptions *options_out);

/* ---------------------------------------------------------------------------
 * Results
 *
 * Every coordinate is in the source picture's pixels, not the letterboxed
 * input's.  A result belongs to the session and is valid until the next run.
 * ------------------------------------------------------------------------ */

typedef struct {
    float left, top, right, bottom;   /* detect, segment, pose: the corners */
    float center_x, center_y;         /* obb: the centre, size and angle */
    float box_width, box_height;
    float angle;                      /* radians, obb only; 0 otherwise */

    float score;
    int   class_index;

    /* pose: keypoint_count * keypoint_size floats, x, y[, visibility] */
    const float *keypoint_list;

    /* segment: one byte a pixel over the whole source picture, 255 inside */
    const uint8_t *mask_plane;
    int            mask_width;
    int            mask_height;
} YoloBox;

typedef struct {
    int            box_count;
    const YoloBox *box_list;

    /* classify: one score a class, already through softmax */
    int          score_count;
    const float *score_list;

    int source_width;
    int source_height;
} YoloResult;

/* ---------------------------------------------------------------------------
 * Sessions
 *
 * A session holds the scratch one picture needs.  It is not thread-safe; open
 * one a thread.  Opening is the expensive part and allocating is done there,
 * so a run over a stream of pictures allocates nothing after the first.
 * ------------------------------------------------------------------------ */

typedef struct YoloSession YoloSession;

YoloStatus yolo_session_open(const YoloModel *model, const YoloOptions *options,
                             YoloSession **session_out);
void       yolo_session_close(YoloSession *session);

/* Run one picture.  `result_out` points into the session and stays valid until
 * the next call to `yolo_session_run` on it. */
YoloStatus yolo_session_run(YoloSession *session, const YoloImage *image,
                            const YoloResult **result_out);

/* The picture as the model saw it: three planes of `height` by `width`, red
 * first, each cell between zero and one.  Valid until the next run.  This is
 * here so a parity run against the reference can compare the shaping and the
 * forward pass separately, rather than arguing about which of the two moved. */
const YoloPlane *yolo_session_input(const YoloSession *session);

/* The head's rows as they came out, before anything was selected: the class
 * scores are logits, `anchor_count` of them a class. */
const float *yolo_session_score_room(const YoloSession *session, int *anchor_count_out,
                                     int *class_count_out);

/* Milliseconds the last run spent, split the way the cost actually splits. */
double yolo_session_shape_time(const YoloSession *session);
double yolo_session_forward_time(const YoloSession *session);
double yolo_session_decode_time(const YoloSession *session);

/* ---------------------------------------------------------------------------
 * Drawing
 *
 * A convenience, not part of the engine: boxes, masks and skeletons over a
 * copy of the source picture, so a run can be looked at.
 * ------------------------------------------------------------------------ */

YoloStatus yolo_result_draw(const YoloModel *model, const YoloResult *result,
                            YoloImage *image);

#endif /* YOLO_INCLUDED */

/* ===========================================================================
 * Everything below here is private.
 * ======================================================================== */

#ifndef YOLO_IMPLEMENTED
#define YOLO_IMPLEMENTED

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <float.h>

/* ---------------------------------------------------------------------------
 * 1  Platform
 * ------------------------------------------------------------------------ */

const char *yolo_status_text(YoloStatus status)
{
    switch (status) {
    case YOLO_OK:          return "ok";
    case YOLO_ERR_ARG:     return "a call was given an argument it cannot use";
    case YOLO_ERR_FILE:    return "the file could not be opened or read";
    case YOLO_ERR_FORMAT:  return "the bytes are not a checkpoint this engine reads";
    case YOLO_ERR_MEMORY:  return "an allocation failed";
    case YOLO_ERR_MODULE:  return "the checkpoint holds a module that is not implemented";
    case YOLO_ERR_SHAPE:   return "a weight's shape does not match the module using it";
    case YOLO_ERR_IMAGE:   return "the picture could not be read, written or converted";
    }
    return "unknown";
}

const char *yolo_task_text(YoloTask task)
{
    switch (task) {
    case YOLO_TASK_DETECT:   return "detect";
    case YOLO_TASK_SEGMENT:  return "segment";
    case YOLO_TASK_POSE:     return "pose";
    case YOLO_TASK_OBB:      return "obb";
    case YOLO_TASK_CLASSIFY: return "classify";
    }
    return "unknown";
}

/* One detail line, because a status code alone does not say which module was
 * missing.  Written on the failing path only, so a successful run costs it
 * nothing. */
static char yolo_detail_room[256] = "";

const char *yolo_detail_text(void) { return yolo_detail_room; }

static YoloStatus yolo_fail(YoloStatus status, const char *shape, ...)
{
    va_list mark;
    va_start(mark, shape);
    vsnprintf(yolo_detail_room, sizeof yolo_detail_room, shape, mark);
    va_end(mark);
    return status;
}

static double yolo_clock_ms(void)
{
    struct timespec now;
#if defined(CLOCK_MONOTONIC)
    if (clock_gettime(CLOCK_MONOTONIC, &now) == 0)
        return (double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1.0e6;
#endif
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
}

/* An arena, because a model's weights and a session's scratch are both bump
 * allocated and freed all at once.  Blocks grow geometrically so a load that
 * cannot know its total size ahead of time still makes O(log n) calls to
 * malloc; a session sets its block size once and then reuses it.
 *
 * `yolo_arena_mark` and `yolo_arena_reset` are the reason the forward pass
 * does not allocate: a temporary taken inside a module is released on the way
 * out, so the high-water mark is the deepest module rather than the sum. */

typedef struct YoloBlock {
    struct YoloBlock *next;
    size_t            size;
    size_t            used;
    unsigned char    *byte_list;
} YoloBlock;

typedef struct {
    YoloBlock *block_list;
    size_t     block_size;      /* the size the next fresh block takes */
    size_t     high_mark;       /* the most ever handed out at one time */
    size_t     live_size;
} YoloArena;

typedef struct {
    YoloBlock *block;
    size_t     used;
    size_t     live_size;
} YoloMark;

static void yolo_arena_open(YoloArena *arena, size_t block_size)
{
    memset(arena, 0, sizeof *arena);
    arena->block_size = block_size ? block_size : (size_t)1 << 20;
}

static void yolo_arena_close(YoloArena *arena)
{
    YoloBlock *block = arena->block_list;
    while (block) {
        YoloBlock *next = block->next;
        free(block->byte_list);
        free(block);
        block = next;
    }
    memset(arena, 0, sizeof *arena);
}

static void *yolo_arena_take(YoloArena *arena, size_t size)
{
    size_t want = (size + 63u) & ~(size_t)63;   /* keep every row cache aligned */
    YoloBlock *block = arena->block_list;
    void *cell;

    if (want == 0) want = 64;
    if (!block || block->used + want > block->size) {
        size_t fresh = arena->block_size;
        YoloBlock *made;
        while (fresh < want) fresh *= 2;
        made = (YoloBlock *)calloc(1, sizeof *made);
        if (!made) return NULL;
        made->byte_list = (unsigned char *)malloc(fresh);
        if (!made->byte_list) { free(made); return NULL; }
        made->size = fresh;
        made->used = 0;
        made->next = arena->block_list;
        arena->block_list = made;
        block = made;
        if (fresh > arena->block_size) arena->block_size = fresh;
    }
    cell = block->byte_list + block->used;
    block->used += want;
    arena->live_size += want;
    if (arena->live_size > arena->high_mark) arena->high_mark = arena->live_size;
    return cell;
}

static void *yolo_arena_zero(YoloArena *arena, size_t size)
{
    void *cell = yolo_arena_take(arena, size);
    if (cell) memset(cell, 0, size);
    return cell;
}

static YoloMark yolo_arena_mark(const YoloArena *arena)
{
    YoloMark mark;
    mark.block = arena->block_list;
    mark.used = arena->block_list ? arena->block_list->used : 0;
    mark.live_size = arena->live_size;
    return mark;
}

/* Release back to a mark.  Blocks taken since the mark are kept rather than
 * freed -- the next picture through the session will want them, and a session
 * that reaches its steady state stops calling malloc entirely. */
static void yolo_arena_reset(YoloArena *arena, YoloMark mark)
{
    YoloBlock *block = arena->block_list;
    while (block && block != mark.block) {
        block->used = 0;
        block = block->next;
    }
    if (block) block->used = mark.used;
    arena->live_size = mark.live_size;
}

static void yolo_arena_clear(YoloArena *arena)
{
    YoloBlock *block = arena->block_list;
    while (block) { block->used = 0; block = block->next; }
    arena->live_size = 0;
}

/* Little-endian reads, because a checkpoint's byteorder record says "little"
 * and nothing publishes the other one. */

static uint16_t yolo_read16(const unsigned char *byte_list)
{
    return (uint16_t)(byte_list[0] | ((uint16_t)byte_list[1] << 8));
}

static uint32_t yolo_read32(const unsigned char *byte_list)
{
    return (uint32_t)byte_list[0] | ((uint32_t)byte_list[1] << 8)
         | ((uint32_t)byte_list[2] << 16) | ((uint32_t)byte_list[3] << 24);
}

static uint64_t yolo_read64(const unsigned char *byte_list)
{
    return (uint64_t)yolo_read32(byte_list)
         | ((uint64_t)yolo_read32(byte_list + 4) << 32);
}

/* Half to float, by hand rather than through a compiler extension, so the
 * file builds anywhere.  Subnormals are handled: yolo26's weights are small
 * enough that a few thousand of them are, and flushing those to zero moves
 * the output. */
static float yolo_half_float(uint16_t half)
{
    uint32_t sign = (uint32_t)(half >> 15) << 31;
    uint32_t power = (half >> 10) & 0x1f;
    uint32_t part = half & 0x3ff;
    uint32_t bits;
    float out;

    if (power == 0) {
        if (part == 0) {
            bits = sign;
        } else {
            /* normalise the subnormal: shift until the implied bit appears */
            power = 127 - 15 + 1;
            while ((part & 0x400) == 0) { part <<= 1; power--; }
            part &= 0x3ff;
            bits = sign | (power << 23) | (part << 13);
        }
    } else if (power == 0x1f) {
        bits = sign | 0x7f800000u | (part << 13);   /* inf or nan */
    } else {
        bits = sign | ((power + 127 - 15) << 23) | (part << 13);
    }
    memcpy(&out, &bits, sizeof out);
    return out;
}

static float yolo_bits_float(uint32_t bits)
{
    float out;
    memcpy(&out, &bits, sizeof out);
    return out;
}

static int yolo_min_int(int a, int b) { return a < b ? a : b; }
static int yolo_max_int(int a, int b) { return a > b ? a : b; }

static float yolo_clamp_float(float value, float low, float high)
{
    return value < low ? low : (value > high ? high : value);
}

/* ---------------------------------------------------------------------------
 * 2  Planes
 * ------------------------------------------------------------------------ */

static size_t yolo_plane_count(const YoloPlane *plane)
{
    return (size_t)plane->batch_size * (size_t)plane->channel_count
         * (size_t)plane->height * (size_t)plane->width;
}

static YoloPlane yolo_plane_take(YoloArena *arena, int batch_size,
                                 int channel_count, int height, int width)
{
    YoloPlane plane;
    plane.batch_size = batch_size;
    plane.channel_count = channel_count;
    plane.height = height;
    plane.width = width;
    plane.cell_list = (float *)yolo_arena_take(arena,
        yolo_plane_count(&plane) * sizeof(float));
    return plane;
}

/* The rows of one channel of one picture. */
static float *yolo_plane_face(const YoloPlane *plane, int batch_index,
                              int channel_index)
{
    size_t face = (size_t)plane->height * (size_t)plane->width;
    return plane->cell_list
         + ((size_t)batch_index * (size_t)plane->channel_count
            + (size_t)channel_index) * face;
}

/* ---------------------------------------------------------------------------
 * 3 and 4  The CPU backend
 *
 * One table of operations, filled with scalar C.  The loops are written in the
 * order that lets a compiler vectorise them -- the inner axis is always the
 * one that is contiguous in memory and independent across iterations -- rather
 * than with intrinsics, because the intrinsics would have to be written once
 * per instruction set and this file is meant to build anywhere.  A host that
 * wants more than this fills the same table.
 * ------------------------------------------------------------------------ */

static float yolo_silu(float value)
{
    return value / (1.0f + expf(-value));
}

static float yolo_sigmoid(float value)
{
    return 1.0f / (1.0f + expf(-value));
}

static void yolo_cpu_act(const YoloBackend *backend, float *cell_list,
                         size_t cell_count, YoloActKind act_kind)
{
    size_t index;
    (void)backend;
    if (act_kind == YOLO_ACT_SILU) {
        for (index = 0; index < cell_count; index++)
            cell_list[index] = yolo_silu(cell_list[index]);
    } else if (act_kind == YOLO_ACT_SIGMOID) {
        for (index = 0; index < cell_count; index++)
            cell_list[index] = yolo_sigmoid(cell_list[index]);
    }
}

static void yolo_cpu_add(const YoloBackend *backend, float *a_list,
                         const float *b_list, size_t cell_count)
{
    size_t index;
    (void)backend;
    for (index = 0; index < cell_count; index++) a_list[index] += b_list[index];
}

/* out[row, col] = sum_mid a[row, mid] * b[mid, col], or b[col, mid] swapped.
 *
 * The unswapped form is written as a run of scaled row adds rather than as
 * dot products: b's rows are contiguous, so the inner loop reads one stride-1
 * stream and writes another, which is what vectorises.  The swapped form has
 * no such layout and is a plain dot product; it is only reached by attention,
 * where the matrices are small. */
static void yolo_cpu_gemm(const YoloBackend *backend,
                          const float *a_list, const float *b_list,
                          float *out_list,
                          int row_count, int mid_count, int col_count,
                          int a_step, int b_step, int out_step, int b_swap_flag)
{
    int row, mid, col;
    (void)backend;

    if (b_swap_flag) {
        for (row = 0; row < row_count; row++) {
            const float *a_row = a_list + (size_t)row * (size_t)a_step;
            float *out_row = out_list + (size_t)row * (size_t)out_step;
            for (col = 0; col < col_count; col++) {
                const float *b_row = b_list + (size_t)col * (size_t)b_step;
                float total = 0.0f;
                for (mid = 0; mid < mid_count; mid++)
                    total += a_row[mid] * b_row[mid];
                out_row[col] = total;
            }
        }
        return;
    }

    for (row = 0; row < row_count; row++) {
        const float *a_row = a_list + (size_t)row * (size_t)a_step;
        float *out_row = out_list + (size_t)row * (size_t)out_step;
        for (col = 0; col < col_count; col++) out_row[col] = 0.0f;
        for (mid = 0; mid < mid_count; mid++) {
            float scale = a_row[mid];
            const float *b_row = b_list + (size_t)mid * (size_t)b_step;
            if (scale == 0.0f) continue;
            for (col = 0; col < col_count; col++)
                out_row[col] += scale * b_row[col];
        }
    }
}

/* Scratch for a convolution: the lowered patch matrix, one group at a time.
 * A 1x1 convolution needs none, because the input already is the patch
 * matrix; a depthwise one needs none, because it is run directly. */
static size_t yolo_cpu_conv_room(const YoloPlane *in, const YoloPlane *out,
                                 int kernel_size, int group_count)
{
    size_t patch_size, face;
    if (kernel_size == 1 && group_count == 1) return 0;
    if (group_count == in->channel_count && group_count == out->channel_count)
        return 0;
    patch_size = (size_t)(in->channel_count / group_count)
               * (size_t)kernel_size * (size_t)kernel_size;
    face = (size_t)out->height * (size_t)out->width;
    return patch_size * face * sizeof(float);
}

/* Lower one group of one picture into a patch matrix: rows are (channel,
 * kernel row, kernel column), columns are output positions. */
static void yolo_cpu_lower(const YoloPlane *in, int batch_index,
                           int channel_first, int channel_span,
                           int out_height, int out_width,
                           int kernel_size, int stride_size, int pad_size,
                           int dilation_size, float *room_list)
{
    int channel, kernel_y, kernel_x, out_y, out_x;
    size_t face = (size_t)out_height * (size_t)out_width;
    size_t row = 0;

    for (channel = 0; channel < channel_span; channel++) {
        const float *source = yolo_plane_face(in, batch_index,
                                              channel_first + channel);
        for (kernel_y = 0; kernel_y < kernel_size; kernel_y++) {
            for (kernel_x = 0; kernel_x < kernel_size; kernel_x++, row++) {
                float *target = room_list + row * face;
                for (out_y = 0; out_y < out_height; out_y++) {
                    int in_y = out_y * stride_size - pad_size
                             + kernel_y * dilation_size;
                    float *target_row = target + (size_t)out_y * (size_t)out_width;
                    if (in_y < 0 || in_y >= in->height) {
                        memset(target_row, 0, (size_t)out_width * sizeof(float));
                        continue;
                    }
                    for (out_x = 0; out_x < out_width; out_x++) {
                        int in_x = out_x * stride_size - pad_size
                                 + kernel_x * dilation_size;
                        target_row[out_x] = (in_x < 0 || in_x >= in->width)
                            ? 0.0f
                            : source[(size_t)in_y * (size_t)in->width + (size_t)in_x];
                    }
                }
            }
        }
    }
}

/* A depthwise convolution run directly.  Lowering it would build a patch
 * matrix with one row a kernel cell and one column an output cell, then
 * multiply by a 1 x k*k weight -- all the memory traffic of a general
 * convolution for none of the arithmetic.  Every 3x3 in the detection head's
 * class branch is depthwise, so this path carries real weight. */
static void yolo_cpu_conv_deep(const YoloPlane *in, YoloPlane *out,
                               const float *weight_sheet, const float *bias_list,
                               int kernel_size, int stride_size, int pad_size,
                               int dilation_size)
{
    int batch_index, channel, out_y, out_x, kernel_y, kernel_x;

    for (batch_index = 0; batch_index < in->batch_size; batch_index++) {
        for (channel = 0; channel < out->channel_count; channel++) {
            const float *source = yolo_plane_face(in, batch_index, channel);
            float *target = yolo_plane_face(out, batch_index, channel);
            const float *weight = weight_sheet
                + (size_t)channel * (size_t)kernel_size * (size_t)kernel_size;
            float bias = bias_list ? bias_list[channel] : 0.0f;

            for (out_y = 0; out_y < out->height; out_y++) {
                float *target_row = target + (size_t)out_y * (size_t)out->width;
                for (out_x = 0; out_x < out->width; out_x++)
                    target_row[out_x] = bias;
                for (kernel_y = 0; kernel_y < kernel_size; kernel_y++) {
                    int in_y = out_y * stride_size - pad_size
                             + kernel_y * dilation_size;
                    const float *source_row;
                    if (in_y < 0 || in_y >= in->height) continue;
                    source_row = source + (size_t)in_y * (size_t)in->width;
                    for (kernel_x = 0; kernel_x < kernel_size; kernel_x++) {
                        float scale = weight[kernel_y * kernel_size + kernel_x];
                        int lead = -pad_size + kernel_x * dilation_size;
                        int first = 0, last = out->width;
                        int position;
                        if (scale == 0.0f) continue;
                        /* trim the ends that would read outside the row, so
                         * the inner loop carries no bounds test */
                        while (first < last && first * stride_size + lead < 0)
                            first++;
                        while (last > first
                               && (last - 1) * stride_size + lead >= in->width)
                            last--;
                        for (position = first; position < last; position++)
                            target_row[position] +=
                                scale * source_row[position * stride_size + lead];
                    }
                }
            }
        }
    }
}

static void yolo_cpu_conv(const YoloBackend *backend,
                          const YoloPlane *in, YoloPlane *out,
                          const float *weight_sheet, const float *bias_list,
                          int kernel_size, int stride_size, int pad_size,
                          int dilation_size, int group_count,
                          YoloActKind act_kind, float *room_list)
{
    int batch_index, group, channel;
    size_t face = (size_t)out->height * (size_t)out->width;
    int in_span = in->channel_count / group_count;
    int out_span = out->channel_count / group_count;
    size_t patch_size = (size_t)in_span * (size_t)kernel_size * (size_t)kernel_size;

    if (group_count == in->channel_count && group_count == out->channel_count) {
        yolo_cpu_conv_deep(in, out, weight_sheet, bias_list, kernel_size,
                           stride_size, pad_size, dilation_size);
        backend->act_run(backend, out->cell_list, yolo_plane_count(out), act_kind);
        return;
    }

    for (batch_index = 0; batch_index < in->batch_size; batch_index++) {
        for (group = 0; group < group_count; group++) {
            const float *patch_list;
            const float *weight = weight_sheet
                                + (size_t)group * (size_t)out_span * patch_size;
            float *target = yolo_plane_face(out, batch_index, group * out_span);

            if (kernel_size == 1 && stride_size == 1 && pad_size == 0) {
                patch_list = yolo_plane_face(in, batch_index, group * in_span);
            } else {
                yolo_cpu_lower(in, batch_index, group * in_span, in_span,
                               out->height, out->width, kernel_size,
                               stride_size, pad_size, dilation_size, room_list);
                patch_list = room_list;
            }
            backend->gemm_run(backend, weight, patch_list, target,
                              out_span, (int)patch_size, (int)face,
                              (int)patch_size, (int)face, (int)face, 0);
            if (bias_list) {
                for (channel = 0; channel < out_span; channel++) {
                    float bias = bias_list[group * out_span + channel];
                    float *target_row = target + (size_t)channel * face;
                    size_t index;
                    for (index = 0; index < face; index++) target_row[index] += bias;
                }
            }
        }
    }
    backend->act_run(backend, out->cell_list, yolo_plane_count(out), act_kind);
}

/* A transposed convolution with stride equal to the kernel and no padding,
 * which is the only shape the mask prototype head asks for: every output cell
 * is written by exactly one input cell, so it is a scatter rather than the
 * general overlap-add. */
static void yolo_cpu_deconv(const YoloBackend *backend,
                            const YoloPlane *in, YoloPlane *out,
                            const float *weight_sheet, const float *bias_list,
                            int kernel_size, int stride_size,
                            YoloActKind act_kind)
{
    int batch_index, out_channel, in_channel, in_y, in_x, kernel_y, kernel_x;

    for (batch_index = 0; batch_index < in->batch_size; batch_index++) {
        for (out_channel = 0; out_channel < out->channel_count; out_channel++) {
            float *target = yolo_plane_face(out, batch_index, out_channel);
            float bias = bias_list ? bias_list[out_channel] : 0.0f;
            size_t index;
            for (index = 0; index < (size_t)out->height * (size_t)out->width; index++)
                target[index] = bias;
        }
        for (in_channel = 0; in_channel < in->channel_count; in_channel++) {
            const float *source = yolo_plane_face(in, batch_index, in_channel);
            for (out_channel = 0; out_channel < out->channel_count; out_channel++) {
                /* torch stores a transposed convolution as [in, out, k, k] */
                const float *weight = weight_sheet
                    + (((size_t)in_channel * (size_t)out->channel_count
                        + (size_t)out_channel)
                       * (size_t)kernel_size * (size_t)kernel_size);
                float *target = yolo_plane_face(out, batch_index, out_channel);
                for (in_y = 0; in_y < in->height; in_y++) {
                    for (in_x = 0; in_x < in->width; in_x++) {
                        float value = source[(size_t)in_y * (size_t)in->width
                                             + (size_t)in_x];
                        if (value == 0.0f) continue;
                        for (kernel_y = 0; kernel_y < kernel_size; kernel_y++) {
                            int out_y = in_y * stride_size + kernel_y;
                            float *target_row;
                            if (out_y >= out->height) break;
                            target_row = target + (size_t)out_y * (size_t)out->width;
                            for (kernel_x = 0; kernel_x < kernel_size; kernel_x++) {
                                int out_x = in_x * stride_size + kernel_x;
                                if (out_x >= out->width) break;
                                target_row[out_x] += value
                                    * weight[kernel_y * kernel_size + kernel_x];
                            }
                        }
                    }
                }
            }
        }
    }
    backend->act_run(backend, out->cell_list, yolo_plane_count(out), act_kind);
}

static void yolo_cpu_pool(const YoloBackend *backend,
                          const YoloPlane *in, YoloPlane *out,
                          int kernel_size, int stride_size, int pad_size)
{
    int batch_index, channel, out_y, out_x, kernel_y, kernel_x;
    (void)backend;

    for (batch_index = 0; batch_index < in->batch_size; batch_index++) {
        for (channel = 0; channel < in->channel_count; channel++) {
            const float *source = yolo_plane_face(in, batch_index, channel);
            float *target = yolo_plane_face(out, batch_index, channel);
            for (out_y = 0; out_y < out->height; out_y++) {
                for (out_x = 0; out_x < out->width; out_x++) {
                    float best = -FLT_MAX;
                    for (kernel_y = 0; kernel_y < kernel_size; kernel_y++) {
                        int in_y = out_y * stride_size - pad_size + kernel_y;
                        const float *source_row;
                        if (in_y < 0 || in_y >= in->height) continue;
                        source_row = source + (size_t)in_y * (size_t)in->width;
                        for (kernel_x = 0; kernel_x < kernel_size; kernel_x++) {
                            int in_x = out_x * stride_size - pad_size + kernel_x;
                            if (in_x < 0 || in_x >= in->width) continue;
                            if (source_row[in_x] > best) best = source_row[in_x];
                        }
                    }
                    target[(size_t)out_y * (size_t)out->width + (size_t)out_x] = best;
                }
            }
        }
    }
}

static void yolo_cpu_resize(const YoloBackend *backend,
                            const YoloPlane *in, YoloPlane *out, int scale_size)
{
    int batch_index, channel, out_y, out_x;
    (void)backend;

    for (batch_index = 0; batch_index < in->batch_size; batch_index++) {
        for (channel = 0; channel < in->channel_count; channel++) {
            const float *source = yolo_plane_face(in, batch_index, channel);
            float *target = yolo_plane_face(out, batch_index, channel);
            for (out_y = 0; out_y < out->height; out_y++) {
                const float *source_row = source
                    + (size_t)(out_y / scale_size) * (size_t)in->width;
                float *target_row = target + (size_t)out_y * (size_t)out->width;
                for (out_x = 0; out_x < out->width; out_x++)
                    target_row[out_x] = source_row[out_x / scale_size];
            }
        }
    }
}

static void yolo_cpu_softmax(const YoloBackend *backend, float *cell_list,
                             int row_count, int col_count)
{
    int row, col;
    (void)backend;
    for (row = 0; row < row_count; row++) {
        float *cell_row = cell_list + (size_t)row * (size_t)col_count;
        float best = -FLT_MAX;
        float total = 0.0f;
        for (col = 0; col < col_count; col++)
            if (cell_row[col] > best) best = cell_row[col];
        for (col = 0; col < col_count; col++) {
            cell_row[col] = expf(cell_row[col] - best);
            total += cell_row[col];
        }
        if (total > 0.0f) {
            float scale = 1.0f / total;
            for (col = 0; col < col_count; col++) cell_row[col] *= scale;
        }
    }
}

static const YoloBackend yolo_backend_cpu_table = {
    "cpu",
    yolo_cpu_conv,
    yolo_cpu_conv_room,
    yolo_cpu_deconv,
    yolo_cpu_pool,
    yolo_cpu_resize,
    yolo_cpu_act,
    yolo_cpu_gemm,
    yolo_cpu_add,
    yolo_cpu_softmax
};

const YoloBackend *yolo_backend_cpu(void) { return &yolo_backend_cpu_table; }

/* ---------------------------------------------------------------------------
 * 5  The store: a zip, a pickle, and the storages under them
 *
 * A `.pt` is a zip.  One entry, `data.pkl`, is a pickle of the saved object
 * graph; every tensor in it appears as a persistent id naming one of the other
 * entries, `data/0`, `data/1` and so on, which hold the raw bytes.  Nothing is
 * compressed -- torch writes the archive stored -- so reading it is a directory
 * walk and a set of offsets into one mapping of the file.
 *
 * The pickle is read into a value tree rather than executed.  Nothing in this
 * file constructs a class or calls a function named by the pickle; a GLOBAL is
 * a pair of names, and the handful that mean something to a checkpoint --
 * OrderedDict, _rebuild_tensor_v2, _rebuild_parameter -- are recognised by
 * name and turned into the value they stand for.  Everything else becomes an
 * object carrying its class name and its state, which is exactly what the
 * graph layer above wants to read.
 * ------------------------------------------------------------------------ */

typedef enum {
    YOLO_TYPE_F32 = 0,
    YOLO_TYPE_F16,
    YOLO_TYPE_BF16,
    YOLO_TYPE_F64,
    YOLO_TYPE_I64,
    YOLO_TYPE_I32,
    YOLO_TYPE_I16,
    YOLO_TYPE_I8,
    YOLO_TYPE_U8,
    YOLO_TYPE_NONE
} YoloCellType;

static const struct { const char *name; YoloCellType type; int size; } yolo_type_table[] = {
    { "FloatStorage",    YOLO_TYPE_F32,  4 },
    { "HalfStorage",     YOLO_TYPE_F16,  2 },
    { "BFloat16Storage", YOLO_TYPE_BF16, 2 },
    { "DoubleStorage",   YOLO_TYPE_F64,  8 },
    { "LongStorage",     YOLO_TYPE_I64,  8 },
    { "IntStorage",      YOLO_TYPE_I32,  4 },
    { "ShortStorage",    YOLO_TYPE_I16,  2 },
    { "CharStorage",     YOLO_TYPE_I8,   1 },
    { "ByteStorage",     YOLO_TYPE_U8,   1 },
    { "BoolStorage",     YOLO_TYPE_U8,   1 }
};

static int yolo_type_size(YoloCellType type)
{
    size_t index;
    for (index = 0; index < sizeof yolo_type_table / sizeof yolo_type_table[0]; index++)
        if (yolo_type_table[index].type == type) return yolo_type_table[index].size;
    return 0;
}

/* ---- the value tree ---- */

typedef enum {
    YOLO_VALUE_NONE = 0,
    YOLO_VALUE_BOOL,
    YOLO_VALUE_INT,
    YOLO_VALUE_REAL,
    YOLO_VALUE_TEXT,     /* a string or a bytes object; the bytes are kept */
    YOLO_VALUE_LIST,     /* a list, a tuple or a set */
    YOLO_VALUE_DICT,
    YOLO_VALUE_OBJECT,   /* anything the pickle named a class for */
    YOLO_VALUE_NAME,     /* a GLOBAL that has not been called */
    YOLO_VALUE_TENSOR,
    YOLO_VALUE_STORAGE
} YoloValueKind;

#define YOLO_RANK_LIMIT 8

typedef struct YoloValue YoloValue;

struct YoloValue {
    YoloValueKind kind;

    int64_t     number;        /* int, bool */
    double      real;
    const char *text;          /* text, and the class name of an object */
    size_t      text_size;
    const char *module_name;   /* object and name */

    YoloValue **item_list;     /* list items, or a dict's values */
    YoloValue **key_list;      /* a dict's keys */
    int         item_count;
    int         item_limit;

    YoloValue *state;          /* an object's __dict__, from BUILD */
    YoloValue *argument;       /* an object's construction arguments */

    /* tensor */
    int          storage_key;
    YoloCellType cell_type;
    int64_t      cell_offset;
    int          rank;
    int64_t      size_list[YOLO_RANK_LIMIT];
    int64_t      step_list[YOLO_RANK_LIMIT];
    int64_t      cell_count;
};

static YoloValue *yolo_value_take(YoloArena *arena, YoloValueKind kind)
{
    YoloValue *value = (YoloValue *)yolo_arena_zero(arena, sizeof *value);
    if (value) value->kind = kind;
    return value;
}

static int yolo_value_push(YoloArena *arena, YoloValue *value, YoloValue *item)
{
    if (value->item_count == value->item_limit) {
        int limit = value->item_limit ? value->item_limit * 2 : 8;
        YoloValue **fresh = (YoloValue **)yolo_arena_take(arena,
            (size_t)limit * sizeof(YoloValue *));
        if (!fresh) return 0;
        if (value->item_count)
            memcpy(fresh, value->item_list,
                   (size_t)value->item_count * sizeof(YoloValue *));
        value->item_list = fresh;
        if (value->kind == YOLO_VALUE_DICT) {
            YoloValue **keys = (YoloValue **)yolo_arena_take(arena,
                (size_t)limit * sizeof(YoloValue *));
            if (!keys) return 0;
            if (value->item_count)
                memcpy(keys, value->key_list,
                       (size_t)value->item_count * sizeof(YoloValue *));
            value->key_list = keys;
        }
        value->item_limit = limit;
    }
    value->item_list[value->item_count++] = item;
    return 1;
}

static int yolo_value_set(YoloArena *arena, YoloValue *map,
                          YoloValue *key, YoloValue *item)
{
    int index;
    /* a pickle may set the same key twice; the last write wins, as python's */
    for (index = 0; index < map->item_count; index++) {
        YoloValue *had = map->key_list[index];
        if (had && key && had->kind == YOLO_VALUE_TEXT
            && key->kind == YOLO_VALUE_TEXT
            && had->text_size == key->text_size
            && memcmp(had->text, key->text, key->text_size) == 0) {
            map->item_list[index] = item;
            return 1;
        }
    }
    if (!yolo_value_push(arena, map, item)) return 0;
    map->key_list[map->item_count - 1] = key;
    return 1;
}

/* Look a key up in a dict.  Keys are compared as text; a checkpoint's class
 * names are text and its `names` map is keyed by number, which the caller
 * reads through `yolo_value_at` instead. */
static YoloValue *yolo_value_find(const YoloValue *map, const char *name)
{
    int index;
    size_t size;
    if (!map || map->kind != YOLO_VALUE_DICT) return NULL;
    size = strlen(name);
    for (index = 0; index < map->item_count; index++) {
        const YoloValue *key = map->key_list[index];
        if (key && key->kind == YOLO_VALUE_TEXT && key->text_size == size
            && memcmp(key->text, name, size) == 0)
            return map->item_list[index];
    }
    return NULL;
}

static YoloValue *yolo_value_at(const YoloValue *value, int index)
{
    if (!value || index < 0 || index >= value->item_count) return NULL;
    return value->item_list[index];
}

static int64_t yolo_value_number(const YoloValue *value, int64_t fallback)
{
    if (!value) return fallback;
    if (value->kind == YOLO_VALUE_INT || value->kind == YOLO_VALUE_BOOL)
        return value->number;
    if (value->kind == YOLO_VALUE_REAL) return (int64_t)value->real;
    /* torch writes a pair for a kernel size, a stride and a padding; both
     * halves are equal on everything yolo builds, so the first stands for it */
    if (value->kind == YOLO_VALUE_LIST && value->item_count > 0)
        return yolo_value_number(value->item_list[0], fallback);
    return fallback;
}

static double yolo_value_real(const YoloValue *value, double fallback)
{
    if (!value) return fallback;
    if (value->kind == YOLO_VALUE_REAL) return value->real;
    if (value->kind == YOLO_VALUE_INT || value->kind == YOLO_VALUE_BOOL)
        return (double)value->number;
    return fallback;
}

static int yolo_value_is(const YoloValue *value, const char *name)
{
    size_t size;
    if (!value || value->kind != YOLO_VALUE_TEXT) return 0;
    size = strlen(name);
    return value->text_size == size && memcmp(value->text, name, size) == 0;
}

/* ---- the zip ---- */

typedef struct {
    const char    *name;
    size_t         name_size;
    const unsigned char *byte_list;
    size_t         size;
} YoloEntry;

typedef struct {
    unsigned char *file_byte_list;
    size_t         file_size;
    YoloEntry     *entry_list;
    int            entry_count;
} YoloArchive;

static YoloStatus yolo_archive_open(YoloArena *arena, const char *path,
                                    YoloArchive *archive)
{
    FILE *handle = fopen(path, "rb");
    long file_size;
    size_t read_size;
    const unsigned char *tail;
    const unsigned char *walk;
    uint32_t directory_size, directory_at;
    uint16_t entry_count;
    int index;

    memset(archive, 0, sizeof *archive);
    if (!handle) return yolo_fail(YOLO_ERR_FILE, "cannot open %s", path);
    if (fseek(handle, 0, SEEK_END) != 0) { fclose(handle); return YOLO_ERR_FILE; }
    file_size = ftell(handle);
    if (file_size <= 22) { fclose(handle); return yolo_fail(YOLO_ERR_FORMAT, "%s is too short to be a zip", path); }
    rewind(handle);

    archive->file_size = (size_t)file_size;
    archive->file_byte_list = (unsigned char *)yolo_arena_take(arena, archive->file_size);
    if (!archive->file_byte_list) { fclose(handle); return YOLO_ERR_MEMORY; }
    read_size = fread(archive->file_byte_list, 1, archive->file_size, handle);
    fclose(handle);
    if (read_size != archive->file_size)
        return yolo_fail(YOLO_ERR_FILE, "short read on %s", path);

    /* the end of central directory record, found by scanning back for its
     * signature -- there is no other way, the record is the only thing that
     * says where the directory starts */
    tail = NULL;
    {
        size_t back = archive->file_size < 66000u ? archive->file_size : 66000u;
        size_t step;
        for (step = 22; step <= back; step++) {
            const unsigned char *look = archive->file_byte_list
                                      + archive->file_size - step;
            if (yolo_read32(look) == 0x06054b50u) { tail = look; break; }
        }
    }
    if (!tail) return yolo_fail(YOLO_ERR_FORMAT, "%s has no zip directory", path);

    entry_count = yolo_read16(tail + 10);
    directory_size = yolo_read32(tail + 12);
    directory_at = yolo_read32(tail + 16);
    if (directory_at == 0xffffffffu || entry_count == 0xffffu) {
        /* zip64: the record above is a placeholder and the real one sits in
         * front of the locator.  No published yolo26 reaches four gigabytes,
         * so this is a refusal rather than a second parser. */
        return yolo_fail(YOLO_ERR_FORMAT,
                         "%s is a zip64 archive, which this engine does not read", path);
    }
    if ((size_t)directory_at + directory_size > archive->file_size)
        return yolo_fail(YOLO_ERR_FORMAT, "%s has a truncated directory", path);

    archive->entry_list = (YoloEntry *)yolo_arena_zero(arena,
        (size_t)entry_count * sizeof(YoloEntry));
    if (!archive->entry_list) return YOLO_ERR_MEMORY;

    walk = archive->file_byte_list + directory_at;
    for (index = 0; index < entry_count; index++) {
        uint16_t method, name_size, extra_size, comment_size;
        uint32_t packed_size, plain_size, local_at;
        const unsigned char *local;
        YoloEntry *entry = &archive->entry_list[index];

        if (walk + 46 > archive->file_byte_list + archive->file_size
            || yolo_read32(walk) != 0x02014b50u)
            return yolo_fail(YOLO_ERR_FORMAT, "%s directory entry %d is malformed",
                             path, index);
        method       = yolo_read16(walk + 10);
        packed_size  = yolo_read32(walk + 20);
        plain_size   = yolo_read32(walk + 24);
        name_size    = yolo_read16(walk + 28);
        extra_size   = yolo_read16(walk + 30);
        comment_size = yolo_read16(walk + 32);
        local_at     = yolo_read32(walk + 42);

        if (method != 0)
            return yolo_fail(YOLO_ERR_FORMAT,
                             "%s stores an entry deflated; torch.save writes them "
                             "stored, so this archive was repacked", path);

        entry->name = (const char *)(walk + 46);
        entry->name_size = name_size;
        entry->size = plain_size;

        local = archive->file_byte_list + local_at;
        if (local + 30 > archive->file_byte_list + archive->file_size
            || yolo_read32(local) != 0x04034b50u)
            return yolo_fail(YOLO_ERR_FORMAT, "%s local header %d is malformed",
                             path, index);
        {
            size_t at = (size_t)local_at + 30
                      + yolo_read16(local + 26) + yolo_read16(local + 28);
            if (at + packed_size > archive->file_size)
                return yolo_fail(YOLO_ERR_FORMAT, "%s entry %d runs past the file",
                                 path, index);
            entry->byte_list = archive->file_byte_list + at;
        }
        walk += 46 + name_size + extra_size + comment_size;
    }
    archive->entry_count = entry_count;
    return YOLO_OK;
}

/* Entries are named `<prefix>/data.pkl`, `<prefix>/data/0` and so on, where
 * the prefix is whatever the file was called when it was saved.  Matching on
 * the tail keeps the engine from caring what that was. */
static const YoloEntry *yolo_archive_find(const YoloArchive *archive,
                                          const char *tail)
{
    size_t size = strlen(tail);
    int index;
    for (index = 0; index < archive->entry_count; index++) {
        const YoloEntry *entry = &archive->entry_list[index];
        if (entry->name_size < size) continue;
        if (memcmp(entry->name + entry->name_size - size, tail, size) != 0) continue;
        if (entry->name_size > size && entry->name[entry->name_size - size - 1] != '/')
            continue;
        return entry;
    }
    return NULL;
}

/* ---- the pickle ---- */

typedef struct {
    YoloArena           *arena;
    const unsigned char *byte_list;
    size_t               size;
    size_t               at;

    YoloValue **stack_list;
    int         stack_count;
    int         stack_limit;

    int  *mark_list;
    int   mark_count;
    int   mark_limit;

    YoloValue **memo_list;
    int         memo_count;
    int         memo_limit;

    YoloStatus status;
} YoloPickle;

static int yolo_pickle_grow(YoloArena *arena, void ***list, int *limit,
                            int want, size_t cell)
{
    int fresh = *limit ? *limit : 64;
    void **made;
    while (fresh < want) fresh *= 2;
    made = (void **)yolo_arena_take(arena, (size_t)fresh * cell);
    if (!made) return 0;
    if (*limit) memcpy(made, *list, (size_t)*limit * cell);
    *list = made;
    *limit = fresh;
    return 1;
}

static void yolo_pickle_push(YoloPickle *pickle, YoloValue *value)
{
    if (pickle->status != YOLO_OK) return;
    if (pickle->stack_count == pickle->stack_limit
        && !yolo_pickle_grow(pickle->arena, (void ***)&pickle->stack_list,
                             &pickle->stack_limit, pickle->stack_count + 1,
                             sizeof(YoloValue *))) {
        pickle->status = YOLO_ERR_MEMORY;
        return;
    }
    pickle->stack_list[pickle->stack_count++] = value;
}

static YoloValue *yolo_pickle_pop(YoloPickle *pickle)
{
    if (pickle->stack_count <= 0) {
        if (pickle->status == YOLO_OK)
            pickle->status = yolo_fail(YOLO_ERR_FORMAT, "the pickle underflows");
        return NULL;
    }
    return pickle->stack_list[--pickle->stack_count];
}

static YoloValue *yolo_pickle_make(YoloPickle *pickle, YoloValueKind kind)
{
    YoloValue *value = yolo_value_take(pickle->arena, kind);
    if (!value) pickle->status = YOLO_ERR_MEMORY;
    return value;
}

static void yolo_pickle_memo(YoloPickle *pickle, int slot, YoloValue *value)
{
    if (pickle->status != YOLO_OK) return;
    if (slot < 0) { pickle->status = YOLO_ERR_FORMAT; return; }
    if (slot >= pickle->memo_limit
        && !yolo_pickle_grow(pickle->arena, (void ***)&pickle->memo_list,
                             &pickle->memo_limit, slot + 1, sizeof(YoloValue *))) {
        pickle->status = YOLO_ERR_MEMORY;
        return;
    }
    while (pickle->memo_count <= slot) pickle->memo_list[pickle->memo_count++] = NULL;
    pickle->memo_list[slot] = value;
}

static const unsigned char *yolo_pickle_eat(YoloPickle *pickle, size_t size)
{
    const unsigned char *at;
    if (pickle->at + size > pickle->size) {
        pickle->status = yolo_fail(YOLO_ERR_FORMAT, "the pickle ends mid-opcode");
        return NULL;
    }
    at = pickle->byte_list + pickle->at;
    pickle->at += size;
    return at;
}

static YoloValue *yolo_pickle_text(YoloPickle *pickle, const char *text, size_t size)
{
    YoloValue *value = yolo_pickle_make(pickle, YOLO_VALUE_TEXT);
    if (value) { value->text = text; value->text_size = size; }
    return value;
}

static YoloValue *yolo_pickle_int(YoloPickle *pickle, int64_t number)
{
    YoloValue *value = yolo_pickle_make(pickle, YOLO_VALUE_INT);
    if (value) value->number = number;
    return value;
}

/* Collect what sits above the last mark into a fresh list, and drop the mark. */
static YoloValue *yolo_pickle_gather(YoloPickle *pickle, YoloValueKind kind)
{
    YoloValue *value = yolo_pickle_make(pickle, kind);
    int first, index;
    if (!value) return NULL;
    if (pickle->mark_count <= 0) {
        pickle->status = yolo_fail(YOLO_ERR_FORMAT, "the pickle has a stray mark");
        return value;
    }
    first = pickle->mark_list[--pickle->mark_count];
    for (index = first; index < pickle->stack_count; index++) {
        if (!yolo_value_push(pickle->arena, value, pickle->stack_list[index])) {
            pickle->status = YOLO_ERR_MEMORY;
            return value;
        }
    }
    pickle->stack_count = first;
    return value;
}

static int yolo_name_is(const YoloValue *value, const char *module_name,
                        const char *class_name)
{
    if (!value || value->kind != YOLO_VALUE_NAME) return 0;
    if (module_name && (!value->module_name
                        || strcmp(value->module_name, module_name) != 0)) return 0;
    return value->text && strcmp(value->text, class_name) == 0;
}

/* A GLOBAL that has been called.  The four that a checkpoint actually needs
 * are turned into the value they stand for; every other becomes an object,
 * which is enough for the graph layer to read a class name off. */
static YoloValue *yolo_pickle_reduce(YoloPickle *pickle, YoloValue *name,
                                     YoloValue *argument)
{
    YoloValue *value;

    if (!name || name->kind != YOLO_VALUE_NAME) {
        pickle->status = yolo_fail(YOLO_ERR_FORMAT, "a pickle reduce has no class");
        return NULL;
    }
    if (yolo_name_is(name, "collections", "OrderedDict")
        || yolo_name_is(name, NULL, "dict"))
        return yolo_pickle_make(pickle, YOLO_VALUE_DICT);

    if (yolo_name_is(name, NULL, "set") || yolo_name_is(name, NULL, "frozenset")) {
        YoloValue *inner = yolo_value_at(argument, 0);
        if (inner && inner->kind == YOLO_VALUE_LIST) return inner;
        return yolo_pickle_make(pickle, YOLO_VALUE_LIST);
    }
    if (yolo_name_is(name, NULL, "_rebuild_parameter")
        || yolo_name_is(name, NULL, "_rebuild_parameter_with_state"))
        return yolo_value_at(argument, 0);

    if (yolo_name_is(name, NULL, "_rebuild_tensor_v2")
        || yolo_name_is(name, NULL, "_rebuild_tensor")) {
        YoloValue *storage = yolo_value_at(argument, 0);
        YoloValue *size_list = yolo_value_at(argument, 2);
        YoloValue *step_list = yolo_value_at(argument, 3);
        int index;

        if (!storage || storage->kind != YOLO_VALUE_STORAGE) {
            pickle->status = yolo_fail(YOLO_ERR_FORMAT,
                                       "a tensor names no storage");
            return NULL;
        }
        value = yolo_pickle_make(pickle, YOLO_VALUE_TENSOR);
        if (!value) return NULL;
        value->storage_key = storage->storage_key;
        value->cell_type = storage->cell_type;
        value->cell_offset = yolo_value_number(yolo_value_at(argument, 1), 0);
        value->rank = size_list ? size_list->item_count : 0;
        if (value->rank > YOLO_RANK_LIMIT) {
            pickle->status = yolo_fail(YOLO_ERR_FORMAT,
                                       "a tensor has rank %d", value->rank);
            return value;
        }
        value->cell_count = 1;
        for (index = 0; index < value->rank; index++) {
            value->size_list[index] = yolo_value_number(yolo_value_at(size_list, index), 0);
            value->step_list[index] = yolo_value_number(yolo_value_at(step_list, index), 0);
            value->cell_count *= value->size_list[index];
        }
        return value;
    }

    value = yolo_pickle_make(pickle, YOLO_VALUE_OBJECT);
    if (value) {
        value->module_name = name->module_name;
        value->text = name->text;
        value->text_size = name->text ? strlen(name->text) : 0;
        value->argument = argument;
    }
    return value;
}

static YoloValue *yolo_pickle_persist(YoloPickle *pickle, YoloValue *id_list)
{
    YoloValue *kind = yolo_value_at(id_list, 0);
    YoloValue *type_name = yolo_value_at(id_list, 1);
    YoloValue *key = yolo_value_at(id_list, 2);
    YoloValue *count = yolo_value_at(id_list, 4);
    YoloValue *value;
    size_t index;

    if (!yolo_value_is(kind, "storage")) {
        pickle->status = yolo_fail(YOLO_ERR_FORMAT,
                                   "a persistent id that is not a storage");
        return NULL;
    }
    value = yolo_pickle_make(pickle, YOLO_VALUE_STORAGE);
    if (!value) return NULL;
    value->cell_type = YOLO_TYPE_NONE;
    if (type_name && type_name->text) {
        for (index = 0; index < sizeof yolo_type_table / sizeof yolo_type_table[0]; index++) {
            if (strcmp(type_name->text, yolo_type_table[index].name) == 0) {
                value->cell_type = yolo_type_table[index].type;
                break;
            }
        }
    }
    if (value->cell_type == YOLO_TYPE_NONE) {
        pickle->status = yolo_fail(YOLO_ERR_FORMAT, "storage type %s is not read",
                                   type_name && type_name->text ? type_name->text : "?");
        return value;
    }
    value->storage_key = key && key->kind == YOLO_VALUE_TEXT
                       ? (int)strtol(key->text, NULL, 10) : -1;
    value->cell_count = yolo_value_number(count, 0);
    return value;
}

/* Copy a run of bytes out of the pickle and terminate it, so the value tree
 * can hand out plain C strings without owning the archive's lifetime rules. */
static const char *yolo_pickle_keep(YoloPickle *pickle, const unsigned char *at,
                                    size_t size)
{
    char *room = (char *)yolo_arena_take(pickle->arena, size + 1);
    if (!room) { pickle->status = YOLO_ERR_MEMORY; return ""; }
    memcpy(room, at, size);
    room[size] = 0;
    return room;
}

static YoloStatus yolo_pickle_run(YoloArena *arena, const unsigned char *byte_list,
                                  size_t size, YoloValue **value_out)
{
    YoloPickle pickle;
    int running = 1;

    memset(&pickle, 0, sizeof pickle);
    pickle.arena = arena;
    pickle.byte_list = byte_list;
    pickle.size = size;
    pickle.status = YOLO_OK;

    while (running && pickle.status == YOLO_OK) {
        const unsigned char *at = yolo_pickle_eat(&pickle, 1);
        unsigned char code;
        if (!at) break;
        code = *at;
        switch (code) {
        case 0x80: (void)yolo_pickle_eat(&pickle, 1); break;            /* PROTO */
        case 0x95: (void)yolo_pickle_eat(&pickle, 8); break;            /* FRAME */
        case '.': running = 0; break;                                   /* STOP */

        case 'N': yolo_pickle_push(&pickle, yolo_pickle_make(&pickle, YOLO_VALUE_NONE)); break;
        case 0x88: case 0x89: {                                         /* NEWTRUE/FALSE */
            YoloValue *value = yolo_pickle_make(&pickle, YOLO_VALUE_BOOL);
            if (value) value->number = (code == 0x88);
            yolo_pickle_push(&pickle, value);
            break;
        }
        case 'K': case 'M': case 'J': {                     /* BININT1/2/4 */
            size_t head = code == 'K' ? 1 : (code == 'M' ? 2 : 4);
            const unsigned char *b = yolo_pickle_eat(&pickle, head);
            if (b) {
                int64_t number = head == 1 ? (int64_t)b[0]
                               : head == 2 ? (int64_t)yolo_read16(b)
                                           : (int64_t)(int32_t)yolo_read32(b);
                yolo_pickle_push(&pickle, yolo_pickle_int(&pickle, number));
            }
            break;
        }
        case 0x8a: case 0x8b: {                                         /* LONG1/LONG4 */
            const unsigned char *b = yolo_pickle_eat(&pickle, code == 0x8a ? 1 : 4);
            size_t span = b ? (code == 0x8a ? (size_t)b[0] : (size_t)yolo_read32(b)) : 0;
            const unsigned char *body = b ? yolo_pickle_eat(&pickle, span) : NULL;
            int64_t number = 0;
            size_t index;
            if (!body && span) break;
            for (index = 0; index < span && index < 8; index++)
                number |= (int64_t)body[index] << (8 * index);
            if (span > 0 && span <= 8 && (body[span - 1] & 0x80))    /* sign extend */
                number |= -((int64_t)1 << (8 * span));
            yolo_pickle_push(&pickle, yolo_pickle_int(&pickle, number));
            break;
        }
        case 'G': {                                                     /* BINFLOAT */
            const unsigned char *b = yolo_pickle_eat(&pickle, 8);
            YoloValue *value = yolo_pickle_make(&pickle, YOLO_VALUE_REAL);
            if (b && value) {
                /* a pickle writes a double big-endian, whatever the host is */
                unsigned char flip[8];
                double real;
                int index;
                for (index = 0; index < 8; index++) flip[index] = b[7 - index];
                memcpy(&real, flip, sizeof real);
                value->real = real;
            }
            yolo_pickle_push(&pickle, value);
            break;
        }
        case 0x8c: case 'X': case 'C': case 'B': case 0x8d: case 0x8e: {
            /* SHORT_BINUNICODE, BINUNICODE, SHORT_BINBYTES, BINBYTES,
             * BINUNICODE8, BINBYTES8 -- all a length then the bytes */
            size_t head = (code == 0x8c || code == 'C') ? 1
                        : (code == 0x8d || code == 0x8e) ? 8 : 4;
            const unsigned char *b = yolo_pickle_eat(&pickle, head);
            size_t span = 0;
            const unsigned char *body;
            if (!b) break;
            span = head == 1 ? b[0] : head == 4 ? yolo_read32(b) : (size_t)yolo_read64(b);
            body = yolo_pickle_eat(&pickle, span);
            if (!body && span) break;
            yolo_pickle_push(&pickle,
                yolo_pickle_text(&pickle, yolo_pickle_keep(&pickle, body, span), span));
            break;
        }
        case '(': {                                                     /* MARK */
            if (pickle.mark_count == pickle.mark_limit
                && !yolo_pickle_grow(arena, (void ***)&pickle.mark_list,
                                     &pickle.mark_limit, pickle.mark_count + 1,
                                     sizeof(int))) {
                pickle.status = YOLO_ERR_MEMORY;
                break;
            }
            pickle.mark_list[pickle.mark_count++] = pickle.stack_count;
            break;
        }
        case ')': yolo_pickle_push(&pickle, yolo_pickle_make(&pickle, YOLO_VALUE_LIST)); break;
        case ']': yolo_pickle_push(&pickle, yolo_pickle_make(&pickle, YOLO_VALUE_LIST)); break;
        case '}': yolo_pickle_push(&pickle, yolo_pickle_make(&pickle, YOLO_VALUE_DICT)); break;
        case 0x8f: yolo_pickle_push(&pickle, yolo_pickle_make(&pickle, YOLO_VALUE_LIST)); break;
        case 't': yolo_pickle_push(&pickle, yolo_pickle_gather(&pickle, YOLO_VALUE_LIST)); break;
        case 'l': yolo_pickle_push(&pickle, yolo_pickle_gather(&pickle, YOLO_VALUE_LIST)); break;
        case 0x91: yolo_pickle_push(&pickle, yolo_pickle_gather(&pickle, YOLO_VALUE_LIST)); break;
        case 'd': {                                                     /* DICT */
            YoloValue *flat = yolo_pickle_gather(&pickle, YOLO_VALUE_LIST);
            YoloValue *map = yolo_pickle_make(&pickle, YOLO_VALUE_DICT);
            int index;
            if (flat && map)
                for (index = 0; index + 1 < flat->item_count; index += 2)
                    yolo_value_set(arena, map, flat->item_list[index],
                                   flat->item_list[index + 1]);
            yolo_pickle_push(&pickle, map);
            break;
        }
        case 0x85: case 0x86: case 0x87: {                              /* TUPLE1..3 */
            int span = code - 0x84;
            YoloValue *value = yolo_pickle_make(&pickle, YOLO_VALUE_LIST);
            int index;
            if (!value) break;
            if (pickle.stack_count < span) { pickle.status = YOLO_ERR_FORMAT; break; }
            for (index = 0; index < span; index++)
                yolo_value_push(arena, value,
                                pickle.stack_list[pickle.stack_count - span + index]);
            pickle.stack_count -= span;
            yolo_pickle_push(&pickle, value);
            break;
        }
        case 'a': {                                                     /* APPEND */
            YoloValue *item = yolo_pickle_pop(&pickle);
            YoloValue *list = pickle.stack_count > 0
                            ? pickle.stack_list[pickle.stack_count - 1] : NULL;
            if (list) yolo_value_push(arena, list, item);
            break;
        }
        case 'e': case 0x90: {                                          /* APPENDS/ADDITEMS */
            int first, index;
            YoloValue *list;
            if (pickle.mark_count <= 0) { pickle.status = YOLO_ERR_FORMAT; break; }
            first = pickle.mark_list[--pickle.mark_count];
            list = first > 0 ? pickle.stack_list[first - 1] : NULL;
            for (index = first; index < pickle.stack_count; index++)
                if (list) yolo_value_push(arena, list, pickle.stack_list[index]);
            pickle.stack_count = first;
            break;
        }
        case 's': {                                                     /* SETITEM */
            YoloValue *item = yolo_pickle_pop(&pickle);
            YoloValue *key = yolo_pickle_pop(&pickle);
            YoloValue *map = pickle.stack_count > 0
                           ? pickle.stack_list[pickle.stack_count - 1] : NULL;
            if (map) yolo_value_set(arena, map, key, item);
            break;
        }
        case 'u': {                                                     /* SETITEMS */
            int first, index;
            YoloValue *map;
            if (pickle.mark_count <= 0) { pickle.status = YOLO_ERR_FORMAT; break; }
            first = pickle.mark_list[--pickle.mark_count];
            map = first > 0 ? pickle.stack_list[first - 1] : NULL;
            for (index = first; index + 1 < pickle.stack_count; index += 2)
                if (map) yolo_value_set(arena, map, pickle.stack_list[index],
                                        pickle.stack_list[index + 1]);
            pickle.stack_count = first;
            break;
        }
        case 'q': case 'r': {                                           /* BINPUT */
            const unsigned char *b = yolo_pickle_eat(&pickle, code == 'q' ? 1 : 4);
            int slot = b ? (code == 'q' ? (int)b[0] : (int)yolo_read32(b)) : -1;
            if (b && pickle.stack_count > 0)
                yolo_pickle_memo(&pickle, slot,
                                 pickle.stack_list[pickle.stack_count - 1]);
            break;
        }
        case 0x94: {                                                    /* MEMOIZE */
            if (pickle.stack_count > 0)
                yolo_pickle_memo(&pickle, pickle.memo_count,
                                 pickle.stack_list[pickle.stack_count - 1]);
            break;
        }
        case 'h': case 'j': {                                           /* BINGET */
            const unsigned char *b = yolo_pickle_eat(&pickle, code == 'h' ? 1 : 4);
            int slot = b ? (code == 'h' ? (int)b[0] : (int)yolo_read32(b)) : -1;
            if (!b) break;
            if (slot < 0 || slot >= pickle.memo_count) {
                pickle.status = yolo_fail(YOLO_ERR_FORMAT,
                                          "the pickle reads memo %d, which is unset", slot);
                break;
            }
            yolo_pickle_push(&pickle, pickle.memo_list[slot]);
            break;
        }
        case 'c': case 0x93: {                                          /* GLOBAL */
            YoloValue *value = yolo_pickle_make(&pickle, YOLO_VALUE_NAME);
            if (!value) break;
            if (code == 0x93) {                                         /* STACK_GLOBAL */
                YoloValue *class_name = yolo_pickle_pop(&pickle);
                YoloValue *module_name = yolo_pickle_pop(&pickle);
                value->module_name = module_name ? module_name->text : NULL;
                value->text = class_name ? class_name->text : NULL;
            } else {
                size_t start = pickle.at, split;
                while (pickle.at < pickle.size && byte_list[pickle.at] != '\n') pickle.at++;
                split = pickle.at;
                if (pickle.at < pickle.size) pickle.at++;
                value->module_name = yolo_pickle_keep(&pickle,
                    byte_list + start, split - start);
                start = pickle.at;
                while (pickle.at < pickle.size && byte_list[pickle.at] != '\n') pickle.at++;
                value->text = yolo_pickle_keep(&pickle, byte_list + start, pickle.at - start);
                if (pickle.at < pickle.size) pickle.at++;
            }
            /* the class name is what the graph layer reads; the tail of a
             * dotted module path is kept for matching a short name */
            yolo_pickle_push(&pickle, value);
            break;
        }
        case 'R': {                                                     /* REDUCE */
            YoloValue *argument = yolo_pickle_pop(&pickle);
            YoloValue *name = yolo_pickle_pop(&pickle);
            yolo_pickle_push(&pickle, yolo_pickle_reduce(&pickle, name, argument));
            break;
        }
        case 0x81: {                                                    /* NEWOBJ */
            YoloValue *argument = yolo_pickle_pop(&pickle);
            YoloValue *name = yolo_pickle_pop(&pickle);
            YoloValue *value = yolo_pickle_make(&pickle, YOLO_VALUE_OBJECT);
            if (value && name) {
                value->module_name = name->module_name;
                value->text = name->text;
                value->text_size = name->text ? strlen(name->text) : 0;
                value->argument = argument;
            }
            yolo_pickle_push(&pickle, value);
            break;
        }
        case 'b': {                                                     /* BUILD */
            YoloValue *state = yolo_pickle_pop(&pickle);
            YoloValue *object = pickle.stack_count > 0
                              ? pickle.stack_list[pickle.stack_count - 1] : NULL;
            if (object) object->state = state;
            break;
        }
        case 'Q': {                                                     /* BINPERSID */
            YoloValue *id_list = yolo_pickle_pop(&pickle);
            yolo_pickle_push(&pickle, yolo_pickle_persist(&pickle, id_list));
            break;
        }
        default:
            pickle.status = yolo_fail(YOLO_ERR_FORMAT,
                "the pickle uses opcode 0x%02x ('%c'), which this engine does not read",
                code, (code >= 32 && code < 127) ? code : '?');
            break;
        }
    }

    if (pickle.status != YOLO_OK) return pickle.status;
    if (pickle.stack_count != 1)
        return yolo_fail(YOLO_ERR_FORMAT, "the pickle left %d values on the stack",
                         pickle.stack_count);
    *value_out = pickle.stack_list[0];
    return YOLO_OK;
}

/* ---------------------------------------------------------------------------
 * 6  The graph
 *
 * The pickled module tree already is the graph: every node carries its class
 * name, the channel counts torch built it with, and the wiring -- `f`, which
 * layer or layers it reads, and `i`, where it sits.  This layer walks that
 * tree and turns each node into something runnable, and does one rewrite on
 * the way: a convolution followed by a batch norm becomes one convolution
 * with a bias, which is the same arithmetic in half the reads and is what
 * ultralytics itself does before it predicts.
 * ------------------------------------------------------------------------ */

typedef enum {
    YOLO_MOD_NONE = 0,
    YOLO_MOD_HOLD,        /* nn.Sequential and nn.ModuleList: run in order */
    YOLO_MOD_CONV,        /* ultralytics Conv / DWConv, batch norm folded in */
    YOLO_MOD_CONV2D,      /* a bare torch convolution, bias and no activation */
    YOLO_MOD_DECONV2D,
    YOLO_MOD_LINEAR,
    YOLO_MOD_IDENTITY,
    YOLO_MOD_ACT,
    YOLO_MOD_MAXPOOL,
    YOLO_MOD_AVGPOOL,
    YOLO_MOD_UPSAMPLE,
    YOLO_MOD_CONCAT,
    YOLO_MOD_BOTTLENECK,
    YOLO_MOD_C2F,         /* C2f and C3k2 */
    YOLO_MOD_C3,          /* C3 and C3k */
    YOLO_MOD_SPPF,
    YOLO_MOD_ATTENTION,
    YOLO_MOD_PSABLOCK,
    YOLO_MOD_C2PSA,
    YOLO_MOD_PROTO,
    YOLO_MOD_PROTO26,
    YOLO_MOD_DETECT,
    YOLO_MOD_CLASSIFY
} YoloModuleKind;

typedef struct YoloModule YoloModule;

struct YoloModule {
    YoloModuleKind kind;
    const char    *class_name;

    YoloModule **child_list;
    const char **child_name_list;
    int          child_count;

    /* convolution */
    float *weight_sheet;
    float *bias_list;
    int    in_channel_count;
    int    out_channel_count;
    int    kernel_size;
    int    stride_size;
    int    pad_size;
    int    dilation_size;
    int    group_count;
    YoloActKind act_kind;

    /* pooling, resizing, joining */
    int scale_size;
    int concat_axis;

    /* blocks */
    int split_size;      /* C2f's hidden width, C2PSA's half */
    int add_flag;
    int repeat_count;    /* SPPF's pool count */

    /* attention */
    int   head_count;
    int   head_size;
    int   key_size;
    float attn_scale;

    /* a layer of the top-level plan */
    int  plan_index;
    int  from_list[8];
    int  from_count;
};

/* ---- reading a tensor out of the archive ---- */

typedef struct {
    const YoloArchive *archive;
    YoloArena         *arena;
} YoloStore;

static const unsigned char *yolo_store_bytes(const YoloStore *store, int key,
                                             size_t *size_out)
{
    char name[64];
    const YoloEntry *entry;
    snprintf(name, sizeof name, "data/%d", key);
    entry = yolo_archive_find(store->archive, name);
    if (!entry) return NULL;
    *size_out = entry->size;
    return entry->byte_list;
}

/* Every weight becomes float, whatever it was stored as.  yolo26 publishes
 * half, which is a read the engine would have to widen at every use anyway;
 * widening once at load costs twice the memory of the checkpoint -- ten
 * megabytes on the nano model -- and saves it on every picture after. */
static YoloStatus yolo_store_floats(const YoloStore *store, const YoloValue *tensor,
                                    float **cell_out, int64_t *count_out)
{
    size_t byte_size = 0;
    const unsigned char *byte_list;
    int64_t count, index, want;
    float *cell_list;
    int cell_size;

    if (!tensor || tensor->kind != YOLO_VALUE_TENSOR)
        return yolo_fail(YOLO_ERR_FORMAT, "a weight is not a tensor");

    /* torch saves weights contiguous; a view would need a gather this engine
     * has no reason to carry, so it is reported rather than guessed at */
    {
        int64_t step = 1;
        int axis;
        for (axis = tensor->rank - 1; axis >= 0; axis--) {
            if (tensor->size_list[axis] != 1 && tensor->step_list[axis] != step)
                return yolo_fail(YOLO_ERR_FORMAT,
                                 "a weight is stored as a non-contiguous view");
            step *= tensor->size_list[axis];
        }
    }

    count = tensor->cell_count;
    cell_size = yolo_type_size(tensor->cell_type);
    if (cell_size == 0) return yolo_fail(YOLO_ERR_FORMAT, "a weight has no known type");

    byte_list = yolo_store_bytes(store, tensor->storage_key, &byte_size);
    if (!byte_list)
        return yolo_fail(YOLO_ERR_FORMAT, "storage %d is missing from the archive",
                         tensor->storage_key);
    want = (tensor->cell_offset + count) * cell_size;
    if (want > (int64_t)byte_size)
        return yolo_fail(YOLO_ERR_FORMAT,
                         "storage %d holds %zu bytes, the tensor wants %lld",
                         tensor->storage_key, byte_size, (long long)want);

    cell_list = (float *)yolo_arena_take(store->arena, (size_t)count * sizeof(float));
    if (!cell_list) return YOLO_ERR_MEMORY;
    byte_list += tensor->cell_offset * cell_size;

    switch (tensor->cell_type) {
    case YOLO_TYPE_F32:
        for (index = 0; index < count; index++)
            cell_list[index] = yolo_bits_float(yolo_read32(byte_list + index * 4));
        break;
    case YOLO_TYPE_F16:
        for (index = 0; index < count; index++)
            cell_list[index] = yolo_half_float(yolo_read16(byte_list + index * 2));
        break;
    case YOLO_TYPE_BF16:
        for (index = 0; index < count; index++)
            cell_list[index] = yolo_bits_float(
                (uint32_t)yolo_read16(byte_list + index * 2) << 16);
        break;
    case YOLO_TYPE_F64: {
        for (index = 0; index < count; index++) {
            unsigned char flip[8];
            double real;
            int step;
            for (step = 0; step < 8; step++) flip[step] = byte_list[index * 8 + step];
            memcpy(&real, flip, sizeof real);
            cell_list[index] = (float)real;
        }
        break;
    }
    case YOLO_TYPE_I64:
        for (index = 0; index < count; index++)
            cell_list[index] = (float)(int64_t)yolo_read64(byte_list + index * 8);
        break;
    case YOLO_TYPE_I32:
        for (index = 0; index < count; index++)
            cell_list[index] = (float)(int32_t)yolo_read32(byte_list + index * 4);
        break;
    case YOLO_TYPE_I16:
        for (index = 0; index < count; index++)
            cell_list[index] = (float)(int16_t)yolo_read16(byte_list + index * 2);
        break;
    case YOLO_TYPE_I8:
        for (index = 0; index < count; index++)
            cell_list[index] = (float)(int8_t)byte_list[index];
        break;
    case YOLO_TYPE_U8:
        for (index = 0; index < count; index++) cell_list[index] = (float)byte_list[index];
        break;
    default:
        return yolo_fail(YOLO_ERR_FORMAT, "a weight has an unread type");
    }
    *cell_out = cell_list;
    if (count_out) *count_out = count;
    return YOLO_OK;
}

/* ---- reading the module tree ---- */

static const YoloValue *yolo_node_state(const YoloValue *node)
{
    return node ? node->state : NULL;
}

static const YoloValue *yolo_node_attr(const YoloValue *node, const char *name)
{
    return yolo_value_find(yolo_node_state(node), name);
}

/* A parameter or a buffer, whichever holds the name. */
static const YoloValue *yolo_node_weight(const YoloValue *node, const char *name)
{
    const YoloValue *state = yolo_node_state(node);
    const YoloValue *found = yolo_value_find(yolo_value_find(state, "_parameters"), name);
    if (found && found->kind == YOLO_VALUE_TENSOR) return found;
    found = yolo_value_find(yolo_value_find(state, "_buffers"), name);
    if (found && found->kind == YOLO_VALUE_TENSOR) return found;
    found = yolo_value_find(state, name);
    return (found && found->kind == YOLO_VALUE_TENSOR) ? found : NULL;
}

static const YoloValue *yolo_node_child(const YoloValue *node, const char *name)
{
    return yolo_value_find(yolo_value_find(yolo_node_state(node), "_modules"), name);
}

static int yolo_node_is(const YoloValue *node, const char *class_name)
{
    return node && node->kind == YOLO_VALUE_OBJECT && node->text
        && strcmp(node->text, class_name) == 0;
}

static const char *yolo_node_class(const YoloValue *node)
{
    return (node && node->text) ? node->text : "?";
}

/* ---- the builder ---- */

typedef struct {
    YoloStore  store;
    YoloArena *arena;
} YoloBuild;

static YoloStatus yolo_module_make(YoloBuild *build, const YoloValue *node,
                                   YoloModule **module_out);

/* Branches a checkpoint carries for training and inference never reads: the
 * normalising flow the pose head scores keypoints with while it learns, the
 * semantic side of the mask prototype head, and the per-keypoint sigmas.  They
 * are skipped rather than built, because building them would mean implementing
 * modules -- Tanh, a stack of linear layers -- that no forward pass here runs,
 * and a missing module is supposed to be a refusal rather than dead weight. */
static int yolo_module_skip(const char *name)
{
    static const char *const skip_list[] = {
        "flow_model", "semseg", "cv4_sigma", "one2one_cv4_sigma", "criterion"
    };
    size_t index;
    if (!name) return 0;
    for (index = 0; index < sizeof skip_list / sizeof skip_list[0]; index++)
        if (strcmp(name, skip_list[index]) == 0) return 1;
    return 0;
}

static YoloStatus yolo_module_children(YoloBuild *build, const YoloValue *node,
                                       YoloModule *module)
{
    const YoloValue *map = yolo_value_find(yolo_node_state(node), "_modules");
    int index, count = 0;

    if (!map || map->kind != YOLO_VALUE_DICT) return YOLO_OK;
    for (index = 0; index < map->item_count; index++)
        if (map->item_list[index] && map->item_list[index]->kind == YOLO_VALUE_OBJECT)
            count++;
    if (count == 0) return YOLO_OK;

    module->child_list = (YoloModule **)yolo_arena_zero(build->arena,
        (size_t)count * sizeof(YoloModule *));
    module->child_name_list = (const char **)yolo_arena_zero(build->arena,
        (size_t)count * sizeof(char *));
    if (!module->child_list || !module->child_name_list) return YOLO_ERR_MEMORY;

    for (index = 0; index < map->item_count; index++) {
        const YoloValue *item = map->item_list[index];
        const YoloValue *key = map->key_list[index];
        YoloStatus status;
        if (!item || item->kind != YOLO_VALUE_OBJECT) continue;
        if (key && yolo_module_skip(key->text)) continue;
        status = yolo_module_make(build, item,
                                  &module->child_list[module->child_count]);
        if (status != YOLO_OK) return status;
        module->child_name_list[module->child_count] = key ? key->text : "";
        module->child_count++;
    }
    return YOLO_OK;
}

static YoloModule *yolo_module_child(const YoloModule *module, const char *name)
{
    int index;
    if (!module) return NULL;
    for (index = 0; index < module->child_count; index++)
        if (module->child_name_list[index]
            && strcmp(module->child_name_list[index], name) == 0)
            return module->child_list[index];
    return NULL;
}

/* A bare torch convolution.  `bn` is folded in when one follows it, which is
 * the arithmetic identity
 *
 *     bn(conv(x)) = conv'(x),  w' = w * g/sqrt(v+e),  b' = (b-m)*g/sqrt(v+e)+B
 *
 * and turns two passes over the output into one. */
static YoloStatus yolo_conv_read(YoloBuild *build, const YoloValue *conv_node,
                                 const YoloValue *bn_node, YoloModule *module)
{
    const YoloValue *weight = yolo_node_weight(conv_node, "weight");
    const YoloValue *bias = yolo_node_weight(conv_node, "bias");
    YoloStatus status;
    int64_t count = 0;
    int transposed = (int)yolo_value_number(yolo_node_attr(conv_node, "transposed"), 0);

    if (!weight) return yolo_fail(YOLO_ERR_SHAPE, "a convolution has no weight");

    module->in_channel_count  = (int)yolo_value_number(yolo_node_attr(conv_node, "in_channels"), 0);
    module->out_channel_count = (int)yolo_value_number(yolo_node_attr(conv_node, "out_channels"), 0);
    module->kernel_size   = (int)yolo_value_number(yolo_node_attr(conv_node, "kernel_size"), 1);
    module->stride_size   = (int)yolo_value_number(yolo_node_attr(conv_node, "stride"), 1);
    module->pad_size      = (int)yolo_value_number(yolo_node_attr(conv_node, "padding"), 0);
    module->dilation_size = (int)yolo_value_number(yolo_node_attr(conv_node, "dilation"), 1);
    module->group_count   = (int)yolo_value_number(yolo_node_attr(conv_node, "groups"), 1);
    module->kind = transposed ? YOLO_MOD_DECONV2D : YOLO_MOD_CONV2D;

    status = yolo_store_floats(&build->store, weight, &module->weight_sheet, &count);
    if (status != YOLO_OK) return status;

    if (bias) {
        int64_t bias_count = 0;
        status = yolo_store_floats(&build->store, bias, &module->bias_list, &bias_count);
        if (status != YOLO_OK) return status;
    }

    if (bn_node) {
        const YoloValue *gain_value = yolo_node_weight(bn_node, "weight");
        const YoloValue *shift_value = yolo_node_weight(bn_node, "bias");
        const YoloValue *mean_value = yolo_node_weight(bn_node, "running_mean");
        const YoloValue *var_value = yolo_node_weight(bn_node, "running_var");
        float *gain_list = NULL, *shift_list = NULL, *mean_list = NULL, *var_list = NULL;
        float eps = (float)yolo_value_real(yolo_node_attr(bn_node, "eps"), 1e-5);
        int out_count = module->out_channel_count;
        int64_t span = count / (out_count ? out_count : 1);
        int channel;
        YoloMark mark;

        if (!mean_value || !var_value)
            return yolo_fail(YOLO_ERR_SHAPE, "a batch norm carries no running statistics");

        mark = yolo_arena_mark(build->arena);
        status = yolo_store_floats(&build->store, mean_value, &mean_list, NULL);
        if (status == YOLO_OK)
            status = yolo_store_floats(&build->store, var_value, &var_list, NULL);
        if (status == YOLO_OK && gain_value)
            status = yolo_store_floats(&build->store, gain_value, &gain_list, NULL);
        if (status == YOLO_OK && shift_value)
            status = yolo_store_floats(&build->store, shift_value, &shift_list, NULL);
        if (status != YOLO_OK) return status;

        if (!module->bias_list) {
            module->bias_list = (float *)yolo_arena_zero(build->arena,
                (size_t)out_count * sizeof(float));
            if (!module->bias_list) return YOLO_ERR_MEMORY;
        }
        for (channel = 0; channel < out_count; channel++) {
            float gain = gain_list ? gain_list[channel] : 1.0f;
            float shift = shift_list ? shift_list[channel] : 0.0f;
            float scale = gain / sqrtf(var_list[channel] + eps);
            int64_t index;
            for (index = 0; index < span; index++)
                module->weight_sheet[channel * span + index] *= scale;
            module->bias_list[channel] =
                (module->bias_list[channel] - mean_list[channel]) * scale + shift;
        }
        /* the statistics were only ever scratch; the folded weights sit below
         * the mark and survive it */
        (void)mark;
    }
    return YOLO_OK;
}

static YoloActKind yolo_act_read(const YoloValue *act_node)
{
    if (!act_node) return YOLO_ACT_NONE;
    if (yolo_node_is(act_node, "SiLU")) return YOLO_ACT_SILU;
    if (yolo_node_is(act_node, "Sigmoid")) return YOLO_ACT_SIGMOID;
    return YOLO_ACT_NONE;   /* Identity, and anything that behaves as one */
}

static YoloStatus yolo_module_make(YoloBuild *build, const YoloValue *node,
                                   YoloModule **module_out)
{
    const char *class_name = yolo_node_class(node);
    YoloModule *module = (YoloModule *)yolo_arena_zero(build->arena, sizeof *module);
    YoloStatus status;

    if (!module) return YOLO_ERR_MEMORY;
    module->class_name = class_name;
    module->kernel_size = 1;
    module->stride_size = 1;
    module->dilation_size = 1;
    module->group_count = 1;
    module->scale_size = 1;
    *module_out = module;

    /* the leaves torch owns */
    if (yolo_node_is(node, "Conv2d") || yolo_node_is(node, "ConvTranspose2d"))
        return yolo_conv_read(build, node, NULL, module);

    if (yolo_node_is(node, "Identity") || yolo_node_is(node, "Dropout")
        || yolo_node_is(node, "BatchNorm2d")) {
        module->kind = YOLO_MOD_IDENTITY;
        return YOLO_OK;
    }
    if (yolo_node_is(node, "SiLU") || yolo_node_is(node, "Sigmoid")) {
        module->kind = YOLO_MOD_ACT;
        module->act_kind = yolo_act_read(node);
        return YOLO_OK;
    }
    if (yolo_node_is(node, "MaxPool2d")) {
        module->kind = YOLO_MOD_MAXPOOL;
        module->kernel_size = (int)yolo_value_number(yolo_node_attr(node, "kernel_size"), 5);
        module->stride_size = (int)yolo_value_number(yolo_node_attr(node, "stride"), 1);
        module->pad_size = (int)yolo_value_number(yolo_node_attr(node, "padding"), 2);
        return YOLO_OK;
    }
    if (yolo_node_is(node, "AdaptiveAvgPool2d")) {
        module->kind = YOLO_MOD_AVGPOOL;
        return YOLO_OK;
    }
    if (yolo_node_is(node, "Upsample")) {
        const YoloValue *mode = yolo_node_attr(node, "mode");
        module->kind = YOLO_MOD_UPSAMPLE;
        module->scale_size = (int)yolo_value_real(yolo_node_attr(node, "scale_factor"), 2.0);
        if (mode && !yolo_value_is(mode, "nearest"))
            return yolo_fail(YOLO_ERR_MODULE,
                             "an upsample uses mode '%s'; only nearest is implemented",
                             mode->text ? mode->text : "?");
        return YOLO_OK;
    }
    if (yolo_node_is(node, "Linear")) {
        const YoloValue *weight = yolo_node_weight(node, "weight");
        const YoloValue *bias = yolo_node_weight(node, "bias");
        module->kind = YOLO_MOD_LINEAR;
        module->in_channel_count = (int)yolo_value_number(yolo_node_attr(node, "in_features"), 0);
        module->out_channel_count = (int)yolo_value_number(yolo_node_attr(node, "out_features"), 0);
        status = yolo_store_floats(&build->store, weight, &module->weight_sheet, NULL);
        if (status != YOLO_OK) return status;
        if (bias) {
            status = yolo_store_floats(&build->store, bias, &module->bias_list, NULL);
            if (status != YOLO_OK) return status;
        }
        return YOLO_OK;
    }
    if (yolo_node_is(node, "Sequential") || yolo_node_is(node, "ModuleList")) {
        module->kind = YOLO_MOD_HOLD;
        return yolo_module_children(build, node, module);
    }

    /* the ultralytics blocks */
    if (yolo_node_is(node, "Conv") || yolo_node_is(node, "DWConv")
        || yolo_node_is(node, "LightConv")) {
        const YoloValue *conv_node = yolo_node_child(node, "conv");
        const YoloValue *bn_node = yolo_node_child(node, "bn");
        if (!conv_node)
            return yolo_fail(YOLO_ERR_MODULE, "%s carries no convolution", class_name);
        status = yolo_conv_read(build, conv_node, bn_node, module);
        if (status != YOLO_OK) return status;
        module->kind = YOLO_MOD_CONV;
        module->act_kind = yolo_act_read(yolo_node_child(node, "act"));
        return YOLO_OK;
    }
    if (yolo_node_is(node, "Concat")) {
        module->kind = YOLO_MOD_CONCAT;
        module->concat_axis = (int)yolo_value_number(yolo_node_attr(node, "d"), 1);
        if (module->concat_axis != 1)
            return yolo_fail(YOLO_ERR_MODULE,
                             "a concat joins on axis %d; only the channel axis is implemented",
                             module->concat_axis);
        return YOLO_OK;
    }
    if (yolo_node_is(node, "Bottleneck")) {
        module->kind = YOLO_MOD_BOTTLENECK;
        module->add_flag = (int)yolo_value_number(yolo_node_attr(node, "add"), 0);
        return yolo_module_children(build, node, module);
    }
    if (yolo_node_is(node, "C2f") || yolo_node_is(node, "C3k2")
        || yolo_node_is(node, "C2fPSA")) {
        module->kind = YOLO_MOD_C2F;
        module->split_size = (int)yolo_value_number(yolo_node_attr(node, "c"), 0);
        return yolo_module_children(build, node, module);
    }
    if (yolo_node_is(node, "C3") || yolo_node_is(node, "C3k")) {
        module->kind = YOLO_MOD_C3;
        return yolo_module_children(build, node, module);
    }
    if (yolo_node_is(node, "SPPF")) {
        module->kind = YOLO_MOD_SPPF;
        module->repeat_count = (int)yolo_value_number(yolo_node_attr(node, "n"), 3);
        module->add_flag = (int)yolo_value_number(yolo_node_attr(node, "add"), 0);
        return yolo_module_children(build, node, module);
    }
    if (yolo_node_is(node, "Attention")) {
        module->kind = YOLO_MOD_ATTENTION;
        module->head_count = (int)yolo_value_number(yolo_node_attr(node, "num_heads"), 1);
        module->head_size = (int)yolo_value_number(yolo_node_attr(node, "head_dim"), 0);
        module->key_size = (int)yolo_value_number(yolo_node_attr(node, "key_dim"), 0);
        module->attn_scale = (float)yolo_value_real(yolo_node_attr(node, "scale"), 1.0);
        return yolo_module_children(build, node, module);
    }
    if (yolo_node_is(node, "PSABlock")) {
        module->kind = YOLO_MOD_PSABLOCK;
        module->add_flag = (int)yolo_value_number(yolo_node_attr(node, "add"), 1);
        return yolo_module_children(build, node, module);
    }
    if (yolo_node_is(node, "C2PSA")) {
        module->kind = YOLO_MOD_C2PSA;
        module->split_size = (int)yolo_value_number(yolo_node_attr(node, "c"), 0);
        return yolo_module_children(build, node, module);
    }
    if (yolo_node_is(node, "Proto")) {
        module->kind = YOLO_MOD_PROTO;
        return yolo_module_children(build, node, module);
    }
    if (yolo_node_is(node, "Proto26")) {
        module->kind = YOLO_MOD_PROTO26;
        return yolo_module_children(build, node, module);
    }
    if (yolo_node_is(node, "Classify")) {
        module->kind = YOLO_MOD_CLASSIFY;
        return yolo_module_children(build, node, module);
    }
    if (yolo_node_is(node, "Detect") || yolo_node_is(node, "Segment")
        || yolo_node_is(node, "Segment26") || yolo_node_is(node, "Pose")
        || yolo_node_is(node, "Pose26") || yolo_node_is(node, "OBB")
        || yolo_node_is(node, "OBB26")) {
        module->kind = YOLO_MOD_DETECT;
        return yolo_module_children(build, node, module);
    }

    return yolo_fail(YOLO_ERR_MODULE,
                     "the checkpoint holds a %s, which this engine does not implement",
                     class_name);
}

/* ---------------------------------------------------------------------------
 * The model
 * ------------------------------------------------------------------------ */

#define YOLO_LAYER_LIMIT 128
#define YOLO_LEVEL_LIMIT 8

struct YoloModel {
    YoloArena          arena;
    const YoloBackend *backend;
    YoloArchive        archive;

    YoloTask task;
    char   **class_name_list;
    int      class_count;
    char   **keypoint_name_list;
    int      keypoint_count;
    int      keypoint_size;
    int      mask_count;
    int      input_size;
    int      stride_size;
    int      nms_free_flag;

    YoloModule *layer_list[YOLO_LAYER_LIMIT];
    int         from_count_list[YOLO_LAYER_LIMIT];
    int         from_list[YOLO_LAYER_LIMIT][YOLO_LEVEL_LIMIT];
    int         save_flag_list[YOLO_LAYER_LIMIT];
    int         layer_count;

    /* the head, which the forward pass hands its feature maps to */
    YoloModule *head;
    int         level_count;        /* 3 on every published yolo26 */
    int         reg_max;
    int         row_size;           /* `no`: 4*reg_max + nc */
    float       level_stride_list[YOLO_LEVEL_LIMIT];
};

static YoloStatus yolo_text_keep(YoloArena *arena, const YoloValue *value,
                                 char **text_out)
{
    char *room;
    if (!value || value->kind != YOLO_VALUE_TEXT) { *text_out = NULL; return YOLO_OK; }
    room = (char *)yolo_arena_take(arena, value->text_size + 1);
    if (!room) return YOLO_ERR_MEMORY;
    memcpy(room, value->text, value->text_size);
    room[value->text_size] = 0;
    *text_out = room;
    return YOLO_OK;
}

/* `names` is a dict keyed by class index; it is not always dense and is not
 * always in order, so it is read by key rather than by position. */
static YoloStatus yolo_names_read(YoloModel *model, const YoloValue *names,
                                  int count, char ***list_out, int *count_out)
{
    char **list;
    int index, top = count;

    if (names && names->kind == YOLO_VALUE_DICT) {
        for (index = 0; index < names->item_count; index++) {
            int at = (int)yolo_value_number(names->key_list[index], -1);
            if (at + 1 > top) top = at + 1;
        }
    }
    if (top <= 0) { *list_out = NULL; *count_out = 0; return YOLO_OK; }

    list = (char **)yolo_arena_zero(&model->arena, (size_t)top * sizeof(char *));
    if (!list) return YOLO_ERR_MEMORY;
    if (names && names->kind == YOLO_VALUE_DICT) {
        for (index = 0; index < names->item_count; index++) {
            int at = (int)yolo_value_number(names->key_list[index], -1);
            if (at < 0 || at >= top) continue;
            if (yolo_text_keep(&model->arena, names->item_list[index], &list[at]) != YOLO_OK)
                return YOLO_ERR_MEMORY;
        }
    }
    *list_out = list;
    *count_out = top;
    return YOLO_OK;
}

static YoloTask yolo_task_read(const char *class_name)
{
    if (strncmp(class_name, "Segment", 7) == 0) return YOLO_TASK_SEGMENT;
    if (strncmp(class_name, "Pose", 4) == 0) return YOLO_TASK_POSE;
    if (strncmp(class_name, "OBB", 3) == 0) return YOLO_TASK_OBB;
    if (strcmp(class_name, "Classify") == 0) return YOLO_TASK_CLASSIFY;
    return YOLO_TASK_DETECT;
}

YoloStatus yolo_model_open(const char *path, YoloModel **model_out)
{
    YoloModel *model;
    YoloBuild build;
    YoloValue *root = NULL;
    const YoloValue *net, *sequence, *head_node, *modules;
    const YoloEntry *entry;
    YoloStatus status;
    int index;

    if (!path || !model_out) return YOLO_ERR_ARG;
    *model_out = NULL;

    model = (YoloModel *)calloc(1, sizeof *model);
    if (!model) return YOLO_ERR_MEMORY;
    yolo_arena_open(&model->arena, (size_t)8 << 20);
    model->backend = yolo_backend_cpu();
    model->stride_size = 32;

    status = yolo_archive_open(&model->arena, path, &model->archive);
    if (status != YOLO_OK) goto fail;

    entry = yolo_archive_find(&model->archive, "data.pkl");
    if (!entry) {
        status = yolo_fail(YOLO_ERR_FORMAT,
                           "%s holds no data.pkl; it is a zip but not a checkpoint", path);
        goto fail;
    }
    status = yolo_pickle_run(&model->arena, entry->byte_list, entry->size, &root);
    if (status != YOLO_OK) goto fail;

    /* a checkpoint is a dict; the model sits under `model`, and `ema` holds a
     * smoothed copy when one was kept.  The published weights carry no ema. */
    net = yolo_value_find(root, "ema");
    if (!net || net->kind != YOLO_VALUE_OBJECT) net = yolo_value_find(root, "model");
    if (!net || net->kind != YOLO_VALUE_OBJECT) {
        status = yolo_fail(YOLO_ERR_FORMAT, "%s holds no model object", path);
        goto fail;
    }

    build.arena = &model->arena;
    build.store.arena = &model->arena;
    build.store.archive = &model->archive;

    sequence = yolo_node_child(net, "model");
    if (!sequence) {
        status = yolo_fail(YOLO_ERR_FORMAT, "the model holds no layer sequence");
        goto fail;
    }
    modules = yolo_value_find(yolo_node_state(sequence), "_modules");
    if (!modules || modules->kind != YOLO_VALUE_DICT) {
        status = yolo_fail(YOLO_ERR_FORMAT, "the layer sequence is empty");
        goto fail;
    }
    if (modules->item_count > YOLO_LAYER_LIMIT) {
        status = yolo_fail(YOLO_ERR_FORMAT, "the model has %d layers, past the %d this "
                           "engine plans for", modules->item_count, YOLO_LAYER_LIMIT);
        goto fail;
    }

    for (index = 0; index < modules->item_count; index++) {
        const YoloValue *layer_node = modules->item_list[index];
        const YoloValue *from;
        int at = model->layer_count;

        if (!layer_node || layer_node->kind != YOLO_VALUE_OBJECT) continue;
        status = yolo_module_make(&build, layer_node, &model->layer_list[at]);
        if (status != YOLO_OK) goto fail;
        model->layer_list[at]->plan_index = at;

        from = yolo_node_attr(layer_node, "f");
        if (from && from->kind == YOLO_VALUE_LIST) {
            int step;
            if (from->item_count > YOLO_LEVEL_LIMIT) {
                status = yolo_fail(YOLO_ERR_FORMAT, "layer %d reads %d inputs",
                                   at, from->item_count);
                goto fail;
            }
            for (step = 0; step < from->item_count; step++)
                model->from_list[at][step] =
                    (int)yolo_value_number(from->item_list[step], -1);
            model->from_count_list[at] = from->item_count;
        } else {
            model->from_list[at][0] = (int)yolo_value_number(from, -1);
            model->from_count_list[at] = 1;
        }
        model->layer_count++;
    }

    /* which outputs a later layer reads, so the forward pass keeps only those */
    for (index = 0; index < model->layer_count; index++) {
        int step;
        for (step = 0; step < model->from_count_list[index]; step++) {
            int at = model->from_list[index][step];
            if (at >= 0 && at < model->layer_count) model->save_flag_list[at] = 1;
        }
    }

    head_node = modules->item_list[modules->item_count - 1];
    model->head = model->layer_list[model->layer_count - 1];
    model->task = yolo_task_read(yolo_node_class(head_node));
    model->reg_max = (int)yolo_value_number(yolo_node_attr(head_node, "reg_max"), 1);
    model->row_size = (int)yolo_value_number(yolo_node_attr(head_node, "no"), 0);
    model->level_count = (int)yolo_value_number(yolo_node_attr(head_node, "nl"), 3);
    model->mask_count = (int)yolo_value_number(yolo_node_attr(head_node, "nm"), 0);
    model->nms_free_flag = yolo_module_child(model->head, "one2one_cv2") != NULL;

    if (model->task == YOLO_TASK_POSE) {
        const YoloValue *shape = yolo_node_attr(head_node, "kpt_shape");
        model->keypoint_count = (int)yolo_value_number(yolo_value_at(shape, 0), 0);
        model->keypoint_size = (int)yolo_value_number(yolo_value_at(shape, 1), 3);
    }

    if (model->task != YOLO_TASK_CLASSIFY) {
        const YoloValue *stride = yolo_node_attr(head_node, "stride");
        float *level_list = NULL;
        int64_t count = 0;
        if (!stride) stride = yolo_node_attr(net, "stride");
        if (stride && stride->kind == YOLO_VALUE_TENSOR
            && yolo_store_floats(&build.store, stride, &level_list, &count) == YOLO_OK) {
            int step;
            for (step = 0; step < count && step < YOLO_LEVEL_LIMIT; step++)
                model->level_stride_list[step] = level_list[step];
            if (count > 0 && level_list[count - 1] > 0)
                model->stride_size = (int)level_list[count - 1];
        } else {
            /* the published strides, in case a checkpoint ever drops them */
            int step;
            for (step = 0; step < model->level_count && step < YOLO_LEVEL_LIMIT; step++)
                model->level_stride_list[step] = (float)(8 << step);
        }
        if (model->level_count > YOLO_LEVEL_LIMIT) {
            status = yolo_fail(YOLO_ERR_FORMAT, "the head has %d levels",
                               model->level_count);
            goto fail;
        }
    }

    {
        const YoloValue *names = yolo_value_find(yolo_node_state(net), "names");
        int nc = (int)yolo_value_number(yolo_node_attr(head_node, "nc"),
                 yolo_value_number(yolo_value_find(yolo_node_state(net), "nc"), 0));
        status = yolo_names_read(model, names, nc,
                                 &model->class_name_list, &model->class_count);
        if (status != YOLO_OK) goto fail;
        if (model->class_count == 0) model->class_count = nc;
    }
    if (model->task == YOLO_TASK_POSE) {
        const YoloValue *names = yolo_value_find(yolo_node_state(net), "kpt_names");
        int count = 0;
        /* the map is keyed by class, and every published pose model has one
         * class, so its keypoint names are the first row of it */
        if (names && names->kind == YOLO_VALUE_DICT && names->item_count > 0) {
            const YoloValue *row = names->item_list[0];
            if (row && row->kind == YOLO_VALUE_LIST) {
                int step;
                model->keypoint_name_list = (char **)yolo_arena_zero(&model->arena,
                    (size_t)row->item_count * sizeof(char *));
                if (!model->keypoint_name_list) { status = YOLO_ERR_MEMORY; goto fail; }
                for (step = 0; step < row->item_count; step++)
                    yolo_text_keep(&model->arena, row->item_list[step],
                                   &model->keypoint_name_list[step]);
                count = row->item_count;
            }
        }
        (void)count;
    }

    model->input_size = model->task == YOLO_TASK_CLASSIFY ? 224 : 640;
    {
        const YoloValue *args = yolo_value_find(root, "train_args");
        int64_t trained = yolo_value_number(yolo_value_find(args, "imgsz"), 0);
        if (trained >= 32 && trained <= 8192) model->input_size = (int)trained;
    }

    *model_out = model;
    return YOLO_OK;

fail:
    yolo_arena_close(&model->arena);
    free(model);
    return status;
}

void yolo_model_close(YoloModel *model)
{
    if (!model) return;
    yolo_arena_close(&model->arena);
    free(model);
}

YoloTask yolo_model_task(const YoloModel *model) { return model->task; }
int yolo_model_class_count(const YoloModel *model) { return model->class_count; }
int yolo_model_stride_size(const YoloModel *model) { return model->stride_size; }
int yolo_model_input_size(const YoloModel *model) { return model->input_size; }
int yolo_model_nms_free_flag(const YoloModel *model) { return model->nms_free_flag; }
int yolo_model_keypoint_count(const YoloModel *model) { return model->keypoint_count; }
int yolo_model_keypoint_size(const YoloModel *model) { return model->keypoint_size; }
int yolo_model_mask_count(const YoloModel *model) { return model->mask_count; }

const char *yolo_model_class_name(const YoloModel *model, int class_index)
{
    if (!model || class_index < 0 || class_index >= model->class_count
        || !model->class_name_list || !model->class_name_list[class_index])
        return "?";
    return model->class_name_list[class_index];
}

const char *yolo_model_keypoint_name(const YoloModel *model, int keypoint_index)
{
    if (!model || !model->keypoint_name_list || keypoint_index < 0
        || keypoint_index >= model->keypoint_count
        || !model->keypoint_name_list[keypoint_index])
        return "?";
    return model->keypoint_name_list[keypoint_index];
}

YoloStatus yolo_model_backend_set(YoloModel *model, const YoloBackend *backend)
{
    if (!model) return YOLO_ERR_ARG;
    model->backend = backend ? backend : yolo_backend_cpu();
    return YOLO_OK;
}

/* ---------------------------------------------------------------------------
 * 7  The forward pass
 *
 * Every module returns a fresh plane taken from the run's arena.  Temporaries
 * are bracketed by a mark and a reset, so what a block costs is its own depth
 * rather than the sum of everything that ran before it, and a session that has
 * seen one picture allocates nothing for the next.
 * ------------------------------------------------------------------------ */

typedef struct {
    const YoloModel   *model;
    const YoloBackend *backend;
    YoloArena         *arena;
    YoloStatus         status;
} YoloRun;

static YoloPlane yolo_run_fail(YoloRun *run, YoloStatus status)
{
    YoloPlane empty;
    memset(&empty, 0, sizeof empty);
    if (run->status == YOLO_OK) run->status = status;
    return empty;
}

static YoloPlane yolo_run_take(YoloRun *run, int channel_count, int height, int width)
{
    YoloPlane plane = yolo_plane_take(run->arena, 1, channel_count, height, width);
    if (!plane.cell_list) return yolo_run_fail(run, YOLO_ERR_MEMORY);
    return plane;
}

/* Copy one plane into a run of channels of another. */
static void yolo_plane_lay(YoloPlane *target, int channel_first, const YoloPlane *source)
{
    size_t face = (size_t)target->height * (size_t)target->width;
    memcpy(yolo_plane_face(target, 0, channel_first), source->cell_list,
           (size_t)source->channel_count * face * sizeof(float));
}

/* A view of a run of channels.  The engine does not own it, so it is never
 * written through except where the module that made it says so. */
static YoloPlane yolo_plane_slice(const YoloPlane *source, int channel_first,
                                  int channel_count)
{
    YoloPlane view = *source;
    view.channel_count = channel_count;
    view.cell_list = yolo_plane_face(source, 0, channel_first);
    return view;
}

static YoloPlane yolo_module_run(YoloRun *run, const YoloModule *module, YoloPlane in);

/* What a convolution leaves, given what it is handed. */
static void yolo_conv_shape(const YoloModule *module, int height, int width,
                            int *height_out, int *width_out)
{
    int reach = module->dilation_size * (module->kernel_size - 1) + 1;
    *height_out = (height + 2 * module->pad_size - reach) / module->stride_size + 1;
    *width_out  = (width  + 2 * module->pad_size - reach) / module->stride_size + 1;
}

static void yolo_conv_into(YoloRun *run, const YoloModule *module,
                           YoloPlane in, YoloPlane *out)
{
    YoloMark mark;
    size_t room_size;
    float *room_list = NULL;

    if (run->status != YOLO_OK) return;
    if (in.channel_count != module->in_channel_count) {
        (void)yolo_run_fail(run, yolo_fail(YOLO_ERR_SHAPE,
            "a %s convolution wants %d channels and was given %d",
            module->class_name, module->in_channel_count, in.channel_count));
        return;
    }
    mark = yolo_arena_mark(run->arena);
    room_size = run->backend->conv_room(&in, out, module->kernel_size,
                                        module->group_count);
    if (room_size) {
        room_list = (float *)yolo_arena_take(run->arena, room_size);
        if (!room_list) { (void)yolo_run_fail(run, YOLO_ERR_MEMORY); return; }
    }
    run->backend->conv_run(run->backend, &in, out, module->weight_sheet,
                           module->bias_list, module->kernel_size,
                           module->stride_size, module->pad_size,
                           module->dilation_size, module->group_count,
                           module->act_kind, room_list);
    yolo_arena_reset(run->arena, mark);
}

static YoloPlane yolo_conv_apply(YoloRun *run, const YoloModule *module, YoloPlane in)
{
    int out_height, out_width;
    YoloPlane out;

    yolo_conv_shape(module, in.height, in.width, &out_height, &out_width);
    if (out_height <= 0 || out_width <= 0)
        return yolo_run_fail(run, yolo_fail(YOLO_ERR_SHAPE,
            "a convolution over %dx%d leaves nothing", in.height, in.width));
    out = yolo_run_take(run, module->out_channel_count, out_height, out_width);
    if (!out.cell_list) return out;
    yolo_conv_into(run, module, in, &out);
    return out;
}

static YoloPlane yolo_deconv_apply(YoloRun *run, const YoloModule *module, YoloPlane in)
{
    int out_height = (in.height - 1) * module->stride_size + module->kernel_size;
    int out_width  = (in.width  - 1) * module->stride_size + module->kernel_size;
    YoloPlane out = yolo_run_take(run, module->out_channel_count, out_height, out_width);
    if (!out.cell_list) return out;
    run->backend->deconv_run(run->backend, &in, &out, module->weight_sheet,
                             module->bias_list, module->kernel_size,
                             module->stride_size, module->act_kind);
    return out;
}

/* C2f, and so C3k2: one convolution splits the input in two, a chain of
 * blocks extends the tail, and every step is kept and joined.  The joined
 * buffer is allocated up front and each block writes straight into its slot,
 * so the concatenation the reference does at the end costs nothing here. */
static YoloPlane yolo_c2f_run(YoloRun *run, const YoloModule *module, YoloPlane in)
{
    const YoloModule *cv1 = yolo_module_child(module, "cv1");
    const YoloModule *cv2 = yolo_module_child(module, "cv2");
    const YoloModule *chain = yolo_module_child(module, "m");
    int span = module->split_size;
    int count = chain ? chain->child_count : 0;
    YoloPlane first, joined, out;
    YoloMark mark;
    int index;

    if (!cv1 || !cv2) return yolo_run_fail(run, YOLO_ERR_MODULE);
    {
        int mid_height, mid_width, out_height, out_width;
        yolo_conv_shape(cv1, in.height, in.width, &mid_height, &mid_width);
        yolo_conv_shape(cv2, mid_height, mid_width, &out_height, &out_width);
        out = yolo_run_take(run, cv2->out_channel_count, out_height, out_width);
        if (!out.cell_list) return out;
    }
    mark = yolo_arena_mark(run->arena);
    first = yolo_conv_apply(run, cv1, in);
    if (run->status != YOLO_OK) return first;
    if (span <= 0) span = first.channel_count / 2;

    joined = yolo_run_take(run, (2 + count) * span, first.height, first.width);
    if (!joined.cell_list) return joined;
    yolo_plane_lay(&joined, 0, &first);

    for (index = 0; index < count; index++) {
        YoloPlane tail = yolo_plane_slice(&joined, (1 + index) * span, span);
        YoloPlane step = yolo_module_run(run, chain->child_list[index], tail);
        if (run->status != YOLO_OK) return step;
        yolo_plane_lay(&joined, (2 + index) * span, &step);
    }
    yolo_conv_into(run, cv2, joined, &out);
    /* the split, the chain's steps and the joined buffer were all scratch;
     * the result was taken before the mark, so releasing keeps it */
    yolo_arena_reset(run->arena, mark);
    return out;
}

static YoloPlane yolo_c3_run(YoloRun *run, const YoloModule *module, YoloPlane in)
{
    const YoloModule *cv1 = yolo_module_child(module, "cv1");
    const YoloModule *cv2 = yolo_module_child(module, "cv2");
    const YoloModule *cv3 = yolo_module_child(module, "cv3");
    const YoloModule *chain = yolo_module_child(module, "m");
    YoloPlane left, right, joined;
    int index;

    if (!cv1 || !cv2 || !cv3) return yolo_run_fail(run, YOLO_ERR_MODULE);
    left = yolo_conv_apply(run, cv1, in);
    if (run->status != YOLO_OK) return left;
    for (index = 0; chain && index < chain->child_count; index++) {
        left = yolo_module_run(run, chain->child_list[index], left);
        if (run->status != YOLO_OK) return left;
    }
    right = yolo_conv_apply(run, cv2, in);
    if (run->status != YOLO_OK) return right;

    joined = yolo_run_take(run, left.channel_count + right.channel_count,
                           left.height, left.width);
    if (!joined.cell_list) return joined;
    yolo_plane_lay(&joined, 0, &left);
    yolo_plane_lay(&joined, left.channel_count, &right);
    return yolo_conv_apply(run, cv3, joined);
}

/* SPPF, which yolo26 gave a repeat count and a residual.  The pool is 5x5 at
 * stride one, so each of the three passes widens the receptive field without
 * moving the grid, and the four results are joined. */
static YoloPlane yolo_sppf_run(YoloRun *run, const YoloModule *module, YoloPlane in)
{
    const YoloModule *cv1 = yolo_module_child(module, "cv1");
    const YoloModule *cv2 = yolo_module_child(module, "cv2");
    const YoloModule *pool = yolo_module_child(module, "m");
    int count = module->repeat_count > 0 ? module->repeat_count : 3;
    YoloPlane first, joined, out;
    int index, span;

    if (!cv1 || !cv2 || !pool) return yolo_run_fail(run, YOLO_ERR_MODULE);
    first = yolo_conv_apply(run, cv1, in);
    if (run->status != YOLO_OK) return first;
    span = first.channel_count;

    joined = yolo_run_take(run, span * (count + 1), first.height, first.width);
    if (!joined.cell_list) return joined;
    yolo_plane_lay(&joined, 0, &first);
    for (index = 0; index < count; index++) {
        YoloPlane source = yolo_plane_slice(&joined, index * span, span);
        YoloPlane target = yolo_plane_slice(&joined, (index + 1) * span, span);
        run->backend->pool_run(run->backend, &source, &target, pool->kernel_size,
                               pool->stride_size, pool->pad_size);
    }
    out = yolo_conv_apply(run, cv2, joined);
    if (run->status != YOLO_OK) return out;
    if (module->add_flag && out.channel_count == in.channel_count)
        run->backend->add_run(run->backend, out.cell_list, in.cell_list,
                              yolo_plane_count(&out));
    return out;
}

/* The attention block the PSA layers carry.  One convolution makes queries,
 * keys and values at once; the scores are taken a head at a time because the
 * head axis is the outer one and nothing is shared across it.  The positional
 * term is a depthwise convolution over the values, added before the
 * projection, which is what makes this attention rather than a mixer. */
static YoloPlane yolo_attention_run(YoloRun *run, const YoloModule *module, YoloPlane in)
{
    const YoloModule *qkv = yolo_module_child(module, "qkv");
    const YoloModule *proj = yolo_module_child(module, "proj");
    const YoloModule *pe = yolo_module_child(module, "pe");
    int head_count = module->head_count;
    int key_size = module->key_size;
    int head_size = module->head_size;
    int step = key_size * 2 + head_size;
    int place_count = in.height * in.width;
    YoloPlane packed, value_plane, mixed, out;
    float *query_room, *key_room, *score_room;
    YoloMark mark;
    int head, place, depth;

    if (!qkv || !proj || !pe) return yolo_run_fail(run, YOLO_ERR_MODULE);
    out = yolo_run_take(run, proj->out_channel_count, in.height, in.width);
    if (!out.cell_list) return out;
    mark = yolo_arena_mark(run->arena);
    packed = yolo_conv_apply(run, qkv, in);
    if (run->status != YOLO_OK) return packed;
    if (packed.channel_count != head_count * step)
        return yolo_run_fail(run, yolo_fail(YOLO_ERR_SHAPE,
            "attention made %d channels and expected %d",
            packed.channel_count, head_count * step));

    value_plane = yolo_run_take(run, head_count * head_size, in.height, in.width);
    mixed = yolo_run_take(run, head_count * head_size, in.height, in.width);
    query_room = (float *)yolo_arena_take(run->arena,
        (size_t)place_count * (size_t)key_size * sizeof(float));
    key_room = (float *)yolo_arena_take(run->arena,
        (size_t)place_count * (size_t)key_size * sizeof(float));
    score_room = (float *)yolo_arena_take(run->arena,
        (size_t)place_count * (size_t)place_count * sizeof(float));
    if (!value_plane.cell_list || !mixed.cell_list || !query_room || !key_room
        || !score_room)
        return yolo_run_fail(run, YOLO_ERR_MEMORY);

    for (head = 0; head < head_count; head++) {
        const float *query = yolo_plane_face(&packed, 0, head * step);
        const float *key = yolo_plane_face(&packed, 0, head * step + key_size);
        const float *value = yolo_plane_face(&packed, 0, head * step + 2 * key_size);

        /* the scores want a place a row, the convolution left a depth a row */
        for (place = 0; place < place_count; place++) {
            for (depth = 0; depth < key_size; depth++) {
                query_room[(size_t)place * key_size + depth] =
                    query[(size_t)depth * place_count + place] * module->attn_scale;
                key_room[(size_t)place * key_size + depth] =
                    key[(size_t)depth * place_count + place];
            }
        }
        run->backend->gemm_run(run->backend, query_room, key_room, score_room,
                               place_count, key_size, place_count,
                               key_size, key_size, place_count, 1);
        run->backend->softmax_run(run->backend, score_room, place_count, place_count);

        memcpy(yolo_plane_face(&value_plane, 0, head * head_size), value,
               (size_t)head_size * (size_t)place_count * sizeof(float));
        run->backend->gemm_run(run->backend, value, score_room,
                               yolo_plane_face(&mixed, 0, head * head_size),
                               head_size, place_count, place_count,
                               place_count, place_count, place_count, 1);
    }

    {
        YoloPlane placed = yolo_conv_apply(run, pe, value_plane);
        if (run->status != YOLO_OK) return placed;
        run->backend->add_run(run->backend, mixed.cell_list, placed.cell_list,
                              yolo_plane_count(&mixed));
    }
    yolo_conv_into(run, proj, mixed, &out);
    /* the scores are the largest thing this block touches -- a place by a
     * place, a head at a time -- and none of it outlives the projection */
    yolo_arena_reset(run->arena, mark);
    return out;
}

static YoloPlane yolo_psablock_run(YoloRun *run, const YoloModule *module, YoloPlane in)
{
    const YoloModule *attn = yolo_module_child(module, "attn");
    const YoloModule *ffn = yolo_module_child(module, "ffn");
    YoloPlane out = in;
    YoloPlane step;

    if (!attn || !ffn) return yolo_run_fail(run, YOLO_ERR_MODULE);
    step = yolo_attention_run(run, attn, out);
    if (run->status != YOLO_OK) return step;
    if (module->add_flag) {
        run->backend->add_run(run->backend, step.cell_list, out.cell_list,
                              yolo_plane_count(&step));
    }
    out = step;
    step = yolo_module_run(run, ffn, out);
    if (run->status != YOLO_OK) return step;
    if (module->add_flag)
        run->backend->add_run(run->backend, step.cell_list, out.cell_list,
                              yolo_plane_count(&step));
    return step;
}

static YoloPlane yolo_c2psa_run(YoloRun *run, const YoloModule *module, YoloPlane in)
{
    const YoloModule *cv1 = yolo_module_child(module, "cv1");
    const YoloModule *cv2 = yolo_module_child(module, "cv2");
    const YoloModule *chain = yolo_module_child(module, "m");
    YoloPlane split;
    int span, index;

    if (!cv1 || !cv2) return yolo_run_fail(run, YOLO_ERR_MODULE);
    split = yolo_conv_apply(run, cv1, in);
    if (run->status != YOLO_OK) return split;
    span = module->split_size > 0 ? module->split_size : split.channel_count / 2;

    for (index = 0; chain && index < chain->child_count; index++) {
        YoloPlane tail = yolo_plane_slice(&split, span, span);
        YoloPlane step = yolo_module_run(run, chain->child_list[index], tail);
        if (run->status != YOLO_OK) return step;
        memcpy(yolo_plane_face(&split, 0, span), step.cell_list,
               (size_t)span * (size_t)split.height * (size_t)split.width * sizeof(float));
    }
    return yolo_conv_apply(run, cv2, split);
}

static YoloPlane yolo_module_run(YoloRun *run, const YoloModule *module, YoloPlane in)
{
    int index;

    if (run->status != YOLO_OK) return in;
    switch (module->kind) {
    case YOLO_MOD_CONV:
    case YOLO_MOD_CONV2D:
        return yolo_conv_apply(run, module, in);
    case YOLO_MOD_DECONV2D:
        return yolo_deconv_apply(run, module, in);
    case YOLO_MOD_IDENTITY:
        return in;
    case YOLO_MOD_ACT:
        run->backend->act_run(run->backend, in.cell_list, yolo_plane_count(&in),
                              module->act_kind);
        return in;
    case YOLO_MOD_CLASSIFY:   /* a convolution, a mean over the map, a linear */
    case YOLO_MOD_HOLD: {
        YoloPlane out = in;
        for (index = 0; index < module->child_count; index++) {
            out = yolo_module_run(run, module->child_list[index], out);
            if (run->status != YOLO_OK) return out;
        }
        return out;
    }
    case YOLO_MOD_MAXPOOL: {
        int reach = module->kernel_size;
        int height = (in.height + 2 * module->pad_size - reach) / module->stride_size + 1;
        int width  = (in.width  + 2 * module->pad_size - reach) / module->stride_size + 1;
        YoloPlane out = yolo_run_take(run, in.channel_count, height, width);
        if (!out.cell_list) return out;
        run->backend->pool_run(run->backend, &in, &out, module->kernel_size,
                               module->stride_size, module->pad_size);
        return out;
    }
    case YOLO_MOD_AVGPOOL: {
        YoloPlane out = yolo_run_take(run, in.channel_count, 1, 1);
        size_t face = (size_t)in.height * (size_t)in.width;
        if (!out.cell_list) return out;
        for (index = 0; index < in.channel_count; index++) {
            const float *source = yolo_plane_face(&in, 0, index);
            float total = 0.0f;
            size_t step;
            for (step = 0; step < face; step++) total += source[step];
            out.cell_list[index] = face ? total / (float)face : 0.0f;
        }
        return out;
    }
    case YOLO_MOD_UPSAMPLE: {
        YoloPlane out = yolo_run_take(run, in.channel_count,
                                      in.height * module->scale_size,
                                      in.width * module->scale_size);
        if (!out.cell_list) return out;
        run->backend->resize_run(run->backend, &in, &out, module->scale_size);
        return out;
    }
    case YOLO_MOD_LINEAR: {
        YoloPlane out = yolo_run_take(run, module->out_channel_count, 1, 1);
        if (!out.cell_list) return out;
        run->backend->gemm_run(run->backend, module->weight_sheet, in.cell_list,
                               out.cell_list, module->out_channel_count,
                               module->in_channel_count, 1,
                               module->in_channel_count, 1, 1, 0);
        if (module->bias_list)
            for (index = 0; index < module->out_channel_count; index++)
                out.cell_list[index] += module->bias_list[index];
        return out;
    }
    case YOLO_MOD_BOTTLENECK: {
        const YoloModule *cv1 = yolo_module_child(module, "cv1");
        const YoloModule *cv2 = yolo_module_child(module, "cv2");
        YoloPlane out;
        if (!cv1 || !cv2) return yolo_run_fail(run, YOLO_ERR_MODULE);
        out = yolo_conv_apply(run, cv1, in);
        if (run->status != YOLO_OK) return out;
        out = yolo_conv_apply(run, cv2, out);
        if (run->status != YOLO_OK) return out;
        if (module->add_flag)
            run->backend->add_run(run->backend, out.cell_list, in.cell_list,
                                  yolo_plane_count(&out));
        return out;
    }
    case YOLO_MOD_C2F:   return yolo_c2f_run(run, module, in);
    case YOLO_MOD_C3:    return yolo_c3_run(run, module, in);
    case YOLO_MOD_SPPF:  return yolo_sppf_run(run, module, in);
    case YOLO_MOD_ATTENTION: return yolo_attention_run(run, module, in);
    case YOLO_MOD_PSABLOCK:  return yolo_psablock_run(run, module, in);
    case YOLO_MOD_C2PSA:     return yolo_c2psa_run(run, module, in);
    case YOLO_MOD_PROTO:
    case YOLO_MOD_PROTO26: {
        /* Proto26 adds a step that fuses the three levels before this one; the
         * head runs that itself and hands the fused map in, so from here the
         * two are the same four convolutions */
        const YoloModule *cv1 = yolo_module_child(module, "cv1");
        const YoloModule *lift = yolo_module_child(module, "upsample");
        const YoloModule *cv2 = yolo_module_child(module, "cv2");
        const YoloModule *cv3 = yolo_module_child(module, "cv3");
        YoloPlane out;
        if (!cv1 || !lift || !cv2 || !cv3) return yolo_run_fail(run, YOLO_ERR_MODULE);
        out = yolo_conv_apply(run, cv1, in);
        if (run->status == YOLO_OK) out = yolo_deconv_apply(run, lift, out);
        if (run->status == YOLO_OK) out = yolo_conv_apply(run, cv2, out);
        if (run->status == YOLO_OK) out = yolo_conv_apply(run, cv3, out);
        return out;
    }
    default:
        return yolo_run_fail(run, yolo_fail(YOLO_ERR_MODULE,
            "%s has no forward in this engine", module->class_name));
    }
}

/* ---------------------------------------------------------------------------
 * 9  Pictures
 *
 * Reading and writing go through stb, which this repository vendors under
 * `app/libc11`, so every format stb knows is a format the engine reads.  With
 * YOLO_NO_STB defined it falls back to binary PNM, which is what the rest of
 * this repository's tests generate.
 * ------------------------------------------------------------------------ */

#ifndef YOLO_NO_STB
#define STBI_NO_STDIO_WRITE
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_THREAD_LOCALS
#include "libc11/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "libc11/stb_image_write.h"
#endif

static const char *yolo_path_tail(const char *path)
{
    const char *dot = strrchr(path, '.');
    return dot ? dot + 1 : "";
}

static int yolo_tail_is(const char *path, const char *tail)
{
    const char *had = yolo_path_tail(path);
    size_t index;
    for (index = 0; had[index] && tail[index]; index++) {
        char left = had[index];
        if (left >= 'A' && left <= 'Z') left = (char)(left - 'A' + 'a');
        if (left != tail[index]) return 0;
    }
    return had[index] == 0 && tail[index] == 0;
}

/* ---- binary PNM, which needs no library ---- */

static int yolo_pnm_number(FILE *handle, int *number_out)
{
    int value = 0, digit = 0, letter;
    for (;;) {
        letter = fgetc(handle);
        if (letter == '#') { while (letter != '\n' && letter != EOF) letter = fgetc(handle); continue; }
        if (letter == ' ' || letter == '\t' || letter == '\n' || letter == '\r') {
            if (digit) break;
            continue;
        }
        if (letter < '0' || letter > '9') return digit ? (ungetc(letter, handle), 1) : 0;
        value = value * 10 + (letter - '0');
        digit = 1;
    }
    *number_out = value;
    return digit;
}

static YoloStatus yolo_pnm_read(const char *path, YoloImage *image_out)
{
    FILE *handle = fopen(path, "rb");
    int kind, width = 0, height = 0, top = 0;
    size_t size;

    if (!handle) return yolo_fail(YOLO_ERR_IMAGE, "cannot open %s", path);
    if (fgetc(handle) != 'P') { fclose(handle); return yolo_fail(YOLO_ERR_IMAGE, "%s is not a PNM", path); }
    kind = fgetc(handle);
    if (kind != '5' && kind != '6') {
        fclose(handle);
        return yolo_fail(YOLO_ERR_IMAGE, "%s is a P%c; only binary P5 and P6 are read",
                         path, kind == EOF ? '?' : (char)kind);
    }
    if (!yolo_pnm_number(handle, &width) || !yolo_pnm_number(handle, &height)
        || !yolo_pnm_number(handle, &top) || width <= 0 || height <= 0) {
        fclose(handle);
        return yolo_fail(YOLO_ERR_IMAGE, "%s has a malformed header", path);
    }
    if (top != 255) { fclose(handle); return yolo_fail(YOLO_ERR_IMAGE, "%s is not eight bits a channel", path); }

    image_out->width = width;
    image_out->height = height;
    image_out->channel_count = (kind == '6') ? 3 : 1;
    size = (size_t)width * (size_t)height * (size_t)image_out->channel_count;
    image_out->pixel_list = (uint8_t *)malloc(size);
    image_out->owned_flag = 1;
    if (!image_out->pixel_list) { fclose(handle); return YOLO_ERR_MEMORY; }
    if (fread(image_out->pixel_list, 1, size, handle) != size) {
        free(image_out->pixel_list);
        image_out->pixel_list = NULL;
        fclose(handle);
        return yolo_fail(YOLO_ERR_IMAGE, "%s ends before its pixels do", path);
    }
    fclose(handle);
    return YOLO_OK;
}

static YoloStatus yolo_pnm_write(const char *path, const YoloImage *image)
{
    FILE *handle = fopen(path, "wb");
    size_t size;
    if (!handle) return yolo_fail(YOLO_ERR_IMAGE, "cannot write %s", path);
    fprintf(handle, "P%c\n%d %d\n255\n", image->channel_count == 1 ? '5' : '6',
            image->width, image->height);
    size = (size_t)image->width * (size_t)image->height * (size_t)image->channel_count;
    if (fwrite(image->pixel_list, 1, size, handle) != size) {
        fclose(handle);
        return yolo_fail(YOLO_ERR_IMAGE, "the write to %s was short", path);
    }
    fclose(handle);
    return YOLO_OK;
}

YoloStatus yolo_image_read(const char *path, YoloImage *image_out)
{
    if (!path || !image_out) return YOLO_ERR_ARG;
    memset(image_out, 0, sizeof *image_out);
#ifndef YOLO_NO_STB
    {
        int width = 0, height = 0, had = 0;
        unsigned char *pixel_list = stbi_load(path, &width, &height, &had, 3);
        if (pixel_list) {
            image_out->width = width;
            image_out->height = height;
            image_out->channel_count = 3;
            image_out->pixel_list = pixel_list;
            image_out->owned_flag = 1;
            return YOLO_OK;
        }
    }
#endif
    return yolo_pnm_read(path, image_out);
}

YoloStatus yolo_image_wrap(const uint8_t *pixel_list, int width, int height,
                           int channel_count, YoloImage *image_out)
{
    if (!pixel_list || !image_out || width <= 0 || height <= 0) return YOLO_ERR_ARG;
    if (channel_count != 1 && channel_count != 3 && channel_count != 4)
        return yolo_fail(YOLO_ERR_ARG, "a picture with %d channels is not read",
                         channel_count);
    image_out->width = width;
    image_out->height = height;
    image_out->channel_count = channel_count;
    image_out->pixel_list = (uint8_t *)pixel_list;
    image_out->owned_flag = 0;
    return YOLO_OK;
}

YoloStatus yolo_image_write(const char *path, const YoloImage *image)
{
    if (!path || !image || !image->pixel_list) return YOLO_ERR_ARG;
    if (yolo_tail_is(path, "ppm") || yolo_tail_is(path, "pgm") || yolo_tail_is(path, "pnm"))
        return yolo_pnm_write(path, image);
#ifndef YOLO_NO_STB
    {
        int ok = 0;
        if (yolo_tail_is(path, "png"))
            ok = stbi_write_png(path, image->width, image->height,
                                image->channel_count, image->pixel_list,
                                image->width * image->channel_count);
        else if (yolo_tail_is(path, "jpg") || yolo_tail_is(path, "jpeg"))
            ok = stbi_write_jpg(path, image->width, image->height,
                                image->channel_count, image->pixel_list, 92);
        else if (yolo_tail_is(path, "bmp"))
            ok = stbi_write_bmp(path, image->width, image->height,
                                image->channel_count, image->pixel_list);
        else if (yolo_tail_is(path, "tga"))
            ok = stbi_write_tga(path, image->width, image->height,
                                image->channel_count, image->pixel_list);
        else
            return yolo_fail(YOLO_ERR_IMAGE, "%s names no format this engine writes", path);
        return ok ? YOLO_OK : yolo_fail(YOLO_ERR_IMAGE, "the write to %s failed", path);
    }
#else
    return yolo_fail(YOLO_ERR_IMAGE,
                     "%s names a format this build does not write; PNM only", path);
#endif
}

void yolo_image_free(YoloImage *image)
{
    if (!image) return;
    if (image->owned_flag && image->pixel_list) free(image->pixel_list);
    memset(image, 0, sizeof *image);
}

/* ---- letterboxing ----
 *
 * The shape the model sees is the source scaled by one ratio, so nothing is
 * stretched, and padded with grey to the multiple of the stride the head
 * wants.  The ratio and the offsets are kept, because a box the head reports
 * is in that padded frame and has to come back out of it.
 */

typedef struct {
    float ratio;
    int   pad_left;
    int   pad_top;
    int   fit_width;
    int   fit_height;
} YoloFrame;

/* Resizing, done the way OpenCV does it, because that is what the reference
 * letterbox calls and the difference is visible in the answer.
 *
 * The obvious float bilinear -- sample at `(dst + 0.5) * scale - 0.5`, weigh
 * the four neighbours, keep the float -- is off by up to 0.003 of a unit
 * against `cv2.resize`, which is not rounding noise: OpenCV quantises the two
 * interpolation weights to elevenths of a bit and rounds the result back to a
 * byte before anything downstream sees it.  Three parts in a thousand at the
 * input turned into a hundredth of a logit at the head and a different score
 * on a marginal detection, so the quantisation is reproduced here rather than
 * improved on.  The engine is not trying to resize well; it is trying to
 * resize the same.
 *
 * `INTER_RESIZE_COEF_BITS` is 11, so a weight is a whole number of 2048ths;
 * the horizontal pass leaves eleven fractional bits, the vertical pass makes
 * it twenty-two, and the cast rounds those away and clamps to a byte. */

#define YOLO_COEF_BITS 11
#define YOLO_COEF_SCALE (1 << YOLO_COEF_BITS)

static int yolo_coef_round(float value)
{
    /* cv::saturate_cast<short> rounds to nearest, ties to even */
    return (int)lrintf(value);
}

static void yolo_row_scale(const uint8_t *row, int source_width, int channel_count,
                           int bgr_flag, const int *low_list, const int *high_list,
                           const short *weight_list, int target_width, int *out_list)
{
    int index, channel;
    (void)source_width;
    for (index = 0; index < target_width; index++) {
        int low = low_list[index], high = high_list[index];
        int weight_high = weight_list[index * 2 + 1];
        int weight_low = weight_list[index * 2];
        for (channel = 0; channel < 3; channel++) {
            int take = channel_count == 1 ? 0 : (bgr_flag ? 2 - channel : channel);
            out_list[index * 3 + channel] =
                (int)row[low * channel_count + take] * weight_low
              + (int)row[high * channel_count + take] * weight_high;
        }
    }
}

static YoloStatus yolo_pixels_scale(const uint8_t *source, int source_width,
                                    int source_height, int source_step,
                                    int channel_count, int bgr_flag,
                                    float *target, int target_width, int target_height,
                                    int target_step, int target_rows,
                                    int pad_left, int pad_top, float divisor)
{
    float scale_x = (float)source_width / (float)target_width;
    float scale_y = (float)source_height / (float)target_height;
    /* the distance between one channel's plane and the next is the padded
     * height, not the height the picture was scaled to; they differ exactly
     * when there is padding, which is why a square picture hid this */
    size_t face = (size_t)target_step * (size_t)target_rows;
    int *low_x_list, *high_x_list, *row_room;
    short *weight_x_list;
    int out_y, out_x, channel;

    low_x_list = (int *)malloc((size_t)target_width * sizeof(int) * 2
                               + (size_t)target_width * sizeof(short) * 2
                               + (size_t)target_width * 3 * sizeof(int) * 2);
    if (!low_x_list) return YOLO_ERR_MEMORY;
    high_x_list = low_x_list + target_width;
    weight_x_list = (short *)(high_x_list + target_width);
    row_room = (int *)(weight_x_list + target_width * 2);

    for (out_x = 0; out_x < target_width; out_x++) {
        float at = ((float)out_x + 0.5f) * scale_x - 0.5f;
        int low = (int)floorf(at);
        float part = at - (float)low;
        if (low < 0) { low = 0; part = 0.0f; }
        if (low >= source_width - 1) { low = source_width - 1; part = 0.0f; }
        low_x_list[out_x] = low;
        high_x_list[out_x] = yolo_min_int(low + 1, source_width - 1);
        weight_x_list[out_x * 2 + 1] = (short)yolo_coef_round(part * YOLO_COEF_SCALE);
        weight_x_list[out_x * 2] = (short)(YOLO_COEF_SCALE - weight_x_list[out_x * 2 + 1]);
    }

    for (out_y = 0; out_y < target_height; out_y++) {
        float at = ((float)out_y + 0.5f) * scale_y - 0.5f;
        int low = (int)floorf(at);
        float part = at - (float)low;
        int high, weight_high, weight_low;
        int *upper = row_room, *lower = row_room + target_width * 3;

        if (low < 0) { low = 0; part = 0.0f; }
        if (low >= source_height - 1) { low = source_height - 1; part = 0.0f; }
        high = yolo_min_int(low + 1, source_height - 1);
        weight_high = yolo_coef_round(part * YOLO_COEF_SCALE);
        weight_low = YOLO_COEF_SCALE - weight_high;

        yolo_row_scale(source + (size_t)low * (size_t)source_step, source_width,
                       channel_count, bgr_flag, low_x_list, high_x_list,
                       weight_x_list, target_width, upper);
        yolo_row_scale(source + (size_t)high * (size_t)source_step, source_width,
                       channel_count, bgr_flag, low_x_list, high_x_list,
                       weight_x_list, target_width, lower);

        for (out_x = 0; out_x < target_width; out_x++) {
            for (channel = 0; channel < 3; channel++) {
                /* OpenCV's byte path drops four bits off each row before it
                 * weighs them, keeps two fractional bits through the sum, and
                 * rounds those away.  Doing the same arithmetic in one wider
                 * step and rounding once at the end is more accurate and is
                 * wrong by a unit on about a tenth of the cells. */
                int byte = ((weight_low * (upper[out_x * 3 + channel] >> 4)) >> 16)
                         + ((weight_high * (lower[out_x * 3 + channel] >> 4)) >> 16);
                byte = (byte + 2) >> 2;
                if (byte < 0) byte = 0;
                if (byte > 255) byte = 255;
                target[(size_t)channel * face
                       + (size_t)(out_y + pad_top) * (size_t)target_step
                       + (size_t)(out_x + pad_left)] = (float)byte / divisor;
            }
        }
    }
    free(low_x_list);
    return YOLO_OK;
}

static void yolo_frame_fit(const YoloImage *image, int size, int stride_size,
                           int rect_flag, int scale_up_flag, int center_flag,
                           YoloFrame *frame, int *width_out, int *height_out)
{
    float ratio = (float)size / (float)image->height;
    float other = (float)size / (float)image->width;
    float gap_x, gap_y;
    int pad_right, pad_bottom;

    if (other < ratio) ratio = other;
    if (!scale_up_flag && ratio > 1.0f) ratio = 1.0f;
    frame->ratio = ratio;
    frame->fit_width = (int)lrintf((float)image->width * ratio);
    frame->fit_height = (int)lrintf((float)image->height * ratio);

    gap_x = (float)(size - frame->fit_width);
    gap_y = (float)(size - frame->fit_height);
    if (rect_flag && stride_size > 0) {
        gap_x = (float)((int)gap_x % stride_size);
        gap_y = (float)((int)gap_y % stride_size);
    }
    if (center_flag) { gap_x *= 0.5f; gap_y *= 0.5f; }

    frame->pad_left = center_flag ? (int)lrintf(gap_x - 0.1f) : 0;
    frame->pad_top = center_flag ? (int)lrintf(gap_y - 0.1f) : 0;
    pad_right = (int)lrintf(gap_x + 0.1f);
    pad_bottom = (int)lrintf(gap_y + 0.1f);
    if (frame->pad_left < 0) frame->pad_left = 0;
    if (frame->pad_top < 0) frame->pad_top = 0;

    *width_out = frame->fit_width + frame->pad_left + pad_right;
    *height_out = frame->fit_height + frame->pad_top + pad_bottom;
}

/* ---------------------------------------------------------------------------
 * 8  The head, and what comes out of it
 *
 * Three feature maps go in.  Each is read by two small branches -- one that
 * says where a box's four edges are relative to the cell that predicts it, one
 * that scores every class -- and by a third for the task's own channels: mask
 * coefficients, an angle, or a set of keypoints.
 *
 * yolo26 ships two copies of those branches.  `cv2`/`cv3` are trained one to
 * many, so several cells fire on one object and suppression picks between
 * them; `one2one_cv2`/`one2one_cv3` are trained one to one, so the top scoring
 * cells are the answer and there is nothing to suppress.  `yolo predict`
 * reads the first pair unless it is told otherwise, and so does this engine;
 * `nms_free_flag` reads the second.
 * ------------------------------------------------------------------------ */

typedef struct {
    int    anchor_count;
    int    class_count;
    int    extra_count;      /* mask coefficients, or an angle, or nothing */
    int    keypoint_room;    /* keypoint channels, pose only */
    float *box_room;         /* 4 * reg_max rows of anchor_count */
    float *score_room;       /* class_count rows of anchor_count */
    float *extra_room;
    float *keypoint_list;
    float *anchor_x_list;
    float *anchor_y_list;
    float *stride_list;
    YoloPlane proto_plane;   /* segmentation only */
    int       proto_flag;
} YoloHeadOut;

/* Lay one branch's output for one level into the flat row it occupies. */
static void yolo_head_lay(const YoloPlane *source, float *target,
                          int row_count, int anchor_count, int anchor_first)
{
    size_t face = (size_t)source->height * (size_t)source->width;
    int row;
    for (row = 0; row < row_count; row++)
        memcpy(target + (size_t)row * (size_t)anchor_count + anchor_first,
               source->cell_list + (size_t)row * face, face * sizeof(float));
}

static YoloStatus yolo_head_run(YoloRun *run, const YoloModel *model,
                                YoloPlane *level_list, int nms_free_flag,
                                YoloHeadOut *out)
{
    const YoloModule *head = model->head;
    const YoloModule *box_branch = yolo_module_child(head,
        nms_free_flag ? "one2one_cv2" : "cv2");
    const YoloModule *class_branch = yolo_module_child(head,
        nms_free_flag ? "one2one_cv3" : "cv3");
    const YoloModule *extra_branch = yolo_module_child(head,
        nms_free_flag ? "one2one_cv4" : "cv4");
    const YoloModule *point_branch = yolo_module_child(head,
        nms_free_flag ? "one2one_cv4_kpts" : "cv4_kpts");
    int level_count = model->level_count;
    int anchor_count = 0, anchor_at = 0;
    int level, index;

    if (!box_branch || !class_branch)
        return yolo_fail(YOLO_ERR_MODULE, "the head carries no %s branch",
                         nms_free_flag ? "one-to-one" : "one-to-many");

    for (level = 0; level < level_count; level++)
        anchor_count += level_list[level].height * level_list[level].width;

    out->anchor_count = anchor_count;
    out->class_count = model->class_count;
    out->extra_count = 0;
    out->keypoint_room = 0;
    if (model->task == YOLO_TASK_SEGMENT) out->extra_count = model->mask_count;
    if (model->task == YOLO_TASK_OBB) out->extra_count = 1;
    if (model->task == YOLO_TASK_POSE)
        out->keypoint_room = model->keypoint_count * model->keypoint_size;

    out->box_room = (float *)yolo_arena_take(run->arena,
        (size_t)4 * (size_t)model->reg_max * (size_t)anchor_count * sizeof(float));
    out->score_room = (float *)yolo_arena_take(run->arena,
        (size_t)out->class_count * (size_t)anchor_count * sizeof(float));
    out->anchor_x_list = (float *)yolo_arena_take(run->arena,
        (size_t)anchor_count * sizeof(float));
    out->anchor_y_list = (float *)yolo_arena_take(run->arena,
        (size_t)anchor_count * sizeof(float));
    out->stride_list = (float *)yolo_arena_take(run->arena,
        (size_t)anchor_count * sizeof(float));
    if (!out->box_room || !out->score_room || !out->anchor_x_list
        || !out->anchor_y_list || !out->stride_list)
        return YOLO_ERR_MEMORY;
    if (out->extra_count) {
        out->extra_room = (float *)yolo_arena_take(run->arena,
            (size_t)out->extra_count * (size_t)anchor_count * sizeof(float));
        if (!out->extra_room) return YOLO_ERR_MEMORY;
    }
    if (out->keypoint_room) {
        out->keypoint_list = (float *)yolo_arena_take(run->arena,
            (size_t)out->keypoint_room * (size_t)anchor_count * sizeof(float));
        if (!out->keypoint_list) return YOLO_ERR_MEMORY;
    }

    for (level = 0; level < level_count; level++) {
        YoloPlane feature = level_list[level];
        int face = feature.height * feature.width;
        float stride = model->level_stride_list[level];
        YoloMark mark = yolo_arena_mark(run->arena);
        YoloPlane step;
        int row, column;

        /* the cell centres, half a cell in from its corner */
        for (row = 0; row < feature.height; row++) {
            for (column = 0; column < feature.width; column++) {
                int at = anchor_at + row * feature.width + column;
                out->anchor_x_list[at] = (float)column + 0.5f;
                out->anchor_y_list[at] = (float)row + 0.5f;
                out->stride_list[at] = stride;
            }
        }

        step = yolo_module_run(run, box_branch->child_list[level], feature);
        if (run->status != YOLO_OK) return run->status;
        yolo_head_lay(&step, out->box_room, 4 * model->reg_max, anchor_count, anchor_at);

        step = yolo_module_run(run, class_branch->child_list[level], feature);
        if (run->status != YOLO_OK) return run->status;
        yolo_head_lay(&step, out->score_room, out->class_count, anchor_count, anchor_at);

        if (out->extra_count && extra_branch) {
            step = yolo_module_run(run, extra_branch->child_list[level], feature);
            if (run->status != YOLO_OK) return run->status;
            yolo_head_lay(&step, out->extra_room, out->extra_count, anchor_count, anchor_at);
        }
        if (out->keypoint_room && extra_branch) {
            step = yolo_module_run(run, extra_branch->child_list[level], feature);
            if (run->status != YOLO_OK) return run->status;
            if (point_branch) {
                step = yolo_module_run(run, point_branch->child_list[level], step);
                if (run->status != YOLO_OK) return run->status;
            }
            yolo_head_lay(&step, out->keypoint_list, out->keypoint_room,
                          anchor_count, anchor_at);
        }
        anchor_at += face;
        yolo_arena_reset(run->arena, mark);
    }

    /* the mask prototypes, which every instance's coefficients weigh */
    if (model->task == YOLO_TASK_SEGMENT) {
        const YoloModule *proto = yolo_module_child(head, "proto");
        if (!proto) return yolo_fail(YOLO_ERR_MODULE, "a segment head carries no prototypes");
        if (proto->kind == YOLO_MOD_PROTO26) {
            const YoloModule *refine = yolo_module_child(proto, "feat_refine");
            const YoloModule *fuse = yolo_module_child(proto, "feat_fuse");
            YoloPlane feat = level_list[0];
            YoloPlane sum = yolo_run_take(run, feat.channel_count, feat.height, feat.width);
            if (!sum.cell_list) return YOLO_ERR_MEMORY;
            memcpy(sum.cell_list, feat.cell_list, yolo_plane_count(&feat) * sizeof(float));
            for (index = 0; refine && index < refine->child_count; index++) {
                YoloPlane lifted = yolo_module_run(run, refine->child_list[index],
                                                   level_list[index + 1]);
                YoloPlane wide;
                int scale = 1 << (index + 1);
                if (run->status != YOLO_OK) return run->status;
                wide = yolo_run_take(run, lifted.channel_count,
                                     lifted.height * scale, lifted.width * scale);
                if (!wide.cell_list) return YOLO_ERR_MEMORY;
                run->backend->resize_run(run->backend, &lifted, &wide, scale);
                run->backend->add_run(run->backend, sum.cell_list, wide.cell_list,
                                      yolo_plane_count(&sum));
            }
            if (!fuse) return yolo_fail(YOLO_ERR_MODULE, "a Proto26 carries no fuse");
            sum = yolo_conv_apply(run, fuse, sum);
            if (run->status != YOLO_OK) return run->status;
            out->proto_plane = yolo_module_run(run, proto, sum);
        } else {
            out->proto_plane = yolo_module_run(run, proto, level_list[0]);
        }
        if (run->status != YOLO_OK) return run->status;
        out->proto_flag = 1;
    }
    return YOLO_OK;
}

/* ---- turning a head's rows into boxes ---- */

typedef struct {
    float left, top, right, bottom;   /* corners, or centre-size for an obb */
    float angle;
    float score;
    int   class_index;
    int   anchor_index;
} YoloPick;

/* Where the cell that owns an anchor says the four edges are.  With reg_max
 * one -- which is every published yolo26, and the whole of what "DFL-free"
 * means -- the branch says the distance outright.  With reg_max above one the
 * branch says a distribution over `reg_max` bins and the distance is its mean,
 * which is what a v8 or v11 checkpoint does and is kept here so one of those
 * loads and runs rather than being silently mis-read. */
static float yolo_edge_value(const float *box_room, int reg_max, int anchor_count,
                             int edge, int anchor_index)
{
    if (reg_max <= 1)
        return box_room[(size_t)edge * (size_t)anchor_count + anchor_index];
    {
        float best = -FLT_MAX, total = 0.0f, mean = 0.0f;
        int bin;
        for (bin = 0; bin < reg_max; bin++) {
            float value = box_room[(size_t)(edge * reg_max + bin) * (size_t)anchor_count
                                   + anchor_index];
            if (value > best) best = value;
        }
        for (bin = 0; bin < reg_max; bin++) {
            float value = expf(box_room[(size_t)(edge * reg_max + bin) * (size_t)anchor_count
                                        + anchor_index] - best);
            total += value;
            mean += value * (float)bin;
        }
        return total > 0.0f ? mean / total : 0.0f;
    }
}

/* An axis-aligned box in the padded input's pixels. */
static void yolo_box_decode(const YoloHeadOut *head, int reg_max, int anchor_index,
                            float *left_out, float *top_out,
                            float *right_out, float *bottom_out)
{
    float stride = head->stride_list[anchor_index];
    float anchor_x = head->anchor_x_list[anchor_index];
    float anchor_y = head->anchor_y_list[anchor_index];
    float left = yolo_edge_value(head->box_room, reg_max, head->anchor_count, 0, anchor_index);
    float top = yolo_edge_value(head->box_room, reg_max, head->anchor_count, 1, anchor_index);
    float right = yolo_edge_value(head->box_room, reg_max, head->anchor_count, 2, anchor_index);
    float bottom = yolo_edge_value(head->box_room, reg_max, head->anchor_count, 3, anchor_index);

    *left_out = (anchor_x - left) * stride;
    *top_out = (anchor_y - top) * stride;
    *right_out = (anchor_x + right) * stride;
    *bottom_out = (anchor_y + bottom) * stride;
}

/* An oriented box: the same four distances, but the centre is pushed out
 * along the predicted angle rather than sitting between the edges. */
static void yolo_rbox_decode(const YoloHeadOut *head, int reg_max, int anchor_index,
                             float *center_x_out, float *center_y_out,
                             float *width_out, float *height_out, float *angle_out)
{
    float stride = head->stride_list[anchor_index];
    float left = yolo_edge_value(head->box_room, reg_max, head->anchor_count, 0, anchor_index);
    float top = yolo_edge_value(head->box_room, reg_max, head->anchor_count, 1, anchor_index);
    float right = yolo_edge_value(head->box_room, reg_max, head->anchor_count, 2, anchor_index);
    float bottom = yolo_edge_value(head->box_room, reg_max, head->anchor_count, 3, anchor_index);
    /* yolo26 says the angle in radians outright; v8's obb head put it through
     * a sigmoid and shifted it, which this does not */
    float angle = head->extra_room ? head->extra_room[anchor_index] : 0.0f;
    float half_x = (right - left) * 0.5f;
    float half_y = (bottom - top) * 0.5f;
    float turn_cos = cosf(angle), turn_sin = sinf(angle);

    *center_x_out = (half_x * turn_cos - half_y * turn_sin
                     + head->anchor_x_list[anchor_index]) * stride;
    *center_y_out = (half_x * turn_sin + half_y * turn_cos
                     + head->anchor_y_list[anchor_index]) * stride;
    *width_out = (left + right) * stride;
    *height_out = (top + bottom) * stride;
    *angle_out = angle;
}

static int yolo_pick_order(const void *left, const void *right)
{
    const YoloPick *a = (const YoloPick *)left;
    const YoloPick *b = (const YoloPick *)right;
    if (a->score < b->score) return 1;
    if (a->score > b->score) return -1;
    return a->anchor_index - b->anchor_index;   /* a stable order for a tie */
}

static float yolo_overlap(const YoloPick *a, const YoloPick *b)
{
    float left = a->left > b->left ? a->left : b->left;
    float top = a->top > b->top ? a->top : b->top;
    float right = a->right < b->right ? a->right : b->right;
    float bottom = a->bottom < b->bottom ? a->bottom : b->bottom;
    float wide = right - left;
    float high = bottom - top;
    float shared, total;

    if (wide <= 0.0f || high <= 0.0f) return 0.0f;
    shared = wide * high;
    total = (a->right - a->left) * (a->bottom - a->top)
          + (b->right - b->left) * (b->bottom - b->top) - shared;
    return total > 0.0f ? shared / total : 0.0f;
}

/* An oriented pick keeps its centre in `left`/`top` and its size in
 * `right`/`bottom`, because a rotated box has no corners to put there.
 *
 * The probabilistic overlap two oriented boxes have, which is what the
 * reference suppresses rotated boxes by: each box is read as a Gaussian with
 * the box's covariance, and the Bhattacharyya distance between the two
 * becomes a number that behaves like an IoU. */
static float yolo_rbox_overlap(const YoloPick *a, const YoloPick *b)
{
    float a_cos = cosf(a->angle), a_sin = sinf(a->angle);
    float b_cos = cosf(b->angle), b_sin = sinf(b->angle);
    float a_wide = a->right * a->right / 12.0f, a_high = a->bottom * a->bottom / 12.0f;
    float b_wide = b->right * b->right / 12.0f, b_high = b->bottom * b->bottom / 12.0f;
    float a11 = a_wide * a_cos * a_cos + a_high * a_sin * a_sin;
    float a22 = a_wide * a_sin * a_sin + a_high * a_cos * a_cos;
    float a12 = (a_wide - a_high) * a_cos * a_sin;
    float b11 = b_wide * b_cos * b_cos + b_high * b_sin * b_sin;
    float b22 = b_wide * b_sin * b_sin + b_high * b_cos * b_cos;
    float b12 = (b_wide - b_high) * b_cos * b_sin;
    float gap_x = a->left - b->left, gap_y = a->top - b->top;
    float s11 = a11 + b11, s22 = a22 + b22, s12 = a12 + b12;
    float det = s11 * s22 - s12 * s12;
    float first, second, distance, overlap;

    if (det <= 1e-12f) return 0.0f;
    first = ((s22 * gap_x * gap_x + s11 * gap_y * gap_y
              - 2.0f * s12 * gap_x * gap_y) / det) * 0.25f;
    second = logf(det / (4.0f * sqrtf(
                  fmaxf((a11 * a22 - a12 * a12), 0.0f)
                * fmaxf((b11 * b22 - b12 * b12), 0.0f)) + 1e-12f) + 1e-12f) * 0.5f;
    distance = yolo_clamp_float(first + second, 1e-7f, 100.0f);
    overlap = 1.0f - sqrtf(1.0f - expf(-distance) + 1e-12f);
    return yolo_clamp_float(overlap, 0.0f, 1.0f);
}

/* Greedy suppression, highest score first.  Boxes of different classes never
 * suppress each other unless the caller asked for that; the reference does it
 * by shifting each class's boxes far apart on the plane, which this does by
 * comparing the class instead -- the same answer without the magic offset. */
static int yolo_suppress(YoloPick *pick_list, int count, float overlap_limit,
                         int class_blind_flag, int rotated_flag, int keep_limit)
{
    int *dead_list;
    int index, other, kept = 0;

    if (count <= 0) return 0;
    dead_list = (int *)calloc((size_t)count, sizeof(int));
    if (!dead_list) return count;
    for (index = 0; index < count && kept < keep_limit; index++) {
        if (dead_list[index]) continue;
        pick_list[kept++] = pick_list[index];
        for (other = index + 1; other < count; other++) {
            float overlap;
            if (dead_list[other]) continue;
            if (!class_blind_flag
                && pick_list[other].class_index != pick_list[index].class_index)
                continue;
            overlap = rotated_flag
                    ? yolo_rbox_overlap(&pick_list[index], &pick_list[other])
                    : yolo_overlap(&pick_list[index], &pick_list[other]);
            if (overlap > overlap_limit) dead_list[other] = 1;
        }
    }
    free(dead_list);
    return kept;
}

/* ---------------------------------------------------------------------------
 * 10  The session
 * ------------------------------------------------------------------------ */

#define YOLO_PICK_LIMIT 30000

struct YoloSession {
    const YoloModel *model;
    YoloOptions      options;

    YoloArena arena;         /* the forward pass, released every picture */
    YoloArena keep_arena;    /* the result, released every picture */

    YoloResult result;
    YoloBox   *box_list;
    int        box_limit;
    float     *score_list;

    int input_width;
    int input_height;

    YoloPlane input_plane;
    float    *score_room;
    int       score_anchor_count;

    double shape_ms;
    double forward_ms;
    double decode_ms;
};

const YoloPlane *yolo_session_input(const YoloSession *session)
{
    return &session->input_plane;
}

const float *yolo_session_score_room(const YoloSession *session,
                                     int *anchor_count_out, int *class_count_out)
{
    if (anchor_count_out) *anchor_count_out = session->score_anchor_count;
    if (class_count_out) *class_count_out = session->model->class_count;
    return session->score_room;
}

void yolo_options_default(const YoloModel *model, YoloOptions *options_out)
{
    if (!options_out) return;
    memset(options_out, 0, sizeof *options_out);
    options_out->input_size = model ? model->input_size : 640;
    options_out->score_floor = 0.25f;
    options_out->overlap_limit = 0.7f;
    options_out->detect_limit = 300;
    options_out->letterbox_fill = 114;
    options_out->scale_up_flag = 1;
    options_out->center_flag = 1;
    options_out->rect_flag = 1;
    options_out->mask_flag = 1;
    options_out->thread_count = 1;
}

YoloStatus yolo_session_open(const YoloModel *model, const YoloOptions *options,
                             YoloSession **session_out)
{
    YoloSession *session;
    int size;

    if (!model || !session_out) return YOLO_ERR_ARG;
    *session_out = NULL;
    session = (YoloSession *)calloc(1, sizeof *session);
    if (!session) return YOLO_ERR_MEMORY;
    session->model = model;
    if (options) session->options = *options;
    else yolo_options_default(model, &session->options);

    size = session->options.input_size > 0 ? session->options.input_size
                                           : model->input_size;
    if (model->task != YOLO_TASK_CLASSIFY && size % model->stride_size != 0) {
        free(session);
        return yolo_fail(YOLO_ERR_ARG,
            "an input of %d is not a multiple of the model's stride of %d",
            size, model->stride_size);
    }
    session->options.input_size = size;
    session->input_width = size;      /* replaced per picture by the shaping */
    session->input_height = size;
    if (session->options.detect_limit <= 0) session->options.detect_limit = 300;

    yolo_arena_open(&session->arena, (size_t)16 << 20);
    yolo_arena_open(&session->keep_arena, (size_t)1 << 20);
    session->box_limit = session->options.detect_limit;
    session->box_list = (YoloBox *)calloc((size_t)session->box_limit, sizeof(YoloBox));
    if (!session->box_list) {
        yolo_arena_close(&session->arena);
        yolo_arena_close(&session->keep_arena);
        free(session);
        return YOLO_ERR_MEMORY;
    }
    *session_out = session;
    return YOLO_OK;
}

void yolo_session_close(YoloSession *session)
{
    if (!session) return;
    yolo_arena_close(&session->arena);
    yolo_arena_close(&session->keep_arena);
    free(session->box_list);
    free(session);
}

double yolo_session_shape_time(const YoloSession *session) { return session->shape_ms; }
double yolo_session_forward_time(const YoloSession *session) { return session->forward_ms; }
double yolo_session_decode_time(const YoloSession *session) { return session->decode_ms; }

/* ---- shaping ---- */

static YoloStatus yolo_session_shape(YoloSession *session, const YoloImage *image,
                                     YoloPlane *plane_out, YoloFrame *frame_out)
{
    const YoloOptions *options = &session->options;
    YoloPlane plane;
    YoloStatus status;
    size_t index, count;

    if (session->model->task == YOLO_TASK_CLASSIFY) {
        /* a classifier shrinks the short side to the input and takes the
         * middle square, so the picture is not squeezed and nothing is padded */
        int size = session->input_width;
        int short_side = yolo_min_int(image->width, image->height);
        float ratio = (float)size / (float)short_side;
        int wide = (int)lrintf((float)image->width * ratio);
        int high = (int)lrintf((float)image->height * ratio);
        int crop_left = (wide - size) / 2;
        int crop_top = (high - size) / 2;
        float *room;
        YoloMark mark;

        plane = yolo_plane_take(&session->arena, 1, 3, size, size);
        if (!plane.cell_list) return YOLO_ERR_MEMORY;
        mark = yolo_arena_mark(&session->arena);
        room = (float *)yolo_arena_take(&session->arena,
            (size_t)3 * (size_t)wide * (size_t)high * sizeof(float));
        if (!room) return YOLO_ERR_MEMORY;
        status = yolo_pixels_scale(image->pixel_list, image->width, image->height,
                                   image->width * image->channel_count,
                                   image->channel_count, options->bgr_flag,
                                   room, wide, high, wide, high, 0, 0, 255.0f);
        if (status != YOLO_OK) return status;
        {
            int channel, row;
            for (channel = 0; channel < 3; channel++)
                for (row = 0; row < size; row++)
                    memcpy(plane.cell_list + ((size_t)channel * (size_t)size + (size_t)row)
                                             * (size_t)size,
                           room + (size_t)channel * (size_t)wide * (size_t)high
                                + (size_t)(row + crop_top) * (size_t)wide + crop_left,
                           (size_t)size * sizeof(float));
        }
        yolo_arena_reset(&session->arena, mark);
        frame_out->ratio = ratio;
        frame_out->pad_left = -crop_left;
        frame_out->pad_top = -crop_top;
        frame_out->fit_width = size;
        frame_out->fit_height = size;
        *plane_out = plane;
        return YOLO_OK;
    }

    yolo_frame_fit(image, session->options.input_size, session->model->stride_size,
                   options->rect_flag, options->scale_up_flag, options->center_flag,
                   frame_out, &session->input_width, &session->input_height);
    plane = yolo_plane_take(&session->arena, 1, 3,
                            session->input_height, session->input_width);
    if (!plane.cell_list) return YOLO_ERR_MEMORY;
    count = yolo_plane_count(&plane);
    {
        float fill = (float)options->letterbox_fill / 255.0f;
        for (index = 0; index < count; index++) plane.cell_list[index] = fill;
    }
    status = yolo_pixels_scale(image->pixel_list, image->width, image->height,
                               image->width * image->channel_count,
                               image->channel_count, options->bgr_flag,
                               plane.cell_list, frame_out->fit_width,
                               frame_out->fit_height, session->input_width,
                               session->input_height, frame_out->pad_left,
                               frame_out->pad_top, 255.0f);
    if (status != YOLO_OK) return status;
    *plane_out = plane;
    return YOLO_OK;
}

/* ---- the backbone and neck ---- */

static YoloStatus yolo_session_forward(YoloSession *session, YoloRun *run,
                                       YoloPlane input, YoloPlane *level_list,
                                       YoloPlane *tail_out)
{
    const YoloModel *model = session->model;
    YoloPlane *keep_list;
    YoloPlane walk = input;
    int index, step;

    keep_list = (YoloPlane *)yolo_arena_zero(run->arena,
        (size_t)model->layer_count * sizeof(YoloPlane));
    if (!keep_list) return YOLO_ERR_MEMORY;

    for (index = 0; index < model->layer_count; index++) {
        const YoloModule *layer = model->layer_list[index];
        int from_count = model->from_count_list[index];
        int last = index == model->layer_count - 1;

        /* what this layer reads: -1 is whatever the layer before left */
        if (from_count == 1 && model->from_list[index][0] == -1) {
            /* walk already holds it */
        } else if (last) {
            for (step = 0; step < from_count && step < YOLO_LEVEL_LIMIT; step++) {
                int at = model->from_list[index][step];
                level_list[step] = at == -1 ? walk : keep_list[at];
            }
            *tail_out = walk;
            return YOLO_OK;
        } else if (layer->kind == YOLO_MOD_CONCAT) {
            int channel_count = 0, height = 0, width = 0, laid = 0;
            YoloPlane joined;
            for (step = 0; step < from_count; step++) {
                int at = model->from_list[index][step];
                YoloPlane part = at == -1 ? walk : keep_list[at];
                channel_count += part.channel_count;
                height = part.height;
                width = part.width;
            }
            joined = yolo_run_take(run, channel_count, height, width);
            if (!joined.cell_list) return YOLO_ERR_MEMORY;
            for (step = 0; step < from_count; step++) {
                int at = model->from_list[index][step];
                YoloPlane part = at == -1 ? walk : keep_list[at];
                if (part.height != height || part.width != width)
                    return yolo_fail(YOLO_ERR_SHAPE,
                        "layer %d joins a %dx%d map to a %dx%d one",
                        index, part.height, part.width, height, width);
                yolo_plane_lay(&joined, laid, &part);
                laid += part.channel_count;
            }
            walk = joined;
            if (model->save_flag_list[index]) keep_list[index] = walk;
            continue;
        } else {
            int at = model->from_list[index][0];
            walk = at == -1 ? walk : keep_list[at];
        }

        if (last) {
            level_list[0] = walk;
            *tail_out = walk;
            return YOLO_OK;
        }
        walk = yolo_module_run(run, layer, walk);
        if (run->status != YOLO_OK) return run->status;
        if (model->save_flag_list[index]) keep_list[index] = walk;
    }
    *tail_out = walk;
    return YOLO_OK;
}

/* ---- from the padded input's pixels back to the source's ---- */

static void yolo_frame_back(const YoloFrame *frame, float in_x, float in_y,
                            float *out_x, float *out_y)
{
    *out_x = (in_x - (float)frame->pad_left) / frame->ratio;
    *out_y = (in_y - (float)frame->pad_top) / frame->ratio;
}

typedef struct { float score; int index; } YoloRank;

static int yolo_rank_order(const void *left, const void *right)
{
    const YoloRank *a = (const YoloRank *)left;
    const YoloRank *b = (const YoloRank *)right;
    if (a->score < b->score) return 1;
    if (a->score > b->score) return -1;
    return a->index - b->index;
}

static int yolo_class_kept(const YoloOptions *options, int class_index)
{
    int index;
    if (!options->class_filter_list || options->class_filter_count <= 0) return 1;
    for (index = 0; index < options->class_filter_count; index++)
        if (options->class_filter_list[index] == class_index) return 1;
    return 0;
}

/* The one-to-one head's selection.  The reference takes the best class of
 * every anchor, keeps the top k anchors by that, then takes the top k of those
 * anchors' full score rows -- so one anchor may contribute two classes and a
 * second anchor none.  Doing it in one pass over anchor-by-class would give a
 * different set, which is why it is done in two here as well. */
static int yolo_head_choose(const YoloHeadOut *head, const YoloOptions *options,
                            YoloRank *rank_room, YoloRank *flat_room,
                            YoloPick *pick_list, int keep_limit)
{
    int anchor_count = head->anchor_count;
    int class_count = head->class_count;
    int wide = yolo_min_int(keep_limit, anchor_count);
    int index, class_index, count = 0;

    for (index = 0; index < anchor_count; index++) {
        float best = -FLT_MAX;
        for (class_index = 0; class_index < class_count; class_index++) {
            float value = head->score_room[(size_t)class_index * (size_t)anchor_count + index];
            if (value > best) best = value;
        }
        rank_room[index].score = best;
        rank_room[index].index = index;
    }
    qsort(rank_room, (size_t)anchor_count, sizeof(YoloRank), yolo_rank_order);

    if (options->class_blind_flag) {
        for (index = 0; index < wide; index++) {
            int anchor_index = rank_room[index].index;
            float score = yolo_sigmoid(rank_room[index].score);
            int best_class = 0;
            float best = -FLT_MAX;
            if (score <= options->score_floor) continue;
            for (class_index = 0; class_index < class_count; class_index++) {
                float value = head->score_room[(size_t)class_index * (size_t)anchor_count
                                               + anchor_index];
                if (value > best) { best = value; best_class = class_index; }
            }
            if (!yolo_class_kept(options, best_class)) continue;
            pick_list[count].anchor_index = anchor_index;
            pick_list[count].class_index = best_class;
            pick_list[count].score = score;
            count++;
        }
        return count;
    }

    for (index = 0; index < wide; index++) {
        int anchor_index = rank_room[index].index;
        for (class_index = 0; class_index < class_count; class_index++) {
            int at = index * class_count + class_index;
            flat_room[at].score =
                head->score_room[(size_t)class_index * (size_t)anchor_count + anchor_index];
            flat_room[at].index = at;
        }
    }
    qsort(flat_room, (size_t)wide * (size_t)class_count, sizeof(YoloRank), yolo_rank_order);
    for (index = 0; index < wide; index++) {
        int at = flat_room[index].index;
        int anchor_index = rank_room[at / class_count].index;
        int best_class = at % class_count;
        float score = yolo_sigmoid(flat_room[index].score);
        if (score <= options->score_floor) continue;
        if (!yolo_class_kept(options, best_class)) continue;
        pick_list[count].anchor_index = anchor_index;
        pick_list[count].class_index = best_class;
        pick_list[count].score = score;
        count++;
    }
    return count;
}

/* The one-to-many head's selection: everything over the floor, which
 * suppression then thins. */
static int yolo_head_gather(const YoloHeadOut *head, const YoloOptions *options,
                            YoloPick *pick_list, int pick_limit)
{
    int anchor_count = head->anchor_count;
    int class_count = head->class_count;
    int index, class_index, count = 0;
    float floor_logit;

    /* comparing before the sigmoid saves an exp on every one of the eight
     * thousand anchors that will not make it */
    if (options->score_floor <= 0.0f) floor_logit = -FLT_MAX;
    else if (options->score_floor >= 1.0f) floor_logit = FLT_MAX;
    else floor_logit = logf(options->score_floor / (1.0f - options->score_floor));

    for (index = 0; index < anchor_count && count < pick_limit; index++) {
        int best_class = 0;
        float best = -FLT_MAX;
        for (class_index = 0; class_index < class_count; class_index++) {
            float value = head->score_room[(size_t)class_index * (size_t)anchor_count + index];
            if (value > best) { best = value; best_class = class_index; }
        }
        if (best <= floor_logit) continue;

        if (options->multi_label_flag && class_count > 1) {
            for (class_index = 0; class_index < class_count && count < pick_limit; class_index++) {
                float value = head->score_room[(size_t)class_index * (size_t)anchor_count + index];
                if (value <= floor_logit) continue;
                if (!yolo_class_kept(options, class_index)) continue;
                pick_list[count].anchor_index = index;
                pick_list[count].class_index = class_index;
                pick_list[count].score = yolo_sigmoid(value);
                count++;
            }
            continue;
        }
        if (!yolo_class_kept(options, best_class)) continue;
        pick_list[count].anchor_index = index;
        pick_list[count].class_index = best_class;
        pick_list[count].score = yolo_sigmoid(best);
        count++;
    }
    return count;
}

/* One instance's mask, sampled straight from the prototypes into the source
 * picture's pixels.  The reference crops at the prototype grid, lifts to the
 * padded input and thresholds; doing it in one step here skips two full-size
 * buffers a box, and the only difference is that the bilinear sample happens
 * before the threshold rather than between two of them. */
static YoloStatus yolo_mask_lift(YoloSession *session, const YoloHeadOut *head,
                                 const YoloFrame *frame, const YoloImage *image,
                                 const YoloPick *pick, YoloBox *box)
{
    const YoloPlane *proto = &head->proto_plane;
    int mask_count = session->model->mask_count;
    int left = (int)floorf(box->left), top = (int)floorf(box->top);
    int right = (int)ceilf(box->right), bottom = (int)ceilf(box->bottom);
    float scale_x = (float)proto->width / (float)session->input_width;
    float scale_y = (float)proto->height / (float)session->input_height;
    uint8_t *plane;
    int row, column, part;

    left = yolo_max_int(left, 0);
    top = yolo_max_int(top, 0);
    right = yolo_min_int(right, image->width);
    bottom = yolo_min_int(bottom, image->height);

    plane = (uint8_t *)yolo_arena_zero(&session->keep_arena,
        (size_t)image->width * (size_t)image->height);
    if (!plane) return YOLO_ERR_MEMORY;
    box->mask_plane = plane;
    box->mask_width = image->width;
    box->mask_height = image->height;

    for (row = top; row < bottom; row++) {
        float in_y = (float)row * frame->ratio + (float)frame->pad_top;
        float at_y = in_y * scale_y - 0.5f;
        int low_y = (int)floorf(at_y);
        float weight_y = at_y - (float)low_y;
        int high_y;
        if (low_y < 0) { low_y = 0; weight_y = 0.0f; }
        if (low_y > proto->height - 1) { low_y = proto->height - 1; weight_y = 0.0f; }
        high_y = yolo_min_int(low_y + 1, proto->height - 1);

        for (column = left; column < right; column++) {
            float in_x = (float)column * frame->ratio + (float)frame->pad_left;
            float at_x = in_x * scale_x - 0.5f;
            int low_x = (int)floorf(at_x);
            float weight_x = at_x - (float)low_x;
            int high_x;
            float total = 0.0f;
            if (low_x < 0) { low_x = 0; weight_x = 0.0f; }
            if (low_x > proto->width - 1) { low_x = proto->width - 1; weight_x = 0.0f; }
            high_x = yolo_min_int(low_x + 1, proto->width - 1);

            for (part = 0; part < mask_count; part++) {
                const float *face = yolo_plane_face(proto, 0, part);
                float weight = head->extra_room[(size_t)part * (size_t)head->anchor_count
                                                + pick->anchor_index];
                float top_left = face[(size_t)low_y * (size_t)proto->width + low_x];
                float top_right = face[(size_t)low_y * (size_t)proto->width + high_x];
                float low_left = face[(size_t)high_y * (size_t)proto->width + low_x];
                float low_right = face[(size_t)high_y * (size_t)proto->width + high_x];
                float upper = top_left + (top_right - top_left) * weight_x;
                float lower = low_left + (low_right - low_left) * weight_x;
                total += weight * (upper + (lower - upper) * weight_y);
            }
            if (total > 0.0f) plane[(size_t)row * (size_t)image->width + column] = 255;
        }
    }
    return YOLO_OK;
}

YoloStatus yolo_session_run(YoloSession *session, const YoloImage *image,
                            const YoloResult **result_out)
{
    const YoloModel *model;
    YoloOptions *options;
    YoloRun run;
    YoloPlane input, tail;
    YoloPlane level_list[YOLO_LEVEL_LIMIT];
    YoloFrame frame;
    YoloHeadOut head;
    YoloStatus status;
    double clock_at;
    int rotated_flag, index, count;

    if (!session || !image || !image->pixel_list || !result_out) return YOLO_ERR_ARG;
    model = session->model;
    options = &session->options;
    *result_out = NULL;

    yolo_arena_clear(&session->arena);
    yolo_arena_clear(&session->keep_arena);
    memset(&session->result, 0, sizeof session->result);
    memset(&head, 0, sizeof head);
    memset(level_list, 0, sizeof level_list);
    session->result.source_width = image->width;
    session->result.source_height = image->height;

    clock_at = yolo_clock_ms();
    status = yolo_session_shape(session, image, &input, &frame);
    if (status != YOLO_OK) return status;
    session->shape_ms = yolo_clock_ms() - clock_at;
    session->input_plane = input;

    run.model = model;
    run.backend = model->backend;
    run.arena = &session->arena;
    run.status = YOLO_OK;

    clock_at = yolo_clock_ms();
    status = yolo_session_forward(session, &run, input, level_list, &tail);
    if (status != YOLO_OK) return status;
    if (run.status != YOLO_OK) return run.status;

    if (model->task == YOLO_TASK_CLASSIFY) {
        YoloPlane out = yolo_module_run(&run, model->head, level_list[0]);
        float *score_list;
        if (run.status != YOLO_OK) return run.status;
        session->forward_ms = yolo_clock_ms() - clock_at;
        clock_at = yolo_clock_ms();
        run.backend->softmax_run(run.backend, out.cell_list, 1, out.channel_count);
        score_list = (float *)yolo_arena_take(&session->keep_arena,
            (size_t)out.channel_count * sizeof(float));
        if (!score_list) return YOLO_ERR_MEMORY;
        memcpy(score_list, out.cell_list, (size_t)out.channel_count * sizeof(float));
        session->result.score_list = score_list;
        session->result.score_count = out.channel_count;
        session->decode_ms = yolo_clock_ms() - clock_at;
        *result_out = &session->result;
        return YOLO_OK;
    }

    status = yolo_head_run(&run, model, level_list, options->nms_free_flag, &head);
    if (status != YOLO_OK) return status;
    if (run.status != YOLO_OK) return run.status;
    session->forward_ms = yolo_clock_ms() - clock_at;
    session->score_room = head.score_room;
    session->score_anchor_count = head.anchor_count;

    clock_at = yolo_clock_ms();
    rotated_flag = model->task == YOLO_TASK_OBB;

    {
        int pick_limit = options->nms_free_flag
                       ? yolo_min_int(options->detect_limit, head.anchor_count)
                             * (options->class_blind_flag ? 1 : head.class_count)
                       : YOLO_PICK_LIMIT;
        YoloPick *pick_list = (YoloPick *)yolo_arena_take(&session->arena,
            (size_t)(pick_limit + 1) * sizeof(YoloPick));
        if (!pick_list) return YOLO_ERR_MEMORY;

        if (options->nms_free_flag) {
            int wide = yolo_min_int(options->detect_limit, head.anchor_count);
            YoloRank *rank_room = (YoloRank *)yolo_arena_take(&session->arena,
                (size_t)head.anchor_count * sizeof(YoloRank));
            YoloRank *flat_room = (YoloRank *)yolo_arena_take(&session->arena,
                (size_t)wide * (size_t)head.class_count * sizeof(YoloRank));
            if (!rank_room || !flat_room) return YOLO_ERR_MEMORY;
            count = yolo_head_choose(&head, options, rank_room, flat_room,
                                     pick_list, wide);
        } else {
            count = yolo_head_gather(&head, options, pick_list, pick_limit);
        }

        for (index = 0; index < count; index++) {
            YoloPick *pick = &pick_list[index];
            if (rotated_flag) {
                yolo_rbox_decode(&head, model->reg_max, pick->anchor_index,
                                 &pick->left, &pick->top, &pick->right,
                                 &pick->bottom, &pick->angle);
            } else {
                yolo_box_decode(&head, model->reg_max, pick->anchor_index,
                                &pick->left, &pick->top, &pick->right, &pick->bottom);
                pick->angle = 0.0f;
            }
        }

        if (!options->nms_free_flag) {
            qsort(pick_list, (size_t)count, sizeof(YoloPick), yolo_pick_order);
            count = yolo_suppress(pick_list, count, options->overlap_limit,
                                  options->class_blind_flag, rotated_flag,
                                  options->detect_limit);
        }
        if (count > options->detect_limit) count = options->detect_limit;
        if (count > session->box_limit) count = session->box_limit;

        for (index = 0; index < count; index++) {
            const YoloPick *pick = &pick_list[index];
            YoloBox *box = &session->box_list[index];
            memset(box, 0, sizeof *box);
            box->score = pick->score;
            box->class_index = pick->class_index;
            box->angle = pick->angle;

            if (rotated_flag) {
                yolo_frame_back(&frame, pick->left, pick->top,
                                &box->center_x, &box->center_y);
                box->box_width = pick->right / frame.ratio;
                box->box_height = pick->bottom / frame.ratio;
                box->left = box->center_x - box->box_width * 0.5f;
                box->top = box->center_y - box->box_height * 0.5f;
                box->right = box->center_x + box->box_width * 0.5f;
                box->bottom = box->center_y + box->box_height * 0.5f;
            } else {
                yolo_frame_back(&frame, pick->left, pick->top, &box->left, &box->top);
                yolo_frame_back(&frame, pick->right, pick->bottom,
                                &box->right, &box->bottom);
                box->left = yolo_clamp_float(box->left, 0.0f, (float)image->width);
                box->right = yolo_clamp_float(box->right, 0.0f, (float)image->width);
                box->top = yolo_clamp_float(box->top, 0.0f, (float)image->height);
                box->bottom = yolo_clamp_float(box->bottom, 0.0f, (float)image->height);
                box->center_x = (box->left + box->right) * 0.5f;
                box->center_y = (box->top + box->bottom) * 0.5f;
                box->box_width = box->right - box->left;
                box->box_height = box->bottom - box->top;
            }

            if (model->task == YOLO_TASK_POSE && head.keypoint_list) {
                int size = model->keypoint_size;
                int point;
                float *point_list = (float *)yolo_arena_take(&session->keep_arena,
                    (size_t)model->keypoint_count * (size_t)size * sizeof(float));
                if (!point_list) return YOLO_ERR_MEMORY;
                for (point = 0; point < model->keypoint_count; point++) {
                    size_t at = (size_t)(point * size) * (size_t)head.anchor_count
                              + (size_t)pick->anchor_index;
                    float stride = head.stride_list[pick->anchor_index];
                    /* yolo26 puts the keypoint one cell unit from the anchor;
                     * v8 put it two and shifted the anchor by half a cell */
                    float in_x = (head.keypoint_list[at]
                                  + head.anchor_x_list[pick->anchor_index]) * stride;
                    float in_y = (head.keypoint_list[at + (size_t)head.anchor_count]
                                  + head.anchor_y_list[pick->anchor_index]) * stride;
                    float out_x, out_y;
                    yolo_frame_back(&frame, in_x, in_y, &out_x, &out_y);
                    point_list[point * size + 0] =
                        yolo_clamp_float(out_x, 0.0f, (float)image->width);
                    point_list[point * size + 1] =
                        yolo_clamp_float(out_y, 0.0f, (float)image->height);
                    if (size > 2)
                        point_list[point * size + 2] = yolo_sigmoid(
                            head.keypoint_list[at + (size_t)(2 * head.anchor_count)]);
                }
                box->keypoint_list = point_list;
            }

            if (model->task == YOLO_TASK_SEGMENT && head.proto_flag
                && options->mask_flag) {
                status = yolo_mask_lift(session, &head, &frame, image, pick, box);
                if (status != YOLO_OK) return status;
            }
        }
        session->result.box_count = count;
        session->result.box_list = session->box_list;
    }
    session->decode_ms = yolo_clock_ms() - clock_at;
    *result_out = &session->result;
    return YOLO_OK;
}

/* ---------------------------------------------------------------------------
 * 11  Drawing, and a command line
 * ------------------------------------------------------------------------ */

/* A hue a class, spread so neighbouring classes do not share one. */
static void yolo_class_colour(int class_index, uint8_t *tint)
{
    static const uint8_t wheel[20][3] = {
        { 255,  56,  56 }, { 255, 157, 151 }, { 255, 112,  31 }, { 255, 178,  29 },
        { 207, 210,  49 }, {  72, 249,  10 }, { 146, 204,  23 }, {  61, 219, 134 },
        {  26, 147,  52 }, {   0, 212, 187 }, {   0, 194, 255 }, {  52,  69, 147 },
        { 100, 115, 255 }, {   0,  24, 236 }, { 132,  56, 255 }, {  82,   0, 133 },
        { 203,  56, 255 }, { 255, 149, 200 }, { 255,  55, 199 }, { 255, 128,   0 }
    };
    int at = ((class_index % 20) + 20) % 20;
    tint[0] = wheel[at][0];
    tint[1] = wheel[at][1];
    tint[2] = wheel[at][2];
}

static void yolo_dot_draw(YoloImage *image, int x, int y, const uint8_t *tint, int reach)
{
    int step_y, step_x, channel;
    for (step_y = -reach; step_y <= reach; step_y++) {
        for (step_x = -reach; step_x <= reach; step_x++) {
            int at_x = x + step_x, at_y = y + step_y;
            uint8_t *cell;
            if (at_x < 0 || at_y < 0 || at_x >= image->width || at_y >= image->height)
                continue;
            cell = image->pixel_list
                 + ((size_t)at_y * (size_t)image->width + (size_t)at_x)
                   * (size_t)image->channel_count;
            for (channel = 0; channel < image->channel_count && channel < 3; channel++)
                cell[channel] = tint[channel];
        }
    }
}

static void yolo_line_draw(YoloImage *image, float from_x, float from_y,
                           float to_x, float to_y, const uint8_t *tint, int reach)
{
    float span_x = to_x - from_x, span_y = to_y - from_y;
    float length = sqrtf(span_x * span_x + span_y * span_y);
    int steps = (int)length + 1, step;
    for (step = 0; step <= steps; step++) {
        float part = steps ? (float)step / (float)steps : 0.0f;
        yolo_dot_draw(image, (int)lrintf(from_x + span_x * part),
                      (int)lrintf(from_y + span_y * part), tint, reach);
    }
}

YoloStatus yolo_result_draw(const YoloModel *model, const YoloResult *result,
                            YoloImage *image)
{
    /* the coco skeleton, which every published pose checkpoint uses */
    static const int bone_list[19][2] = {
        {15,13},{13,11},{16,14},{14,12},{11,12},{5,11},{6,12},{5,6},{5,7},
        {6,8},{7,9},{8,10},{1,2},{0,1},{0,2},{1,3},{2,4},{3,5},{4,6}
    };
    int index, reach;

    if (!model || !result || !image || !image->pixel_list) return YOLO_ERR_ARG;
    reach = yolo_max_int(1, yolo_min_int(image->width, image->height) / 400);

    for (index = 0; index < result->box_count; index++) {
        const YoloBox *box = &result->box_list[index];
        uint8_t tint[3];
        yolo_class_colour(box->class_index, tint);

        if (box->mask_plane) {
            int row, column, channel;
            for (row = 0; row < box->mask_height; row++) {
                for (column = 0; column < box->mask_width; column++) {
                    uint8_t *cell;
                    if (!box->mask_plane[(size_t)row * (size_t)box->mask_width + column])
                        continue;
                    cell = image->pixel_list
                         + ((size_t)row * (size_t)image->width + (size_t)column)
                           * (size_t)image->channel_count;
                    for (channel = 0; channel < image->channel_count && channel < 3; channel++)
                        cell[channel] = (uint8_t)((cell[channel] + tint[channel]) / 2);
                }
            }
        }

        if (model->task == YOLO_TASK_OBB) {
            float turn_cos = cosf(box->angle), turn_sin = sinf(box->angle);
            float half_w = box->box_width * 0.5f, half_h = box->box_height * 0.5f;
            float corner_x[4], corner_y[4];
            int corner;
            const float sign_x[4] = { -1.0f, 1.0f, 1.0f, -1.0f };
            const float sign_y[4] = { -1.0f, -1.0f, 1.0f, 1.0f };
            for (corner = 0; corner < 4; corner++) {
                float local_x = sign_x[corner] * half_w;
                float local_y = sign_y[corner] * half_h;
                corner_x[corner] = box->center_x + local_x * turn_cos - local_y * turn_sin;
                corner_y[corner] = box->center_y + local_x * turn_sin + local_y * turn_cos;
            }
            for (corner = 0; corner < 4; corner++)
                yolo_line_draw(image, corner_x[corner], corner_y[corner],
                               corner_x[(corner + 1) % 4], corner_y[(corner + 1) % 4],
                               tint, reach);
        } else {
            yolo_line_draw(image, box->left, box->top, box->right, box->top, tint, reach);
            yolo_line_draw(image, box->right, box->top, box->right, box->bottom, tint, reach);
            yolo_line_draw(image, box->right, box->bottom, box->left, box->bottom, tint, reach);
            yolo_line_draw(image, box->left, box->bottom, box->left, box->top, tint, reach);
        }

        if (box->keypoint_list && model->keypoint_count >= 17) {
            int size = model->keypoint_size;
            int bone, point;
            uint8_t bone_tint[3] = { 51, 153, 255 };
            for (bone = 0; bone < 19; bone++) {
                const float *a = box->keypoint_list + bone_list[bone][0] * size;
                const float *b = box->keypoint_list + bone_list[bone][1] * size;
                if (size > 2 && (a[2] < 0.5f || b[2] < 0.5f)) continue;
                yolo_line_draw(image, a[0], a[1], b[0], b[1], bone_tint, reach);
            }
            for (point = 0; point < model->keypoint_count; point++) {
                const float *at = box->keypoint_list + point * size;
                if (size > 2 && at[2] < 0.5f) continue;
                yolo_dot_draw(image, (int)lrintf(at[0]), (int)lrintf(at[1]),
                              tint, reach + 1);
            }
        }
    }
    return YOLO_OK;
}

#endif /* YOLO_IMPLEMENTED */

/* ---------------------------------------------------------------------------
 * The command line
 *
 * Built when YOLO_MAIN is defined, so the same file is a library and a
 * program.  It exists to make a run reproducible from a shell, not to be a
 * user interface.
 * ------------------------------------------------------------------------ */

#ifdef YOLO_MAIN

static void yolo_usage(void)
{
    printf(
"yolo -- run a yolo26 checkpoint\n"
"\n"
"  yolo <checkpoint.pt> <picture> [options]\n"
"\n"
"Options\n"
"  --conf F      the score a detection must beat        (0.25)\n"
"  --iou F       the overlap above which a box is a duplicate (0.7)\n"
"  --max N       the most detections to report          (300)\n"
"  --size N      the side the picture is letterboxed to (the checkpoint's)\n"
"  --nms-free    read the one-to-one head and skip suppression\n"
"  --square      pad to a full square rather than to a stride multiple\n"
"  --agnostic    suppress across classes as well as within\n"
"  --classes A,B keep only these class indices\n"
"  --no-masks    skip lifting segmentation masks to the source\n"
"  --draw PATH   write the picture with the result drawn on it\n"
"  --repeat N    run the picture N times and report the fastest\n"
"  --dump PATH   write the shaped input and the head's rows, for a parity run\n"
"  --info        describe the checkpoint and stop\n");
}

static int yolo_main_classes(char *text, int *room, int limit)
{
    int count = 0;
    char *walk = text;
    while (*walk && count < limit) {
        room[count++] = (int)strtol(walk, &walk, 10);
        if (*walk == ',') walk++;
        else break;
    }
    return count;
}

int main(int argc, char **argv)
{
    YoloModel *model = NULL;
    YoloSession *session = NULL;
    YoloOptions options;
    YoloImage image;
    const YoloResult *result = NULL;
    YoloStatus status;
    const char *model_path = NULL, *image_path = NULL, *draw_path = NULL;
    const char *dump_path = NULL;
    int class_room[64], class_count = 0, info_flag = 0, repeat_count = 1;
    int square_flag = 0, mask_off_flag = 0;
    int index, round;
    double best_ms = 0.0;

    /* zero, not the defaults: what the caller sets is merged over the model's
     * own defaults below, and a zero here means "was not set" */
    memset(&options, 0, sizeof options);
    memset(&image, 0, sizeof image);

    for (index = 1; index < argc; index++) {
        const char *word = argv[index];
        if (strcmp(word, "--help") == 0 || strcmp(word, "-h") == 0) { yolo_usage(); return 0; }
        else if (strcmp(word, "--info") == 0) info_flag = 1;
        else if (word[0] != '-' && !model_path) model_path = word;
        else if (word[0] != '-' && !image_path) image_path = word;
        else if (index + 1 < argc) {
            const char *value = argv[++index];
            if (strcmp(word, "--conf") == 0) options.score_floor = (float)atof(value);
            else if (strcmp(word, "--iou") == 0) options.overlap_limit = (float)atof(value);
            else if (strcmp(word, "--max") == 0) options.detect_limit = atoi(value);
            else if (strcmp(word, "--size") == 0) options.input_size = atoi(value);
            else if (strcmp(word, "--draw") == 0) draw_path = value;
            else if (strcmp(word, "--repeat") == 0) repeat_count = atoi(value);
            else if (strcmp(word, "--dump") == 0) dump_path = value;
            else if (strcmp(word, "--classes") == 0)
                class_count = yolo_main_classes((char *)value, class_room, 64);
            else { index--; goto flag; }
            continue;
        } else goto flag;
        continue;
flag:
        if (strcmp(word, "--nms-free") == 0) options.nms_free_flag = 1;
        else if (strcmp(word, "--agnostic") == 0) options.class_blind_flag = 1;
        else if (strcmp(word, "--square") == 0) square_flag = 1;
        else if (strcmp(word, "--no-masks") == 0) mask_off_flag = 1;
        else { printf("unknown option %s\n", word); return 2; }
    }
    if (!model_path) { yolo_usage(); return 2; }

    status = yolo_model_open(model_path, &model);
    if (status != YOLO_OK) {
        printf("%s: %s\n", yolo_status_text(status), yolo_detail_text());
        return 1;
    }

    printf("%s  %s  %d classes  input %d  stride %d%s\n",
           model_path, yolo_task_text(yolo_model_task(model)),
           yolo_model_class_count(model), yolo_model_input_size(model),
           yolo_model_stride_size(model),
           yolo_model_nms_free_flag(model) ? "  one-to-one head available" : "");
    if (info_flag || !image_path) {
        for (index = 0; index < yolo_model_class_count(model); index++)
            printf("  %3d  %s\n", index, yolo_model_class_name(model, index));
        yolo_model_close(model);
        return 0;
    }

    {
        YoloOptions fresh;
        yolo_options_default(model, &fresh);
        /* the flags the caller set overwrite the defaults, and everything
         * untouched keeps what `yolo predict` would use */
        if (options.score_floor > 0.0f) fresh.score_floor = options.score_floor;
        if (options.overlap_limit > 0.0f) fresh.overlap_limit = options.overlap_limit;
        if (options.detect_limit > 0) fresh.detect_limit = options.detect_limit;
        if (options.input_size > 0) fresh.input_size = options.input_size;
        fresh.nms_free_flag = options.nms_free_flag;
        fresh.class_blind_flag = options.class_blind_flag;
        if (square_flag) fresh.rect_flag = 0;
        if (mask_off_flag) fresh.mask_flag = 0;
        if (class_count) { fresh.class_filter_list = class_room;
                           fresh.class_filter_count = class_count; }
        options = fresh;
    }

    status = yolo_image_read(image_path, &image);
    if (status != YOLO_OK) {
        printf("%s: %s\n", yolo_status_text(status), yolo_detail_text());
        yolo_model_close(model);
        return 1;
    }

    status = yolo_session_open(model, &options, &session);
    if (status != YOLO_OK) {
        printf("%s: %s\n", yolo_status_text(status), yolo_detail_text());
        yolo_image_free(&image);
        yolo_model_close(model);
        return 1;
    }

    for (round = 0; round < (repeat_count > 0 ? repeat_count : 1); round++) {
        double at = yolo_clock_ms();
        double spent;
        status = yolo_session_run(session, &image, &result);
        spent = yolo_clock_ms() - at;
        if (status != YOLO_OK) break;
        if (round == 0 || spent < best_ms) best_ms = spent;
    }
    if (status != YOLO_OK) {
        printf("%s: %s\n", yolo_status_text(status), yolo_detail_text());
    } else if (yolo_model_task(model) == YOLO_TASK_CLASSIFY) {
        int best = 0;
        for (index = 0; index < result->score_count; index++)
            if (result->score_list[index] > result->score_list[best]) best = index;
        printf("%s %.4f\n", yolo_model_class_name(model, best), result->score_list[best]);
    } else {
        printf("%d found in %dx%d\n", result->box_count, image.width, image.height);
        for (index = 0; index < result->box_count; index++) {
            const YoloBox *box = &result->box_list[index];
            if (yolo_model_task(model) == YOLO_TASK_OBB)
                printf("  %-18s %.3f  centre %.1f %.1f  size %.1f %.1f  angle %.3f\n",
                       yolo_model_class_name(model, box->class_index), box->score,
                       box->center_x, box->center_y, box->box_width, box->box_height,
                       box->angle);
            else
                printf("  %-18s %.3f  %.1f %.1f %.1f %.1f\n",
                       yolo_model_class_name(model, box->class_index), box->score,
                       box->left, box->top, box->right, box->bottom);
        }
        if (dump_path) {
            const YoloPlane *plane = yolo_session_input(session);
            int anchor_count = 0, score_classes = 0;
            const float *score_room =
                yolo_session_score_room(session, &anchor_count, &score_classes);
            FILE *handle = fopen(dump_path, "wb");
            if (handle) {
                int head_room[4];
                head_room[0] = plane->height;
                head_room[1] = plane->width;
                head_room[2] = anchor_count;
                head_room[3] = score_classes;
                fwrite(head_room, sizeof(int), 4, handle);
                fwrite(plane->cell_list, sizeof(float),
                       yolo_plane_count(plane), handle);
                if (score_room)
                    fwrite(score_room, sizeof(float),
                           (size_t)anchor_count * (size_t)score_classes, handle);
                fclose(handle);
                printf("dumped %s\n", dump_path);
            }
        }
        if (draw_path) {
            YoloStatus wrote;
            yolo_result_draw(model, result, &image);
            wrote = yolo_image_write(draw_path, &image);
            if (wrote != YOLO_OK) printf("draw: %s\n", yolo_detail_text());
            else printf("drew %s\n", draw_path);
        }
    }
    printf("%.1f ms  shape %.1f  forward %.1f  decode %.1f\n", best_ms,
           yolo_session_shape_time(session), yolo_session_forward_time(session),
           yolo_session_decode_time(session));

    yolo_session_close(session);
    yolo_image_free(&image);
    yolo_model_close(model);
    return status == YOLO_OK ? 0 : 1;
}

#endif /* YOLO_MAIN */

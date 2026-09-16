/* ============================================================================
 * app/core.c -- Liquid inference engine.
 *
 * A single-translation-unit engine that runs the Liquid Foundation Model (LFM2
 * family) directly from a Hugging Face checkpoint folder: config.json, one or
 * more *.safetensors shards, and tokenizer.json.  Pure C11, no third-party
 * dependencies, no build system requirements beyond a C compiler.
 *
 * The file is laid out as a header followed by its implementation.  Callers
 * include it once:
 *
 *     #include "core.c"
 *
 * app/main.c and test/test.c both do exactly that, which keeps the project
 * flat while still giving the engine a single, clean, documented API surface.
 *
 * Layout
 *   part 1   public interface
 *   part 2   platform layer -- memory, files, clocks, threads, cpu probes
 *   part 3   number formats -- f16 and bf16
 *   part 4   vector vocabulary -- the one width-agnostic kernel dialect
 *   part 5   block quantisation -- the q8 format and its dot product
 *   part 6   json reader
 *   part 7   safetensors store
 *   part 8   compute kernels
 *   part 9   backend seam -- the accelerator hand-off point
 *   part 10  model load -- arch, weights, layout
 *   part 11  model state -- key/value cache, stride cache, scratch
 *   part 12  forward pass -- attention blocks, conv blocks, feed forward
 *   part 13  vocabulary -- byte level bpe
 *   part 14  sampler
 *   part 15  facade
 *
 * Licensed under the terms in LICENSE.
 * ==========================================================================*/

#ifndef ILL_CORE_INCLUDED
#define ILL_CORE_INCLUDED

/* Ask the host headers for the POSIX surface the platform layer uses. */
#if !defined(_WIN32)
#  ifndef _POSIX_C_SOURCE
#    define _POSIX_C_SOURCE 200809L
#  endif
#  ifndef _DEFAULT_SOURCE
#    define _DEFAULT_SOURCE 1
#  endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define ILL_SPARE __attribute__((unused))
#else
#  define ILL_SPARE
#endif

/* ============================================================================
 * part 1 -- public interface
 * ==========================================================================*/

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ILL_VERSION_MAJOR 1
#define ILL_VERSION_MINOR 0
#define ILL_VERSION_PATCH 0
#define ILL_VERSION_TEXT  "1.0.0"

/* -- results ------------------------------------------------------------- */

typedef enum IllResult {
    ILL_OK = 0,      /* call succeeded                                      */
    ILL_ARGS,        /* caller passed something impossible                  */
    ILL_ALLOC,       /* out of memory                                       */
    ILL_FILE,        /* a file could not be opened, mapped, or read         */
    ILL_PARSE,       /* a json or safetensors document is malformed         */
    ILL_SHAPE,       /* a tensor has an unexpected rank or extent           */
    ILL_MODEL,       /* the checkpoint is not a model this engine runs      */
    ILL_VOCAB,       /* the tokenizer is missing or of an unsupported kind  */
    ILL_LIMIT,       /* a caller supplied buffer or window was too small    */
    ILL_STATE        /* the call does not fit the current engine state      */
} IllResult;

/* Stable, human readable name for a result code. Never returns NULL. */
const char *ill_result_text(IllResult code);

/* -- number formats ------------------------------------------------------- */

typedef enum IllType {
    ILL_TYPE_KEEP = -1, /* load time request: leave weights as stored       */
    ILL_TYPE_F32  =  0,
    ILL_TYPE_F16,
    ILL_TYPE_BF16,
    ILL_TYPE_Q8,        /* 32 value blocks, one f32 scale per block         */
    ILL_TYPE_COUNT
} IllType;

const char *ill_type_text(IllType type);
size_t      ill_type_size(IllType type, size_t count); /* bytes for count values */

/* -- architecture --------------------------------------------------------- */

typedef enum IllLayer {
    ILL_LAYER_ATTN = 0,  /* grouped query attention with rotary positions   */
    ILL_LAYER_CONV = 1   /* liquid short convolution operator               */
} IllLayer;

typedef struct IllArch {
    int32_t  vocab_size;    /* token count                                  */
    int32_t  model_dim;     /* residual stream width                        */
    int32_t  inner_dim;     /* feed forward width                           */
    int32_t  layer_count;   /* blocks in the stack                          */
    int32_t  head_count;    /* query heads                                  */
    int32_t  group_count;   /* key/value heads                              */
    int32_t  head_dim;      /* width of one head                            */
    int32_t  conv_width;    /* short convolution taps                       */
    int32_t  conv_bias;     /* short convolution carries bias terms         */
    int32_t  bound_embed;   /* output projection is tied to the embedding   */
    int32_t  token_span;    /* trained position span                        */
    float    norm_eps;      /* rms norm epsilon                             */
    float    rope_base;     /* rotary frequency base                        */
    int32_t  begin_token;   /* bos                                          */
    int32_t  end_token;     /* eos                                          */
    int32_t  pad_token;     /* pad                                          */
    uint8_t *layer_kind;    /* layer_count entries of IllLayer              */
    int32_t  attn_count;    /* attention layers in the stack                */
    int32_t  conv_count;    /* convolution layers in the stack              */
} IllArch;

/* -- load plan ------------------------------------------------------------ */

typedef struct IllPlan {
    const char *model_path;    /* checkpoint folder, or a .safetensors file */
    IllType     weight_type;   /* ILL_TYPE_KEEP or ILL_TYPE_Q8              */
    const char *backend_name;  /* NULL selects the default backend          */
    int32_t     thread_count;  /* 0 asks the platform for a sensible count  */
    int32_t     batch_span;    /* prefill chunk in tokens, 0 picks 256      */
    int32_t     quiet_load;    /* suppress progress notes on stderr         */
} IllPlan;

/* Fills a plan with the engine defaults. */
void ill_plan_init(IllPlan *plan);

/* -- opaque objects ------------------------------------------------------- */

typedef struct IllModel   IllModel;   /* weights, arch, vocabulary          */
typedef struct IllState   IllState;   /* one sequence: caches and scratch   */
typedef struct IllVocab   IllVocab;   /* tokenizer                          */
typedef struct IllSampler IllSampler; /* token choice policy plus history   */

/* -- model ---------------------------------------------------------------- */

IllResult      ill_model_load(IllModel **model, const IllPlan *plan);
void           ill_model_free(IllModel *model);
const IllArch *ill_model_arch(const IllModel *model);
const IllVocab *ill_model_vocab(const IllModel *model);
size_t         ill_model_bytes(const IllModel *model);  /* resident weights */
const char    *ill_model_backend(const IllModel *model);
int32_t        ill_model_threads(const IllModel *model);

/* -- state ---------------------------------------------------------------- */

IllResult ill_state_make(IllState **state, IllModel *model, int32_t context_span);
void      ill_state_free(IllState *state);
void      ill_state_reset(IllState *state);
int32_t   ill_state_fill(const IllState *state);   /* cached token count    */
int32_t   ill_state_span(const IllState *state);   /* cache capacity        */
size_t    ill_state_bytes(const IllState *state);

/* Rewinds the sequence to `fill` tokens.  Only the key/value cache can be
 * rewound exactly; the stride cache is recurrent, so a rewind past the last
 * `conv_width - 1` tokens is reported as ILL_STATE. */
IllResult ill_state_crop(IllState *state, int32_t fill);

/* -- forward pass --------------------------------------------------------- */

typedef struct IllBatch {
    const int32_t *tokens;  /* token ids to append to the sequence          */
    int32_t        count;   /* how many                                     */
    int32_t        every;   /* 1 yields logits for every token, 0 the last  */
} IllBatch;

/* Runs `batch` through the stack and appends it to the sequence held in
 * `state`.  On return *logits points at engine owned memory holding either one
 * row of vocab_size floats, or `count` rows when batch->every is set.  The
 * pointer stays valid until the next call on the same state. */
IllResult ill_model_apply(IllModel *model, IllState *state,
                          const IllBatch *batch, float **logits);

/* -- vocabulary ----------------------------------------------------------- */

int32_t     ill_vocab_size(const IllVocab *vocab);
int32_t     ill_vocab_stops(const IllVocab *vocab);   /* how many end a turn */
int32_t     ill_vocab_find(const IllVocab *vocab, const char *piece);
const char *ill_vocab_name(const IllVocab *vocab, int32_t token);
int32_t     ill_vocab_stop(const IllVocab *vocab, int32_t token);

/* Encodes UTF-8 text into token ids.  Writes at most `limit` ids and always
 * reports the full requirement through `count`, so a caller can size a buffer
 * with a first pass that passes limit 0. */
IllResult ill_vocab_encode(const IllVocab *vocab, const char *text, size_t text_len,
                           int32_t add_begin, int32_t *tokens, int32_t limit,
                           int32_t *count);

/* Decodes one token into UTF-8 bytes, resolving byte level escapes. */
IllResult ill_vocab_decode(const IllVocab *vocab, int32_t token,
                           char *text, int32_t limit, int32_t *len);

/* Renders a chat exchange into the checkpoint's prompt format. */
typedef struct IllTurn {
    const char *role;  /* "system", "user", "assistant"                     */
    const char *text;
} IllTurn;

IllResult ill_vocab_prompt(const IllVocab *vocab, const IllTurn *turns, int32_t count,
                           int32_t open_reply, char *text, size_t limit, size_t *len);

/* -- sampler -------------------------------------------------------------- */

typedef struct IllTuning {
    float    temperature;    /* 0 selects greedy decoding                   */
    float    top_p;          /* nucleus mass, 1 disables                    */
    float    min_p;          /* floor relative to the peak, 0 disables      */
    int32_t  top_k;          /* candidate cap, 0 disables                   */
    float    repeat_penalty; /* 1 disables                                  */
    int32_t  repeat_span;    /* history depth the penalty looks back over   */
    uint64_t seed;
} IllTuning;

void      ill_tuning_init(IllTuning *tuning);
IllResult ill_sampler_make(IllSampler **sampler, const IllTuning *tuning, int32_t vocab_size);
void      ill_sampler_free(IllSampler *sampler);
void      ill_sampler_wipe(IllSampler *sampler);
void      ill_sampler_note(IllSampler *sampler, int32_t token);
int32_t   ill_sampler_pick(IllSampler *sampler, float *logits);

/* -- compute backend ------------------------------------------------------
 *
 * The accelerator seam.  IllBackend and its six operations are declared in
 * part 9, next to the cpu implementation that serves as their reference; these
 * three calls are the whole registration surface. */

typedef struct IllBackend IllBackend;

IllResult   ill_backend_join(IllBackend *backend);          /* before load    */
int32_t     ill_backend_list(const char **names, int32_t limit);
const char *ill_backend_simd(void);   /* the vector width this build targets */

/* -- clocks --------------------------------------------------------------- */

double ill_clock_now(void);   /* monotonic seconds */

#ifdef __cplusplus
}
#endif

/* ============================================================================
 * part 2 -- platform layer
 *
 * Everything the engine needs from the host operating system lives here:
 * aligned memory, whole-file mapping, a monotonic clock, a thread pool, and a
 * cpu feature probe.  Three ports are provided -- POSIX, Windows, and a bare
 * fallback that keeps the engine compiling on a freestanding toolchain.
 * ==========================================================================*/

#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <ctype.h>
#include <limits.h>
#include <float.h>

#if defined(_WIN32)
#  define ILL_HOST_WINDOWS 1
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN 1
#  endif
#  include <windows.h>
#else
#  define ILL_HOST_POSIX 1
#  include <unistd.h>
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <sys/mman.h>
#  include <dirent.h>
#  include <time.h>
#  include <pthread.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__) && \
    (defined(_M_IX86) || defined(_M_X64)) && !defined(ILL_NO_THREADS)
/* MSVC reports __STDC_NO_ATOMICS__ and ships no working C11 <stdatomic.h>,
 * but the thread pool needs only a sliver of the atomic surface.  Back it
 * with the interlocked intrinsics instead.
 *
 * Scoped to x86 and x64 deliberately.  The loads below are plain volatile
 * reads, which carry acquire ordering on this architecture once the
 * compiler is told not to sink them.  On ARM64 they would not: MSVC
 * defaults to /volatile:iso there and the hardware reorders freely, so a
 * worker could read pool->chore before it observed the epoch that
 * published it.  That target keeps the single threaded fallback until the
 * loads below grow real barriers.  clang-cl also defines _MSC_VER but
 * ships a working <stdatomic.h>, so it skips this shim entirely. */
#  include <intrin.h>
#  define ILL_HAS_ATOMIC 1
typedef volatile LONG IllAtomic32;
#  define atomic_int_least32_t   IllAtomic32
#  define atomic_uint_least32_t  IllAtomic32
#  define atomic_int             IllAtomic32
#  define memory_order_relaxed   0
#  define memory_order_acquire   2
#  define memory_order_release   3
static __forceinline LONG ill_msvc_load_acquire(volatile const LONG *cell)
{
    LONG seen = *cell;
    _ReadWriteBarrier();   /* x86 loads are acquire; stop the compiler alone */
    return seen;
}
#  define atomic_load(p)                      ill_msvc_load_acquire((volatile const LONG *)(p))
#  define atomic_load_explicit(p, mo)         ill_msvc_load_acquire((volatile const LONG *)(p))
#  define atomic_store(p, v)                  ((void)_InterlockedExchange((volatile LONG *)(p), (LONG)(v)))
#  define atomic_fetch_add(p, v)              _InterlockedExchangeAdd((volatile LONG *)(p), (LONG)(v))
#  define atomic_fetch_add_explicit(p, v, mo) _InterlockedExchangeAdd((volatile LONG *)(p), (LONG)(v))
#elif defined(__STDC_NO_ATOMICS__) || (defined(__STDC_VERSION__) && __STDC_VERSION__ < 201112L)
#  define ILL_HAS_ATOMIC 0
#else
#  define ILL_HAS_ATOMIC 1
#  include <stdatomic.h>
#endif

#if !ILL_HAS_ATOMIC
#  define ILL_WITH_THREADS 0
#elif defined(ILL_HOST_WINDOWS) || defined(ILL_HOST_POSIX)
#  define ILL_WITH_THREADS 1
#else
#  define ILL_WITH_THREADS 0
#endif

#define ILL_ALIGN_BYTES 64
#define ILL_THREAD_MAX  128
#define ILL_PATH_MAX    1024

#define ILL_MIN(a, b) ((a) < (b) ? (a) : (b))
#define ILL_MAX(a, b) ((a) > (b) ? (a) : (b))
#define ILL_UP(a, b)  (((a) + (b) - 1) / (b) * (b))

/* -- diagnostics ---------------------------------------------------------- */

static int ill_noise_level = 1;   /* 0 silent, 1 warnings, 2 progress */

static void ill_note(int level, const char *form, ...)
{
    va_list args;
    if (level > ill_noise_level) return;
    va_start(args, form);
    fputs("ill: ", stderr);
    vfprintf(stderr, form, args);
    fputc('\n', stderr);
    va_end(args);
}

/* -- memory --------------------------------------------------------------- */

/* Aligned allocation with a one byte back-pointer, so the same code path works
 * on every host regardless of aligned_alloc availability. */
static void *ill_block_make(size_t bytes)
{
    void *raw;
    void *fit;
    if (bytes == 0) bytes = 1;
    raw = malloc(bytes + ILL_ALIGN_BYTES + sizeof(void *));
    if (!raw) return NULL;
    fit = (void *)ILL_UP((uintptr_t)raw + sizeof(void *), (uintptr_t)ILL_ALIGN_BYTES);
    ((void **)fit)[-1] = raw;
    return fit;
}

static void ill_block_free(void *fit)
{
    if (fit) free(((void **)fit)[-1]);
}

static void *ill_block_zero(size_t bytes)
{
    void *fit = ill_block_make(bytes);
    if (fit) memset(fit, 0, bytes);
    return fit;
}

/* -- files ---------------------------------------------------------------- */

typedef struct IllFile {
    const uint8_t *data;
    size_t         size;
    int            mapped;
#if defined(ILL_HOST_WINDOWS)
    HANDLE         file;
    HANDLE         view;
#endif
} IllFile;

static IllResult ill_file_open(IllFile *file, const char *path)
{
    memset(file, 0, sizeof(*file));
#if defined(ILL_HOST_POSIX)
    {
        int fd;
        struct stat info;
        void *base;
        fd = open(path, O_RDONLY);
        if (fd < 0) return ILL_FILE;
        if (fstat(fd, &info) != 0 || info.st_size <= 0) { close(fd); return ILL_FILE; }
        base = mmap(NULL, (size_t)info.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (base == MAP_FAILED) return ILL_FILE;
        file->data   = (const uint8_t *)base;
        file->size   = (size_t)info.st_size;
        file->mapped = 1;
        return ILL_OK;
    }
#elif defined(ILL_HOST_WINDOWS)
    {
        LARGE_INTEGER span;
        file->file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (file->file == INVALID_HANDLE_VALUE) return ILL_FILE;
        if (!GetFileSizeEx(file->file, &span) || span.QuadPart <= 0) {
            CloseHandle(file->file); file->file = NULL; return ILL_FILE;
        }
        file->view = CreateFileMappingA(file->file, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!file->view) { CloseHandle(file->file); file->file = NULL; return ILL_FILE; }
        file->data = (const uint8_t *)MapViewOfFile(file->view, FILE_MAP_READ, 0, 0, 0);
        if (!file->data) {
            CloseHandle(file->view); CloseHandle(file->file);
            file->view = NULL; file->file = NULL; return ILL_FILE;
        }
        file->size   = (size_t)span.QuadPart;
        file->mapped = 1;
        return ILL_OK;
    }
#else
    {
        FILE *fh = fopen(path, "rb");
        uint8_t *base;
        long span;
        if (!fh) return ILL_FILE;
        if (fseek(fh, 0, SEEK_END) != 0) { fclose(fh); return ILL_FILE; }
        span = ftell(fh);
        if (span <= 0) { fclose(fh); return ILL_FILE; }
        rewind(fh);
        base = (uint8_t *)ill_block_make((size_t)span);
        if (!base) { fclose(fh); return ILL_ALLOC; }
        if (fread(base, 1, (size_t)span, fh) != (size_t)span) {
            ill_block_free(base); fclose(fh); return ILL_FILE;
        }
        fclose(fh);
        file->data = base;
        file->size = (size_t)span;
        return ILL_OK;
    }
#endif
}

static void ill_file_close(IllFile *file)
{
    if (!file || !file->data) return;
#if defined(ILL_HOST_POSIX)
    munmap((void *)file->data, file->size);
#elif defined(ILL_HOST_WINDOWS)
    UnmapViewOfFile((LPCVOID)file->data);
    if (file->view) CloseHandle(file->view);
    if (file->file) CloseHandle(file->file);
#else
    ill_block_free((void *)file->data);
#endif
    memset(file, 0, sizeof(*file));
}

static int ill_file_here(const char *path)
{
#if defined(ILL_HOST_POSIX)
    struct stat info;
    return stat(path, &info) == 0 && S_ISREG(info.st_mode);
#else
    FILE *fh = fopen(path, "rb");
    if (!fh) return 0;
    fclose(fh);
    return 1;
#endif
}

static int ill_path_dir(const char *path)
{
#if defined(ILL_HOST_POSIX)
    struct stat info;
    return stat(path, &info) == 0 && S_ISDIR(info.st_mode);
#elif defined(ILL_HOST_WINDOWS)
    DWORD bits = GetFileAttributesA(path);
    return bits != INVALID_FILE_ATTRIBUTES && (bits & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
    (void)path;
    return 0;
#endif
}

/* Joins a folder and a leaf into `out`, using the host separator. */
static void ill_path_join(char *out, size_t limit, const char *dir, const char *leaf)
{
#if defined(ILL_HOST_WINDOWS)
    const char slash = '\\';
#else
    const char slash = '/';
#endif
    size_t used = 0;
    size_t span = strlen(dir);
    if (span >= limit) span = limit - 1;
    memcpy(out, dir, span);
    used = span;
    if (used && out[used - 1] != '/' && out[used - 1] != '\\' && used + 1 < limit)
        out[used++] = slash;
    span = strlen(leaf);
    if (used + span >= limit) span = limit - used - 1;
    memcpy(out + used, leaf, span);
    out[used + span] = '\0';
}

/* -- clock ---------------------------------------------------------------- */

double ill_clock_now(void)
{
#if defined(ILL_HOST_POSIX)
    struct timespec tick;
    clock_gettime(CLOCK_MONOTONIC, &tick);
    return (double)tick.tv_sec + (double)tick.tv_nsec * 1e-9;
#elif defined(ILL_HOST_WINDOWS)
    LARGE_INTEGER rate, tick;
    QueryPerformanceFrequency(&rate);
    QueryPerformanceCounter(&tick);
    return (double)tick.QuadPart / (double)rate.QuadPart;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

/* -- cpu probe ------------------------------------------------------------ */

static int32_t ill_cpu_count(void)
{
#if defined(ILL_HOST_WINDOWS)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return (int32_t)info.dwNumberOfProcessors;
#elif defined(_SC_NPROCESSORS_ONLN)
    long span = sysconf(_SC_NPROCESSORS_ONLN);
    return span > 0 ? (int32_t)span : 1;
#else
    return 1;
#endif
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define ILL_ARCH_X86 1
#endif
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
#  define ILL_ARCH_ARM 1
#endif

static void ill_cpu_pause(void)
{
#if defined(ILL_ARCH_X86) && defined(__GNUC__)
    __asm__ __volatile__("pause");
#elif defined(ILL_ARCH_X86) && defined(_MSC_VER)
    _mm_pause();
#elif defined(ILL_ARCH_ARM) && defined(__GNUC__)
    __asm__ __volatile__("yield");
#endif
}

static void ill_cpu_yield(void)
{
#if defined(ILL_HOST_WINDOWS)
    SwitchToThread();
#elif defined(ILL_HOST_POSIX)
    sched_yield();
#endif
}

/* -- thread pool ----------------------------------------------------------
 *
 * The pool runs one shared task at a time.  A task is a function plus an
 * argument plus a chunk count; workers claim chunks with an atomic counter, so
 * uneven chunks self balance.  Handshakes use a monotonically rising epoch,
 * which keeps the fast path free of locks: a decode step issues a few hundred
 * of these, and a mutex round trip per issue would dominate the step.
 * ------------------------------------------------------------------------*/

typedef void (*IllChore)(void *args, int32_t chunk, int32_t chunks, int32_t worker);

#if ILL_WITH_THREADS

typedef struct IllPool IllPool;

typedef struct IllHand {
    IllPool *pool;
    int32_t  index;
#if defined(ILL_HOST_WINDOWS)
    HANDLE   thread;
#else
    pthread_t thread;
#endif
} IllHand;

struct IllPool {
    IllHand           hands[ILL_THREAD_MAX];
    int32_t           count;        /* workers plus the calling thread */
    IllChore          chore;
    void             *args;
    int32_t           chunks;
    atomic_int_least32_t claim;     /* next chunk to hand out */
    atomic_int_least32_t done;      /* workers finished with this epoch */
    atomic_uint_least32_t epoch;    /* rises once per issued task */
    atomic_int           halt;
};

static void ill_pool_work(IllPool *pool, int32_t worker)
{
    IllChore chore  = pool->chore;
    void    *args   = pool->args;
    int32_t  chunks = pool->chunks;
    for (;;) {
        int32_t chunk = (int32_t)atomic_fetch_add(&pool->claim, 1);
        if (chunk >= chunks) break;
        chore(args, chunk, chunks, worker);
    }
    atomic_fetch_add(&pool->done, 1);
}

static void ill_pool_loop(IllHand *hand)
{
    IllPool *pool = hand->pool;
    uint32_t seen = 0;
    for (;;) {
        uint32_t spins = 0;
        uint32_t epoch;
        for (;;) {
            if (atomic_load(&pool->halt)) return;
            epoch = (uint32_t)atomic_load_explicit(&pool->epoch, memory_order_acquire);
            if (epoch != seen) break;
            if (++spins < 4096u) ill_cpu_pause();
            else ill_cpu_yield();
        }
        seen = epoch;
        ill_pool_work(pool, hand->index);
    }
}

#if defined(ILL_HOST_WINDOWS)
static DWORD WINAPI ill_pool_gate(LPVOID hand) { ill_pool_loop((IllHand *)hand); return 0; }
#else
static void *ill_pool_gate(void *hand) { ill_pool_loop((IllHand *)hand); return NULL; }
#endif

static IllResult ill_pool_make(IllPool **out, int32_t count)
{
    IllPool *pool;
    int32_t  index;
    if (count < 1) count = 1;
    if (count > ILL_THREAD_MAX) count = ILL_THREAD_MAX;
    pool = (IllPool *)ill_block_zero(sizeof(IllPool));
    if (!pool) return ILL_ALLOC;
    pool->count = count;
    atomic_store(&pool->claim, 0);
    atomic_store(&pool->done, 0);
    atomic_store(&pool->epoch, 0);
    atomic_store(&pool->halt, 0);
    for (index = 1; index < count; ++index) {
        pool->hands[index].pool  = pool;
        pool->hands[index].index = index;
#if defined(ILL_HOST_WINDOWS)
        pool->hands[index].thread = CreateThread(NULL, 0, ill_pool_gate,
                                                 &pool->hands[index], 0, NULL);
        if (!pool->hands[index].thread) { pool->count = index; break; }
#else
        if (pthread_create(&pool->hands[index].thread, NULL,
                           ill_pool_gate, &pool->hands[index]) != 0) {
            pool->count = index;
            break;
        }
#endif
    }
    *out = pool;
    return ILL_OK;
}

static void ill_pool_free(IllPool *pool)
{
    int32_t index;
    if (!pool) return;
    atomic_store(&pool->halt, 1);
    atomic_fetch_add(&pool->epoch, 1);
    for (index = 1; index < pool->count; ++index) {
#if defined(ILL_HOST_WINDOWS)
        WaitForSingleObject(pool->hands[index].thread, INFINITE);
        CloseHandle(pool->hands[index].thread);
#else
        pthread_join(pool->hands[index].thread, NULL);
#endif
    }
    ill_block_free(pool);
}

/* Issues `chunks` units of `chore` and returns once every unit has run. */
static void ill_pool_fork(IllPool *pool, IllChore chore, void *args, int32_t chunks)
{
    int32_t  helpers;
    uint32_t spins = 0;
    if (chunks <= 0) return;
    if (!pool || pool->count <= 1 || chunks == 1) {
        int32_t chunk;
        for (chunk = 0; chunk < chunks; ++chunk) chore(args, chunk, chunks, 0);
        return;
    }
    helpers = pool->count - 1;
    pool->chore  = chore;
    pool->args   = args;
    pool->chunks = chunks;
    atomic_store(&pool->claim, 0);
    atomic_store(&pool->done, 0);
    atomic_fetch_add_explicit(&pool->epoch, 1, memory_order_release);
    ill_pool_work(pool, 0);
    /* The caller counted itself in `done`; wait for the workers. */
    while (atomic_load_explicit(&pool->done, memory_order_acquire) < helpers + 1) {
        if (++spins < 4096u) ill_cpu_pause();
        else ill_cpu_yield();
    }
}

static int32_t ill_pool_size(const IllPool *pool) { return pool ? pool->count : 1; }

#else /* !ILL_WITH_THREADS -- single threaded fallback */

typedef struct IllPool { int32_t count; } IllPool;

static IllResult ill_pool_make(IllPool **out, int32_t count)
{
    IllPool *pool = (IllPool *)ill_block_zero(sizeof(IllPool));
    (void)count;
    if (!pool) return ILL_ALLOC;
    pool->count = 1;
    *out = pool;
    return ILL_OK;
}
static void ill_pool_free(IllPool *pool) { ill_block_free(pool); }
static void ill_pool_fork(IllPool *pool, IllChore chore, void *args, int32_t chunks)
{
    int32_t chunk;
    (void)pool;
    for (chunk = 0; chunk < chunks; ++chunk) chore(args, chunk, chunks, 0);
}
static int32_t ill_pool_size(const IllPool *pool) { (void)pool; return 1; }

#endif

/* Splits `total` rows across `chunks` slices and reports slice `chunk`. */
static void ill_span_split(int32_t total, int32_t chunk, int32_t chunks,
                           int32_t *head, int32_t *tail)
{
    int32_t base = total / chunks;
    int32_t over = total % chunks;
    int32_t from = chunk * base + (chunk < over ? chunk : over);
    int32_t span = base + (chunk < over ? 1 : 0);
    *head = from;
    *tail = from + span;
}

/* ============================================================================
 * part 3 -- number formats
 *
 * The engine keeps activations in f32 and weights in whichever of f32, f16,
 * bf16, or q8 the plan asked for.  Widening a stored value is exact in every
 * direction that matters, so a checkpoint held at half width and the same
 * checkpoint held at full width agree once loaded.
 * ==========================================================================*/

const char *ill_type_text(IllType type)
{
    switch (type) {
        case ILL_TYPE_KEEP: return "keep";
        case ILL_TYPE_F32:  return "f32";
        case ILL_TYPE_F16:  return "f16";
        case ILL_TYPE_BF16: return "bf16";
        case ILL_TYPE_Q8:   return "q8";
        default:            return "?";
    }
}

size_t ill_type_size(IllType type, size_t count)
{
    switch (type) {
        case ILL_TYPE_F32:  return count * 4;
        case ILL_TYPE_F16:
        case ILL_TYPE_BF16: return count * 2;
        case ILL_TYPE_Q8:   return count;   /* scales are held separately */
        default:            return 0;
    }
}

static float ill_bf16_cast(uint16_t bits)
{
    union { uint32_t u; float f; } cell;
    cell.u = (uint32_t)bits << 16;
    return cell.f;
}

ILL_SPARE static uint16_t ill_bf16_pack(float value)
{
    union { uint32_t u; float f; } cell;
    uint32_t bias;
    cell.f = value;
    if (((cell.u >> 23) & 0xFF) == 0xFF) return (uint16_t)(cell.u >> 16); /* nan/inf */
    bias = ((cell.u >> 16) & 1u) + 0x7FFFu;                              /* round to even */
    return (uint16_t)((cell.u + bias) >> 16);
}

static float ill_f16_cast(uint16_t bits)
{
    union { uint32_t u; float f; } cell;
    uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    uint32_t code = (uint32_t)(bits >> 10) & 0x1Fu;
    uint32_t frac = (uint32_t)(bits & 0x03FFu);
    if (code == 0) {
        if (frac == 0) { cell.u = sign; return cell.f; }
        /* subnormal: renormalise by hand */
        while (!(frac & 0x0400u)) { frac <<= 1; --code; }
        ++code;
        frac &= 0x03FFu;
    } else if (code == 0x1F) {
        cell.u = sign | 0x7F800000u | (frac << 13);
        return cell.f;
    }
    cell.u = sign | ((code + 112u) << 23) | (frac << 13);
    return cell.f;
}

ILL_SPARE static uint16_t ill_f16_pack(float value)
{
    union { uint32_t u; float f; } cell;
    uint32_t sign, rest;
    int32_t  code;
    cell.f = value;
    sign = (cell.u >> 16) & 0x8000u;
    rest = cell.u & 0x7FFFFFFFu;
    if (rest >= 0x7F800000u)                     /* nan or inf */
        return (uint16_t)(sign | 0x7C00u | (rest > 0x7F800000u ? 0x200u : 0u));
    code = (int32_t)(rest >> 23) - 127 + 15;
    if (code >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    if (code <= 0) {
        uint32_t frac;
        if (code < -10) return (uint16_t)sign;
        frac = (rest & 0x007FFFFFu) | 0x00800000u;
        frac >>= (uint32_t)(1 - code);
        return (uint16_t)(sign | ((frac + 0x00001000u) >> 13));
    }
    return (uint16_t)(sign | (uint32_t)((((code << 23) | (int32_t)(rest & 0x007FFFFFu))
                                         + 0x00001000) >> 13));
}

/* ============================================================================
 * part 4 -- vector vocabulary
 *
 * Every kernel in this engine is written once, against the handful of
 * operations below -- zero, widen, load, store, fuse, reduce -- plus a loader
 * per stored format.  The vocabulary is instantiated at whatever width the
 * target offers: 16 for AVX-512, 8 for AVX2, 4 for NEON, and 1 for a plain C
 * build.  The width-1 instantiation is the reference: it is the same source,
 * so a scalar host and a vector host agree by construction rather than by
 * review.
 *
 * The build picks a width at compile time.  util/make.py probes the host and
 * passes the matching flags; `make.py build --portable` pins the baseline.
 * ==========================================================================*/

#if defined(ILL_ARCH_X86) && defined(__AVX512F__) && !defined(ILL_NO_SIMD)
#  define ILL_SIMD_NAME "avx512"
#  define ILL_VW 16
#  include <immintrin.h>
typedef __m512 IllVec;
static inline IllVec ill_vec_zero(void)                        { return _mm512_setzero_ps(); }
static inline IllVec ill_vec_wide(float v)                     { return _mm512_set1_ps(v); }
static inline IllVec ill_vec_load(const float *p)              { return _mm512_loadu_ps(p); }
static inline void   ill_vec_save(float *p, IllVec v)          { _mm512_storeu_ps(p, v); }
static inline IllVec ill_vec_add(IllVec a, IllVec b)           { return _mm512_add_ps(a, b); }
static inline IllVec ill_vec_mul(IllVec a, IllVec b)           { return _mm512_mul_ps(a, b); }
static inline IllVec ill_vec_fma(IllVec a, IllVec b, IllVec c) { return _mm512_fmadd_ps(a, b, c); }
static inline float  ill_vec_sum(IllVec v)                     { return _mm512_reduce_add_ps(v); }
static inline IllVec ill_vec_abs(IllVec v)                      { return _mm512_abs_ps(v); }
static inline IllVec ill_vec_max(IllVec a, IllVec b)           { return _mm512_max_ps(a, b); }
static inline float  ill_vec_top(IllVec v)                     { return _mm512_reduce_max_ps(v); }
static inline IllVec ill_vec_bf16(const uint16_t *p)
{
    __m512i wide = _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)p)), 16);
    return _mm512_castsi512_ps(wide);
}
static inline IllVec ill_vec_f16(const uint16_t *p)
{
    return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p));
}

#elif defined(ILL_ARCH_X86) && defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER)) && !defined(ILL_NO_SIMD)
#  define ILL_SIMD_NAME "avx2"
#  define ILL_VW 8
#  include <immintrin.h>
typedef __m256 IllVec;
static inline IllVec ill_vec_zero(void)                        { return _mm256_setzero_ps(); }
static inline IllVec ill_vec_wide(float v)                     { return _mm256_set1_ps(v); }
static inline IllVec ill_vec_load(const float *p)              { return _mm256_loadu_ps(p); }
static inline void   ill_vec_save(float *p, IllVec v)          { _mm256_storeu_ps(p, v); }
static inline IllVec ill_vec_add(IllVec a, IllVec b)           { return _mm256_add_ps(a, b); }
static inline IllVec ill_vec_mul(IllVec a, IllVec b)           { return _mm256_mul_ps(a, b); }
static inline IllVec ill_vec_fma(IllVec a, IllVec b, IllVec c) { return _mm256_fmadd_ps(a, b, c); }
static inline float  ill_vec_sum(IllVec v)
{
    __m128 half = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    half = _mm_add_ps(half, _mm_movehl_ps(half, half));
    half = _mm_add_ss(half, _mm_shuffle_ps(half, half, 0x55));
    return _mm_cvtss_f32(half);
}
static inline IllVec ill_vec_abs(IllVec v)
{
    return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v);
}
static inline IllVec ill_vec_max(IllVec a, IllVec b)           { return _mm256_max_ps(a, b); }
static inline float  ill_vec_top(IllVec v)
{
    __m128 half = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    half = _mm_max_ps(half, _mm_movehl_ps(half, half));
    half = _mm_max_ss(half, _mm_shuffle_ps(half, half, 0x55));
    return _mm_cvtss_f32(half);
}
static inline IllVec ill_vec_bf16(const uint16_t *p)
{
    __m256i wide = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)), 16);
    return _mm256_castsi256_ps(wide);
}
static inline IllVec ill_vec_f16(const uint16_t *p)
{
#if defined(__F16C__) || defined(_MSC_VER)
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
#else
    float cell[8];
    int   slot;
    for (slot = 0; slot < 8; ++slot) cell[slot] = ill_f16_cast(p[slot]);
    return _mm256_loadu_ps(cell);
#endif
}

#elif defined(ILL_ARCH_ARM) && defined(__ARM_NEON) && !defined(ILL_NO_SIMD)
#  define ILL_SIMD_NAME "neon"
#  define ILL_VW 4
#  include <arm_neon.h>
typedef float32x4_t IllVec;
static inline IllVec ill_vec_zero(void)                        { return vdupq_n_f32(0.0f); }
static inline IllVec ill_vec_wide(float v)                     { return vdupq_n_f32(v); }
static inline IllVec ill_vec_load(const float *p)              { return vld1q_f32(p); }
static inline void   ill_vec_save(float *p, IllVec v)          { vst1q_f32(p, v); }
static inline IllVec ill_vec_add(IllVec a, IllVec b)           { return vaddq_f32(a, b); }
static inline IllVec ill_vec_mul(IllVec a, IllVec b)           { return vmulq_f32(a, b); }
static inline IllVec ill_vec_fma(IllVec a, IllVec b, IllVec c) { return vfmaq_f32(c, a, b); }
static inline float  ill_vec_sum(IllVec v)
{
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float32x2_t half = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(half, half), 0);
#endif
}
static inline IllVec ill_vec_abs(IllVec v)                      { return vabsq_f32(v); }
static inline IllVec ill_vec_max(IllVec a, IllVec b)           { return vmaxq_f32(a, b); }
static inline float  ill_vec_top(IllVec v)
{
#if defined(__aarch64__)
    return vmaxvq_f32(v);
#else
    float32x2_t half = vmax_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpmax_f32(half, half), 0);
#endif
}
static inline IllVec ill_vec_bf16(const uint16_t *p)
{
    uint32x4_t wide = vshll_n_u16(vld1_u16(p), 16);
    return vreinterpretq_f32_u32(wide);
}
static inline IllVec ill_vec_f16(const uint16_t *p)
{
#if defined(__ARM_FP16_FORMAT_IEEE)
    return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p)));
#else
    float cell[4];
    int   slot;
    for (slot = 0; slot < 4; ++slot) cell[slot] = ill_f16_cast(p[slot]);
    return vld1q_f32(cell);
#endif
}

#else
#  define ILL_SIMD_NAME "scalar"
#  define ILL_VW 1
typedef float IllVec;
static inline IllVec ill_vec_zero(void)                        { return 0.0f; }
static inline IllVec ill_vec_wide(float v)                     { return v; }
static inline IllVec ill_vec_load(const float *p)              { return *p; }
static inline void   ill_vec_save(float *p, IllVec v)          { *p = v; }
static inline IllVec ill_vec_add(IllVec a, IllVec b)           { return a + b; }
static inline IllVec ill_vec_mul(IllVec a, IllVec b)           { return a * b; }
static inline IllVec ill_vec_fma(IllVec a, IllVec b, IllVec c) { return a * b + c; }
static inline float  ill_vec_sum(IllVec v)                     { return v; }
static inline IllVec ill_vec_abs(IllVec v)                      { return v < 0.0f ? -v : v; }
static inline IllVec ill_vec_max(IllVec a, IllVec b)           { return a > b ? a : b; }
static inline float  ill_vec_top(IllVec v)                     { return v; }
static inline IllVec ill_vec_bf16(const uint16_t *p)           { return ill_bf16_cast(*p); }
static inline IllVec ill_vec_f16(const uint16_t *p)            { return ill_f16_cast(*p); }
#endif

/* ============================================================================
 * part 5 -- block quantisation
 *
 * q8 is a symmetric per-block format: 32 signed bytes share one f32 scale, and
 * blocks run along the input dimension so a dot product consumes weight scales
 * and activation scales in the same stride order.  Integer products accumulate
 * into a float vector that is scaled per block, which leaves the only
 * horizontal reduction at the end of a row instead of the end of every block.
 * ==========================================================================*/

#define ILL_Q8_BLOCK 32

/* Number of q8 blocks needed to cover `count` values. */
static int32_t ill_q8_blocks(int32_t count)
{
    return (count + ILL_Q8_BLOCK - 1) / ILL_Q8_BLOCK;
}

/* Quantises one row of f32 into signed bytes plus per-block scales. */
static void ill_q8_pack(const float *src, int32_t count, int8_t *quant, float *scale)
{
    int32_t base;
    for (base = 0; base < count; base += ILL_Q8_BLOCK) {
        int32_t span = ILL_MIN(ILL_Q8_BLOCK, count - base);
        const float *cell = src + base;
        int8_t      *dest = quant + base;
        IllVec  wide = ill_vec_zero();
        float   peak = 0.0f, step, back;
        int32_t slot = 0;
        for (; slot + ILL_VW <= span; slot += ILL_VW)
            wide = ill_vec_max(wide, ill_vec_abs(ill_vec_load(cell + slot)));
        peak = ill_vec_top(wide);
        for (; slot < span; ++slot) {
            float mag = cell[slot] < 0.0f ? -cell[slot] : cell[slot];
            if (mag > peak) peak = mag;
        }
        step = peak / 127.0f;
        back = step > 0.0f ? 1.0f / step : 0.0f;
        scale[base / ILL_Q8_BLOCK] = step;
        /* Branch free so the compiler can widen this loop as well. */
        for (slot = 0; slot < span; ++slot) {
            float   tick = cell[slot] * back;
            float   snap = tick + (tick < 0.0f ? -0.5f : 0.5f);
            int32_t cast = (int32_t)snap;
            cast = cast >  127 ?  127 : cast;
            cast = cast < -127 ? -127 : cast;
            dest[slot] = (int8_t)cast;
        }
        for (; slot < ILL_Q8_BLOCK && base + slot < count; ++slot) dest[slot] = 0;
    }
}


/* -- q8 dot ----------------------------------------------------------------
 *
 * One block is 32 signed bytes with an f32 scale on each side.  Integer
 * products accumulate in a float vector scaled per block, so the reduction
 * happens once per row rather than once per block.
 * ------------------------------------------------------------------------*/

#if defined(ILL_ARCH_X86) && defined(__AVX512BW__) && defined(__AVX512F__) && !defined(ILL_NO_SIMD)
typedef __m512 IllQAcc;
static inline IllQAcc ill_q8_zero(void) { return _mm512_setzero_ps(); }
static inline float   ill_q8_fold(IllQAcc acc) { return _mm512_reduce_add_ps(acc); }
static inline IllQAcc ill_q8_step(IllQAcc acc, const int8_t *w, const int8_t *a, float step)
{
    /* One block is 32 bytes, which widens to exactly one 512 bit register of
     * int16, so a whole block costs a single multiply-add. */
    __m512i wv  = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)w));
    __m512i av  = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)a));
    __m512i tot = _mm512_madd_epi16(wv, av);
    return _mm512_fmadd_ps(_mm512_cvtepi32_ps(tot), _mm512_set1_ps(step), acc);
}

#elif defined(ILL_ARCH_X86) && defined(__AVX2__) && !defined(ILL_NO_SIMD)
typedef __m256 IllQAcc;
static inline IllQAcc ill_q8_zero(void) { return _mm256_setzero_ps(); }
static inline float   ill_q8_fold(IllQAcc acc)
{
    __m128 half = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    half = _mm_add_ps(half, _mm_movehl_ps(half, half));
    half = _mm_add_ss(half, _mm_shuffle_ps(half, half, 0x55));
    return _mm_cvtss_f32(half);
}
static inline IllQAcc ill_q8_step(IllQAcc acc, const int8_t *w, const int8_t *a, float step)
{
    __m256i wv = _mm256_loadu_si256((const __m256i *)w);
    __m256i av = _mm256_loadu_si256((const __m256i *)a);
    __m256i lo = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(wv)),
                                   _mm256_cvtepi8_epi16(_mm256_castsi256_si128(av)));
    __m256i hi = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(wv, 1)),
                                   _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av, 1)));
    __m256i tot = _mm256_add_epi32(lo, hi);
#if defined(__FMA__)
    return _mm256_fmadd_ps(_mm256_cvtepi32_ps(tot), _mm256_set1_ps(step), acc);
#else
    return _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(tot), _mm256_set1_ps(step)));
#endif
}

#elif defined(ILL_ARCH_ARM) && defined(__ARM_NEON) && !defined(ILL_NO_SIMD)
typedef float32x4_t IllQAcc;
static inline IllQAcc ill_q8_zero(void) { return vdupq_n_f32(0.0f); }
static inline float   ill_q8_fold(IllQAcc acc)
{
#if defined(__aarch64__)
    return vaddvq_f32(acc);
#else
    float32x2_t half = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    return vget_lane_f32(vpadd_f32(half, half), 0);
#endif
}
static inline IllQAcc ill_q8_step(IllQAcc acc, const int8_t *w, const int8_t *a, float step)
{
    int32x4_t tot = vdupq_n_s32(0);
    int32_t   half;
    for (half = 0; half < 32; half += 16) {
        int8x16_t wv = vld1q_s8(w + half);
        int8x16_t av = vld1q_s8(a + half);
        int16x8_t lo = vmull_s8(vget_low_s8(wv), vget_low_s8(av));
        int16x8_t hi = vmull_s8(vget_high_s8(wv), vget_high_s8(av));
        tot = vpadalq_s16(tot, lo);
        tot = vpadalq_s16(tot, hi);
    }
    return vfmaq_f32(acc, vcvtq_f32_s32(tot), vdupq_n_f32(step));
}

#else
typedef float IllQAcc;
static inline IllQAcc ill_q8_zero(void) { return 0.0f; }
static inline float   ill_q8_fold(IllQAcc acc) { return acc; }
static inline IllQAcc ill_q8_step(IllQAcc acc, const int8_t *w, const int8_t *a, float step)
{
    int32_t tot = 0, slot;
    for (slot = 0; slot < ILL_Q8_BLOCK; ++slot) tot += (int32_t)w[slot] * (int32_t)a[slot];
    return acc + (float)tot * step;
}
#endif

/* ============================================================================
 * part 6 -- json reader
 *
 * A compact, allocation-light DOM.  Every node is an index into one flat node
 * array, so the whole document frees with two calls.  Strings are decoded in
 * place into a scratch pool, with \uXXXX escapes folded to UTF-8.
 * ==========================================================================*/

typedef enum IllNodeKind {
    ILL_NODE_NULL = 0, ILL_NODE_TRUE, ILL_NODE_FALSE,
    ILL_NODE_NUMBER, ILL_NODE_STRING, ILL_NODE_ARRAY, ILL_NODE_OBJECT
} IllNodeKind;

typedef struct IllNode {
    uint8_t kind;
    int32_t name;    /* text offset of the key, object members only, else -1 */
    int32_t text;    /* text offset for strings                              */
    double  number;
    int32_t head;    /* first child, arrays and objects                      */
    int32_t next;    /* next sibling                                         */
    int32_t count;   /* child count                                          */
} IllNode;

typedef struct IllJson {
    IllNode *nodes;
    int32_t  count;
    int32_t  limit;
    char    *text;
    size_t   used;
    size_t   room;
    const char *scan;
    const char *stop;
    int      fault;
} IllJson;

static void ill_json_free(IllJson *doc)
{
    if (!doc) return;
    ill_block_free(doc->nodes);
    ill_block_free(doc->text);
    memset(doc, 0, sizeof(*doc));
}

static int32_t ill_json_node(IllJson *doc, IllNodeKind kind)
{
    IllNode *slot;
    if (doc->count == doc->limit) {
        int32_t  grown = doc->limit ? doc->limit * 2 : 256;
        IllNode *fresh = (IllNode *)ill_block_make((size_t)grown * sizeof(IllNode));
        if (!fresh) { doc->fault = 1; return -1; }
        if (doc->nodes) memcpy(fresh, doc->nodes, (size_t)doc->count * sizeof(IllNode));
        ill_block_free(doc->nodes);
        doc->nodes = fresh;
        doc->limit = grown;
    }
    slot = &doc->nodes[doc->count];
    memset(slot, 0, sizeof(*slot));
    slot->kind = (uint8_t)kind;
    slot->name = -1;
    slot->text = -1;
    slot->head = -1;
    slot->next = -1;
    return doc->count++;
}

static int32_t ill_json_room(IllJson *doc, size_t need)
{
    if (doc->used + need > doc->room) {
        size_t grown = doc->room ? doc->room * 2 : 4096;
        char  *fresh;
        while (grown < doc->used + need) grown *= 2;
        fresh = (char *)ill_block_make(grown);
        if (!fresh) { doc->fault = 1; return -1; }
        if (doc->text) memcpy(fresh, doc->text, doc->used);
        ill_block_free(doc->text);
        doc->text = fresh;
        doc->room = grown;
    }
    return (int32_t)doc->used;
}

static void ill_json_skip(IllJson *doc)
{
    while (doc->scan < doc->stop) {
        char cell = *doc->scan;
        if (cell == ' ' || cell == '\t' || cell == '\n' || cell == '\r') ++doc->scan;
        else break;
    }
}

/* Appends one code point as UTF-8. */
static void ill_json_rune(IllJson *doc, uint32_t rune)
{
    char cell[4];
    int  span = 0;
    if (rune < 0x80u) { cell[span++] = (char)rune; }
    else if (rune < 0x800u) {
        cell[span++] = (char)(0xC0u | (rune >> 6));
        cell[span++] = (char)(0x80u | (rune & 0x3Fu));
    } else if (rune < 0x10000u) {
        cell[span++] = (char)(0xE0u | (rune >> 12));
        cell[span++] = (char)(0x80u | ((rune >> 6) & 0x3Fu));
        cell[span++] = (char)(0x80u | (rune & 0x3Fu));
    } else {
        cell[span++] = (char)(0xF0u | (rune >> 18));
        cell[span++] = (char)(0x80u | ((rune >> 12) & 0x3Fu));
        cell[span++] = (char)(0x80u | ((rune >> 6) & 0x3Fu));
        cell[span++] = (char)(0x80u | (rune & 0x3Fu));
    }
    if (ill_json_room(doc, (size_t)span) < 0) return;
    memcpy(doc->text + doc->used, cell, (size_t)span);
    doc->used += (size_t)span;
}

static uint32_t ill_json_hex4(const char *scan)
{
    uint32_t code = 0;
    int      slot;
    for (slot = 0; slot < 4; ++slot) {
        char cell = scan[slot];
        code <<= 4;
        if (cell >= '0' && cell <= '9') code |= (uint32_t)(cell - '0');
        else if (cell >= 'a' && cell <= 'f') code |= (uint32_t)(cell - 'a' + 10);
        else if (cell >= 'A' && cell <= 'F') code |= (uint32_t)(cell - 'A' + 10);
    }
    return code;
}

/* Reads a quoted string and returns its offset in the text pool. */
static int32_t ill_json_text(IllJson *doc)
{
    int32_t base;
    if (doc->scan >= doc->stop || *doc->scan != '"') { doc->fault = 1; return -1; }
    ++doc->scan;
    base = ill_json_room(doc, 1);
    if (base < 0) return -1;
    while (doc->scan < doc->stop && *doc->scan != '"') {
        char cell = *doc->scan;
        if (cell == '\\') {
            ++doc->scan;
            if (doc->scan >= doc->stop) { doc->fault = 1; return -1; }
            cell = *doc->scan++;
            switch (cell) {
                case 'n': cell = '\n'; break;
                case 't': cell = '\t'; break;
                case 'r': cell = '\r'; break;
                case 'b': cell = '\b'; break;
                case 'f': cell = '\f'; break;
                case '/': case '\\': case '"': break;
                case 'u': {
                    uint32_t rune;
                    if (doc->stop - doc->scan < 4) { doc->fault = 1; return -1; }
                    rune = ill_json_hex4(doc->scan);
                    doc->scan += 4;
                    if (rune >= 0xD800u && rune <= 0xDBFFu &&
                        doc->stop - doc->scan >= 6 &&
                        doc->scan[0] == '\\' && doc->scan[1] == 'u') {
                        uint32_t tail = ill_json_hex4(doc->scan + 2);
                        if (tail >= 0xDC00u && tail <= 0xDFFFu) {
                            rune = 0x10000u + ((rune - 0xD800u) << 10) + (tail - 0xDC00u);
                            doc->scan += 6;
                        }
                    }
                    ill_json_rune(doc, rune);
                    continue;
                }
                default: doc->fault = 1; return -1;
            }
        } else {
            ++doc->scan;
        }
        if (ill_json_room(doc, 1) < 0) return -1;
        doc->text[doc->used++] = cell;
    }
    if (doc->scan >= doc->stop) { doc->fault = 1; return -1; }
    ++doc->scan;
    if (ill_json_room(doc, 1) < 0) return -1;
    doc->text[doc->used++] = '\0';
    return base;
}

static int32_t ill_json_read(IllJson *doc);

static int32_t ill_json_pack(IllJson *doc, IllNodeKind kind, char open, char shut)
{
    int32_t self = ill_json_node(doc, kind);
    int32_t last = -1;
    int     owes = 0;      /* a comma promised one more member */
    if (self < 0) return -1;
    ++doc->scan; /* consume open */
    (void)open;
    for (;;) {
        int32_t item;
        int32_t name = -1;
        ill_json_skip(doc);
        if (doc->scan >= doc->stop) { doc->fault = 1; return -1; }
        if (*doc->scan == shut) {
            if (owes) { doc->fault = 1; return -1; }   /* no trailing commas */
            ++doc->scan;
            break;
        }
        if (kind == ILL_NODE_OBJECT) {
            name = ill_json_text(doc);
            if (doc->fault) return -1;
            ill_json_skip(doc);
            if (doc->scan >= doc->stop || *doc->scan != ':') { doc->fault = 1; return -1; }
            ++doc->scan;
        }
        item = ill_json_read(doc);
        if (item < 0) return -1;
        doc->nodes[item].name = name;
        if (last < 0) doc->nodes[self].head = item;
        else          doc->nodes[last].next = item;
        last = item;
        ++doc->nodes[self].count;
        ill_json_skip(doc);
        if (doc->scan < doc->stop && *doc->scan == ',') { ++doc->scan; owes = 1; continue; }
        if (doc->scan < doc->stop && *doc->scan == shut) { ++doc->scan; break; }
        doc->fault = 1;
        return -1;
    }
    return self;
}

static int32_t ill_json_read(IllJson *doc)
{
    ill_json_skip(doc);
    if (doc->scan >= doc->stop) { doc->fault = 1; return -1; }
    switch (*doc->scan) {
        case '{': return ill_json_pack(doc, ILL_NODE_OBJECT, '{', '}');
        case '[': return ill_json_pack(doc, ILL_NODE_ARRAY,  '[', ']');
        case '"': {
            int32_t self = ill_json_node(doc, ILL_NODE_STRING);
            int32_t text;
            if (self < 0) return -1;
            text = ill_json_text(doc);
            if (doc->fault) return -1;
            doc->nodes[self].text = text;
            return self;
        }
        case 't':
            if (doc->stop - doc->scan < 4 || memcmp(doc->scan, "true", 4)) { doc->fault = 1; return -1; }
            doc->scan += 4;
            return ill_json_node(doc, ILL_NODE_TRUE);
        case 'f':
            if (doc->stop - doc->scan < 5 || memcmp(doc->scan, "false", 5)) { doc->fault = 1; return -1; }
            doc->scan += 5;
            return ill_json_node(doc, ILL_NODE_FALSE);
        case 'n':
            if (doc->stop - doc->scan < 4 || memcmp(doc->scan, "null", 4)) { doc->fault = 1; return -1; }
            doc->scan += 4;
            return ill_json_node(doc, ILL_NODE_NULL);
        default: {
            char    cell[64];
            int32_t span = 0;
            int32_t self;
            while (doc->scan < doc->stop && span < 63) {
                char peek = *doc->scan;
                if ((peek >= '0' && peek <= '9') || peek == '-' || peek == '+' ||
                    peek == '.' || peek == 'e' || peek == 'E') cell[span++] = *doc->scan++;
                else break;
            }
            if (span == 0) { doc->fault = 1; return -1; }
            cell[span] = '\0';
            self = ill_json_node(doc, ILL_NODE_NUMBER);
            if (self < 0) return -1;
            doc->nodes[self].number = strtod(cell, NULL);
            return self;
        }
    }
}

static IllResult ill_json_load(IllJson *doc, const char *body, size_t span)
{
    int32_t root;
    memset(doc, 0, sizeof(*doc));
    doc->scan = body;
    doc->stop = body + span;
    root = ill_json_read(doc);
    if (root != 0 || doc->fault) { ill_json_free(doc); return ILL_PARSE; }
    return ILL_OK;
}

/* -- json lookups --------------------------------------------------------- */

static const char *ill_json_word(const IllJson *doc, int32_t offset)
{
    return offset >= 0 ? doc->text + offset : NULL;
}

static int32_t ill_json_pick(const IllJson *doc, int32_t node, const char *name)
{
    int32_t item;
    if (node < 0 || doc->nodes[node].kind != ILL_NODE_OBJECT) return -1;
    for (item = doc->nodes[node].head; item >= 0; item = doc->nodes[item].next) {
        const char *key = ill_json_word(doc, doc->nodes[item].name);
        if (key && strcmp(key, name) == 0) return item;
    }
    return -1;
}

static int32_t ill_json_item(const IllJson *doc, int32_t node, int32_t index)
{
    int32_t item;
    if (node < 0) return -1;
    for (item = doc->nodes[node].head; item >= 0; item = doc->nodes[item].next)
        if (index-- == 0) return item;
    return -1;
}

static double ill_json_real(const IllJson *doc, int32_t node, double spare)
{
    if (node < 0) return spare;
    if (doc->nodes[node].kind == ILL_NODE_NUMBER) return doc->nodes[node].number;
    if (doc->nodes[node].kind == ILL_NODE_TRUE)   return 1.0;
    if (doc->nodes[node].kind == ILL_NODE_FALSE)  return 0.0;
    return spare;
}

static int32_t ill_json_long(const IllJson *doc, int32_t node, int32_t spare)
{
    return node < 0 ? spare : (int32_t)ill_json_real(doc, node, (double)spare);
}

static const char *ill_json_str(const IllJson *doc, int32_t node, const char *spare)
{
    if (node < 0 || doc->nodes[node].kind != ILL_NODE_STRING) return spare;
    return ill_json_word(doc, doc->nodes[node].text);
}

/* Reads a numeric field from an object, following a fallback key if needed. */
static double ill_json_find(const IllJson *doc, int32_t node, const char *name, double spare)
{
    return ill_json_real(doc, ill_json_pick(doc, node, name), spare);
}

/* ============================================================================
 * part 7 -- safetensors store
 *
 * A store maps every shard of a checkpoint and indexes the tensors inside
 * them.  Nothing is copied: a slab points straight into the mapping, so a
 * checkpoint costs address space rather than resident memory until the loader
 * touches it.
 * ==========================================================================*/

#define ILL_RANK_MAX 4

typedef struct IllSlab {
    const char    *name;
    const uint8_t *cells;
    size_t         bytes;
    IllType        type;
    int32_t        rank;
    int64_t        dims[ILL_RANK_MAX];
} IllSlab;

typedef struct IllStore {
    IllFile *files;
    int32_t  file_count;
    IllJson *heads;        /* one parsed header per shard */
    IllSlab *slabs;
    int32_t  slab_count;
    int32_t  slab_limit;
    size_t   bytes;
} IllStore;

static IllType ill_store_type(const char *name)
{
    if (!name) return ILL_TYPE_COUNT;
    if (!strcmp(name, "F32")  || !strcmp(name, "float32"))  return ILL_TYPE_F32;
    if (!strcmp(name, "F16")  || !strcmp(name, "float16"))  return ILL_TYPE_F16;
    if (!strcmp(name, "BF16") || !strcmp(name, "bfloat16")) return ILL_TYPE_BF16;
    return ILL_TYPE_COUNT;
}

static IllResult ill_store_grow(IllStore *store, int32_t need)
{
    IllSlab *fresh;
    int32_t  grown;
    if (store->slab_count + need <= store->slab_limit) return ILL_OK;
    grown = store->slab_limit ? store->slab_limit : 64;
    while (grown < store->slab_count + need) grown *= 2;
    fresh = (IllSlab *)ill_block_make((size_t)grown * sizeof(IllSlab));
    if (!fresh) return ILL_ALLOC;
    if (store->slabs) memcpy(fresh, store->slabs, (size_t)store->slab_count * sizeof(IllSlab));
    ill_block_free(store->slabs);
    store->slabs = fresh;
    store->slab_limit = grown;
    return ILL_OK;
}

/* Maps one shard and folds its tensors into the store index. */
static IllResult ill_store_shard(IllStore *store, const char *path)
{
    IllFile *file = &store->files[store->file_count];
    IllJson *head = &store->heads[store->file_count];
    uint64_t span;
    IllResult code;
    int32_t   node;
    const uint8_t *base;

    code = ill_file_open(file, path);
    if (code != ILL_OK) { ill_note(1, "cannot open %s", path); return code; }
    if (file->size < 8) { ill_file_close(file); return ILL_PARSE; }
    memcpy(&span, file->data, 8);
    if (span == 0 || span + 8 > file->size) { ill_file_close(file); return ILL_PARSE; }
    code = ill_json_load(head, (const char *)file->data + 8, (size_t)span);
    if (code != ILL_OK) { ill_file_close(file); return code; }
    store->file_count++;
    base = file->data + 8 + span;

    for (node = head->nodes[0].head; node >= 0; node = head->nodes[node].next) {
        const char *name = ill_json_word(head, head->nodes[node].name);
        int32_t     kind, shape, marks, axis, item;
        IllSlab    *slab;
        uint64_t    from, upto;
        if (!name || !strcmp(name, "__metadata__")) continue;
        kind  = ill_json_pick(head, node, "dtype");
        shape = ill_json_pick(head, node, "shape");
        marks = ill_json_pick(head, node, "data_offsets");
        if (kind < 0 || shape < 0 || marks < 0) return ILL_PARSE;
        if (ill_store_grow(store, 1) != ILL_OK) return ILL_ALLOC;
        slab = &store->slabs[store->slab_count];
        memset(slab, 0, sizeof(*slab));
        slab->name = name;
        slab->type = ill_store_type(ill_json_str(head, kind, NULL));
        if (slab->type == ILL_TYPE_COUNT) {
            ill_note(1, "tensor %s uses unsupported dtype %s", name,
                     ill_json_str(head, kind, "?"));
            continue;   /* skip rather than fail: extras may be unused */
        }
        axis = 0;
        for (item = head->nodes[shape].head; item >= 0 && axis < ILL_RANK_MAX;
             item = head->nodes[item].next)
            slab->dims[axis++] = (int64_t)ill_json_real(head, item, 0.0);
        slab->rank = axis;
        from = (uint64_t)ill_json_real(head, ill_json_item(head, marks, 0), 0.0);
        upto = (uint64_t)ill_json_real(head, ill_json_item(head, marks, 1), 0.0);
        if (upto < from || 8 + span + upto > file->size) return ILL_PARSE;
        slab->cells = base + from;
        slab->bytes = (size_t)(upto - from);
        store->bytes += slab->bytes;
        store->slab_count++;
    }
    return ILL_OK;
}

static void ill_store_close(IllStore *store)
{
    int32_t index;
    if (!store) return;
    for (index = 0; index < store->file_count; ++index) {
        ill_json_free(&store->heads[index]);
        ill_file_close(&store->files[index]);
    }
    ill_block_free(store->files);
    ill_block_free(store->heads);
    ill_block_free(store->slabs);
    memset(store, 0, sizeof(*store));
}

static const IllSlab *ill_store_find(const IllStore *store, const char *name)
{
    int32_t index;
    for (index = 0; index < store->slab_count; ++index)
        if (!strcmp(store->slabs[index].name, name)) return &store->slabs[index];
    return NULL;
}

/* -- shard discovery ------------------------------------------------------ */

typedef struct IllList {
    char  **names;
    int32_t count;
    int32_t limit;
} IllList;

static void ill_list_free(IllList *list)
{
    int32_t index;
    for (index = 0; index < list->count; ++index) ill_block_free(list->names[index]);
    ill_block_free(list->names);
    memset(list, 0, sizeof(*list));
}

static int ill_list_push(IllList *list, const char *name)
{
    int32_t index;
    char   *copy;
    size_t  span;
    for (index = 0; index < list->count; ++index)
        if (!strcmp(list->names[index], name)) return 1;   /* already present */
    if (list->count == list->limit) {
        int32_t grown = list->limit ? list->limit * 2 : 8;
        char  **fresh = (char **)ill_block_make((size_t)grown * sizeof(char *));
        if (!fresh) return 0;
        if (list->names) memcpy(fresh, list->names, (size_t)list->count * sizeof(char *));
        ill_block_free(list->names);
        list->names = fresh;
        list->limit = grown;
    }
    span = strlen(name) + 1;
    copy = (char *)ill_block_make(span);
    if (!copy) return 0;
    memcpy(copy, name, span);
    list->names[list->count++] = copy;
    return 1;
}

static int ill_name_tail(const char *name, const char *tail)
{
    size_t a = strlen(name), b = strlen(tail);
    return a >= b && !strcmp(name + a - b, tail);
}

/* Collects *.safetensors leaf names inside a folder, sorted for determinism. */
static void ill_dir_scan(const char *dir, IllList *list)
{
#if defined(ILL_HOST_WINDOWS)
    char             glob[ILL_PATH_MAX];
    WIN32_FIND_DATAA found;
    HANDLE           walk;
    ill_path_join(glob, sizeof(glob), dir, "*.safetensors");
    walk = FindFirstFileA(glob, &found);
    if (walk == INVALID_HANDLE_VALUE) return;
    do {
        if (!(found.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
            ill_list_push(list, found.cFileName);
    } while (FindNextFileA(walk, &found));
    FindClose(walk);
#elif defined(ILL_HOST_POSIX)
    DIR *walk = opendir(dir);
    struct dirent *item;
    if (!walk) return;
    while ((item = readdir(walk)) != NULL)
        if (ill_name_tail(item->d_name, ".safetensors")) ill_list_push(list, item->d_name);
    closedir(walk);
#else
    (void)dir; (void)list;
#endif
    {
        int32_t a, b;
        for (a = 1; a < list->count; ++a)
            for (b = a; b > 0 && strcmp(list->names[b - 1], list->names[b]) > 0; --b) {
                char *swap = list->names[b - 1];
                list->names[b - 1] = list->names[b];
                list->names[b] = swap;
            }
    }
}

/* Opens every shard belonging to a checkpoint given a folder or a file. */
static IllResult ill_store_open(IllStore *store, const char *path)
{
    IllList   list;
    char      leaf[ILL_PATH_MAX];
    IllResult code = ILL_OK;
    int32_t   index;

    memset(store, 0, sizeof(*store));
    memset(&list, 0, sizeof(list));

    if (!ill_path_dir(path)) {
        if (!ill_file_here(path)) return ILL_FILE;
        store->files = (IllFile *)ill_block_zero(sizeof(IllFile));
        store->heads = (IllJson *)ill_block_zero(sizeof(IllJson));
        if (!store->files || !store->heads) { ill_store_close(store); return ILL_ALLOC; }
        code = ill_store_shard(store, path);
        if (code != ILL_OK) ill_store_close(store);
        return code;
    }

    ill_path_join(leaf, sizeof(leaf), path, "model.safetensors.index.json");
    if (ill_file_here(leaf)) {
        IllFile  book;
        IllJson  doc;
        int32_t  map, item;
        if (ill_file_open(&book, leaf) != ILL_OK) return ILL_FILE;
        code = ill_json_load(&doc, (const char *)book.data, book.size);
        if (code == ILL_OK) {
            map = ill_json_pick(&doc, 0, "weight_map");
            for (item = map >= 0 ? doc.nodes[map].head : -1; item >= 0;
                 item = doc.nodes[item].next) {
                const char *shard = ill_json_str(&doc, item, NULL);
                if (shard) ill_list_push(&list, shard);
            }
            ill_json_free(&doc);
        }
        ill_file_close(&book);
    }
    if (list.count == 0) {
        ill_path_join(leaf, sizeof(leaf), path, "model.safetensors");
        if (ill_file_here(leaf)) ill_list_push(&list, "model.safetensors");
    }
    if (list.count == 0) ill_dir_scan(path, &list);
    if (list.count == 0) return ILL_FILE;

    store->files = (IllFile *)ill_block_zero((size_t)list.count * sizeof(IllFile));
    store->heads = (IllJson *)ill_block_zero((size_t)list.count * sizeof(IllJson));
    if (!store->files || !store->heads) { ill_list_free(&list); ill_store_close(store); return ILL_ALLOC; }

    for (index = 0; index < list.count; ++index) {
        ill_path_join(leaf, sizeof(leaf), path, list.names[index]);
        code = ill_store_shard(store, leaf);
        if (code != ILL_OK) break;
    }
    ill_list_free(&list);
    if (code != ILL_OK) ill_store_close(store);
    return code;
}

/* ============================================================================
 * part 8 -- compute kernels
 *
 * The shapes of arithmetic the Liquid stack performs, written in the vector
 * vocabulary of part 4 and the block format of part 5.  Dense is the one that
 * decides throughput: a weight row is streamed once and fused against up to
 * four activation rows, which is what keeps prefill from re-reading the whole
 * model once per token.  The tile count is a literal at every instantiation,
 * so the accumulators stay in registers.
 * ==========================================================================*/

/* -- a linear weight ------------------------------------------------------ */

typedef struct IllPlane {
    const void  *cells;   /* rows * cols values in `type`                    */
    const float *steps;   /* q8 only: rows * blocks scales                   */
    IllType      type;
    int32_t      rows;    /* output features                                 */
    int32_t      cols;    /* input features                                  */
    int32_t      blocks;  /* q8 only: scales per row                         */
} IllPlane;

/* -- dense: y = W x, tiled over activation rows ---------------------------
 *
 * A weight row is streamed once and fused against up to four activation rows,
 * which is what keeps prefill from re-reading the whole model per token.  The
 * tile count is a literal at every instantiation so the accumulators live in
 * registers.
 * ------------------------------------------------------------------------*/

#define ILL_TILE_MAX 4

#define ILL_DENSE_BODY(N, LOADW, CTYPE, CASTW)                                        \
    do {                                                                              \
        int32_t r;                                                                    \
        for (r = r0; r < r1; ++r) {                                                   \
            const CTYPE *w = (const CTYPE *)plane->cells + (size_t)r * (size_t)cols;  \
            IllVec  acc[N];                                                           \
            float   sum[N];                                                           \
            int32_t t, j;                                                             \
            for (t = 0; t < N; ++t) acc[t] = ill_vec_zero();                          \
            for (j = 0; j + ILL_VW <= cols; j += ILL_VW) {                            \
                IllVec wv = LOADW(w + j);                                             \
                for (t = 0; t < N; ++t)                                               \
                    acc[t] = ill_vec_fma(wv, ill_vec_load(x + (size_t)t * xstep + j), \
                                         acc[t]);                                     \
            }                                                                         \
            for (t = 0; t < N; ++t) sum[t] = ill_vec_sum(acc[t]);                     \
            for (; j < cols; ++j) {                                                   \
                float wf = CASTW(w[j]);                                               \
                for (t = 0; t < N; ++t) sum[t] += wf * x[(size_t)t * xstep + j];      \
            }                                                                         \
            for (t = 0; t < N; ++t) y[(size_t)t * ystep + r] = sum[t];                \
        }                                                                             \
    } while (0)

#define ILL_CAST_KEEP(v) (v)

#define ILL_DENSE_TYPED(N)                                                            \
    switch (plane->type) {                                                            \
        case ILL_TYPE_F32:                                                            \
            ILL_DENSE_BODY(N, ill_vec_load, float, ILL_CAST_KEEP); break;             \
        case ILL_TYPE_BF16:                                                           \
            ILL_DENSE_BODY(N, ill_vec_bf16, uint16_t, ill_bf16_cast); break;          \
        case ILL_TYPE_F16:                                                            \
            ILL_DENSE_BODY(N, ill_vec_f16, uint16_t, ill_f16_cast); break;            \
        default: break;                                                               \
    }

static void ill_dense_real(const IllPlane *plane, const float *x, size_t xstep,
                           int32_t tiles, int32_t r0, int32_t r1,
                           float *y, size_t ystep)
{
    const int32_t cols = plane->cols;
    switch (tiles) {
        case 1:  ILL_DENSE_TYPED(1); break;
        case 2:  ILL_DENSE_TYPED(2); break;
        case 3:  ILL_DENSE_TYPED(3); break;
        default: ILL_DENSE_TYPED(4); break;
    }
}

#define ILL_DENSE_Q8_BODY(N)                                                          \
    do {                                                                              \
        int32_t r;                                                                    \
        for (r = r0; r < r1; ++r) {                                                   \
            const int8_t *w  = (const int8_t *)plane->cells + (size_t)r * (size_t)cols; \
            const float  *ws = plane->steps + (size_t)r * (size_t)blocks;             \
            IllQAcc acc[N];                                                           \
            int32_t t, b;                                                             \
            for (t = 0; t < N; ++t) acc[t] = ill_q8_zero();                           \
            for (b = 0; b + 1 <= full; ++b) {                                         \
                const int8_t *wb = w + (size_t)b * ILL_Q8_BLOCK;                      \
                for (t = 0; t < N; ++t)                                               \
                    acc[t] = ill_q8_step(acc[t],                                      \
                                wb, xq + (size_t)t * qstep + (size_t)b * ILL_Q8_BLOCK, \
                                ws[b] * xs[(size_t)t * sstep + b]);                   \
            }                                                                         \
            for (t = 0; t < N; ++t) {                                                 \
                float sum = ill_q8_fold(acc[t]);                                      \
                for (b = full; b < blocks; ++b) {                                     \
                    int32_t base = b * ILL_Q8_BLOCK;                                  \
                    int32_t span = ILL_MIN(ILL_Q8_BLOCK, cols - base);                \
                    int32_t tot = 0, slot;                                            \
                    for (slot = 0; slot < span; ++slot)                               \
                        tot += (int32_t)w[base + slot] *                              \
                               (int32_t)xq[(size_t)t * qstep + base + slot];          \
                    sum += (float)tot * ws[b] * xs[(size_t)t * sstep + b];            \
                }                                                                     \
                y[(size_t)t * ystep + r] = sum;                                       \
            }                                                                         \
        }                                                                             \
    } while (0)

static void ill_dense_byte(const IllPlane *plane, const int8_t *xq, size_t qstep,
                           const float *xs, size_t sstep, int32_t tiles,
                           int32_t r0, int32_t r1, float *y, size_t ystep)
{
    const int32_t cols   = plane->cols;
    const int32_t blocks = plane->blocks;
    const int32_t full   = cols / ILL_Q8_BLOCK;   /* whole blocks only */
    switch (tiles) {
        case 1:  ILL_DENSE_Q8_BODY(1); break;
        case 2:  ILL_DENSE_Q8_BODY(2); break;
        case 3:  ILL_DENSE_Q8_BODY(3); break;
        default: ILL_DENSE_Q8_BODY(4); break;
    }
}

/* -- elementwise stages --------------------------------------------------- */

/* rms norm over `width`, matching the reference: reduce in f32, scale, gain. */
static void ill_norm_rms(const float *src, const float *gain, float *dst,
                         int32_t width, float eps)
{
    IllVec  wide = ill_vec_zero();
    float   mass = 0.0f, scale;
    int32_t j;
    for (j = 0; j + ILL_VW <= width; j += ILL_VW) {
        IllVec cell = ill_vec_load(src + j);
        wide = ill_vec_fma(cell, cell, wide);
    }
    mass = ill_vec_sum(wide);
    for (; j < width; ++j) mass += src[j] * src[j];
    scale = 1.0f / sqrtf(mass / (float)width + eps);
    for (j = 0; j + ILL_VW <= width; j += ILL_VW)
        ill_vec_save(dst + j, ill_vec_mul(ill_vec_wide(scale),
                     ill_vec_mul(ill_vec_load(src + j), ill_vec_load(gain + j))));
    for (; j < width; ++j) dst[j] = src[j] * scale * gain[j];
}

static float ill_silu_one(float v) { return v / (1.0f + expf(-v)); }

/* dst = silu(gate) * lift, in place friendly. */
static void ill_glue_swi(const float *gate, const float *lift, float *dst, int32_t width)
{
    int32_t j;
    for (j = 0; j < width; ++j) dst[j] = ill_silu_one(gate[j]) * lift[j];
}

static void ill_soft_max(float *cells, int32_t width)
{
    float   peak = -FLT_MAX, mass = 0.0f, back;
    int32_t j;
    for (j = 0; j < width; ++j) if (cells[j] > peak) peak = cells[j];
    for (j = 0; j < width; ++j) { cells[j] = expf(cells[j] - peak); mass += cells[j]; }
    back = mass > 0.0f ? 1.0f / mass : 0.0f;
    for (j = 0; j < width; ++j) cells[j] *= back;
}

/* dst += weight * src, the attention value mix and the conv tap. */
static void ill_axpy_add(float *dst, const float *src, float weight, int32_t width)
{
    IllVec  wide = ill_vec_wide(weight);
    int32_t j;
    for (j = 0; j + ILL_VW <= width; j += ILL_VW)
        ill_vec_save(dst + j, ill_vec_fma(wide, ill_vec_load(src + j), ill_vec_load(dst + j)));
    for (; j < width; ++j) dst[j] += weight * src[j];
}

static float ill_dot_real(const float *a, const float *b, int32_t width)
{
    IllVec  wide = ill_vec_zero();
    float   sum;
    int32_t j;
    for (j = 0; j + ILL_VW <= width; j += ILL_VW)
        wide = ill_vec_fma(ill_vec_load(a + j), ill_vec_load(b + j), wide);
    sum = ill_vec_sum(wide);
    for (; j < width; ++j) sum += a[j] * b[j];
    return sum;
}

/* Reads one row out of a weight plane into f32, whatever the stored format. */
static void ill_plane_row(const IllPlane *plane, int32_t row, float *dst)
{
    const int32_t cols = plane->cols;
    int32_t j;
    switch (plane->type) {
        case ILL_TYPE_F32:
            memcpy(dst, (const float *)plane->cells + (size_t)row * cols,
                   (size_t)cols * sizeof(float));
            break;
        case ILL_TYPE_BF16: {
            const uint16_t *w = (const uint16_t *)plane->cells + (size_t)row * cols;
            for (j = 0; j < cols; ++j) dst[j] = ill_bf16_cast(w[j]);
            break;
        }
        case ILL_TYPE_F16: {
            const uint16_t *w = (const uint16_t *)plane->cells + (size_t)row * cols;
            for (j = 0; j < cols; ++j) dst[j] = ill_f16_cast(w[j]);
            break;
        }
        case ILL_TYPE_Q8: {
            const int8_t *w  = (const int8_t *)plane->cells + (size_t)row * cols;
            const float  *ws = plane->steps + (size_t)row * plane->blocks;
            for (j = 0; j < cols; ++j) dst[j] = (float)w[j] * ws[j / ILL_Q8_BLOCK];
            break;
        }
        default: memset(dst, 0, (size_t)cols * sizeof(float)); break;
    }
}

/* ============================================================================
 * part 9 -- backend seam
 *
 * The forward pass never calls a kernel directly.  It calls through an
 * IllBackend, a small table of the six shapes of work the Liquid stack
 * actually performs.  The cpu backend below fills that table with the kernels
 * from part 6 spread over a thread pool; an accelerator fills the same table
 * with device kernels and keeps its queue in `inner`.  Nothing above this line
 * needs to change for that to happen.
 * ==========================================================================*/

typedef struct IllPad {           /* per sequence scratch handed to a backend */
    int8_t *quant;                /* activation bytes, rows * cols            */
    float  *steps;                /* activation scales, rows * blocks         */
    int32_t rows;
    int32_t cols;
} IllPad;

typedef struct IllHeedJob {
    const float *query;   /* tokens * heads * width                          */
    const float *keys;    /* groups * span * width                           */
    const float *vals;    /* groups * span * width                           */
    float       *value;   /* tokens * heads * width, the mixed output        */
    float       *board;   /* workers * span scratch for scores               */
    int32_t      tokens, heads, groups, width, span, base;
    float        scale;
} IllHeedJob;

typedef struct IllFlowJob {
    const float *src;     /* tokens * dim                                    */
    float       *dst;     /* tokens * dim                                    */
    float       *hist;    /* dim * (width - 1) rolling state                 */
    const float *taps;    /* dim * width                                     */
    const float *bias;    /* dim, or NULL                                    */
    int32_t      tokens, dim, width;
} IllFlowJob;

struct IllBackend {
    const char *name;
    IllResult (*setup)(IllBackend *self, int32_t threads);
    void      (*close)(IllBackend *self);
    int32_t   (*width)(const IllBackend *self);   /* lanes of parallelism    */
    void      (*dense)(IllBackend *self, const IllPlane *plane, const float *src,
                       float *dst, int32_t tokens, IllPad *pad);
    void      (*rmsnorm)(IllBackend *self, const float *src, const float *gain,
                         float *dst, int32_t tokens, int32_t width, float eps);
    void      (*swiglu)(IllBackend *self, const float *gate, const float *lift,
                        float *dst, int32_t tokens, int32_t width);
    void      (*rope)(IllBackend *self, float *cells, const float *table,
                      int32_t tokens, int32_t heads, int32_t width, int32_t stride);
    void      (*attend)(IllBackend *self, const IllHeedJob *job);
    void      (*conv1d)(IllBackend *self, const IllFlowJob *job);
    void      *inner;
};

/* -- cpu backend ---------------------------------------------------------- */

typedef struct IllCpu {
    IllPool *pool;
    int32_t  lanes;
} IllCpu;

static IllResult ill_cpu_setup(IllBackend *self, int32_t threads)
{
    IllCpu   *cpu;
    IllResult code;
    if (threads <= 0) threads = ill_cpu_count();
    cpu = (IllCpu *)ill_block_zero(sizeof(IllCpu));
    if (!cpu) return ILL_ALLOC;
    code = ill_pool_make(&cpu->pool, threads);
    if (code != ILL_OK) { ill_block_free(cpu); return code; }
    cpu->lanes = ill_pool_size(cpu->pool);
    self->inner = cpu;
    return ILL_OK;
}

static void ill_cpu_close(IllBackend *self)
{
    IllCpu *cpu = (IllCpu *)self->inner;
    if (!cpu) return;
    ill_pool_free(cpu->pool);
    ill_block_free(cpu);
    self->inner = NULL;
}

static int32_t ill_cpu_width(const IllBackend *self)
{
    const IllCpu *cpu = (const IllCpu *)self->inner;
    return cpu ? cpu->lanes : 1;
}

/* dense ------------------------------------------------------------------ */

typedef struct IllDenseArgs {
    const IllPlane *plane;
    const float    *src;
    const int8_t   *quant;
    const float    *steps;
    float          *dst;
    int32_t         tokens;
} IllDenseArgs;

static void ill_cpu_dense_chore(void *args, int32_t chunk, int32_t chunks, int32_t worker)
{
    IllDenseArgs *job   = (IllDenseArgs *)args;
    const IllPlane *pl  = job->plane;
    int32_t r0, r1, tile;
    (void)worker;
    ill_span_split(pl->rows, chunk, chunks, &r0, &r1);
    if (r0 >= r1) return;
    for (tile = 0; tile < job->tokens; tile += ILL_TILE_MAX) {
        int32_t span = ILL_MIN(ILL_TILE_MAX, job->tokens - tile);
        if (pl->type == ILL_TYPE_Q8)
            ill_dense_byte(pl, job->quant + (size_t)tile * pl->cols, (size_t)pl->cols,
                           job->steps + (size_t)tile * pl->blocks, (size_t)pl->blocks,
                           span, r0, r1, job->dst + (size_t)tile * pl->rows,
                           (size_t)pl->rows);
        else
            ill_dense_real(pl, job->src + (size_t)tile * pl->cols, (size_t)pl->cols,
                           span, r0, r1, job->dst + (size_t)tile * pl->rows,
                           (size_t)pl->rows);
    }
}

/* Fork threshold: below this many multiply-adds the handshake costs more than
 * the work it spreads. */
#define ILL_FORK_FLOOR 65536

static void ill_cpu_dense(IllBackend *self, const IllPlane *plane, const float *src,
                          float *dst, int32_t tokens, IllPad *pad)
{
    IllCpu      *cpu = (IllCpu *)self->inner;
    IllDenseArgs job;
    int64_t      load = (int64_t)plane->rows * plane->cols * tokens;
    int32_t      chunks;

    job.plane  = plane;
    job.src    = src;
    job.dst    = dst;
    job.tokens = tokens;
    job.quant  = NULL;
    job.steps  = NULL;

    if (plane->type == ILL_TYPE_Q8) {
        int32_t row;
        for (row = 0; row < tokens; ++row)
            ill_q8_pack(src + (size_t)row * plane->cols, plane->cols,
                        pad->quant + (size_t)row * plane->cols,
                        pad->steps + (size_t)row * plane->blocks);
        job.quant = pad->quant;
        job.steps = pad->steps;
    }

    chunks = cpu ? cpu->lanes : 1;
    if (load < ILL_FORK_FLOOR || chunks < 2) chunks = 1;
    if (chunks > plane->rows) chunks = plane->rows;
    ill_pool_fork(cpu ? cpu->pool : NULL, ill_cpu_dense_chore, &job, chunks);
}

/* rms norm --------------------------------------------------------------- */

typedef struct IllNormArgs {
    const float *src;
    const float *gain;
    float       *dst;
    int32_t      tokens, width;
    float        eps;
} IllNormArgs;

static void ill_cpu_norm_chore(void *args, int32_t chunk, int32_t chunks, int32_t worker)
{
    IllNormArgs *job = (IllNormArgs *)args;
    int32_t t0, t1, t;
    (void)worker;
    ill_span_split(job->tokens, chunk, chunks, &t0, &t1);
    for (t = t0; t < t1; ++t)
        ill_norm_rms(job->src + (size_t)t * job->width, job->gain,
                     job->dst + (size_t)t * job->width, job->width, job->eps);
}

static void ill_cpu_rmsnorm(IllBackend *self, const float *src, const float *gain,
                            float *dst, int32_t tokens, int32_t width, float eps)
{
    IllCpu     *cpu = (IllCpu *)self->inner;
    IllNormArgs job;
    int32_t     chunks;
    job.src = src; job.gain = gain; job.dst = dst;
    job.tokens = tokens; job.width = width; job.eps = eps;
    chunks = cpu ? cpu->lanes : 1;
    if ((int64_t)tokens * width < ILL_FORK_FLOOR) chunks = 1;
    if (chunks > tokens) chunks = tokens;
    ill_pool_fork(cpu ? cpu->pool : NULL, ill_cpu_norm_chore, &job, chunks);
}

/* swiglu ----------------------------------------------------------------- */

typedef struct IllGlueArgs {
    const float *gate;
    const float *lift;
    float       *dst;
    int32_t      total;
} IllGlueArgs;

static void ill_cpu_glue_chore(void *args, int32_t chunk, int32_t chunks, int32_t worker)
{
    IllGlueArgs *job = (IllGlueArgs *)args;
    int32_t j0, j1;
    (void)worker;
    ill_span_split(job->total, chunk, chunks, &j0, &j1);
    ill_glue_swi(job->gate + j0, job->lift + j0, job->dst + j0, j1 - j0);
}

static void ill_cpu_swiglu(IllBackend *self, const float *gate, const float *lift,
                           float *dst, int32_t tokens, int32_t width)
{
    IllCpu     *cpu = (IllCpu *)self->inner;
    IllGlueArgs job;
    int32_t     chunks;
    job.gate = gate; job.lift = lift; job.dst = dst;
    job.total = tokens * width;
    chunks = cpu ? cpu->lanes : 1;
    if (job.total < ILL_FORK_FLOOR) chunks = 1;
    ill_pool_fork(cpu ? cpu->pool : NULL, ill_cpu_glue_chore, &job, chunks);
}

/* rope ------------------------------------------------------------------- */

/* `table` holds tokens * width floats: the first half of each row is cosine,
 * the second half sine, matching the reference layout of cat(freqs, freqs). */
static void ill_cpu_rope(IllBackend *self, float *cells, const float *table,
                         int32_t tokens, int32_t heads, int32_t width, int32_t stride)
{
    int32_t t, h, i;
    int32_t half = width / 2;
    (void)self;
    for (t = 0; t < tokens; ++t) {
        const float *cos = table + (size_t)t * width;
        const float *sin = cos + half;
        for (h = 0; h < heads; ++h) {
            float *vec = cells + (size_t)t * stride + (size_t)h * width;
            for (i = 0; i < half; ++i) {
                float lo = vec[i];
                float hi = vec[i + half];
                vec[i]        = lo * cos[i] - hi * sin[i];
                vec[i + half] = hi * cos[i] + lo * sin[i];
            }
        }
    }
}

/* attention -------------------------------------------------------------- */

static void ill_cpu_heed_chore(void *args, int32_t chunk, int32_t chunks, int32_t worker)
{
    const IllHeedJob *job = (const IllHeedJob *)args;
    int32_t units = job->tokens * job->heads;
    int32_t spread = job->heads / job->groups;
    int32_t u0, u1, unit;
    ill_span_split(units, chunk, chunks, &u0, &u1);
    for (unit = u0; unit < u1; ++unit) {
        int32_t t     = unit / job->heads;
        int32_t h     = unit % job->heads;
        int32_t g     = h / spread;
        int32_t reach = job->base + t + 1;          /* causal horizon */
        const float *q = job->query + (size_t)t * job->heads * job->width
                                    + (size_t)h * job->width;
        const float *k = job->keys + (size_t)g * job->span * job->width;
        const float *v = job->vals + (size_t)g * job->span * job->width;
        float *score = job->board + (size_t)worker * job->span;
        float *out   = job->value + (size_t)t * job->heads * job->width
                                  + (size_t)h * job->width;
        int32_t j;
        for (j = 0; j < reach; ++j)
            score[j] = ill_dot_real(q, k + (size_t)j * job->width, job->width) * job->scale;
        ill_soft_max(score, reach);
        memset(out, 0, (size_t)job->width * sizeof(float));
        for (j = 0; j < reach; ++j)
            ill_axpy_add(out, v + (size_t)j * job->width, score[j], job->width);
    }
}

static void ill_cpu_attend(IllBackend *self, const IllHeedJob *job)
{
    IllCpu *cpu = (IllCpu *)self->inner;
    int32_t units  = job->tokens * job->heads;
    int32_t chunks = cpu ? cpu->lanes : 1;
    if (chunks > units) chunks = units;
    ill_pool_fork(cpu ? cpu->pool : NULL, ill_cpu_heed_chore, (void *)job, chunks);
}

/* short convolution ------------------------------------------------------ */

static void ill_cpu_flow_chore(void *args, int32_t chunk, int32_t chunks, int32_t worker)
{
    const IllFlowJob *job = (const IllFlowJob *)args;
    int32_t keep = job->width - 1;
    int32_t c0, c1, c;
    (void)worker;
    ill_span_split(job->dim, chunk, chunks, &c0, &c1);
    for (c = c0; c < c1; ++c) {
        float  ring[16];
        float *hist = job->hist + (size_t)c * keep;
        const float *taps = job->taps + (size_t)c * job->width;
        float  seed = job->bias ? job->bias[c] : 0.0f;
        int32_t t, j;
        for (j = 0; j < keep; ++j) ring[j] = hist[j];
        for (t = 0; t < job->tokens; ++t) {
            float acc = seed;
            ring[keep] = job->src[(size_t)t * job->dim + c];
            for (j = 0; j < job->width; ++j) acc += taps[j] * ring[j];
            job->dst[(size_t)t * job->dim + c] = acc;
            for (j = 0; j < keep; ++j) ring[j] = ring[j + 1];
        }
        for (j = 0; j < keep; ++j) hist[j] = ring[j];
    }
}

static void ill_cpu_conv1d(IllBackend *self, const IllFlowJob *job)
{
    IllCpu *cpu = (IllCpu *)self->inner;
    int32_t chunks = cpu ? cpu->lanes : 1;
    if ((int64_t)job->dim * job->tokens * job->width < ILL_FORK_FLOOR) chunks = 1;
    if (chunks > job->dim) chunks = job->dim;
    ill_pool_fork(cpu ? cpu->pool : NULL, ill_cpu_flow_chore, (void *)job, chunks);
}

/* -- registry ------------------------------------------------------------- */

static IllBackend ill_backend_cpu = {
    "cpu",
    ill_cpu_setup, ill_cpu_close, ill_cpu_width,
    ill_cpu_dense, ill_cpu_rmsnorm, ill_cpu_swiglu,
    ill_cpu_rope, ill_cpu_attend, ill_cpu_conv1d,
    NULL
};

#define ILL_BACKEND_MAX 8
static IllBackend *ill_backend_slot[ILL_BACKEND_MAX] = { &ill_backend_cpu, NULL };
static int32_t     ill_backend_used = 1;

/* Registers an accelerator.  Call before ill_model_load. */
IllResult ill_backend_join(IllBackend *backend)
{
    if (!backend || !backend->name || !backend->dense) return ILL_ARGS;
    if (ill_backend_used >= ILL_BACKEND_MAX) return ILL_LIMIT;
    ill_backend_slot[ill_backend_used++] = backend;
    return ILL_OK;
}

int32_t ill_backend_list(const char **names, int32_t limit)
{
    int32_t index;
    for (index = 0; index < ill_backend_used && index < limit; ++index)
        names[index] = ill_backend_slot[index]->name;
    return ill_backend_used;
}

static IllBackend *ill_backend_find(const char *name)
{
    int32_t index;
    if (!name || !*name) return &ill_backend_cpu;
    for (index = 0; index < ill_backend_used; ++index)
        if (!strcmp(ill_backend_slot[index]->name, name)) return ill_backend_slot[index];
    return NULL;
}

const char *ill_backend_simd(void) { return ILL_SIMD_NAME; }

/* ============================================================================
 * part 10 -- model load
 *
 * A checkpoint becomes an IllModel in three passes: read config.json into an
 * IllArch, bind every tensor into an IllPlane, and optionally repack the large
 * planes into q8.  Small tensors -- norm gains, convolution taps -- are widened
 * to f32 once at load, which keeps the kernels free of format branches on the
 * paths where the branch would cost more than the storage saves.
 * ==========================================================================*/

typedef struct IllBlock {
    uint8_t      kind;         /* IllLayer                                   */
    /* attention */
    IllPlane     query, keyed, value, joint;
    const float *query_gain, *keyed_gain;
    /* short convolution */
    IllPlane     mixin, mixout;
    const float *taps, *tilt;
    /* feed forward */
    IllPlane     gate, fall, rise;   /* w1, w2, w3 of the swiglu    */
    /* norms */
    const float *op_gain, *ff_gain;
    int32_t      slot;         /* index among layers of the same kind        */
} IllBlock;

struct IllModel {
    IllArch     arch;
    IllStore    store;
    IllBackend *backend;
    IllBlock   *blocks;
    IllPlane    embed;         /* vocab x model_dim                          */
    IllPlane    crown;         /* output projection, tied to embed when bound */
    const float *out_gain;
    IllVocab   *vocab;
    IllType     weight_type;
    int32_t     batch_span;
    size_t      bytes;
    void      **owned;
    int32_t     owned_count, owned_limit;
};

static void *ill_model_own(IllModel *model, void *block)
{
    if (!block) return NULL;
    if (model->owned_count == model->owned_limit) {
        int32_t grown = model->owned_limit ? model->owned_limit * 2 : 64;
        void  **fresh = (void **)ill_block_make((size_t)grown * sizeof(void *));
        if (!fresh) { ill_block_free(block); return NULL; }
        if (model->owned) memcpy(fresh, model->owned, (size_t)model->owned_count * sizeof(void *));
        ill_block_free(model->owned);
        model->owned = fresh;
        model->owned_limit = grown;
    }
    model->owned[model->owned_count++] = block;
    return block;
}

/* Locates a tensor, retrying without the "model." prefix that a bare
 * Lfm2Model checkpoint omits. */
static const IllSlab *ill_model_slab(const IllStore *store, const char *name)
{
    const IllSlab *slab = ill_store_find(store, name);
    if (slab) return slab;
    if (!strncmp(name, "model.", 6)) return ill_store_find(store, name + 6);
    return NULL;
}

static const IllSlab *ill_model_pick(const IllStore *store, const char *form, ...)
{
    char    name[256];
    va_list args;
    va_start(args, form);
    vsnprintf(name, sizeof(name), form, args);
    va_end(args);
    return ill_model_slab(store, name);
}

/* Widens a small tensor to an owned f32 vector. */
static const float *ill_model_vec(IllModel *model, const IllSlab *slab, int32_t count)
{
    float  *cells;
    int32_t j;
    if (!slab) return NULL;
    cells = (float *)ill_block_make((size_t)count * sizeof(float));
    if (!cells) return NULL;
    switch (slab->type) {
        case ILL_TYPE_F32:
            memcpy(cells, slab->cells, (size_t)count * sizeof(float));
            break;
        case ILL_TYPE_BF16:
            for (j = 0; j < count; ++j)
                cells[j] = ill_bf16_cast(((const uint16_t *)slab->cells)[j]);
            break;
        case ILL_TYPE_F16:
            for (j = 0; j < count; ++j)
                cells[j] = ill_f16_cast(((const uint16_t *)slab->cells)[j]);
            break;
        default:
            ill_block_free(cells);
            return NULL;
    }
    return (const float *)ill_model_own(model, cells);
}

/* -- q8 repack ------------------------------------------------------------ */

typedef struct IllPackArgs {
    const IllPlane *from;
    int8_t         *quant;
    float          *steps;
    int32_t         blocks;
} IllPackArgs;

static void ill_model_pack_chore(void *args, int32_t chunk, int32_t chunks, int32_t worker)
{
    IllPackArgs *job = (IllPackArgs *)args;
    const IllPlane *from = job->from;
    float *row;
    int32_t r0, r1, r;
    (void)worker;
    ill_span_split(from->rows, chunk, chunks, &r0, &r1);
    if (r0 >= r1) return;
    row = (float *)ill_block_make((size_t)from->cols * sizeof(float));
    if (!row) return;
    for (r = r0; r < r1; ++r) {
        ill_plane_row(from, r, row);
        ill_q8_pack(row, from->cols,
                    job->quant + (size_t)r * from->cols,
                    job->steps + (size_t)r * job->blocks);
    }
    ill_block_free(row);
}

/* Rewrites a plane into q8 in freshly owned memory. */
static IllResult ill_model_pack(IllModel *model, IllPlane *plane)
{
    IllPlane    from = *plane;
    IllPackArgs job;
    IllCpu     *cpu  = (IllCpu *)model->backend->inner;
    int32_t     blocks = ill_q8_blocks(from.cols);
    int8_t     *quant;
    float      *steps;

    quant = (int8_t *)ill_block_make((size_t)from.rows * from.cols);
    steps = (float *)ill_block_make((size_t)from.rows * blocks * sizeof(float));
    if (!quant || !steps) { ill_block_free(quant); ill_block_free(steps); return ILL_ALLOC; }
    if (!ill_model_own(model, quant) || !ill_model_own(model, steps)) return ILL_ALLOC;

    job.from = &from; job.quant = quant; job.steps = steps; job.blocks = blocks;
    if (model->backend == &ill_backend_cpu && cpu)
        ill_pool_fork(cpu->pool, ill_model_pack_chore, &job, cpu->lanes);
    else
        ill_model_pack_chore(&job, 0, 1, 0);

    plane->cells  = quant;
    plane->steps  = steps;
    plane->type   = ILL_TYPE_Q8;
    plane->blocks = blocks;
    model->bytes += (size_t)from.rows * from.cols + (size_t)from.rows * blocks * sizeof(float);
    return ILL_OK;
}

/* Binds one linear weight, checking its extent and repacking when asked. */
static IllResult ill_model_plane(IllModel *model, IllPlane *plane, const IllSlab *slab,
                                 int32_t rows, int32_t cols, const char *label)
{
    if (!slab) { ill_note(1, "missing tensor: %s", label); return ILL_MODEL; }
    if (slab->rank != 2) { ill_note(1, "tensor %s has rank %d", label, slab->rank); return ILL_SHAPE; }
    if ((rows && slab->dims[0] != rows) || (cols && slab->dims[1] != cols)) {
        ill_note(1, "tensor %s is %lldx%lld, expected %dx%d", label,
                 (long long)slab->dims[0], (long long)slab->dims[1], rows, cols);
        return ILL_SHAPE;
    }
    plane->cells  = slab->cells;
    plane->steps  = NULL;
    plane->type   = slab->type;
    plane->rows   = (int32_t)slab->dims[0];
    plane->cols   = (int32_t)slab->dims[1];
    plane->blocks = 0;
    model->bytes += slab->bytes;
    if (model->weight_type == ILL_TYPE_Q8) {
        model->bytes -= slab->bytes;
        return ill_model_pack(model, plane);
    }
    return ILL_OK;
}

/* -- config --------------------------------------------------------------- */

static IllResult ill_arch_read(IllModel *model, const IllJson *doc)
{
    IllArch *arch = &model->arch;
    int32_t  node, item, index;
    const char *kind;

    kind = ill_json_str(doc, ill_json_pick(doc, 0, "model_type"), "lfm2");
    if (strcmp(kind, "lfm2") != 0) {
        ill_note(1, "checkpoint declares model_type \"%s\"; this engine runs \"lfm2\"", kind);
        return ILL_MODEL;
    }

    arch->vocab_size  = (int32_t)ill_json_find(doc, 0, "vocab_size", 65536);
    arch->model_dim   = (int32_t)ill_json_find(doc, 0, "hidden_size", 2560);
    arch->layer_count = (int32_t)ill_json_find(doc, 0, "num_hidden_layers", 32);
    arch->head_count  = (int32_t)ill_json_find(doc, 0, "num_attention_heads", 32);
    arch->group_count = (int32_t)ill_json_find(doc, 0, "num_key_value_heads", arch->head_count);
    arch->token_span  = (int32_t)ill_json_find(doc, 0, "max_position_embeddings", 128000);
    arch->norm_eps    = (float)ill_json_find(doc, 0, "norm_eps",
                          ill_json_find(doc, 0, "rms_norm_eps", 1e-5));
    arch->conv_width  = (int32_t)ill_json_find(doc, 0, "conv_L_cache", 3);
    arch->conv_bias   = (int32_t)ill_json_find(doc, 0, "conv_bias", 0);
    arch->bound_embed = (int32_t)ill_json_find(doc, 0, "tie_word_embeddings", 1);
    arch->begin_token = (int32_t)ill_json_find(doc, 0, "bos_token_id", 1);
    arch->pad_token   = (int32_t)ill_json_find(doc, 0, "pad_token_id", 0);
    arch->head_dim    = (int32_t)ill_json_find(doc, 0, "head_dim",
                          arch->head_count ? arch->model_dim / arch->head_count : 0);

    node = ill_json_pick(doc, 0, "eos_token_id");
    if (node >= 0 && doc->nodes[node].kind == ILL_NODE_ARRAY)
        arch->end_token = ill_json_long(doc, ill_json_item(doc, node, 0), 2);
    else
        arch->end_token = ill_json_long(doc, node, 2);

    /* rope base lives under rope_parameters in transformers v5, at the top
     * level in v4; accept either spelling. */
    node = ill_json_pick(doc, 0, "rope_parameters");
    if (node >= 0) {
        const char *style = ill_json_str(doc, ill_json_pick(doc, node, "rope_type"), "default");
        arch->rope_base = (float)ill_json_find(doc, node, "rope_theta", 1000000.0);
        if (strcmp(style, "default") != 0 && strcmp(style, "linear") != 0)
            ill_note(1, "rope_type \"%s\" is not modelled; running plain rotary", style);
    } else {
        arch->rope_base = (float)ill_json_find(doc, 0, "rope_theta", 1000000.0);
    }

    if (arch->layer_count <= 0 || arch->model_dim <= 0 || arch->head_count <= 0 ||
        arch->group_count <= 0 || arch->head_dim <= 0 || arch->vocab_size <= 0)
        return ILL_MODEL;
    if (arch->head_count % arch->group_count) {
        ill_note(1, "head_count %d is not a multiple of group_count %d",
                 arch->head_count, arch->group_count);
        return ILL_MODEL;
    }
    if (arch->conv_width < 1 || arch->conv_width > 16) {
        ill_note(1, "conv_L_cache %d is outside the supported range 1..16", arch->conv_width);
        return ILL_MODEL;
    }

    arch->layer_kind = (uint8_t *)ill_block_zero((size_t)arch->layer_count);
    if (!arch->layer_kind) return ILL_ALLOC;
    if (!ill_model_own(model, arch->layer_kind)) return ILL_ALLOC;

    node = ill_json_pick(doc, 0, "layer_types");
    if (node >= 0 && doc->nodes[node].kind == ILL_NODE_ARRAY) {
        index = 0;
        for (item = doc->nodes[node].head; item >= 0 && index < arch->layer_count;
             item = doc->nodes[item].next) {
            const char *word = ill_json_str(doc, item, "full_attention");
            arch->layer_kind[index++] = (uint8_t)(!strcmp(word, "conv") ? ILL_LAYER_CONV
                                                                       : ILL_LAYER_ATTN);
        }
    } else {
        node = ill_json_pick(doc, 0, "full_attn_idxs");
        if (node >= 0 && doc->nodes[node].kind == ILL_NODE_ARRAY) {
            for (index = 0; index < arch->layer_count; ++index)
                arch->layer_kind[index] = (uint8_t)ILL_LAYER_CONV;
            for (item = doc->nodes[node].head; item >= 0; item = doc->nodes[item].next) {
                int32_t slot = ill_json_long(doc, item, -1);
                if (slot >= 0 && slot < arch->layer_count)
                    arch->layer_kind[slot] = (uint8_t)ILL_LAYER_ATTN;
            }
        } else {
            for (index = 0; index < arch->layer_count; ++index)
                arch->layer_kind[index] = (uint8_t)ILL_LAYER_ATTN;
        }
    }
    for (index = 0; index < arch->layer_count; ++index) {
        if (arch->layer_kind[index] == ILL_LAYER_ATTN) arch->attn_count++;
        else                                           arch->conv_count++;
    }
    return ILL_OK;
}

/* -- assembly ------------------------------------------------------------- */

static IllResult ill_vocab_load(IllVocab **vocab, const char *dir, const IllArch *arch);
static void      ill_vocab_free(IllVocab *vocab);

void ill_plan_init(IllPlan *plan)
{
    memset(plan, 0, sizeof(*plan));
    plan->weight_type = ILL_TYPE_KEEP;
    plan->batch_span  = 256;
}

IllResult ill_model_load(IllModel **out, const IllPlan *plan)
{
    IllModel *model;
    IllJson   doc;
    IllFile   book;
    IllResult code;
    IllArch  *arch;
    char      leaf[ILL_PATH_MAX];
    const char *dir;
    int32_t   index, attn_slot = 0, conv_slot = 0;

    if (!out || !plan || !plan->model_path) return ILL_ARGS;
    *out = NULL;
    ill_noise_level = plan->quiet_load ? 1 : 2;

    model = (IllModel *)ill_block_zero(sizeof(IllModel));
    if (!model) return ILL_ALLOC;
    model->weight_type = plan->weight_type == ILL_TYPE_Q8 ? ILL_TYPE_Q8 : ILL_TYPE_KEEP;
    model->batch_span  = plan->batch_span > 0 ? plan->batch_span : 256;
    arch = &model->arch;

    model->backend = ill_backend_find(plan->backend_name);
    if (!model->backend) {
        ill_note(1, "no backend named \"%s\"", plan->backend_name);
        ill_block_free(model);
        return ILL_ARGS;
    }
    code = model->backend->setup(model->backend, plan->thread_count);
    if (code != ILL_OK) { ill_block_free(model); return code; }

    dir = plan->model_path;
    if (ill_path_dir(dir)) ill_path_join(leaf, sizeof(leaf), dir, "config.json");
    else                   snprintf(leaf, sizeof(leaf), "%s", dir);

    if (!ill_path_dir(dir)) {
        ill_note(1, "model_path must be a checkpoint folder holding config.json");
        code = ILL_ARGS;
        goto undo;
    }
    code = ill_file_open(&book, leaf);
    if (code != ILL_OK) { ill_note(1, "cannot read %s", leaf); goto undo; }
    code = ill_json_load(&doc, (const char *)book.data, book.size);
    ill_file_close(&book);
    if (code != ILL_OK) { ill_note(1, "config.json is malformed"); goto undo; }
    code = ill_arch_read(model, &doc);
    ill_json_free(&doc);
    if (code != ILL_OK) goto undo;

    code = ill_store_open(&model->store, dir);
    if (code != ILL_OK) { ill_note(1, "no safetensors shards under %s", dir); goto undo; }
    ill_note(2, "checkpoint: %d shards, %d tensors, %.2f GiB on disk",
             model->store.file_count, model->store.slab_count,
             (double)model->store.bytes / 1073741824.0);

    model->blocks = (IllBlock *)ill_block_zero((size_t)arch->layer_count * sizeof(IllBlock));
    if (!model->blocks) { code = ILL_ALLOC; goto undo; }

    /* embedding and output projection */
    code = ill_model_plane(model, &model->embed,
                           ill_model_pick(&model->store, "model.embed_tokens.weight"),
                           arch->vocab_size, arch->model_dim, "embed_tokens");
    if (code != ILL_OK) goto undo;
    {
        const IllSlab *slab = ill_model_slab(&model->store, "lm_head.weight");
        if (slab) {
            code = ill_model_plane(model, &model->crown, slab,
                                   arch->vocab_size, arch->model_dim, "lm_head");
            if (code != ILL_OK) goto undo;
            arch->bound_embed = 0;
        } else {
            model->crown = model->embed;
            arch->bound_embed = 1;
        }
    }
    model->out_gain = ill_model_vec(model,
                        ill_model_pick(&model->store, "model.embedding_norm.weight"),
                        arch->model_dim);
    if (!model->out_gain) { ill_note(1, "missing embedding_norm"); code = ILL_MODEL; goto undo; }

    /* the feed forward width is taken from the checkpoint, not recomputed */
    {
        const IllSlab *probe = ill_model_pick(&model->store,
                                              "model.layers.0.feed_forward.w1.weight");
        if (!probe || probe->rank != 2) { code = ILL_MODEL; goto undo; }
        arch->inner_dim = (int32_t)probe->dims[0];
    }

    for (index = 0; index < arch->layer_count; ++index) {
        IllBlock *block = &model->blocks[index];
        int32_t   qdim  = arch->head_count * arch->head_dim;
        int32_t   kdim  = arch->group_count * arch->head_dim;
        block->kind = arch->layer_kind[index];

        block->op_gain = ill_model_vec(model,
            ill_model_pick(&model->store, "model.layers.%d.operator_norm.weight", index),
            arch->model_dim);
        block->ff_gain = ill_model_vec(model,
            ill_model_pick(&model->store, "model.layers.%d.ffn_norm.weight", index),
            arch->model_dim);
        if (!block->op_gain || !block->ff_gain) {
            ill_note(1, "layer %d is missing a norm gain", index);
            code = ILL_MODEL; goto undo;
        }

        if (block->kind == ILL_LAYER_ATTN) {
            block->slot = attn_slot++;
            code = ill_model_plane(model, &block->query,
                ill_model_pick(&model->store, "model.layers.%d.self_attn.q_proj.weight", index),
                qdim, arch->model_dim, "q_proj");
            if (code == ILL_OK) code = ill_model_plane(model, &block->keyed,
                ill_model_pick(&model->store, "model.layers.%d.self_attn.k_proj.weight", index),
                kdim, arch->model_dim, "k_proj");
            if (code == ILL_OK) code = ill_model_plane(model, &block->value,
                ill_model_pick(&model->store, "model.layers.%d.self_attn.v_proj.weight", index),
                kdim, arch->model_dim, "v_proj");
            if (code == ILL_OK) code = ill_model_plane(model, &block->joint,
                ill_model_pick(&model->store, "model.layers.%d.self_attn.out_proj.weight", index),
                arch->model_dim, qdim, "out_proj");
            if (code != ILL_OK) goto undo;
            block->query_gain = ill_model_vec(model,
                ill_model_pick(&model->store, "model.layers.%d.self_attn.q_layernorm.weight", index),
                arch->head_dim);
            block->keyed_gain = ill_model_vec(model,
                ill_model_pick(&model->store, "model.layers.%d.self_attn.k_layernorm.weight", index),
                arch->head_dim);
            if (!block->query_gain || !block->keyed_gain) {
                ill_note(1, "layer %d is missing a head norm gain", index);
                code = ILL_MODEL; goto undo;
            }
        } else {
            const IllSlab *taps;
            block->slot = conv_slot++;
            code = ill_model_plane(model, &block->mixin,
                ill_model_pick(&model->store, "model.layers.%d.conv.in_proj.weight", index),
                3 * arch->model_dim, arch->model_dim, "conv.in_proj");
            if (code == ILL_OK) code = ill_model_plane(model, &block->mixout,
                ill_model_pick(&model->store, "model.layers.%d.conv.out_proj.weight", index),
                arch->model_dim, arch->model_dim, "conv.out_proj");
            if (code != ILL_OK) goto undo;
            taps = ill_model_pick(&model->store, "model.layers.%d.conv.conv.weight", index);
            if (!taps || taps->dims[0] != arch->model_dim ||
                taps->dims[taps->rank - 1] != arch->conv_width) {
                ill_note(1, "layer %d has an unexpected convolution kernel", index);
                code = ILL_SHAPE; goto undo;
            }
            block->taps = ill_model_vec(model, taps, arch->model_dim * arch->conv_width);
            if (!block->taps) { code = ILL_ALLOC; goto undo; }
            if (arch->conv_bias) {
                block->tilt = ill_model_vec(model,
                    ill_model_pick(&model->store, "model.layers.%d.conv.conv.bias", index),
                    arch->model_dim);
            }
        }

        code = ill_model_plane(model, &block->gate,
            ill_model_pick(&model->store, "model.layers.%d.feed_forward.w1.weight", index),
            arch->inner_dim, arch->model_dim, "feed_forward.w1");
        if (code == ILL_OK) code = ill_model_plane(model, &block->fall,
            ill_model_pick(&model->store, "model.layers.%d.feed_forward.w2.weight", index),
            arch->model_dim, arch->inner_dim, "feed_forward.w2");
        if (code == ILL_OK) code = ill_model_plane(model, &block->rise,
            ill_model_pick(&model->store, "model.layers.%d.feed_forward.w3.weight", index),
            arch->inner_dim, arch->model_dim, "feed_forward.w3");
        if (code != ILL_OK) goto undo;

        if (!plan->quiet_load && ((index + 1) % 8 == 0 || index + 1 == arch->layer_count))
            ill_note(2, "bound layer %d/%d", index + 1, arch->layer_count);
    }

    code = ill_vocab_load(&model->vocab, dir, arch);
    if (code != ILL_OK)
        ill_note(1, "no usable tokenizer under %s; token ids only", dir);
    else if (ill_vocab_size(model->vocab) > arch->vocab_size)
        ill_note(1, "tokenizer holds %d ids but the model has room for %d; "
                    "the two do not belong to the same checkpoint",
                 ill_vocab_size(model->vocab), arch->vocab_size);

    ill_note(2, "arch: %d layers (%d attention, %d convolution), dim %d, ffn %d, "
                "heads %d/%d, head_dim %d, vocab %d",
             arch->layer_count, arch->attn_count, arch->conv_count, arch->model_dim,
             arch->inner_dim, arch->head_count, arch->group_count, arch->head_dim,
             arch->vocab_size);
    ill_note(2, "weights: %s, %.2f GiB resident, backend %s/%s, %d threads",
             model->weight_type == ILL_TYPE_Q8 ? "q8" : "as stored",
             (double)model->bytes / 1073741824.0, model->backend->name, ILL_SIMD_NAME,
             model->backend->width(model->backend));

    *out = model;
    return ILL_OK;

undo:
    ill_model_free(model);
    return code;
}

void ill_model_free(IllModel *model)
{
    int32_t index;
    if (!model) return;
    ill_vocab_free(model->vocab);
    for (index = 0; index < model->owned_count; ++index) ill_block_free(model->owned[index]);
    ill_block_free(model->owned);
    ill_block_free(model->blocks);
    ill_store_close(&model->store);
    if (model->backend && model->backend->close) model->backend->close(model->backend);
    ill_block_free(model);
}

const IllArch  *ill_model_arch(const IllModel *model)  { return model ? &model->arch : NULL; }
const IllVocab *ill_model_vocab(const IllModel *model) { return model ? model->vocab : NULL; }
size_t          ill_model_bytes(const IllModel *model) { return model ? model->bytes : 0; }
const char     *ill_model_backend(const IllModel *model)
{
    return model && model->backend ? model->backend->name : "none";
}
int32_t ill_model_threads(const IllModel *model)
{
    return model && model->backend ? model->backend->width(model->backend) : 1;
}

/* ============================================================================
 * part 11 -- model state
 *
 * One IllState is one sequence.  It owns the two things the Liquid stack
 * carries between steps -- the key/value cache of the attention layers and the
 * rolling window of the convolution layers -- plus every scratch buffer the
 * forward pass touches, so a step allocates nothing.
 * ==========================================================================*/

struct IllState {
    IllModel *model;
    int32_t   span;        /* cache capacity in tokens                       */
    int32_t   fill;        /* tokens already cached                          */
    int32_t   batch;       /* scratch width in tokens                        */

    float    *keys;        /* attn_count * groups * span * head_dim          */
    float    *vals;
    float    *hist;        /* conv_count * model_dim * (conv_width - 1)      */

    float    *lane;        /* residual stream, batch * model_dim             */
    float    *rest;        /* normalised stream                              */
    float    *step;        /* block output before the residual add           */
    float    *query, *keyed, *value, *mixed;
    float    *widen;       /* conv in_proj, batch * 3 * model_dim            */
    float    *curl;        /* conv working buffer, batch * model_dim         */
    float    *gate, *rise; /* feed forward limbs, batch * inner_dim          */
    float    *board;       /* attention scores, lanes * span                 */
    float    *table;       /* rotary cos/sin, batch * head_dim               */
    double   *wave;        /* rotary inverse frequencies, head_dim / 2       */
    IllPad    pad;         /* q8 activation scratch                          */

    float    *crumb;       /* logits                                         */
    int32_t   crumb_rows;

    void    **owned;
    int32_t   owned_count, owned_limit;
    size_t    bytes;
};

static void *ill_state_own(IllState *state, size_t bytes)
{
    void *block;
    if (state->owned_count == state->owned_limit) {
        int32_t grown = state->owned_limit ? state->owned_limit * 2 : 32;
        void  **fresh = (void **)ill_block_make((size_t)grown * sizeof(void *));
        if (!fresh) return NULL;
        if (state->owned) memcpy(fresh, state->owned, (size_t)state->owned_count * sizeof(void *));
        ill_block_free(state->owned);
        state->owned = fresh;
        state->owned_limit = grown;
    }
    block = ill_block_zero(bytes);
    if (!block) return NULL;
    state->owned[state->owned_count++] = block;
    state->bytes += bytes;
    return block;
}

#define ILL_CLAIM(field, count)                                        \
    do {                                                               \
        state->field = (float *)ill_state_own(state,                   \
                          (size_t)(count) * sizeof(float));            \
        if (!state->field) { ill_state_free(state); return ILL_ALLOC; } \
    } while (0)

IllResult ill_state_make(IllState **out, IllModel *model, int32_t span)
{
    IllState *state;
    const IllArch *arch;
    int32_t   lanes, qdim, kdim, wide, i, half;

    if (!out || !model) return ILL_ARGS;
    *out = NULL;
    arch = &model->arch;
    if (span <= 0) span = ILL_MIN(arch->token_span, 4096);
    if (span <= 0) return ILL_ARGS;

    state = (IllState *)ill_block_zero(sizeof(IllState));
    if (!state) return ILL_ALLOC;
    state->model = model;
    state->span  = span;
    state->batch = ILL_MIN(model->batch_span, span);
    if (state->batch < 1) state->batch = 1;

    lanes = model->backend->width(model->backend);
    qdim  = arch->head_count * arch->head_dim;
    kdim  = arch->group_count * arch->head_dim;
    wide  = ILL_MAX(arch->model_dim, arch->inner_dim);
    wide  = ILL_MAX(wide, qdim);

    if (arch->attn_count > 0) {
        size_t cells = (size_t)arch->attn_count * arch->group_count *
                       (size_t)span * arch->head_dim;
        ILL_CLAIM(keys, cells);
        ILL_CLAIM(vals, cells);
        ILL_CLAIM(board, (size_t)lanes * span);
        ILL_CLAIM(query, (size_t)state->batch * qdim);
        ILL_CLAIM(keyed, (size_t)state->batch * kdim);
        ILL_CLAIM(value, (size_t)state->batch * kdim);
        ILL_CLAIM(mixed, (size_t)state->batch * qdim);
        ILL_CLAIM(table, (size_t)state->batch * arch->head_dim);
    }
    if (arch->conv_count > 0) {
        ILL_CLAIM(hist, (size_t)arch->conv_count * arch->model_dim *
                        (size_t)ILL_MAX(arch->conv_width - 1, 1));
        ILL_CLAIM(widen, (size_t)state->batch * 3 * arch->model_dim);
        ILL_CLAIM(curl,  (size_t)state->batch * arch->model_dim);
    }
    ILL_CLAIM(lane, (size_t)state->batch * arch->model_dim);
    ILL_CLAIM(rest, (size_t)state->batch * arch->model_dim);
    ILL_CLAIM(step, (size_t)state->batch * arch->model_dim);
    ILL_CLAIM(gate, (size_t)state->batch * arch->inner_dim);
    ILL_CLAIM(rise, (size_t)state->batch * arch->inner_dim);

    if (model->weight_type == ILL_TYPE_Q8) {
        state->pad.rows  = state->batch;
        state->pad.cols  = wide;
        state->pad.quant = (int8_t *)ill_state_own(state, (size_t)state->batch * wide);
        state->pad.steps = (float *)ill_state_own(state,
                             (size_t)state->batch * ill_q8_blocks(wide) * sizeof(float));
        if (!state->pad.quant || !state->pad.steps) { ill_state_free(state); return ILL_ALLOC; }
    }

    half = arch->head_dim / 2;
    state->wave = (double *)ill_state_own(state, (size_t)ILL_MAX(half, 1) * sizeof(double));
    if (!state->wave) { ill_state_free(state); return ILL_ALLOC; }
    for (i = 0; i < half; ++i)
        state->wave[i] = pow((double)arch->rope_base,
                             -(double)(2 * i) / (double)arch->head_dim);

    *out = state;
    return ILL_OK;
}

#undef ILL_CLAIM

void ill_state_free(IllState *state)
{
    int32_t index;
    if (!state) return;
    for (index = 0; index < state->owned_count; ++index) ill_block_free(state->owned[index]);
    ill_block_free(state->owned);
    ill_block_free(state->crumb);
    ill_block_free(state);
}

void ill_state_reset(IllState *state)
{
    const IllArch *arch;
    if (!state) return;
    arch = &state->model->arch;
    state->fill = 0;
    if (state->hist)
        memset(state->hist, 0, (size_t)arch->conv_count * arch->model_dim *
                               (size_t)ILL_MAX(arch->conv_width - 1, 1) * sizeof(float));
}

int32_t ill_state_fill(const IllState *state) { return state ? state->fill : 0; }
int32_t ill_state_span(const IllState *state) { return state ? state->span : 0; }
size_t  ill_state_bytes(const IllState *state) { return state ? state->bytes : 0; }

IllResult ill_state_crop(IllState *state, int32_t fill)
{
    if (!state || fill < 0 || fill > state->fill) return ILL_ARGS;
    if (fill == state->fill) return ILL_OK;
    if (fill == 0) { ill_state_reset(state); return ILL_OK; }
    if (state->model->arch.conv_count > 0) {
        /* The convolution window is recurrent: dropping tokens from the tail
         * cannot be undone without replaying the sequence. */
        return ILL_STATE;
    }
    state->fill = fill;
    return ILL_OK;
}

/* ============================================================================
 * part 12 -- forward pass
 *
 * One chunk of tokens moves through the stack here.  Each block is
 *
 *     h <- h + operator(rmsnorm(h))
 *     h <- h + swiglu(rmsnorm(h))
 *
 * where the operator is either grouped query attention with rotary positions
 * or the liquid short convolution, chosen per layer by the checkpoint.
 * ==========================================================================*/

/* Fills `table` with cos/sin for positions base..base+count-1.  Layout mirrors
 * the reference cat(freqs, freqs): cosines first, then sines. */
static void ill_turn_table(IllState *state, int32_t base, int32_t count)
{
    const IllArch *arch = &state->model->arch;
    const int32_t  half = arch->head_dim / 2;
    const double   ring = 6.283185307179586476925286766559;
    int32_t t, i;
    for (t = 0; t < count; ++t) {
        float *cos_row = state->table + (size_t)t * arch->head_dim;
        float *sin_row = cos_row + half;
        double pos = (double)(base + t);
        for (i = 0; i < half; ++i) {
            double phase = fmod(pos * state->wave[i], ring);
            cos_row[i] = (float)cos(phase);
            sin_row[i] = (float)sin(phase);
        }
    }
}

static void ill_lane_add(float *lane, const float *step, int32_t count)
{
    int32_t j;
    for (j = 0; j + ILL_VW <= count; j += ILL_VW)
        ill_vec_save(lane + j, ill_vec_add(ill_vec_load(lane + j), ill_vec_load(step + j)));
    for (; j < count; ++j) lane[j] += step[j];
}

static void ill_stack_attn(IllModel *model, IllState *state, const IllBlock *block,
                           int32_t count, int32_t base)
{
    const IllArch *arch = &model->arch;
    IllBackend *back = model->backend;
    const int32_t hd    = arch->head_dim;
    const int32_t qdim  = arch->head_count * hd;
    const int32_t kdim  = arch->group_count * hd;
    const size_t  reach = (size_t)state->span * hd;
    IllHeedJob    job;
    int32_t t, g;

    back->dense(back, &block->query, state->rest, state->query, count, &state->pad);
    back->dense(back, &block->keyed, state->rest, state->keyed, count, &state->pad);
    back->dense(back, &block->value, state->rest, state->value, count, &state->pad);

    /* Per-head rms norm: the head axis is contiguous, so a head is just a row. */
    back->rmsnorm(back, state->query, block->query_gain, state->query,
                  count * arch->head_count, hd, arch->norm_eps);
    back->rmsnorm(back, state->keyed, block->keyed_gain, state->keyed,
                  count * arch->group_count, hd, arch->norm_eps);

    ill_turn_table(state, base, count);
    back->rope(back, state->query, state->table, count, arch->head_count, hd, qdim);
    back->rope(back, state->keyed, state->table, count, arch->group_count, hd, kdim);

    for (t = 0; t < count; ++t) {
        for (g = 0; g < arch->group_count; ++g) {
            size_t slot = ((size_t)block->slot * arch->group_count + g) * reach
                          + (size_t)(base + t) * hd;
            memcpy(state->keys + slot, state->keyed + (size_t)t * kdim + (size_t)g * hd,
                   (size_t)hd * sizeof(float));
            memcpy(state->vals + slot, state->value + (size_t)t * kdim + (size_t)g * hd,
                   (size_t)hd * sizeof(float));
        }
    }

    job.query  = state->query;
    job.keys   = state->keys + (size_t)block->slot * arch->group_count * reach;
    job.vals   = state->vals + (size_t)block->slot * arch->group_count * reach;
    job.value  = state->mixed;
    job.board  = state->board;
    job.tokens = count;
    job.heads  = arch->head_count;
    job.groups = arch->group_count;
    job.width  = hd;
    job.span   = state->span;
    job.base   = base;
    job.scale  = 1.0f / sqrtf((float)hd);
    back->attend(back, &job);

    back->dense(back, &block->joint, state->mixed, state->step, count, &state->pad);
}

static void ill_stack_conv(IllModel *model, IllState *state, const IllBlock *block,
                           int32_t count, int32_t base)
{
    const IllArch *arch = &model->arch;
    IllBackend *back = model->backend;
    const int32_t dim  = arch->model_dim;
    const int32_t keep = ILL_MAX(arch->conv_width - 1, 1);
    IllFlowJob    job;
    int32_t t, c;
    (void)base;

    back->dense(back, &block->mixin, state->rest, state->widen, count, &state->pad);

    /* in_proj yields three stacked halves: gate B, gate C, and the signal x. */
    for (t = 0; t < count; ++t) {
        const float *row = state->widen + (size_t)t * 3 * dim;
        float       *dst = state->curl + (size_t)t * dim;
        for (c = 0; c < dim; ++c) dst[c] = row[c] * row[2 * dim + c];
    }

    job.src    = state->curl;
    job.dst    = state->rest;          /* rest is free once in_proj has run */
    job.hist   = state->hist + (size_t)block->slot * dim * keep;
    job.taps   = block->taps;
    job.bias   = block->tilt;
    job.tokens = count;
    job.dim    = dim;
    job.width  = arch->conv_width;
    back->conv1d(back, &job);

    for (t = 0; t < count; ++t) {
        const float *row = state->widen + (size_t)t * 3 * dim;
        float       *dst = state->rest + (size_t)t * dim;
        for (c = 0; c < dim; ++c) dst[c] *= row[dim + c];
    }

    back->dense(back, &block->mixout, state->rest, state->step, count, &state->pad);
}

static IllResult ill_stack_run(IllModel *model, IllState *state,
                               const int32_t *tokens, int32_t count,
                               int32_t want, float *crumb)
{
    const IllArch *arch = &model->arch;
    IllBackend *back = model->backend;
    const int32_t dim  = arch->model_dim;
    const int32_t base = state->fill;
    int32_t index, t;

    for (t = 0; t < count; ++t) {
        if (tokens[t] < 0 || tokens[t] >= arch->vocab_size) {
            ill_note(1, "token id %d is outside the model vocabulary of %d",
                     tokens[t], arch->vocab_size);
            return ILL_ARGS;
        }
        ill_plane_row(&model->embed, tokens[t], state->lane + (size_t)t * dim);
    }

    for (index = 0; index < arch->layer_count; ++index) {
        const IllBlock *block = &model->blocks[index];
        back->rmsnorm(back, state->lane, block->op_gain, state->rest,
                      count, dim, arch->norm_eps);
        if (block->kind == ILL_LAYER_ATTN) ill_stack_attn(model, state, block, count, base);
        else                               ill_stack_conv(model, state, block, count, base);
        ill_lane_add(state->lane, state->step, count * dim);

        back->rmsnorm(back, state->lane, block->ff_gain, state->rest,
                      count, dim, arch->norm_eps);
        back->dense(back, &block->gate, state->rest, state->gate, count, &state->pad);
        back->dense(back, &block->rise, state->rest, state->rise, count, &state->pad);
        back->swiglu(back, state->gate, state->rise, state->gate, count, arch->inner_dim);
        back->dense(back, &block->fall, state->gate, state->step, count, &state->pad);
        ill_lane_add(state->lane, state->step, count * dim);
    }

    state->fill = base + count;

    if (want > 0) {
        const float *tail = state->lane + (size_t)(count - want) * dim;
        back->rmsnorm(back, tail, model->out_gain, state->rest, want, dim, arch->norm_eps);
        back->dense(back, &model->crown, state->rest, crumb, want, &state->pad);
    }
    return ILL_OK;
}

IllResult ill_model_apply(IllModel *model, IllState *state,
                          const IllBatch *batch, float **logits)
{
    int32_t rows, done;
    if (!model || !state || !batch || !batch->tokens || batch->count <= 0) return ILL_ARGS;
    if (state->model != model) return ILL_ARGS;
    if (state->fill + batch->count > state->span) return ILL_LIMIT;

    rows = batch->every ? batch->count : 1;
    if (rows > state->crumb_rows) {
        ill_block_free(state->crumb);
        state->crumb = (float *)ill_block_make((size_t)rows * model->arch.vocab_size *
                                               sizeof(float));
        if (!state->crumb) { state->crumb_rows = 0; return ILL_ALLOC; }
        state->crumb_rows = rows;
    }

    for (done = 0; done < batch->count; ) {
        int32_t span = ILL_MIN(state->batch, batch->count - done);
        int32_t last = done + span == batch->count;
        int32_t want = batch->every ? span : (last ? 1 : 0);
        float  *dest = batch->every
                     ? state->crumb + (size_t)done * model->arch.vocab_size
                     : state->crumb;
        IllResult code = ill_stack_run(model, state, batch->tokens + done, span, want, dest);
        if (code != ILL_OK) return code;
        done += span;
    }
    if (logits) *logits = state->crumb;
    return ILL_OK;
}

/* ============================================================================
 * part 13 -- vocabulary
 *
 * A byte level BPE tokenizer read straight from tokenizer.json.  Three tables
 * carry the whole job: pieces indexed by id, a piece to id map, and a merge
 * map keyed by the pair of ids being joined.  Because every merge result is
 * itself a vocabulary entry, the inner loop never touches a string -- it walks
 * ids -- which is what keeps encoding linear in practice.
 *
 * Supported: model.type "BPE" with a byte level alphabet, the GPT-2 and
 * Llama-3 pre-tokenizer patterns, and literal added tokens.  Anything else is
 * reported rather than approximated.
 * ==========================================================================*/

typedef struct IllPiece {
    const char *text;
    int32_t     span;
    uint8_t     special;
} IllPiece;

typedef enum IllSplit { ILL_SPLIT_GPT2 = 0, ILL_SPLIT_LLAMA3 } IllSplit;
typedef enum IllChat  { ILL_CHAT_PLAIN = 0, ILL_CHAT_ML } IllChat;

struct IllVocab {
    IllPiece *pieces;
    int32_t   count;

    int32_t  *bins;        /* open addressed piece -> id                     */
    uint32_t  bin_count;

    uint64_t *pair_key;    /* (a * count + b) + 1, zero marks an empty slot  */
    int32_t  *pair_rank;
    int32_t  *pair_join;
    uint32_t  pair_bins;

    int32_t   byte_id[256];
    int16_t   rune_byte[512];

    int32_t  *extra;       /* ids of added tokens, longest content first     */
    int32_t   extra_count;

    int32_t   begin_token, end_token, pad_token;
    int32_t   stops[8];
    int32_t   stop_count;

    uint8_t   split_kind;
    uint8_t   add_prefix;
    uint8_t   chat_kind;

    char     *pool;
    size_t    pool_used, pool_room;
    void    **owned;
    int32_t   owned_count, owned_limit;
};

static void *ill_vocab_own(IllVocab *vocab, size_t bytes)
{
    void *block;
    if (vocab->owned_count == vocab->owned_limit) {
        int32_t grown = vocab->owned_limit ? vocab->owned_limit * 2 : 16;
        void  **fresh = (void **)ill_block_make((size_t)grown * sizeof(void *));
        if (!fresh) return NULL;
        if (vocab->owned) memcpy(fresh, vocab->owned, (size_t)vocab->owned_count * sizeof(void *));
        ill_block_free(vocab->owned);
        vocab->owned = fresh;
        vocab->owned_limit = grown;
    }
    block = ill_block_zero(bytes);
    if (!block) return NULL;
    vocab->owned[vocab->owned_count++] = block;
    return block;
}

static void ill_vocab_free(IllVocab *vocab)
{
    int32_t index;
    if (!vocab) return;
    for (index = 0; index < vocab->owned_count; ++index) ill_block_free(vocab->owned[index]);
    ill_block_free(vocab->owned);
    ill_block_free(vocab);
}

/* -- utf-8 and the byte alphabet ------------------------------------------ */

static int32_t ill_utf8_read(const char *text, int32_t len, int32_t pos, uint32_t *rune)
{
    const unsigned char *p = (const unsigned char *)text + pos;
    int32_t rest = len - pos;
    if (rest <= 0) { *rune = 0; return 0; }
    if (p[0] < 0x80) { *rune = p[0]; return 1; }
    if ((p[0] & 0xE0) == 0xC0 && rest >= 2) {
        *rune = ((uint32_t)(p[0] & 0x1F) << 6) | (uint32_t)(p[1] & 0x3F);
        return 2;
    }
    if ((p[0] & 0xF0) == 0xE0 && rest >= 3) {
        *rune = ((uint32_t)(p[0] & 0x0F) << 12) | ((uint32_t)(p[1] & 0x3F) << 6) |
                (uint32_t)(p[2] & 0x3F);
        return 3;
    }
    if ((p[0] & 0xF8) == 0xF0 && rest >= 4) {
        *rune = ((uint32_t)(p[0] & 0x07) << 18) | ((uint32_t)(p[1] & 0x3F) << 12) |
                ((uint32_t)(p[2] & 0x3F) << 6) | (uint32_t)(p[3] & 0x3F);
        return 4;
    }
    *rune = p[0];
    return 1;
}

static int32_t ill_utf8_write(uint32_t rune, char *out)
{
    if (rune < 0x80u) { out[0] = (char)rune; return 1; }
    if (rune < 0x800u) {
        out[0] = (char)(0xC0u | (rune >> 6));
        out[1] = (char)(0x80u | (rune & 0x3Fu));
        return 2;
    }
    if (rune < 0x10000u) {
        out[0] = (char)(0xE0u | (rune >> 12));
        out[1] = (char)(0x80u | ((rune >> 6) & 0x3Fu));
        out[2] = (char)(0x80u | (rune & 0x3Fu));
        return 3;
    }
    out[0] = (char)(0xF0u | (rune >> 18));
    out[1] = (char)(0x80u | ((rune >> 12) & 0x3Fu));
    out[2] = (char)(0x80u | ((rune >> 6) & 0x3Fu));
    out[3] = (char)(0x80u | (rune & 0x3Fu));
    return 4;
}

/* The GPT-2 byte alphabet: printable bytes stand for themselves, the rest are
 * lifted into the private range starting at U+0100. */
static uint32_t ill_byte_rune(int32_t byte)
{
    static uint32_t table[256];
    static int      built = 0;
    if (!built) {
        int32_t b, spare = 0;
        for (b = 0; b < 256; ++b) {
            int keep = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) ||
                       (b >= 0xAE && b <= 0xFF);
            table[b] = keep ? (uint32_t)b : 256u + (uint32_t)(spare++);
        }
        built = 1;
    }
    return table[byte & 0xFF];
}

/* -- character classes ----------------------------------------------------
 *
 * The pre-tokenizer patterns lean on \p{L} and \p{N}.  Carrying the whole
 * Unicode database would dwarf the engine, so runes below 0x80 use the exact
 * ASCII rule and runes above it are letters unless they fall in a listed
 * range of spaces, punctuation, or symbols.  The ranges cover the blocks that
 * appear in ordinary prose; see TODO.md for the exact remaining gap.
 * ------------------------------------------------------------------------*/

typedef enum IllClass { ILL_CLASS_OTHER = 0, ILL_CLASS_LETTER, ILL_CLASS_DIGIT,
                        ILL_CLASS_SPACE } IllClass;

static int ill_rune_span(uint32_t rune, const uint32_t *pairs, int32_t count)
{
    int32_t index;
    for (index = 0; index < count; index += 2)
        if (rune >= pairs[index] && rune <= pairs[index + 1]) return 1;
    return 0;
}

static IllClass ill_rune_kind(uint32_t rune)
{
    static const uint32_t blanks[] = {
        0x0009, 0x000D, 0x0020, 0x0020, 0x0085, 0x0085, 0x00A0, 0x00A0,
        0x1680, 0x1680, 0x2000, 0x200A, 0x2028, 0x2029, 0x202F, 0x202F,
        0x205F, 0x205F, 0x3000, 0x3000
    };
    static const uint32_t marks[] = {
        0x00A1, 0x00BF, 0x00D7, 0x00D7, 0x00F7, 0x00F7,
        0x2010, 0x2027, 0x202A, 0x205E, 0x2060, 0x206F,
        0x20A0, 0x20CF, 0x2100, 0x2BFF, 0x2E00, 0x2E7F,
        0x3001, 0x303F, 0xFE10, 0xFE6F, 0xFF01, 0xFF20,
        0xFF3B, 0xFF40, 0xFF5B, 0xFF65, 0xFFE0, 0xFFEF,
        0x1F000, 0x1FAFF
    };
    static const uint32_t figures[] = {
        0x0660, 0x0669, 0x06F0, 0x06F9, 0x0966, 0x096F, 0x09E6, 0x09EF,
        0x0E50, 0x0E59, 0xFF10, 0xFF19
    };
    if (rune < 0x80u) {
        if (rune == ' ' || (rune >= 0x09u && rune <= 0x0Du)) return ILL_CLASS_SPACE;
        if (rune >= '0' && rune <= '9') return ILL_CLASS_DIGIT;
        if ((rune >= 'a' && rune <= 'z') || (rune >= 'A' && rune <= 'Z')) return ILL_CLASS_LETTER;
        return ILL_CLASS_OTHER;
    }
    if (ill_rune_span(rune, blanks, (int32_t)(sizeof(blanks) / sizeof(blanks[0]))))
        return ILL_CLASS_SPACE;
    if (ill_rune_span(rune, figures, (int32_t)(sizeof(figures) / sizeof(figures[0]))))
        return ILL_CLASS_DIGIT;
    if (ill_rune_span(rune, marks, (int32_t)(sizeof(marks) / sizeof(marks[0]))))
        return ILL_CLASS_OTHER;
    return ILL_CLASS_LETTER;
}

/* -- hashing -------------------------------------------------------------- */

static uint64_t ill_hash_text(const char *text, size_t span)
{
    uint64_t seed = 1469598103934665603ULL;
    size_t   index;
    for (index = 0; index < span; ++index) {
        seed ^= (unsigned char)text[index];
        seed *= 1099511628211ULL;
    }
    return seed;
}

static uint64_t ill_hash_pair(uint64_t key)
{
    key ^= key >> 33; key *= 0xFF51AFD7ED558CCDULL;
    key ^= key >> 33; key *= 0xC4CEB9FE1A85EC53ULL;
    key ^= key >> 33;
    return key;
}

static void ill_vocab_bind(IllVocab *vocab, int32_t id)
{
    uint64_t hash = ill_hash_text(vocab->pieces[id].text, (size_t)vocab->pieces[id].span);
    uint32_t slot = (uint32_t)(hash & (vocab->bin_count - 1));
    while (vocab->bins[slot] >= 0) slot = (slot + 1) & (vocab->bin_count - 1);
    vocab->bins[slot] = id;
}

int32_t ill_vocab_find(const IllVocab *vocab, const char *piece)
{
    size_t   span;
    uint64_t hash;
    uint32_t slot;
    if (!vocab || !piece) return -1;
    span = strlen(piece);
    hash = ill_hash_text(piece, span);
    slot = (uint32_t)(hash & (vocab->bin_count - 1));
    while (vocab->bins[slot] >= 0) {
        int32_t id = vocab->bins[slot];
        if (vocab->pieces[id].span == (int32_t)span &&
            !memcmp(vocab->pieces[id].text, piece, span)) return id;
        slot = (slot + 1) & (vocab->bin_count - 1);
    }
    return -1;
}

static int32_t ill_vocab_scan(const IllVocab *vocab, const char *piece, size_t span)
{
    uint64_t hash = ill_hash_text(piece, span);
    uint32_t slot = (uint32_t)(hash & (vocab->bin_count - 1));
    while (vocab->bins[slot] >= 0) {
        int32_t id = vocab->bins[slot];
        if (vocab->pieces[id].span == (int32_t)span &&
            !memcmp(vocab->pieces[id].text, piece, span)) return id;
        slot = (slot + 1) & (vocab->bin_count - 1);
    }
    return -1;
}

static void ill_pair_bind(IllVocab *vocab, int32_t a, int32_t b, int32_t rank, int32_t join)
{
    uint64_t key  = (uint64_t)a * (uint64_t)vocab->count + (uint64_t)b + 1;
    uint32_t slot = (uint32_t)(ill_hash_pair(key) & (vocab->pair_bins - 1));
    while (vocab->pair_key[slot]) {
        if (vocab->pair_key[slot] == key) return;      /* first rank wins */
        slot = (slot + 1) & (vocab->pair_bins - 1);
    }
    vocab->pair_key[slot]  = key;
    vocab->pair_rank[slot] = rank;
    vocab->pair_join[slot] = join;
}

static int ill_pair_seek(const IllVocab *vocab, int32_t a, int32_t b,
                         int32_t *rank, int32_t *join)
{
    uint64_t key  = (uint64_t)a * (uint64_t)vocab->count + (uint64_t)b + 1;
    uint32_t slot = (uint32_t)(ill_hash_pair(key) & (vocab->pair_bins - 1));
    while (vocab->pair_key[slot]) {
        if (vocab->pair_key[slot] == key) {
            *rank = vocab->pair_rank[slot];
            *join = vocab->pair_join[slot];
            return 1;
        }
        slot = (slot + 1) & (vocab->pair_bins - 1);
    }
    return 0;
}

static uint32_t ill_bins_for(int32_t count)
{
    uint32_t bins = 16;
    while (bins < (uint32_t)count * 2u) bins <<= 1;
    return bins;
}

int32_t     ill_vocab_size(const IllVocab *vocab) { return vocab ? vocab->count : 0; }
const char *ill_vocab_name(const IllVocab *vocab, int32_t token)
{
    if (!vocab || token < 0 || token >= vocab->count) return NULL;
    return vocab->pieces[token].text;
}
int32_t ill_vocab_stops(const IllVocab *vocab) { return vocab ? vocab->stop_count : 0; }
int32_t ill_vocab_stop(const IllVocab *vocab, int32_t token)
{
    int32_t index;
    if (!vocab) return 0;
    for (index = 0; index < vocab->stop_count; ++index)
        if (vocab->stops[index] == token) return 1;
    return 0;
}

/* -- loading -------------------------------------------------------------- */

/* The piece pool is a bump allocator sized once from the parsed document, so
 * every piece pointer stays valid for the life of the vocabulary. */
static char *ill_vocab_keep(IllVocab *vocab, const char *text, size_t span)
{
    char *slot;
    if (vocab->pool_used + span + 1 > vocab->pool_room) return NULL;
    slot = vocab->pool + vocab->pool_used;
    memcpy(slot, text, span);
    slot[span] = '\0';
    vocab->pool_used += span + 1;
    return slot;
}

/* Records a stop token, keeping the list short and duplicate free. */
static void ill_vocab_halt(IllVocab *vocab, int32_t token)
{
    int32_t index;
    if (token < 0 || vocab->stop_count >= (int32_t)(sizeof(vocab->stops) / sizeof(int32_t)))
        return;
    for (index = 0; index < vocab->stop_count; ++index)
        if (vocab->stops[index] == token) return;
    vocab->stops[vocab->stop_count++] = token;
}

static IllResult ill_vocab_read(IllVocab *vocab, const IllJson *doc, const IllArch *arch)
{
    int32_t node, item, index, high = -1;
    const char *kind;

    node = ill_json_pick(doc, 0, "model");
    if (node < 0) return ILL_VOCAB;
    kind = ill_json_str(doc, ill_json_pick(doc, node, "type"), "BPE");
    if (strcmp(kind, "BPE") != 0) {
        ill_note(1, "tokenizer model \"%s\" is not supported; this engine reads BPE", kind);
        return ILL_VOCAB;
    }

    /* pass one: how many ids are there */
    {
        int32_t table = ill_json_pick(doc, node, "vocab");
        if (table < 0) return ILL_VOCAB;
        for (item = doc->nodes[table].head; item >= 0; item = doc->nodes[item].next) {
            int32_t id = ill_json_long(doc, item, -1);
            if (id > high) high = id;
        }
        item = ill_json_pick(doc, 0, "added_tokens");
        if (item >= 0)
            for (index = doc->nodes[item].head; index >= 0; index = doc->nodes[index].next) {
                int32_t id = ill_json_long(doc, ill_json_pick(doc, index, "id"), -1);
                if (id > high) high = id;
            }
    }
    if (high < 0) return ILL_VOCAB;
    vocab->count = ILL_MAX(high + 1, arch->vocab_size);

    vocab->pool_room = doc->used + (size_t)vocab->count * 24 + 4096;
    vocab->pool = (char *)ill_vocab_own(vocab, vocab->pool_room);
    if (!vocab->pool) return ILL_ALLOC;
    vocab->pieces = (IllPiece *)ill_vocab_own(vocab, (size_t)vocab->count * sizeof(IllPiece));
    if (!vocab->pieces) return ILL_ALLOC;
    vocab->bin_count = ill_bins_for(vocab->count);
    vocab->bins = (int32_t *)ill_vocab_own(vocab, (size_t)vocab->bin_count * sizeof(int32_t));
    if (!vocab->bins) return ILL_ALLOC;
    for (index = 0; index < (int32_t)vocab->bin_count; ++index) vocab->bins[index] = -1;

    /* pass two: pieces */
    {
        int32_t table = ill_json_pick(doc, node, "vocab");
        for (item = doc->nodes[table].head; item >= 0; item = doc->nodes[item].next) {
            const char *word = ill_json_word(doc, doc->nodes[item].name);
            int32_t     id   = ill_json_long(doc, item, -1);
            size_t      span;
            if (!word || id < 0 || id >= vocab->count) continue;
            span = strlen(word);
            vocab->pieces[id].text = ill_vocab_keep(vocab, word, span);
            vocab->pieces[id].span = (int32_t)span;
            if (!vocab->pieces[id].text) return ILL_ALLOC;
        }
    }
    item = ill_json_pick(doc, 0, "added_tokens");
    if (item >= 0)
        for (index = doc->nodes[item].head; index >= 0; index = doc->nodes[index].next) {
            int32_t     id   = ill_json_long(doc, ill_json_pick(doc, index, "id"), -1);
            const char *word = ill_json_str(doc, ill_json_pick(doc, index, "content"), NULL);
            int32_t     mark = ill_json_pick(doc, index, "special");
            size_t      span;
            if (!word || id < 0 || id >= vocab->count) continue;
            span = strlen(word);
            vocab->pieces[id].text = ill_vocab_keep(vocab, word, span);
            vocab->pieces[id].span = (int32_t)span;
            vocab->pieces[id].special = (uint8_t)(mark < 0 ||
                                        doc->nodes[mark].kind != ILL_NODE_FALSE);
            if (!vocab->pieces[id].text) return ILL_ALLOC;
            vocab->extra_count++;
        }

    /* fill any hole so every id decodes to something printable */
    for (index = 0; index < vocab->count; ++index)
        if (!vocab->pieces[index].text) {
            char spare[32];
            int  span = snprintf(spare, sizeof(spare), "<|unused_%d|>", index);
            vocab->pieces[index].text = ill_vocab_keep(vocab, spare, (size_t)span);
            vocab->pieces[index].span = span;
            vocab->pieces[index].special = 1;
            if (!vocab->pieces[index].text) return ILL_ALLOC;
        }
    for (index = 0; index < vocab->count; ++index) ill_vocab_bind(vocab, index);

    /* added tokens, longest content first, so literal matching is greedy */
    if (vocab->extra_count > 0) {
        int32_t fill = 0, a, b;
        vocab->extra = (int32_t *)ill_vocab_own(vocab,
                          (size_t)vocab->extra_count * sizeof(int32_t));
        if (!vocab->extra) return ILL_ALLOC;
        item = ill_json_pick(doc, 0, "added_tokens");
        for (index = doc->nodes[item].head; index >= 0 && fill < vocab->extra_count;
             index = doc->nodes[index].next) {
            int32_t id = ill_json_long(doc, ill_json_pick(doc, index, "id"), -1);
            if (id >= 0 && id < vocab->count) vocab->extra[fill++] = id;
        }
        vocab->extra_count = fill;
        for (a = 1; a < fill; ++a)
            for (b = a; b > 0 && vocab->pieces[vocab->extra[b - 1]].span <
                                 vocab->pieces[vocab->extra[b]].span; --b) {
                int32_t swap = vocab->extra[b - 1];
                vocab->extra[b - 1] = vocab->extra[b];
                vocab->extra[b] = swap;
            }
    }

    /* merges */
    {
        int32_t list = ill_json_pick(doc, node, "merges");
        int32_t rank = 0, total = 0;
        if (list < 0) return ILL_VOCAB;
        total = doc->nodes[list].count;
        vocab->pair_bins = ill_bins_for(total > 0 ? total : 1);
        vocab->pair_key  = (uint64_t *)ill_vocab_own(vocab,
                              (size_t)vocab->pair_bins * sizeof(uint64_t));
        vocab->pair_rank = (int32_t *)ill_vocab_own(vocab,
                              (size_t)vocab->pair_bins * sizeof(int32_t));
        vocab->pair_join = (int32_t *)ill_vocab_own(vocab,
                              (size_t)vocab->pair_bins * sizeof(int32_t));
        if (!vocab->pair_key || !vocab->pair_rank || !vocab->pair_join) return ILL_ALLOC;

        for (item = doc->nodes[list].head; item >= 0; item = doc->nodes[item].next, ++rank) {
            char        line[512];
            const char *left = NULL, *right = NULL;
            int32_t     a, b, c;
            if (doc->nodes[item].kind == ILL_NODE_ARRAY) {
                left  = ill_json_str(doc, ill_json_item(doc, item, 0), NULL);
                right = ill_json_str(doc, ill_json_item(doc, item, 1), NULL);
            } else {
                const char *word = ill_json_str(doc, item, NULL);
                char       *gap;
                if (!word) continue;
                snprintf(line, sizeof(line), "%s", word);
                gap = strchr(line, ' ');
                if (!gap) continue;
                *gap  = '\0';
                left  = line;
                right = gap + 1;
            }
            if (!left || !right) continue;
            a = ill_vocab_scan(vocab, left, strlen(left));
            b = ill_vocab_scan(vocab, right, strlen(right));
            if (a < 0 || b < 0) continue;
            {
                char joined[512];
                size_t la = strlen(left), lb = strlen(right);
                if (la + lb >= sizeof(joined)) continue;
                memcpy(joined, left, la);
                memcpy(joined + la, right, lb);
                joined[la + lb] = '\0';
                c = ill_vocab_scan(vocab, joined, la + lb);
            }
            if (c < 0) continue;
            ill_pair_bind(vocab, a, b, rank, c);
        }
    }

    /* byte alphabet */
    for (index = 0; index < 512; ++index) vocab->rune_byte[index] = -1;
    for (index = 0; index < 256; ++index) {
        char     cell[8];
        int32_t  span = ill_utf8_write(ill_byte_rune(index), cell);
        uint32_t rune = ill_byte_rune(index);
        vocab->byte_id[index] = ill_vocab_scan(vocab, cell, (size_t)span);
        if (rune < 512u) vocab->rune_byte[rune] = (int16_t)index;
    }

    /* pre-tokenizer flavour */
    vocab->split_kind = ILL_SPLIT_GPT2;
    vocab->add_prefix = 0;
    {
        int32_t pre = ill_json_pick(doc, 0, "pre_tokenizer");
        int32_t bag = pre >= 0 ? ill_json_pick(doc, pre, "pretokenizers") : -1;
        int32_t walk;
        int32_t seen_split = 0;
        for (walk = bag >= 0 ? doc->nodes[bag].head : pre; walk >= 0;
             walk = bag >= 0 ? doc->nodes[walk].next : -1) {
            const char *style = ill_json_str(doc, ill_json_pick(doc, walk, "type"), "");
            if (!strcmp(style, "ByteLevel")) {
                int32_t flag = ill_json_pick(doc, walk, "add_prefix_space");
                vocab->add_prefix = (uint8_t)(flag >= 0 &&
                                    doc->nodes[flag].kind == ILL_NODE_TRUE);
            } else if (!strcmp(style, "Split")) {
                int32_t form = ill_json_pick(doc, walk, "pattern");
                const char *rule = ill_json_str(doc, ill_json_pick(doc, form, "Regex"), "");
                seen_split = 1;
                if (strstr(rule, "[^\\r\\n") || strstr(rule, "\\p{N}{1,3}"))
                    vocab->split_kind = ILL_SPLIT_LLAMA3;
            }
            if (bag < 0) break;
        }
        (void)seen_split;
    }

    /* stop tokens and chat shape */
    vocab->begin_token = arch->begin_token;
    vocab->end_token   = arch->end_token;
    vocab->pad_token   = arch->pad_token;
    ill_vocab_halt(vocab, arch->end_token);
    {
        int32_t shut = ill_vocab_find(vocab, "<|im_end|>");
        int32_t open = ill_vocab_find(vocab, "<|im_start|>");
        if (open >= 0 && shut >= 0) {
            vocab->chat_kind = ILL_CHAT_ML;
            ill_vocab_halt(vocab, shut);
        }
        ill_vocab_halt(vocab, ill_vocab_find(vocab, "<|endoftext|>"));
    }
    return ILL_OK;
}

/* Folds in what tokenizer_config.json adds: the real bos/eos spellings. */
static void ill_vocab_extra(IllVocab *vocab, const IllJson *doc)
{
    static const char *keys[] = { "eos_token", "bos_token", "pad_token" };
    int32_t index;
    for (index = 0; index < 3; ++index) {
        int32_t     node = ill_json_pick(doc, 0, keys[index]);
        const char *word = NULL;
        int32_t     id;
        if (node < 0) continue;
        if (doc->nodes[node].kind == ILL_NODE_STRING) word = ill_json_str(doc, node, NULL);
        else word = ill_json_str(doc, ill_json_pick(doc, node, "content"), NULL);
        if (!word) continue;
        id = ill_vocab_find(vocab, word);
        if (id < 0) continue;
        if (index == 0) { vocab->end_token = id; ill_vocab_halt(vocab, id); }
        if (index == 1) vocab->begin_token = id;
        if (index == 2) vocab->pad_token = id;
    }
}

static IllResult ill_vocab_load(IllVocab **out, const char *dir, const IllArch *arch)
{
    IllVocab *vocab;
    IllFile   book;
    IllJson   doc;
    IllResult code;
    char      leaf[ILL_PATH_MAX];

    *out = NULL;
    ill_path_join(leaf, sizeof(leaf), dir, "tokenizer.json");
    if (ill_file_open(&book, leaf) != ILL_OK) return ILL_VOCAB;

    vocab = (IllVocab *)ill_block_zero(sizeof(IllVocab));
    if (!vocab) { ill_file_close(&book); return ILL_ALLOC; }

    code = ill_json_load(&doc, (const char *)book.data, book.size);
    if (code == ILL_OK) {
        code = ill_vocab_read(vocab, &doc, arch);
        ill_json_free(&doc);
    }
    ill_file_close(&book);
    if (code != ILL_OK) { ill_vocab_free(vocab); return code; }

    ill_path_join(leaf, sizeof(leaf), dir, "tokenizer_config.json");
    if (ill_file_open(&book, leaf) == ILL_OK) {
        if (ill_json_load(&doc, (const char *)book.data, book.size) == ILL_OK) {
            ill_vocab_extra(vocab, &doc);
            ill_json_free(&doc);
        }
        ill_file_close(&book);
    }

    ill_note(2, "tokenizer: %d pieces, %d added, %s split, %s chat, %d stop tokens",
             vocab->count, vocab->extra_count,
             vocab->split_kind == ILL_SPLIT_LLAMA3 ? "llama3" : "gpt2",
             vocab->chat_kind == ILL_CHAT_ML ? "chatml" : "plain", vocab->stop_count);
    *out = vocab;
    return ILL_OK;
}

/* -- pre-tokenizer --------------------------------------------------------
 *
 * Both supported patterns are alternations tried in order, so the splitter is
 * a direct transcription of that order rather than a regex engine.
 * ------------------------------------------------------------------------*/

static int32_t ill_split_tail(const char *text, int32_t len, int32_t pos, int fold)
{
    static const char *forms[] = { "s", "t", "re", "ve", "m", "ll", "d" };
    int32_t index;
    if (pos >= len || text[pos] != '\'') return 0;
    for (index = 0; index < 7; ++index) {
        int32_t span = (int32_t)strlen(forms[index]);
        int32_t slot;
        if (pos + 1 + span > len) continue;
        for (slot = 0; slot < span; ++slot) {
            char cell = text[pos + 1 + slot];
            char want = forms[index][slot];
            if (cell != want && !(fold && cell == want - 32)) break;
        }
        if (slot == span) return 1 + span;
    }
    return 0;
}

static int32_t ill_split_kindrun(const char *text, int32_t len, int32_t pos, IllClass want)
{
    int32_t walk = pos;
    while (walk < len) {
        uint32_t rune;
        int32_t  step = ill_utf8_read(text, len, walk, &rune);
        if (step <= 0 || ill_rune_kind(rune) != want) break;
        walk += step;
    }
    return walk - pos;
}

static int32_t ill_split_markrun(const char *text, int32_t len, int32_t pos)
{
    int32_t walk = pos;
    while (walk < len) {
        uint32_t rune;
        int32_t  step = ill_utf8_read(text, len, walk, &rune);
        IllClass kind;
        if (step <= 0) break;
        kind = ill_rune_kind(rune);
        if (kind != ILL_CLASS_OTHER) break;
        walk += step;
    }
    return walk - pos;
}

/* Trailing whitespace rule: `\s+(?!\S)` first, then a plain `\s+`.  The
 * lookahead means a run that is followed by visible text gives up its final
 * character, so the next word keeps its leading space. */
static int32_t ill_split_blank(const char *text, int32_t len, int32_t pos)
{
    int32_t run = ill_split_kindrun(text, len, pos, ILL_CLASS_SPACE);
    int32_t last = pos, walk = pos;
    if (run <= 0) return 0;
    if (pos + run >= len) return run;
    while (walk < pos + run) {
        uint32_t rune;
        int32_t  step = ill_utf8_read(text, len, walk, &rune);
        if (step <= 0) break;
        last = walk;
        walk += step;
    }
    return last > pos ? last - pos : run;
}

static int32_t ill_split_take(uint8_t kind, const char *text, int32_t len, int32_t pos)
{
    uint32_t rune;
    int32_t  step, span, head;
    IllClass klass;

    step = ill_utf8_read(text, len, pos, &rune);
    if (step <= 0) return 0;

    span = ill_split_tail(text, len, pos, kind == ILL_SPLIT_LLAMA3);
    if (span) return span;

    if (kind == ILL_SPLIT_LLAMA3) {
        /* [^\r\n\p{L}\p{N}]? \p{L}+ */
        klass = ill_rune_kind(rune);
        head  = (klass != ILL_CLASS_LETTER && klass != ILL_CLASS_DIGIT &&
                 rune != '\r' && rune != '\n') ? step : 0;
        span = ill_split_kindrun(text, len, pos + head, ILL_CLASS_LETTER);
        if (span > 0) return head + span;
        /* \p{N}{1,3} */
        if (klass == ILL_CLASS_DIGIT) {
            int32_t walk = pos, taken = 0;
            while (walk < len && taken < 3) {
                uint32_t peek;
                int32_t  wide = ill_utf8_read(text, len, walk, &peek);
                if (wide <= 0 || ill_rune_kind(peek) != ILL_CLASS_DIGIT) break;
                walk += wide;
                ++taken;
            }
            return walk - pos;
        }
        /*  ?[^\s\p{L}\p{N}]+[\r\n]* */
        head = (rune == ' ') ? 1 : 0;
        span = ill_split_markrun(text, len, pos + head);
        if (span > 0) {
            int32_t walk = pos + head + span;
            while (walk < len && (text[walk] == '\r' || text[walk] == '\n')) ++walk;
            return walk - pos;
        }
        /* \s*[\r\n]+ */
        {
            int32_t run = ill_split_kindrun(text, len, pos, ILL_CLASS_SPACE);
            int32_t walk = pos + run;
            if (run > 0) {
                int32_t back = walk;
                while (back > pos && (text[back - 1] == '\r' || text[back - 1] == '\n')) --back;
                if (back < walk) return walk - pos;
            }
        }
        return ill_split_blank(text, len, pos);
    }

    /*  ?\p{L}+ |  ?\p{N}+ |  ?[^\s\p{L}\p{N}]+ */
    head = (rune == ' ') ? 1 : 0;
    span = ill_split_kindrun(text, len, pos + head, ILL_CLASS_LETTER);
    if (span > 0) return head + span;
    span = ill_split_kindrun(text, len, pos + head, ILL_CLASS_DIGIT);
    if (span > 0) return head + span;
    span = ill_split_markrun(text, len, pos + head);
    if (span > 0) return head + span;
    return ill_split_blank(text, len, pos);
}

/* -- byte pair merge ------------------------------------------------------- */

typedef struct IllSym  { int32_t id, prev, next; } IllSym;
typedef struct IllPair { int32_t rank, a, b, ida, idb; } IllPair;

typedef struct IllWeld {
    IllSym  *syms;
    IllPair *heap;
    int32_t  heap_count, heap_limit, sym_count;
} IllWeld;

static void ill_weld_push(IllWeld *weld, IllPair item)
{
    int32_t slot;
    if (weld->heap_count >= weld->heap_limit) return;
    slot = weld->heap_count++;
    weld->heap[slot] = item;
    while (slot > 0) {
        int32_t up = (slot - 1) / 2;
        if (weld->heap[up].rank < weld->heap[slot].rank ||
            (weld->heap[up].rank == weld->heap[slot].rank &&
             weld->heap[up].a <= weld->heap[slot].a)) break;
        item = weld->heap[up]; weld->heap[up] = weld->heap[slot]; weld->heap[slot] = item;
        slot = up;
    }
}

static int ill_weld_pull(IllWeld *weld, IllPair *out)
{
    int32_t slot = 0;
    if (weld->heap_count == 0) return 0;
    *out = weld->heap[0];
    weld->heap[0] = weld->heap[--weld->heap_count];
    for (;;) {
        int32_t kid = slot * 2 + 1;
        int32_t win = slot;
        IllPair swap;
        if (kid < weld->heap_count &&
            (weld->heap[kid].rank < weld->heap[win].rank ||
             (weld->heap[kid].rank == weld->heap[win].rank &&
              weld->heap[kid].a < weld->heap[win].a))) win = kid;
        if (kid + 1 < weld->heap_count &&
            (weld->heap[kid + 1].rank < weld->heap[win].rank ||
             (weld->heap[kid + 1].rank == weld->heap[win].rank &&
              weld->heap[kid + 1].a < weld->heap[win].a))) win = kid + 1;
        if (win == slot) break;
        swap = weld->heap[win]; weld->heap[win] = weld->heap[slot]; weld->heap[slot] = swap;
        slot = win;
    }
    return 1;
}

static void ill_weld_link(const IllVocab *vocab, IllWeld *weld, int32_t a, int32_t b)
{
    IllPair item;
    int32_t rank, join;
    if (a < 0 || b < 0) return;
    if (!ill_pair_seek(vocab, weld->syms[a].id, weld->syms[b].id, &rank, &join)) return;
    item.rank = rank; item.a = a; item.b = b;
    item.ida = weld->syms[a].id; item.idb = weld->syms[b].id;
    ill_weld_push(weld, item);
}

/* -- encode ---------------------------------------------------------------- */

typedef struct IllQuill {
    int32_t *tokens;
    int32_t  limit;
    int32_t  count;
} IllQuill;

static void ill_quill_put(IllQuill *quill, int32_t token)
{
    if (quill->tokens && quill->count < quill->limit) quill->tokens[quill->count] = token;
    ++quill->count;
}

static void ill_vocab_chunk(const IllVocab *vocab, IllWeld *weld, IllQuill *quill,
                            const char *chunk, int32_t span)
{
    int32_t index, head, walk;
    IllPair item;

    weld->sym_count  = 0;
    weld->heap_count = 0;
    for (index = 0; index < span; ++index) {
        int32_t id = vocab->byte_id[(unsigned char)chunk[index]];
        if (id < 0) continue;
        weld->syms[weld->sym_count].id   = id;
        weld->syms[weld->sym_count].prev = weld->sym_count - 1;
        weld->syms[weld->sym_count].next = weld->sym_count + 1;
        ++weld->sym_count;
    }
    if (weld->sym_count == 0) return;
    weld->syms[weld->sym_count - 1].next = -1;
    head = 0;

    for (index = 0; index + 1 < weld->sym_count; ++index)
        ill_weld_link(vocab, weld, index, index + 1);

    while (ill_weld_pull(weld, &item)) {
        IllSym *a = &weld->syms[item.a];
        IllSym *b;
        int32_t rank, join;
        if (a->id != item.ida || a->next != item.b) continue;
        b = &weld->syms[item.b];
        if (b->id != item.idb) continue;
        if (!ill_pair_seek(vocab, item.ida, item.idb, &rank, &join)) continue;
        a->id   = join;
        a->next = b->next;
        if (b->next >= 0) weld->syms[b->next].prev = item.a;
        b->id = -1;
        ill_weld_link(vocab, weld, a->prev, item.a);
        ill_weld_link(vocab, weld, item.a, a->next);
    }

    for (walk = head; walk >= 0; walk = weld->syms[walk].next)
        ill_quill_put(quill, weld->syms[walk].id);
}

static void ill_vocab_slice(const IllVocab *vocab, IllWeld *weld, IllQuill *quill,
                            const char *text, int32_t len)
{
    int32_t pos = 0;
    while (pos < len) {
        int32_t span = ill_split_take(vocab->split_kind, text, len, pos);
        if (span <= 0) span = 1;
        if (pos + span > len) span = len - pos;
        ill_vocab_chunk(vocab, weld, quill, text + pos, span);
        pos += span;
    }
}

IllResult ill_vocab_encode(const IllVocab *vocab, const char *text, size_t text_len,
                           int32_t add_begin, int32_t *tokens, int32_t limit,
                           int32_t *count)
{
    IllQuill quill;
    IllWeld  weld;
    char    *lead = NULL;
    int32_t  span = (int32_t)text_len;
    int32_t  pos  = 0, seg = 0;
    IllResult code = ILL_OK;

    if (!vocab || (!text && text_len)) return ILL_ARGS;
    quill.tokens = tokens;
    quill.limit  = tokens ? limit : 0;
    quill.count  = 0;

    if (add_begin && vocab->begin_token >= 0 && vocab->begin_token < vocab->count)
        ill_quill_put(&quill, vocab->begin_token);

    if (vocab->add_prefix && span >= 0) {
        lead = (char *)ill_block_make((size_t)span + 2);
        if (!lead) return ILL_ALLOC;
        lead[0] = ' ';
        memcpy(lead + 1, text, (size_t)span);
        text = lead;
        span += 1;
    }

    weld.syms = (IllSym *)ill_block_make((size_t)(span + 1) * sizeof(IllSym));
    weld.heap_limit = 4 * span + 16;
    weld.heap = (IllPair *)ill_block_make((size_t)weld.heap_limit * sizeof(IllPair));
    if (!weld.syms || !weld.heap) { code = ILL_ALLOC; goto done; }

    while (pos < span) {
        int32_t hit = -1, hit_span = 0, index;
        for (index = 0; index < vocab->extra_count; ++index) {
            int32_t id   = vocab->extra[index];
            int32_t want = vocab->pieces[id].span;
            if (want <= 0 || pos + want > span) continue;
            if (!memcmp(text + pos, vocab->pieces[id].text, (size_t)want)) {
                hit = id; hit_span = want; break;   /* extras are longest first */
            }
        }
        if (hit < 0) { ++pos; continue; }
        if (pos > seg) ill_vocab_slice(vocab, &weld, &quill, text + seg, pos - seg);
        ill_quill_put(&quill, hit);
        pos += hit_span;
        seg = pos;
    }
    if (span > seg) ill_vocab_slice(vocab, &weld, &quill, text + seg, span - seg);

    if (count) *count = quill.count;
    if (tokens && quill.count > limit) code = ILL_LIMIT;

done:
    ill_block_free(weld.syms);
    ill_block_free(weld.heap);
    ill_block_free(lead);
    if (code != ILL_OK && count) *count = quill.count;
    return code;
}

IllResult ill_vocab_decode(const IllVocab *vocab, int32_t token,
                           char *text, int32_t limit, int32_t *len)
{
    const IllPiece *piece;
    int32_t used = 0, pos = 0;

    if (!vocab || token < 0 || token >= vocab->count) return ILL_ARGS;
    piece = &vocab->pieces[token];

    if (piece->special) {
        if (len) *len = piece->span;
        if (!text || piece->span >= limit) return ILL_LIMIT;
        memcpy(text, piece->text, (size_t)piece->span);
        text[piece->span] = '\0';
        return ILL_OK;
    }

    while (pos < piece->span) {
        uint32_t rune;
        int32_t  step = ill_utf8_read(piece->text, piece->span, pos, &rune);
        if (step <= 0) break;
        pos += step;
        if (rune < 512u && vocab->rune_byte[rune] >= 0) {
            if (text && used < limit - 1) text[used] = (char)vocab->rune_byte[rune];
            ++used;
        } else {
            char cell[4];
            int32_t wide = ill_utf8_write(rune, cell);
            int32_t slot;
            for (slot = 0; slot < wide; ++slot) {
                if (text && used < limit - 1) text[used] = cell[slot];
                ++used;
            }
        }
    }
    if (len) *len = used;
    if (!text || used >= limit) return ILL_LIMIT;
    text[used] = '\0';
    return ILL_OK;
}

/* -- prompt shaping -------------------------------------------------------- */

typedef struct IllScribe {
    char  *text;
    size_t limit;
    size_t used;
} IllScribe;

static void ill_scribe_put(IllScribe *scribe, const char *word)
{
    size_t span = strlen(word);
    if (scribe->text && scribe->used + span < scribe->limit)
        memcpy(scribe->text + scribe->used, word, span);
    scribe->used += span;
}

IllResult ill_vocab_prompt(const IllVocab *vocab, const IllTurn *turns, int32_t count,
                           int32_t open_reply, char *text, size_t limit, size_t *len)
{
    IllScribe scribe;
    int32_t   index;

    if (!vocab || (!turns && count > 0)) return ILL_ARGS;
    scribe.text = text;
    scribe.limit = limit;
    scribe.used = 0;

    for (index = 0; index < count; ++index) {
        const char *role = turns[index].role ? turns[index].role : "user";
        const char *body = turns[index].text ? turns[index].text : "";
        if (vocab->chat_kind == ILL_CHAT_ML) {
            ill_scribe_put(&scribe, "<|im_start|>");
            ill_scribe_put(&scribe, role);
            ill_scribe_put(&scribe, "\n");
            ill_scribe_put(&scribe, body);
            ill_scribe_put(&scribe, "<|im_end|>\n");
        } else {
            ill_scribe_put(&scribe, role);
            ill_scribe_put(&scribe, ": ");
            ill_scribe_put(&scribe, body);
            ill_scribe_put(&scribe, "\n");
        }
    }
    if (open_reply) {
        if (vocab->chat_kind == ILL_CHAT_ML) ill_scribe_put(&scribe, "<|im_start|>assistant\n");
        else                                 ill_scribe_put(&scribe, "assistant: ");
    }

    if (len) *len = scribe.used;
    if (!text || scribe.used >= limit) return ILL_LIMIT;
    text[scribe.used] = '\0';
    return ILL_OK;
}

/* ============================================================================
 * part 14 -- sampler
 *
 * The filters run in the order that keeps them meaningful: repetition penalty
 * on raw logits, then temperature, then the candidate cuts (top-k, top-p,
 * min-p), then one draw from what survives.  A temperature of zero short
 * circuits the whole chain to an argmax.
 * ==========================================================================*/

typedef struct IllCand { float mass; int32_t token; } IllCand;

struct IllSampler {
    IllTuning tuning;
    int32_t   vocab_size;
    IllCand  *cands;
    int32_t  *recent;      /* ring of recent tokens for the penalty          */
    int32_t   recent_span, recent_fill, recent_head;
    uint64_t  seed;
};

void ill_tuning_init(IllTuning *tuning)
{
    memset(tuning, 0, sizeof(*tuning));
    tuning->temperature    = 0.7f;
    tuning->top_p          = 0.95f;
    tuning->min_p          = 0.0f;
    tuning->top_k          = 0;
    tuning->repeat_penalty = 1.0f;
    tuning->repeat_span    = 64;
    tuning->seed           = 0;
}

/* splitmix64: small, well distributed, and reproducible across hosts. */
static uint64_t ill_seed_next(uint64_t *seed)
{
    uint64_t cell = (*seed += 0x9E3779B97F4A7C15ULL);
    cell = (cell ^ (cell >> 30)) * 0xBF58476D1CE4E5B9ULL;
    cell = (cell ^ (cell >> 27)) * 0x94D049BB133111EBULL;
    return cell ^ (cell >> 31);
}

static float ill_seed_real(uint64_t *seed)
{
    return (float)((ill_seed_next(seed) >> 40) * (1.0 / 16777216.0));
}

IllResult ill_sampler_make(IllSampler **out, const IllTuning *tuning, int32_t vocab_size)
{
    IllSampler *sampler;
    if (!out || vocab_size <= 0) return ILL_ARGS;
    *out = NULL;
    sampler = (IllSampler *)ill_block_zero(sizeof(IllSampler));
    if (!sampler) return ILL_ALLOC;
    if (tuning) sampler->tuning = *tuning;
    else        ill_tuning_init(&sampler->tuning);
    sampler->vocab_size  = vocab_size;
    sampler->recent_span = ILL_MAX(sampler->tuning.repeat_span, 1);
    sampler->cands  = (IllCand *)ill_block_make((size_t)vocab_size * sizeof(IllCand));
    sampler->recent = (int32_t *)ill_block_zero((size_t)sampler->recent_span * sizeof(int32_t));
    if (!sampler->cands || !sampler->recent) {
        ill_block_free(sampler->cands);
        ill_block_free(sampler->recent);
        ill_block_free(sampler);
        return ILL_ALLOC;
    }
    sampler->seed = sampler->tuning.seed ? sampler->tuning.seed : 0x2545F4914F6CDD1DULL;
    return (*out = sampler), ILL_OK;
}

void ill_sampler_free(IllSampler *sampler)
{
    if (!sampler) return;
    ill_block_free(sampler->cands);
    ill_block_free(sampler->recent);
    ill_block_free(sampler);
}

void ill_sampler_wipe(IllSampler *sampler)
{
    if (!sampler) return;
    sampler->recent_fill = 0;
    sampler->recent_head = 0;
    sampler->seed = sampler->tuning.seed ? sampler->tuning.seed : 0x2545F4914F6CDD1DULL;
}

void ill_sampler_note(IllSampler *sampler, int32_t token)
{
    if (!sampler) return;
    sampler->recent[sampler->recent_head] = token;
    sampler->recent_head = (sampler->recent_head + 1) % sampler->recent_span;
    if (sampler->recent_fill < sampler->recent_span) ++sampler->recent_fill;
}

static int ill_cand_order(const void *a, const void *b)
{
    const IllCand *x = (const IllCand *)a;
    const IllCand *y = (const IllCand *)b;
    if (x->mass > y->mass) return -1;
    if (x->mass < y->mass) return  1;
    return x->token < y->token ? -1 : 1;
}

int32_t ill_sampler_pick(IllSampler *sampler, float *logits)
{
    const IllTuning *tune;
    int32_t count, index, keep;
    float   peak, mass, draw;

    if (!sampler || !logits) return -1;
    tune  = &sampler->tuning;
    count = sampler->vocab_size;

    if (tune->repeat_penalty != 1.0f && tune->repeat_penalty > 0.0f) {
        for (index = 0; index < sampler->recent_fill; ++index) {
            int32_t token = sampler->recent[index];
            if (token < 0 || token >= count) continue;
            logits[token] = logits[token] > 0.0f ? logits[token] / tune->repeat_penalty
                                                 : logits[token] * tune->repeat_penalty;
        }
    }

    if (tune->temperature <= 0.0f) {
        int32_t best = 0;
        for (index = 1; index < count; ++index)
            if (logits[index] > logits[best]) best = index;
        return best;
    }

    peak = -FLT_MAX;
    for (index = 0; index < count; ++index) if (logits[index] > peak) peak = logits[index];
    mass = 0.0f;
    for (index = 0; index < count; ++index) {
        float cell = expf((logits[index] - peak) / tune->temperature);
        sampler->cands[index].mass  = cell;
        sampler->cands[index].token = index;
        mass += cell;
    }
    for (index = 0; index < count; ++index) sampler->cands[index].mass /= mass;

    qsort(sampler->cands, (size_t)count, sizeof(IllCand), ill_cand_order);
    keep = count;

    if (tune->top_k > 0 && tune->top_k < keep) keep = tune->top_k;

    if (tune->min_p > 0.0f) {
        float floor_mass = tune->min_p * sampler->cands[0].mass;
        int32_t cut = 1;
        while (cut < keep && sampler->cands[cut].mass >= floor_mass) ++cut;
        keep = cut;
    }

    if (tune->top_p > 0.0f && tune->top_p < 1.0f) {
        float run = 0.0f;
        int32_t cut = 0;
        while (cut < keep) {
            run += sampler->cands[cut].mass;
            ++cut;
            if (run >= tune->top_p) break;
        }
        keep = cut;
    }
    if (keep < 1) keep = 1;

    mass = 0.0f;
    for (index = 0; index < keep; ++index) mass += sampler->cands[index].mass;
    draw = ill_seed_real(&sampler->seed) * mass;
    for (index = 0; index < keep; ++index) {
        draw -= sampler->cands[index].mass;
        if (draw <= 0.0f) return sampler->cands[index].token;
    }
    return sampler->cands[keep - 1].token;
}

/* ============================================================================
 * part 15 -- facade
 * ==========================================================================*/

const char *ill_result_text(IllResult code)
{
    switch (code) {
        case ILL_OK:    return "ok";
        case ILL_ARGS:  return "bad argument";
        case ILL_ALLOC: return "out of memory";
        case ILL_FILE:  return "file not readable";
        case ILL_PARSE: return "malformed document";
        case ILL_SHAPE: return "unexpected tensor shape";
        case ILL_MODEL: return "unsupported checkpoint";
        case ILL_VOCAB: return "unsupported tokenizer";
        case ILL_LIMIT: return "buffer or window too small";
        case ILL_STATE: return "not valid in this state";
        default:        return "unknown";
    }
}

#ifdef __cplusplus
}
#endif

#endif /* ILL_CORE_INCLUDED */

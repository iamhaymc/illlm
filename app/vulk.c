/* ============================================================================
 * app/vulk.c -- the Vulkan compute backend for both engines.
 *
 * `app/core.c` and `app/yolo.c` each end their kernel section with a table of
 * function pointers -- IllBackend's six operations and YoloBackend's nine --
 * and each fills that table with a CPU implementation that is the reference
 * for what the operation means.  Nothing had ever been written on the other
 * side of either seam.  This file is that other side, once, for both, because
 * the halves that differ are fourteen compute shaders and the half that is the
 * same -- finding a device, allocating on it, building a pipeline, getting a
 * dispatch to run and the answer back -- is about two thousand lines that
 * neither engine should own twice.
 *
 * Callers include it once, after whichever engines they want it to serve:
 *
 *     #include "core.c"
 *     #include "vulk.c"
 *     ...
 *     vulk_join();                      -- adds "vulkan" to the registry
 *     plan.backend_name = "vulkan";     -- or --backend vulkan on the command line
 *
 * and for the picture engine, which has a setter rather than a registry:
 *
 *     yolo_model_backend_set(model, yolo_backend_vulkan());
 *
 * It includes `core.c` and `yolo.c` itself if they are not already in, so the
 * include order above is a convenience rather than a requirement.  Define
 * VULK_NO_TEXT or VULK_NO_PICTURE to leave one of the two halves out; the
 * command line builds the text engine without the picture one so that `ill`
 * does not grow the image reader it has no use for.
 *
 * ---------------------------------------------------------------------------
 * Three decisions shape the whole file, and each of them is a refusal.
 *
 * **No dependency, at any level.**  There is no vulkan.h here, no SDK, no
 * `-lvulkan`, and nothing added to `util/make.py` beyond one define.  This
 * file declares the two dozen structures and fifty entry points it uses
 * against the published ABI and resolves them through `dlopen` at run time.
 * A build on a host with no Vulkan at all still compiles and still runs; the
 * backend reports itself unavailable and the caller keeps the CPU one.  That
 * is the only arrangement that does not make a GPU a build requirement for a
 * project whose whole claim is that a C compiler is enough.
 *
 * **The shaders are assembled here, at run time, into SPIR-V.**  The usual
 * arrangement -- GLSL compiled by `glslc` into a blob pasted into the source
 * -- needs a shader compiler in the build and leaves twelve kilobytes of hex
 * in the tree that nobody can read or check.  Part 3 is a SPIR-V assembler,
 * about seven hundred lines, and part 4 writes the fourteen kernels against
 * it.
 * The cost is that a kernel reads as a sequence of calls rather than as
 * arithmetic; the gain is that every instruction the device runs is visible
 * in this file, the build gains no step, and a reader can follow a kernel from
 * its first load to its last store without a second tool.
 *
 * **Quantised weights are widened once, on the way to the device.**  A q4
 * plane is a fifth of the size of the f32 it came from, and that is why the
 * text engine has it; on the device it is expanded back to f32 as it is
 * uploaded, and the saving is lost.  The alternative -- unpacking blocks
 * inside the dense kernel -- is a better use of device memory and a worse use
 * of the first version of this file, because it needs one kernel per weight
 * format and the formats are where the parity argument is hardest.  Widening
 * is exact for f32, f16 and bf16 and reproduces the CPU's own rounding for q8
 * and q4, which means a device run and a CPU run differ only by the order the
 * additions happen in.  `TODO.md` item 52 is the packed form.
 *
 * ---------------------------------------------------------------------------
 * What this is worth, and where it is not worth anything.
 *
 * Both seams pass host pointers.  `dense` is handed a `const float *src` and
 * a `float *dst`, and `conv_run` is handed two `YoloPlane`s whose cells are
 * ordinary malloc memory, so a device backend has to upload its inputs and
 * download its outputs on **every** call -- there is nowhere in either
 * interface to say "the answer is already on the device, leave it there".
 * Weights are cached, because they never change after load, and that is most
 * of the traffic; activations are not, because nothing in the seam says when
 * the host last wrote them.
 *
 * The consequence is worth stating plainly rather than discovering.  A
 * single-token step of a 2.6B text model is a hundred-odd dispatches of a few
 * microseconds each, and the submit-and-wait around every one of them costs
 * more than the arithmetic inside it; a 320 picture through yolo26n is 139
 * submissions and 46.4 MiB moved against 12 MiB of resident weights.  Whether
 * a large enough shape -- a prefill, a wide vocabulary projection -- has
 * enough work per round trip to come out ahead is **not measured**, and is not
 * claimed here: the only Vulkan device this was written on is a software
 * rasterizer, and nothing it does is a GPU rate.  `TODO.md` item 55 is the
 * measurement and item 51 is the seam change that would make it moot; the
 * latter is a change to `core.c` and `yolo.c` rather than to this file.
 *
 * Layout
 *   part 1   the ABI -- what this file declares of Vulkan, and the loader
 *   part 2   device -- instance, queue, memory, buffers, submission
 *   part 3   spirv -- the assembler the kernels are written against
 *   part 4   kernels -- the fourteen shaders
 *   part 5   pipelines -- modules, layouts, descriptors, dispatch
 *   part 6   residence -- device copies of host memory, and the weight cache
 *   part 7   opening and closing a device
 *   part 8   one operation, end to end
 *   part 9   the IllBackend table
 *   part 10  the YoloBackend table
 *   part 11  the comparison, under VULK_MAIN or VULK_PROBE
 *
 * Licensed under the terms in LICENSE.
 * ==========================================================================*/

#ifndef VULK_INCLUDED
#define VULK_INCLUDED

/* The engines this file serves.  Each guards its own contents, and the test
 * suite includes both before it includes this, so the tests are what these
 * conditions are for. */
#if !defined(VULK_NO_TEXT) && !defined(ILL_CORE_INCLUDED)
#include "core.c"
#endif
#if !defined(VULK_NO_PICTURE) && !defined(YOLO_INCLUDED)
#include "yolo.c"
#endif

#include <stddef.h>
#include <stdint.h>

typedef enum VulkStatus {
    VULK_OK = 0,
    VULK_NO_LOADER,   /* no Vulkan loader on this host                      */
    VULK_NO_DEVICE,   /* a loader, but no device with a compute queue       */
    VULK_NO_MEMORY,   /* the device refused an allocation                   */
    VULK_NO_SHADER,   /* a pipeline would not build                         */
    VULK_LOST,        /* a submission failed or the device went away        */
    VULK_ARGS,        /* a shape this backend does not accept               */
    VULK_STATUS_COUNT
} VulkStatus;

const char *vulk_status_text(VulkStatus status);

/* Opens the device.  `want` names a device by a substring of its reported
 * name, or is null for the first one that carries a compute queue.  Calling it
 * twice is harmless; the second call returns the state of the first. */
VulkStatus  vulk_open(const char *want);
void        vulk_close(void);
int         vulk_ready(void);               /* non-zero once a device is up */
const char *vulk_device_name(void);         /* "none" before vulk_open      */

/* Bytes of device memory this backend is holding, and the ceiling it will
 * hold in cached weights before it starts evicting.  The budget may be set
 * before or after the device is opened. */
size_t vulk_memory_used(void);
void   vulk_memory_budget_set(size_t bytes);

/* What a run cost the host: how many times it waited on the device, and how
 * many bytes crossed in either direction.  Both seams pass host pointers, so
 * these two numbers are the ones that say whether the interface rather than
 * the arithmetic is what is in the way. */
void vulk_report(uint64_t *sent_out, uint64_t *moved_out);

#ifndef VULK_NO_TEXT
/* The text engine's table.  Registering it does not open a device -- that
 * happens in `setup`, when a model is loaded against it -- so this may be
 * called unconditionally at start up on a host with no Vulkan. */
IllResult vulk_join(void);
#endif

#ifndef VULK_NO_PICTURE
/* The picture engine's table.  Null until `vulk_open` has succeeded, because
 * `yolo_model_backend_set` has no way to report a failure later. */
const YoloBackend *yolo_backend_vulkan(void);
#endif

#endif /* VULK_INCLUDED */

/* ==========================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <float.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

/* ============================================================================
 * part 1 -- the ABI
 *
 * Everything below is a transcription of the Vulkan 1.0 ABI, which is frozen:
 * a structure that has shipped never changes its layout, and a new version
 * adds structures rather than editing them.  Only what this file calls is
 * here.  Two conventions from the specification are load bearing and easy to
 * get wrong, so they are stated rather than assumed:
 *
 *   - **Dispatchable handles are pointers** (instance, physical device,
 *     device, queue, command buffer) and **non-dispatchable handles are always
 *     64 bits**, on a 32 bit host as much as a 64 bit one.  Declaring a buffer
 *     handle as a pointer builds and then corrupts the stack on a 32 bit
 *     build, which is the one mistake in this area that does not show up on
 *     the machine it was written on.
 *
 *   - **Every structure begins with an `sType` tag and a `pNext` pointer**,
 *     and a driver reads the tag rather than trusting the call it arrived
 *     through.  The tags are the values from the specification, written out
 *     here rather than derived, because the derivation is a macro in a header
 *     this file refuses to depend on.
 * ==========================================================================*/

typedef void         *VulkHandle;    /* a dispatchable handle               */
typedef uint64_t      VulkObject;    /* a non-dispatchable handle           */
typedef uint64_t      VulkSize;      /* VkDeviceSize                        */
typedef int32_t       VulkResult;    /* VkResult, negative on failure       */

#define VULK_SUCCESS            0

/* structure tags */
#define VULK_S_APPLICATION            0
#define VULK_S_INSTANCE               1
#define VULK_S_DEVICE_QUEUE           2
#define VULK_S_DEVICE                 3
#define VULK_S_SUBMIT                 4
#define VULK_S_MEMORY_ALLOCATE        5
#define VULK_S_FENCE                  8
#define VULK_S_BUFFER                 12
#define VULK_S_SHADER_MODULE          16
#define VULK_S_PIPELINE_SHADER_STAGE  18
#define VULK_S_COMPUTE_PIPELINE       29
#define VULK_S_PIPELINE_LAYOUT        30
#define VULK_S_DESCRIPTOR_POOL        33
#define VULK_S_DESCRIPTOR_SET_ALLOC   34
#define VULK_S_DESCRIPTOR_SET_LAYOUT  32
#define VULK_S_WRITE_DESCRIPTOR_SET   35
#define VULK_S_COMMAND_POOL           39
#define VULK_S_COMMAND_BUFFER_ALLOC   40
#define VULK_S_COMMAND_BUFFER_BEGIN   42
#define VULK_S_MEMORY_BARRIER         46

/* queue flags */
#define VULK_QUEUE_GRAPHICS  0x01
#define VULK_QUEUE_COMPUTE   0x02

/* memory property flags */
#define VULK_MEM_DEVICE_LOCAL   0x01
#define VULK_MEM_HOST_VISIBLE   0x02
#define VULK_MEM_HOST_COHERENT  0x04

/* buffer usage flags */
#define VULK_USE_TRANSFER_SRC   0x01
#define VULK_USE_TRANSFER_DST   0x02
#define VULK_USE_STORAGE        0x20

/* descriptor type: storage buffer */
#define VULK_DESC_STORAGE_BUFFER 7

/* pipeline stage and access flags, for the one barrier this file emits */
#define VULK_STAGE_TRANSFER        0x1000
#define VULK_STAGE_COMPUTE_SHADER  0x0800
#define VULK_STAGE_HOST            0x4000
#define VULK_ACCESS_SHADER_READ    0x0020
#define VULK_ACCESS_SHADER_WRITE   0x0040
#define VULK_ACCESS_TRANSFER_READ  0x0800
#define VULK_ACCESS_TRANSFER_WRITE 0x1000
#define VULK_ACCESS_HOST_READ      0x2000
#define VULK_ACCESS_HOST_WRITE     0x4000

#define VULK_PIPELINE_COMPUTE   1
#define VULK_STAGE_BIT_COMPUTE  0x20
#define VULK_CMD_ONE_TIME       0x01
#define VULK_SHARING_EXCLUSIVE  0
#define VULK_WHOLE_SIZE         (~(VulkSize)0)

typedef struct VulkAppInfo {
    uint32_t    type;
    const void *next;
    const char *app_name;
    uint32_t    app_version;
    const char *engine_name;
    uint32_t    engine_version;
    uint32_t    api_version;
} VulkAppInfo;

typedef struct VulkInstanceInfo {
    uint32_t            type;
    const void         *next;
    uint32_t            flags;
    const VulkAppInfo  *app;
    uint32_t            layer_count;
    const char *const  *layer_list;
    uint32_t            ext_count;
    const char *const  *ext_list;
} VulkInstanceInfo;

/* VkPhysicalDeviceLimits, up to the compute fields this file reads.  The tail
 * is padding: the driver writes the whole structure, so the space has to be
 * there even though nothing below reads it, and 1024 bytes is comfortably more
 * than the 504 the structure occupies on any host. */
typedef struct VulkLimits {
    uint32_t max_image_1d, max_image_2d, max_image_3d, max_image_cube;
    uint32_t max_image_layers, max_texel_elements;
    uint32_t max_uniform_range, max_storage_range;
    uint32_t max_push_bytes, max_allocations, max_samplers;
    VulkSize buffer_image_grain, sparse_space;
    uint32_t max_bound_sets;
    uint32_t stage_samplers, stage_uniforms, stage_storages, stage_sampled,
             stage_images, stage_inputs, stage_resources;
    uint32_t set_samplers, set_uniforms, set_uniforms_dyn, set_storages,
             set_storages_dyn, set_sampled, set_images, set_inputs;
    uint32_t vertex_attributes, vertex_bindings, vertex_offset, vertex_stride,
             vertex_outputs;
    uint32_t tess_level, tess_patch, tess_in_vertex, tess_out_vertex,
             tess_out_patch, tess_out_total, tess_eval_in, tess_eval_out;
    uint32_t geom_invocations, geom_in, geom_out, geom_vertices, geom_total;
    uint32_t frag_in, frag_attachments, frag_dual, frag_resources;
    uint32_t max_shared_bytes;
    uint32_t max_group_count[3];
    uint32_t max_group_invocations;
    uint32_t max_group_size[3];
    uint8_t  tail[1024];
} VulkLimits;

typedef struct VulkPhysProps {
    uint32_t   api_version;
    uint32_t   driver_version;
    uint32_t   vendor_id;
    uint32_t   device_id;
    uint32_t   device_kind;      /* 1 integrated, 2 discrete, 4 cpu         */
    char       device_name[256];
    uint8_t    cache_uuid[16];
    VulkLimits limits;
    uint8_t    sparse[64];
} VulkPhysProps;

typedef struct VulkQueueProps {
    uint32_t flags;
    uint32_t count;
    uint32_t timestamp_bits;
    uint32_t grain[3];
} VulkQueueProps;

typedef struct VulkMemType { uint32_t flags; uint32_t heap; } VulkMemType;
typedef struct VulkMemHeap { VulkSize size; uint32_t flags; uint32_t pad; } VulkMemHeap;

typedef struct VulkMemProps {
    uint32_t    type_count;
    VulkMemType type_list[32];
    uint32_t    heap_count;
    VulkMemHeap heap_list[16];
} VulkMemProps;

typedef struct VulkQueueInfo {
    uint32_t     type;
    const void  *next;
    uint32_t     flags;
    uint32_t     family;
    uint32_t     count;
    const float *priority_list;
} VulkQueueInfo;

typedef struct VulkDeviceInfo {
    uint32_t             type;
    const void          *next;
    uint32_t             flags;
    uint32_t             queue_count;
    const VulkQueueInfo *queue_list;
    uint32_t             layer_count;
    const char *const   *layer_list;
    uint32_t             ext_count;
    const char *const   *ext_list;
    const void          *features;
} VulkDeviceInfo;

typedef struct VulkBufferInfo {
    uint32_t        type;
    const void     *next;
    uint32_t        flags;
    VulkSize        size;
    uint32_t        usage;
    uint32_t        sharing;
    uint32_t        family_count;
    const uint32_t *family_list;
} VulkBufferInfo;

typedef struct VulkMemNeeds { VulkSize size; VulkSize align; uint32_t bits; } VulkMemNeeds;

typedef struct VulkMemAlloc {
    uint32_t    type;
    const void *next;
    VulkSize    size;
    uint32_t    index;
} VulkMemAlloc;

typedef struct VulkBindingInfo {
    uint32_t          binding;
    uint32_t          kind;
    uint32_t          count;
    uint32_t          stages;
    const VulkObject *samplers;
} VulkBindingInfo;

typedef struct VulkSetLayoutInfo {
    uint32_t               type;
    const void            *next;
    uint32_t               flags;
    uint32_t               binding_count;
    const VulkBindingInfo *binding_list;
} VulkSetLayoutInfo;

typedef struct VulkPoolSize { uint32_t kind; uint32_t count; } VulkPoolSize;

typedef struct VulkPoolInfo {
    uint32_t            type;
    const void         *next;
    uint32_t            flags;
    uint32_t            max_sets;
    uint32_t            size_count;
    const VulkPoolSize *size_list;
} VulkPoolInfo;

typedef struct VulkSetAllocInfo {
    uint32_t          type;
    const void       *next;
    VulkObject        pool;
    uint32_t          count;
    const VulkObject *layout_list;
} VulkSetAllocInfo;

typedef struct VulkBufferBind {
    VulkObject buffer; VulkSize offset; VulkSize range;
} VulkBufferBind;

typedef struct VulkWriteSet {
    uint32_t              type;
    const void           *next;
    VulkObject            set;
    uint32_t              binding;
    uint32_t              element;
    uint32_t              count;
    uint32_t              kind;
    const void           *images;
    const VulkBufferBind *buffers;
    const void           *texels;
} VulkWriteSet;

typedef struct VulkPushRange { uint32_t stages; uint32_t offset; uint32_t size; } VulkPushRange;

typedef struct VulkLayoutInfo {
    uint32_t             type;
    const void          *next;
    uint32_t             flags;
    uint32_t             set_count;
    const VulkObject    *set_list;
    uint32_t             push_count;
    const VulkPushRange *push_list;
} VulkLayoutInfo;

typedef struct VulkModuleInfo {
    uint32_t        type;
    const void     *next;
    uint32_t        flags;
    size_t          bytes;
    const uint32_t *words;
} VulkModuleInfo;

typedef struct VulkStageInfo {
    uint32_t    type;
    const void *next;
    uint32_t    flags;
    uint32_t    stage;
    VulkObject  module;
    const char *entry;
    const void *special;
} VulkStageInfo;

typedef struct VulkComputeInfo {
    uint32_t      type;
    const void   *next;
    uint32_t      flags;
    VulkStageInfo stage;
    VulkObject    layout;
    VulkObject    parent;
    int32_t       parent_index;
} VulkComputeInfo;

typedef struct VulkCmdPoolInfo {
    uint32_t    type;
    const void *next;
    uint32_t    flags;
    uint32_t    family;
} VulkCmdPoolInfo;

typedef struct VulkCmdAllocInfo {
    uint32_t    type;
    const void *next;
    VulkObject  pool;
    uint32_t    level;
    uint32_t    count;
} VulkCmdAllocInfo;

typedef struct VulkCmdBeginInfo {
    uint32_t    type;
    const void *next;
    uint32_t    flags;
    const void *inherit;
} VulkCmdBeginInfo;

typedef struct VulkSubmitInfo {
    uint32_t          type;
    const void       *next;
    uint32_t          wait_count;
    const VulkObject *wait_list;
    const uint32_t   *wait_stages;
    uint32_t          command_count;
    const VulkHandle *command_list;
    uint32_t          signal_count;
    const VulkObject *signal_list;
} VulkSubmitInfo;

typedef struct VulkFenceInfo { uint32_t type; const void *next; uint32_t flags; } VulkFenceInfo;

typedef struct VulkBarrier {
    uint32_t    type;
    const void *next;
    uint32_t    src_access;
    uint32_t    dst_access;
} VulkBarrier;

typedef struct VulkCopy { VulkSize src; VulkSize dst; VulkSize size; } VulkCopy;

/* -- the entry points ----------------------------------------------------- */

typedef void *(*VulkGetProc)(VulkHandle owner, const char *name);

typedef struct VulkApi {
    /* loader and instance */
    VulkResult (*instance_make)(const VulkInstanceInfo *, const void *, VulkHandle *);
    void       (*instance_drop)(VulkHandle, const void *);
    VulkResult (*phys_list)(VulkHandle, uint32_t *, VulkHandle *);
    void       (*phys_props)(VulkHandle, VulkPhysProps *);
    void       (*phys_queues)(VulkHandle, uint32_t *, VulkQueueProps *);
    void       (*phys_memory)(VulkHandle, VulkMemProps *);
    VulkResult (*device_make)(VulkHandle, const VulkDeviceInfo *, const void *, VulkHandle *);
    void      *(*device_proc)(VulkHandle, const char *);
    /* device */
    void       (*device_drop)(VulkHandle, const void *);
    VulkResult (*device_idle)(VulkHandle);
    void       (*queue_get)(VulkHandle, uint32_t, uint32_t, VulkHandle *);
    VulkResult (*queue_submit)(VulkHandle, uint32_t, const VulkSubmitInfo *, VulkObject);
    /* memory */
    VulkResult (*buffer_make)(VulkHandle, const VulkBufferInfo *, const void *, VulkObject *);
    void       (*buffer_drop)(VulkHandle, VulkObject, const void *);
    void       (*buffer_needs)(VulkHandle, VulkObject, VulkMemNeeds *);
    VulkResult (*memory_make)(VulkHandle, const VulkMemAlloc *, const void *, VulkObject *);
    void       (*memory_drop)(VulkHandle, VulkObject, const void *);
    VulkResult (*buffer_bind)(VulkHandle, VulkObject, VulkObject, VulkSize);
    VulkResult (*memory_map)(VulkHandle, VulkObject, VulkSize, VulkSize, uint32_t, void **);
    void       (*memory_unmap)(VulkHandle, VulkObject);
    /* descriptors and pipelines */
    VulkResult (*set_layout_make)(VulkHandle, const VulkSetLayoutInfo *,
                                 const void *, VulkObject *);
    void       (*set_layout_drop)(VulkHandle, VulkObject, const void *);
    VulkResult (*pool_make)(VulkHandle, const VulkPoolInfo *, const void *, VulkObject *);
    void       (*pool_drop)(VulkHandle, VulkObject, const void *);
    VulkResult (*set_alloc)(VulkHandle, const VulkSetAllocInfo *, VulkObject *);
    void       (*set_write)(VulkHandle, uint32_t, const VulkWriteSet *, uint32_t, const void *);
    VulkResult (*layout_make)(VulkHandle, const VulkLayoutInfo *, const void *, VulkObject *);
    void       (*layout_drop)(VulkHandle, VulkObject, const void *);
    VulkResult (*module_make)(VulkHandle, const VulkModuleInfo *, const void *, VulkObject *);
    void       (*module_drop)(VulkHandle, VulkObject, const void *);
    VulkResult (*compute_make)(VulkHandle, VulkObject, uint32_t, const VulkComputeInfo *,
                               const void *, VulkObject *);
    void       (*pipeline_drop)(VulkHandle, VulkObject, const void *);
    /* commands */
    VulkResult (*cmd_pool_make)(VulkHandle, const VulkCmdPoolInfo *, const void *, VulkObject *);
    void       (*cmd_pool_drop)(VulkHandle, VulkObject, const void *);
    VulkResult (*cmd_alloc)(VulkHandle, const VulkCmdAllocInfo *, VulkHandle *);
    VulkResult (*cmd_begin)(VulkHandle, const VulkCmdBeginInfo *);
    VulkResult (*cmd_end)(VulkHandle);
    VulkResult (*cmd_reset)(VulkHandle, uint32_t);
    void       (*cmd_bind_pipeline)(VulkHandle, uint32_t, VulkObject);
    void       (*cmd_bind_sets)(VulkHandle, uint32_t, VulkObject, uint32_t, uint32_t,
                                const VulkObject *, uint32_t, const uint32_t *);
    void       (*cmd_push)(VulkHandle, VulkObject, uint32_t, uint32_t, uint32_t, const void *);
    void       (*cmd_dispatch)(VulkHandle, uint32_t, uint32_t, uint32_t);
    void       (*cmd_barrier)(VulkHandle, uint32_t, uint32_t, uint32_t,
                              uint32_t, const VulkBarrier *,
                              uint32_t, const void *, uint32_t, const void *);
    void       (*cmd_copy)(VulkHandle, VulkObject, VulkObject, uint32_t, const VulkCopy *);
    /* fences */
    VulkResult (*fence_make)(VulkHandle, const VulkFenceInfo *, const void *, VulkObject *);
    void       (*fence_drop)(VulkHandle, VulkObject, const void *);
    VulkResult (*fence_reset)(VulkHandle, uint32_t, const VulkObject *);
    VulkResult (*fence_wait)(VulkHandle, uint32_t, const VulkObject *, uint32_t, uint64_t);
} VulkApi;

/* The loader is opened by name, in the order a host is likely to have one.
 * Nothing is linked: a build on a machine with no Vulkan at all still runs,
 * and this is the function that says so. */
static void *vulk_lib_open(void)
{
#if defined(_WIN32)
    return (void *)LoadLibraryA("vulkan-1.dll");
#elif defined(__APPLE__)
    void *lib = dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("libMoltenVK.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("libvulkan.dylib", RTLD_NOW | RTLD_LOCAL);
    return lib;
#else
    void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    return lib;
#endif
}

static void vulk_lib_close(void *lib)
{
    if (!lib) return;
#if defined(_WIN32)
    FreeLibrary((HMODULE)lib);
#else
    dlclose(lib);
#endif
}

static void *vulk_lib_find(void *lib, const char *name)
{
#if defined(_WIN32)
    return (void *)(uintptr_t)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

/* ============================================================================
 * part 2 -- device
 *
 * One instance, one physical device, one compute queue, one command buffer and
 * one fence, held in a single file-static context because both engines share
 * them and neither has anywhere to put a handle.  Everything is created on the
 * first `vulk_open` and torn down on `vulk_close`; a model that outlives the
 * close has its `setup` state closed first, which is why `close` on the
 * IllBackend table calls through to here rather than the other way round.
 *
 * The one decision worth explaining is how memory is chosen.  A discrete card
 * has device memory the host cannot see and host memory the device reads over
 * the bus, so every buffer needs a staging copy and a `vkCmdCopyBuffer`.  An
 * integrated part -- which is what a laptop running this actually has -- often
 * reports a memory type that is both DEVICE_LOCAL and HOST_VISIBLE, and there
 * the staging copy is pure loss: the pointer `vkMapMemory` hands back is the
 * memory the shader reads.  `unified_flag` is that case, and it removes both
 * the second allocation and the copy from every upload and download in the
 * file.  On SwiftShader, which is where this was developed, everything is
 * unified, so `ILL_VULKAN_STAGED` in the environment makes the picker refuse
 * the unified type and take the staged path instead -- which every discrete
 * card takes, and which nothing on this host would otherwise run.  The test
 * suite passes under it as well as without it.
 * ==========================================================================*/

typedef struct VulkSlab {
    VulkObject buffer;
    VulkObject memory;
    VulkSize   bytes;
    void      *mapped;     /* null unless the memory is host visible        */
    int        host_flag;  /* allocated from the host visible type          */
} VulkSlab;

#define VULK_SLAB_FREE 32   /* scratch buffers kept alive between operations */

typedef struct VulkHold {   /* one cached upload of immutable host memory   */
    const void *key;        /* the host pointer it was read from            */
    size_t      bytes;      /* how much of it, so a re-shape is a miss       */
    uint64_t    seen;       /* the tick it was last used, for eviction      */
    int         pin;        /* in use by the operation being set up         */
    VulkSlab    slab;
} VulkHold;

#define VULK_HOLD_MAX 512

typedef struct VulkDev {
    void       *lib;
    VulkApi     api;
    VulkHandle  instance;
    VulkHandle  phys;
    VulkHandle  device;
    VulkHandle  queue;
    VulkHandle  cmd;
    VulkObject  cmd_pool;
    VulkObject  fence;
    uint32_t    family;
    char        name[256];
    uint32_t    kind;
    uint32_t    group_limit;     /* workgroups per dispatch dimension       */
    uint32_t    shared_limit;    /* bytes of workgroup memory               */
    uint32_t    storage_limit;   /* bytes a single storage binding may span */
    int         unified_flag;
    uint32_t    device_type;     /* memory type index for compute buffers   */
    uint32_t    host_type;       /* memory type index for staging           */
    /* the scratch pool: buffers kept between operations so that a forward
     * pass does not allocate, which is the one thing a driver is slow at */
    VulkSlab    free_list[VULK_SLAB_FREE];
    int32_t     free_used;
    /* the weight cache */
    VulkHold    hold_list[VULK_HOLD_MAX];
    int32_t     hold_used;
    uint64_t    hold_tick;
    size_t      held;            /* bytes the cache is holding              */
    size_t      used;            /* bytes this file has allocated in total  */
    size_t      budget;
    /* what a run cost, for the report at close: submissions is the number of
     * times the host waited on the device, and is the figure that says whether
     * a seam passing host pointers is what is in the way */
    uint64_t    sent;
    uint64_t    moved;
    VulkStatus  state;
    int         open_flag;
} VulkDev;

static VulkDev vulk_dev;

const char *vulk_status_text(VulkStatus status)
{
    switch (status) {
        case VULK_OK:        return "ok";
        case VULK_NO_LOADER: return "no vulkan loader on this host";
        case VULK_NO_DEVICE: return "no vulkan device with a compute queue";
        case VULK_NO_MEMORY: return "the device refused an allocation";
        case VULK_NO_SHADER: return "a compute pipeline would not build";
        case VULK_LOST:      return "the device stopped answering";
        case VULK_ARGS:      return "a shape this backend does not accept";
        default:             return "unknown";
    }
}

/* Reporting goes through the text engine's own note when it is in, so that
 * `--quiet` and `-v` mean the same thing here as everywhere else, and to
 * stderr when only the picture engine is. */
static void vulk_say(int level, const char *form, ...)
{
    va_list args;
    va_start(args, form);
#if defined(ILL_CORE_INCLUDED)
    if (level > ill_noise_level) { va_end(args); return; }
#else
    if (level > 1) { va_end(args); return; }
#endif
    fputs("vulk: ", stderr);
    vfprintf(stderr, form, args);
    fputc('\n', stderr);
    va_end(args);
}

/* -- memory --------------------------------------------------------------- */

/* Picks the memory type for compute buffers and the one for staging.  A type
 * that is both device local and host visible is taken for both, which is what
 * `unified_flag` records; otherwise the first device local type carries the
 * buffers and the first host visible coherent type carries the staging.
 *
 * Every host visible type this file will take is also **coherent**, which is
 * why `vkFlushMappedMemoryRanges` and `vkInvalidateMappedMemoryRanges` appear
 * nowhere below: a write through the mapping is visible to the device and a
 * write by the device is visible through the mapping, with the submission's
 * own barriers ordering them.  A cached, non-coherent type would be faster to
 * read back on some parts and would need both calls; taking only coherent
 * memory is the simpler correctness story and the one this file makes. */
static int vulk_memory_pick(VulkDev *dev)
{
    VulkMemProps props;
    uint32_t     index;
    int32_t      fast = -1, host = -1, both = -1;

    memset(&props, 0, sizeof props);
    dev->api.phys_memory(dev->phys, &props);
    for (index = 0; index < props.type_count && index < 32; ++index) {
        /* skipped below when ILL_VULKAN_STAGED is set, which is how the
         * staging path gets exercised on a host whose only device is unified.
         * Every discrete card takes that path and nothing here would otherwise
         * run it, so it is a switch rather than a comment. */
        uint32_t flags = props.type_list[index].flags;
        int local = (flags & VULK_MEM_DEVICE_LOCAL) != 0;
        int seen  = (flags & VULK_MEM_HOST_VISIBLE) != 0;
        int fused = (flags & VULK_MEM_HOST_COHERENT) != 0;
        if (local && seen && fused && both < 0 && !getenv("ILL_VULKAN_STAGED"))
            both = (int32_t)index;
        if (local && fast < 0) fast = (int32_t)index;
        if (seen && fused && host < 0) host = (int32_t)index;
    }
    if (both >= 0) {
        dev->device_type  = (uint32_t)both;
        dev->host_type    = (uint32_t)both;
        dev->unified_flag = 1;
        return 1;
    }
    if (fast < 0 || host < 0) return 0;
    dev->device_type  = (uint32_t)fast;
    dev->host_type    = (uint32_t)host;
    dev->unified_flag = 0;
    return 1;
}

static void vulk_slab_drop(VulkDev *dev, VulkSlab *slab)
{
    if (!slab->buffer && !slab->memory) return;
    if (slab->mapped) dev->api.memory_unmap(dev->device, slab->memory);
    if (slab->buffer) dev->api.buffer_drop(dev->device, slab->buffer, NULL);
    if (slab->memory) dev->api.memory_drop(dev->device, slab->memory, NULL);
    if (dev->used >= (size_t)slab->bytes) dev->used -= (size_t)slab->bytes;
    memset(slab, 0, sizeof *slab);
}

/* Every slab gets its own allocation.  Sub-allocating out of a few large ones
 * would be fewer objects and is what a serious allocator does, but it also
 * means honouring `minStorageBufferOffsetAlignment` on every binding and
 * tracking free ranges; the pool below recycles whole slabs instead, which
 * removes the allocation from the steady state without any of that. */
static int vulk_slab_make(VulkDev *dev, VulkSlab *slab, size_t bytes, int host_flag)
{
    VulkBufferInfo info;
    VulkMemNeeds   needs;
    VulkMemAlloc   want;
    uint32_t       type = host_flag ? dev->host_type : dev->device_type;

    memset(slab, 0, sizeof *slab);
    if (bytes == 0) bytes = 4;
    memset(&info, 0, sizeof info);
    info.type    = VULK_S_BUFFER;
    info.size    = (VulkSize)bytes;
    info.usage   = VULK_USE_STORAGE | VULK_USE_TRANSFER_SRC | VULK_USE_TRANSFER_DST;
    info.sharing = VULK_SHARING_EXCLUSIVE;
    if (dev->api.buffer_make(dev->device, &info, NULL, &slab->buffer) != VULK_SUCCESS)
        return 0;

    memset(&needs, 0, sizeof needs);
    dev->api.buffer_needs(dev->device, slab->buffer, &needs);
    if (!(needs.bits & (1u << type))) {
        /* the driver will not back this buffer with the type that was picked;
         * fall back to the lowest type it will take */
        uint32_t scan;
        for (scan = 0; scan < 32; ++scan) if (needs.bits & (1u << scan)) break;
        if (scan == 32) { vulk_slab_drop(dev, slab); return 0; }
        type = scan;
        host_flag = 0;
    }

    memset(&want, 0, sizeof want);
    want.type  = VULK_S_MEMORY_ALLOCATE;
    want.size  = needs.size;
    want.index = type;
    if (dev->api.memory_make(dev->device, &want, NULL, &slab->memory) != VULK_SUCCESS) {
        dev->api.buffer_drop(dev->device, slab->buffer, NULL);
        memset(slab, 0, sizeof *slab);
        return 0;
    }
    if (dev->api.buffer_bind(dev->device, slab->buffer, slab->memory, 0) != VULK_SUCCESS) {
        slab->bytes = 0;
        vulk_slab_drop(dev, slab);
        return 0;
    }
    slab->bytes     = (VulkSize)bytes;
    slab->host_flag = host_flag || dev->unified_flag;
    if (slab->host_flag) {
        if (dev->api.memory_map(dev->device, slab->memory, 0, VULK_WHOLE_SIZE,
                                0, &slab->mapped) != VULK_SUCCESS)
            slab->mapped = NULL;
    }
    dev->used += bytes;
    return 1;
}

/* The scratch pool.  A forward pass asks for the same handful of sizes over
 * and over, so a slab is kept when it is given back and handed out again to
 * the next request it is large enough for.  "Large enough" rather than "the
 * same size" costs some memory and saves the allocation, and the pool is
 * small enough that the linear scan is not worth indexing. */
static int vulk_scratch_take(VulkDev *dev, VulkSlab *slab, size_t bytes)
{
    int32_t index, best = -1;
    for (index = 0; index < dev->free_used; ++index) {
        if (dev->free_list[index].bytes < (VulkSize)bytes) continue;
        if (best < 0 || dev->free_list[index].bytes < dev->free_list[best].bytes)
            best = index;
    }
    if (best >= 0) {
        *slab = dev->free_list[best];
        dev->free_list[best] = dev->free_list[--dev->free_used];
        return 1;
    }
    return vulk_slab_make(dev, slab, bytes, 0);
}

static void vulk_scratch_give(VulkDev *dev, VulkSlab *slab)
{
    if (!slab->buffer) return;
    if (dev->free_used < VULK_SLAB_FREE) {
        dev->free_list[dev->free_used++] = *slab;
        memset(slab, 0, sizeof *slab);
        return;
    }
    /* the pool is full: drop the smallest of it and this, so what survives is
     * the set of sizes a pass actually reuses */
    {
        int32_t index, small = 0;
        for (index = 1; index < dev->free_used; ++index)
            if (dev->free_list[index].bytes < dev->free_list[small].bytes) small = index;
        if (dev->free_list[small].bytes < slab->bytes) {
            VulkSlab spill = dev->free_list[small];
            dev->free_list[small] = *slab;
            *slab = spill;
        }
    }
    vulk_slab_drop(dev, slab);
}

static void vulk_scratch_clear(VulkDev *dev)
{
    while (dev->free_used > 0) vulk_slab_drop(dev, &dev->free_list[--dev->free_used]);
}

/* -- submission ----------------------------------------------------------- */

/* One command buffer, recorded and submitted and waited on.  Both seams are
 * synchronous -- they hand over host pointers and expect the answer in host
 * memory when they return -- so there is nothing to overlap with, and a fence
 * wait per operation is the honest shape rather than a missing optimisation.
 * `vulk_send_open` starts a recording, the caller adds its dispatches, and
 * `vulk_send_wait` closes, submits and blocks. */
static int vulk_send_open(VulkDev *dev)
{
    VulkCmdBeginInfo begin;
    if (dev->api.cmd_reset(dev->cmd, 0) != VULK_SUCCESS) return 0;
    memset(&begin, 0, sizeof begin);
    begin.type  = VULK_S_COMMAND_BUFFER_BEGIN;
    begin.flags = VULK_CMD_ONE_TIME;
    return dev->api.cmd_begin(dev->cmd, &begin) == VULK_SUCCESS;
}

/* The one barrier this file emits, between every pair of commands: everything
 * written by a transfer or a shader is visible to the next transfer, shader or
 * host read.  A finer barrier would name the buffers; with one dispatch per
 * submission there is nothing for a finer one to overlap, so the blunt one
 * costs nothing and cannot be wrong. */
static void vulk_send_fence(VulkDev *dev)
{
    VulkBarrier bar;
    memset(&bar, 0, sizeof bar);
    bar.type       = VULK_S_MEMORY_BARRIER;
    bar.src_access = VULK_ACCESS_SHADER_WRITE | VULK_ACCESS_TRANSFER_WRITE |
                     VULK_ACCESS_HOST_WRITE;
    bar.dst_access = VULK_ACCESS_SHADER_READ | VULK_ACCESS_SHADER_WRITE |
                     VULK_ACCESS_TRANSFER_READ | VULK_ACCESS_HOST_READ;
    dev->api.cmd_barrier(dev->cmd,
                         VULK_STAGE_TRANSFER | VULK_STAGE_COMPUTE_SHADER | VULK_STAGE_HOST,
                         VULK_STAGE_TRANSFER | VULK_STAGE_COMPUTE_SHADER | VULK_STAGE_HOST,
                         0, 1, &bar, 0, NULL, 0, NULL);
}

static int vulk_send_wait(VulkDev *dev)
{
    VulkSubmitInfo submit;
    VulkResult     code;
    if (dev->api.cmd_end(dev->cmd) != VULK_SUCCESS) return 0;
    memset(&submit, 0, sizeof submit);
    submit.type          = VULK_S_SUBMIT;
    submit.command_count = 1;
    submit.command_list  = &dev->cmd;
    if (dev->api.fence_reset(dev->device, 1, &dev->fence) != VULK_SUCCESS) return 0;
    if (dev->api.queue_submit(dev->queue, 1, &submit, dev->fence) != VULK_SUCCESS) return 0;
    /* ten seconds.  A dispatch that has not finished by then is a hung device
     * rather than a slow one, and returning is better than never waking. */
    code = dev->api.fence_wait(dev->device, 1, &dev->fence, 1, 10000000000ull);
    ++dev->sent;
    return code == VULK_SUCCESS;
}

/* -- moving bytes --------------------------------------------------------- */

/* Writes host memory into a slab.  On a unified device this is a memcpy into
 * the mapping the shader reads; otherwise it is a memcpy into a staging slab
 * and a copy command, and the caller has a recording open for the copy to go
 * into.  `stage` is handed back so the caller can give it to the scratch pool
 * once the submission it belongs to has completed -- not before, because the
 * copy reads it. */
static int vulk_write(VulkDev *dev, VulkSlab *slab, const void *src, size_t bytes,
                      VulkSlab *stage)
{
    memset(stage, 0, sizeof *stage);
    if (bytes == 0) return 1;
    dev->moved += bytes;
    if (slab->mapped) {
        memcpy(slab->mapped, src, bytes);
        return 1;
    }
    if (!vulk_slab_make(dev, stage, bytes, 1)) return 0;
    if (!stage->mapped) { vulk_slab_drop(dev, stage); return 0; }
    memcpy(stage->mapped, src, bytes);
    {
        VulkCopy span;
        span.src = 0; span.dst = 0; span.size = (VulkSize)bytes;
        dev->api.cmd_copy(dev->cmd, stage->buffer, slab->buffer, 1, &span);
    }
    return 1;
}

/* Reads a slab back.  On a unified device the mapping is the answer, so the
 * whole of this is a memcpy after the fence; otherwise a copy command has to
 * be recorded before the submission and the staging read after it, which is
 * why this is two functions rather than one. */
static int vulk_read_stage(VulkDev *dev, VulkSlab *slab, size_t bytes, VulkSlab *stage)
{
    memset(stage, 0, sizeof *stage);
    if (bytes == 0 || slab->mapped) return 1;
    if (!vulk_slab_make(dev, stage, bytes, 1)) return 0;
    if (!stage->mapped) { vulk_slab_drop(dev, stage); return 0; }
    {
        VulkCopy span;
        span.src = 0; span.dst = 0; span.size = (VulkSize)bytes;
        dev->api.cmd_copy(dev->cmd, slab->buffer, stage->buffer, 1, &span);
    }
    return 1;
}

static void vulk_read_take(VulkDev *dev, VulkSlab *slab, VulkSlab *stage,
                           void *dst, size_t bytes)
{
    if (bytes == 0) return;
    dev->moved += bytes;
    if (slab->mapped) { memcpy(dst, slab->mapped, bytes); return; }
    if (stage->mapped) memcpy(dst, stage->mapped, bytes);
}

/* ============================================================================
 * part 3 -- the SPIR-V assembler
 *
 * A compute shader is a SPIR-V module: a five word header and then a stream of
 * instructions, each one a word of `(length << 16) | opcode` followed by its
 * operands.  Every value is a numbered result id, assigned once, and the
 * module is ordered -- capabilities, then the memory model, then the entry
 * point, then decorations, then types and constants, then the function.  That
 * order is why this assembler keeps five buffers and joins them at the end
 * rather than writing straight out.
 *
 * Two shortcuts keep the whole thing small enough to read.
 *
 * **No phi nodes.**  A value that crosses a branch is normally an `OpPhi` with
 * one entry per incoming block, and getting those right is most of the work in
 * a real compiler back end.  Every variable here is an `OpVariable` in the
 * Function storage class instead, read with `OpLoad` and written with
 * `OpStore`, which is exactly what glslang emits before its own mem2roa pass
 * and what every driver's optimiser turns back into registers.  It costs
 * nothing at run time and removes the only genuinely hard part of emitting
 * structured control flow by hand.
 *
 * **One buffer shape.**  Every binding is `struct { float cell[]; }` in the
 * Uniform storage class with the BufferBlock decoration, which is how a
 * storage buffer is spelled in SPIR-V 1.0 and therefore works on a driver as
 * old as Vulkan 1.0.  Integer data rides in the same buffers through
 * `OpBitcast`, and the push constants are sixteen raw words read the same way,
 * so there is one descriptor layout and one push range for all fourteen
 * kernels.
 * ==========================================================================*/

#define SPV_MAGIC   0x07230203u
#define SPV_VERSION 0x00010000u   /* 1.0: the widest floor there is         */

/* opcodes, in the order the specification numbers them */
#define SPV_OP_EXT_IMPORT      11
#define SPV_OP_EXT_INST        12
#define SPV_OP_MEMORY_MODEL    14
#define SPV_OP_ENTRY_POINT     15
#define SPV_OP_EXEC_MODE       16
#define SPV_OP_CAPABILITY      17
#define SPV_OP_TYPE_VOID       19
#define SPV_OP_TYPE_BOOL       20
#define SPV_OP_TYPE_INT        21
#define SPV_OP_TYPE_FLOAT      22
#define SPV_OP_TYPE_VECTOR     23
#define SPV_OP_TYPE_ARRAY      28
#define SPV_OP_TYPE_RUNTIME    29
#define SPV_OP_TYPE_STRUCT     30
#define SPV_OP_TYPE_POINTER    32
#define SPV_OP_TYPE_FUNCTION   33
#define SPV_OP_CONSTANT        43
#define SPV_OP_FUNCTION        54
#define SPV_OP_FUNCTION_END    56
#define SPV_OP_VARIABLE        59
#define SPV_OP_LOAD            61
#define SPV_OP_STORE           62
#define SPV_OP_ACCESS_CHAIN    65
#define SPV_OP_DECORATE        71
#define SPV_OP_MEMBER_DECORATE 72
#define SPV_OP_CONVERT_F_TO_U  109
#define SPV_OP_CONVERT_F_TO_S  110
#define SPV_OP_CONVERT_S_TO_F  111
#define SPV_OP_CONVERT_U_TO_F  112
#define SPV_OP_BITCAST         124
#define SPV_OP_FNEGATE         127
#define SPV_OP_IADD            128
#define SPV_OP_FADD            129
#define SPV_OP_ISUB            130
#define SPV_OP_FSUB            131
#define SPV_OP_IMUL            132
#define SPV_OP_FMUL            133
#define SPV_OP_UDIV            134
#define SPV_OP_SDIV            135
#define SPV_OP_FDIV            136
#define SPV_OP_UMOD            137
#define SPV_OP_SMOD            139
#define SPV_OP_LOGICAL_OR      166
#define SPV_OP_LOGICAL_AND     167
#define SPV_OP_LOGICAL_NOT     168
#define SPV_OP_SELECT          169
#define SPV_OP_IEQUAL          170
#define SPV_OP_INOTEQUAL       171
#define SPV_OP_UGREATER        172
#define SPV_OP_SGREATER        173
#define SPV_OP_SGREATER_EQ     175
#define SPV_OP_ULESS           176
#define SPV_OP_SLESS           177
#define SPV_OP_SLESS_EQ        179
#define SPV_OP_FEQUAL          180
#define SPV_OP_FLESS           184
#define SPV_OP_FGREATER        186
#define SPV_OP_FLESS_EQ        188
#define SPV_OP_FGREATER_EQ     190
#define SPV_OP_SHIFT_RIGHT     194
#define SPV_OP_SHIFT_LEFT      196
#define SPV_OP_BIT_OR          197
#define SPV_OP_BIT_AND         199
#define SPV_OP_CONTROL_BARRIER 224
#define SPV_OP_LOOP_MERGE      246
#define SPV_OP_SELECT_MERGE    247
#define SPV_OP_LABEL           248
#define SPV_OP_BRANCH          249
#define SPV_OP_BRANCH_COND     250
#define SPV_OP_RETURN          253

/* GLSL.std.450 instruction numbers */
#define SPV_GL_FABS      4
#define SPV_GL_FLOOR     8
#define SPV_GL_EXP       27
#define SPV_GL_LOG       28
#define SPV_GL_SQRT      31
#define SPV_GL_RSQRT     32
#define SPV_GL_FMIN      37
#define SPV_GL_FMAX      40
#define SPV_GL_FMA       50

/* storage classes, decorations, builtins */
#define SPV_SC_INPUT      1
#define SPV_SC_UNIFORM    2
#define SPV_SC_WORKGROUP  4
#define SPV_SC_FUNCTION   7
#define SPV_SC_PUSH       9
#define SPV_DEC_BLOCK        2
#define SPV_DEC_BUFFER_BLOCK 3
#define SPV_DEC_ARRAY_STRIDE 6
#define SPV_DEC_BUILTIN      11
#define SPV_DEC_BINDING      33
#define SPV_DEC_SET          34
#define SPV_DEC_OFFSET       35
#define SPV_BUILTIN_WORKGROUP_COUNT 24
#define SPV_BUILTIN_WORKGROUP_ID    26
#define SPV_BUILTIN_LOCAL_ID        27
#define SPV_BUILTIN_GLOBAL_ID       28

#define VULK_BIND_MAX   6     /* the widest kernel here binds five buffers  */
#define VULK_PUSH_WORDS 16    /* 64 bytes: every host guarantees at least 128 */
#define VULK_LOCAL      64    /* invocations per workgroup, everywhere      */

typedef struct VulkSeg {
    uint32_t *word;
    int32_t   used;
    int32_t   room;
} VulkSeg;

typedef struct VulkNumber { uint32_t type; uint32_t bits; uint32_t id; } VulkNumber;

typedef struct VulkAsm {
    VulkSeg   head;   /* capability, import, memory model, entry, exec mode */
    VulkSeg   deco;   /* decorations                                       */
    VulkSeg   type;   /* types, constants, globals                         */
    VulkSeg   vars;   /* OpVariable Function, spliced after the entry label */
    VulkSeg   code;   /* the body                                          */
    uint32_t  next;
    uint32_t  glsl;
    uint32_t  entry;
    /* types */
    uint32_t  t_void, t_fn, t_bool, t_u32, t_i32, t_f32, t_uvec3;
    uint32_t  p_fun_u32, p_fun_f32, p_in_uvec3, p_in_u32;
    uint32_t  p_buf_f32, p_push_u32, p_work_f32;
    uint32_t  t_store, t_push;
    uint32_t  v_push, v_gid, v_lid, v_wid, v_nwg;
    uint32_t  bind_list[VULK_BIND_MAX];
    int32_t   bind_used;
    VulkNumber num_list[256];
    int32_t   num_used;
    uint32_t  local_x;
    int       fault;
} VulkAsm;

static void spv_seg_word(VulkSeg *seg, uint32_t word, int *fault)
{
    if (seg->used == seg->room) {
        int32_t   room = seg->room ? seg->room * 2 : 256;
        uint32_t *grow = (uint32_t *)realloc(seg->word, (size_t)room * sizeof(uint32_t));
        if (!grow) { *fault = 1; return; }
        seg->word = grow;
        seg->room = room;
    }
    seg->word[seg->used++] = word;
}

/* One instruction: the header word carries the total length, so the operands
 * are counted before anything is written. */
static void spv_ins(VulkAsm *a, VulkSeg *seg, uint32_t op, const uint32_t *arg, int32_t count)
{
    int32_t index;
    spv_seg_word(seg, ((uint32_t)(count + 1) << 16) | op, &a->fault);
    for (index = 0; index < count; ++index) spv_seg_word(seg, arg[index], &a->fault);
}

static void spv_op0(VulkAsm *a, VulkSeg *s, uint32_t op)
{ spv_ins(a, s, op, NULL, 0); }
static void spv_op1(VulkAsm *a, VulkSeg *s, uint32_t op, uint32_t x)
{ spv_ins(a, s, op, &x, 1); }
static void spv_op2(VulkAsm *a, VulkSeg *s, uint32_t op, uint32_t x, uint32_t y)
{ uint32_t w[2]; w[0] = x; w[1] = y; spv_ins(a, s, op, w, 2); }
static void spv_op3(VulkAsm *a, VulkSeg *s, uint32_t op, uint32_t x, uint32_t y, uint32_t z)
{ uint32_t w[3]; w[0] = x; w[1] = y; w[2] = z; spv_ins(a, s, op, w, 3); }
static void spv_op4(VulkAsm *a, VulkSeg *s, uint32_t op, uint32_t x, uint32_t y,
                    uint32_t z, uint32_t v)
{ uint32_t w[4]; w[0] = x; w[1] = y; w[2] = z; w[3] = v; spv_ins(a, s, op, w, 4); }
static void spv_op5(VulkAsm *a, VulkSeg *s, uint32_t op, uint32_t x, uint32_t y,
                    uint32_t z, uint32_t v, uint32_t u)
{ uint32_t w[5]; w[0] = x; w[1] = y; w[2] = z; w[3] = v; w[4] = u; spv_ins(a, s, op, w, 5); }

static uint32_t spv_id(VulkAsm *a) { return a->next++; }

/* A literal string is UTF-8 packed four bytes to a word, little end first,
 * always with a terminating zero and always padded out to a whole word. */
static int32_t spv_text(const char *text, uint32_t *out)
{
    int32_t  used = 0;
    uint32_t pack = 0;
    int32_t  slot = 0;
    for (;; ++text) {
        pack |= (uint32_t)(unsigned char)*text << (8 * slot);
        if (++slot == 4) { out[used++] = pack; pack = 0; slot = 0; }
        if (!*text) break;
    }
    if (slot) out[used++] = pack;
    return used;
}

/* -- types and constants --------------------------------------------------- */

static uint32_t spv_type_void(VulkAsm *a)
{ uint32_t id = spv_id(a); spv_op1(a, &a->type, SPV_OP_TYPE_VOID, id); return id; }

static uint32_t spv_type_int(VulkAsm *a, uint32_t bits, uint32_t signed_flag)
{
    uint32_t id = spv_id(a);
    spv_op3(a, &a->type, SPV_OP_TYPE_INT, id, bits, signed_flag);
    return id;
}

static uint32_t spv_type_float(VulkAsm *a, uint32_t bits)
{ uint32_t id = spv_id(a); spv_op2(a, &a->type, SPV_OP_TYPE_FLOAT, id, bits); return id; }

static uint32_t spv_type_bool(VulkAsm *a)
{ uint32_t id = spv_id(a); spv_op1(a, &a->type, SPV_OP_TYPE_BOOL, id); return id; }

static uint32_t spv_type_vector(VulkAsm *a, uint32_t of, uint32_t count)
{ uint32_t id = spv_id(a); spv_op3(a, &a->type, SPV_OP_TYPE_VECTOR, id, of, count); return id; }

static uint32_t spv_type_pointer(VulkAsm *a, uint32_t storage, uint32_t of)
{ uint32_t id = spv_id(a); spv_op3(a, &a->type, SPV_OP_TYPE_POINTER, id, storage, of); return id; }

/* The constant cache is a flat scan: a kernel uses a few dozen distinct
 * constants and a repeated OpConstant of the same type and bits is not merely
 * wasteful but invalid, so this is a correctness structure rather than a
 * performance one. */
static uint32_t spv_number(VulkAsm *a, uint32_t type, uint32_t bits)
{
    int32_t  index;
    uint32_t id;
    for (index = 0; index < a->num_used; ++index)
        if (a->num_list[index].type == type && a->num_list[index].bits == bits)
            return a->num_list[index].id;
    id = spv_id(a);
    spv_op3(a, &a->type, SPV_OP_CONSTANT, type, id, bits);
    if (a->num_used >= (int32_t)(sizeof a->num_list / sizeof a->num_list[0])) {
        /* the next constant of this type and value would be emitted a second
         * time, and a duplicate OpConstant is not valid SPIR-V.  Fail the
         * module here rather than hand a driver something it may or may not
         * reject.  The widest kernel in this file uses eighteen constants and
         * the table holds 256, so reaching this means a new kernel needs a
         * bigger table rather than a cleverer one. */
        a->fault = 1;
        return id;
    }
    a->num_list[a->num_used].type = type;
    a->num_list[a->num_used].bits = bits;
    a->num_list[a->num_used].id   = id;
    a->num_used++;
    return id;
}

static inline uint32_t spv_u(VulkAsm *a, uint32_t value) { return spv_number(a, a->t_u32, value); }
static inline uint32_t spv_f(VulkAsm *a, float value)
{ uint32_t bits; memcpy(&bits, &value, sizeof bits); return spv_number(a, a->t_f32, bits); }

/* -- values ---------------------------------------------------------------- */

/* The one-line vocabulary below is `static inline` rather than plain `static`
 * for one reason: a build that leaves out one of the two engines does not use
 * all of it, and an unused inline is not a warning where an unused static is.
 * Nothing here is inlined for speed -- it runs once, when a shader is built. */
static uint32_t spv_bin(VulkAsm *a, uint32_t op, uint32_t type, uint32_t x, uint32_t y)
{ uint32_t id = spv_id(a); spv_op4(a, &a->code, op, type, id, x, y); return id; }

static uint32_t spv_un(VulkAsm *a, uint32_t op, uint32_t type, uint32_t x)
{ uint32_t id = spv_id(a); spv_op3(a, &a->code, op, type, id, x); return id; }

static inline uint32_t spv_fadd(VulkAsm *a, uint32_t x, uint32_t y)
{ return spv_bin(a, SPV_OP_FADD, a->t_f32, x, y); }
static inline uint32_t spv_fsub(VulkAsm *a, uint32_t x, uint32_t y)
{ return spv_bin(a, SPV_OP_FSUB, a->t_f32, x, y); }
static inline uint32_t spv_fmul(VulkAsm *a, uint32_t x, uint32_t y)
{ return spv_bin(a, SPV_OP_FMUL, a->t_f32, x, y); }
static inline uint32_t spv_fdiv(VulkAsm *a, uint32_t x, uint32_t y)
{ return spv_bin(a, SPV_OP_FDIV, a->t_f32, x, y); }
static inline uint32_t spv_uadd(VulkAsm *a, uint32_t x, uint32_t y)
{ return spv_bin(a, SPV_OP_IADD, a->t_u32, x, y); }
static inline uint32_t spv_usub(VulkAsm *a, uint32_t x, uint32_t y)
{ return spv_bin(a, SPV_OP_ISUB, a->t_u32, x, y); }
static inline uint32_t spv_umul(VulkAsm *a, uint32_t x, uint32_t y)
{ return spv_bin(a, SPV_OP_IMUL, a->t_u32, x, y); }
static inline uint32_t spv_udiv(VulkAsm *a, uint32_t x, uint32_t y)
{ return spv_bin(a, SPV_OP_UDIV, a->t_u32, x, y); }
static inline uint32_t spv_umod(VulkAsm *a, uint32_t x, uint32_t y)
{ return spv_bin(a, SPV_OP_UMOD, a->t_u32, x, y); }
static inline uint32_t spv_iadd(VulkAsm *a, uint32_t x, uint32_t y)
{ return spv_bin(a, SPV_OP_IADD, a->t_i32, x, y); }
static inline uint32_t spv_ult(VulkAsm *a, uint32_t x, uint32_t y)
{ return spv_bin(a, SPV_OP_ULESS, a->t_bool, x, y); }
static inline uint32_t spv_ueq(VulkAsm *a, uint32_t x, uint32_t y)
{ return spv_bin(a, SPV_OP_IEQUAL, a->t_bool, x, y); }
static inline uint32_t spv_fgt(VulkAsm *a, uint32_t x, uint32_t y)
{ return spv_bin(a, SPV_OP_FGREATER, a->t_bool, x, y); }
static inline uint32_t spv_band(VulkAsm *a, uint32_t x, uint32_t y)
{ return spv_bin(a, SPV_OP_LOGICAL_AND, a->t_bool, x, y); }
static inline uint32_t spv_fneg(VulkAsm *a, uint32_t x)
{ return spv_un(a, SPV_OP_FNEGATE, a->t_f32, x); }
static inline uint32_t spv_u2i(VulkAsm *a, uint32_t x)
{ return spv_un(a, SPV_OP_BITCAST, a->t_i32, x); }
static inline uint32_t spv_i2u(VulkAsm *a, uint32_t x)
{ return spv_un(a, SPV_OP_BITCAST, a->t_u32, x); }
static inline uint32_t spv_u2f(VulkAsm *a, uint32_t x)
{ return spv_un(a, SPV_OP_CONVERT_U_TO_F, a->t_f32, x); }

static uint32_t spv_pick(VulkAsm *a, uint32_t type, uint32_t cond, uint32_t yes, uint32_t no)
{
    uint32_t id = spv_id(a);
    spv_op5(a, &a->code, SPV_OP_SELECT, type, id, cond, yes, no);
    return id;
}

static uint32_t spv_call1(VulkAsm *a, uint32_t which, uint32_t x)
{
    uint32_t id = spv_id(a);
    uint32_t w[5];
    w[0] = a->t_f32; w[1] = id; w[2] = a->glsl; w[3] = which; w[4] = x;
    spv_ins(a, &a->code, SPV_OP_EXT_INST, w, 5);
    return id;
}

static uint32_t spv_call2(VulkAsm *a, uint32_t which, uint32_t x, uint32_t y)
{
    uint32_t id = spv_id(a);
    uint32_t w[6];
    w[0] = a->t_f32; w[1] = id; w[2] = a->glsl; w[3] = which; w[4] = x; w[5] = y;
    spv_ins(a, &a->code, SPV_OP_EXT_INST, w, 6);
    return id;
}

/* -- memory ---------------------------------------------------------------- */

/* An OpVariable outside the Function storage class lives in the types and
 * globals section and takes its storage class as a third operand. */
static uint32_t spv_global(VulkAsm *a, uint32_t ptr_type, uint32_t storage)
{
    uint32_t id = spv_id(a);
    spv_op3(a, &a->type, SPV_OP_VARIABLE, ptr_type, id, storage);
    return id;
}

static uint32_t spv_var_f(VulkAsm *a)
{
    uint32_t id = spv_id(a);
    spv_op3(a, &a->vars, SPV_OP_VARIABLE, a->p_fun_f32, id, SPV_SC_FUNCTION);
    return id;
}

static uint32_t spv_var_u(VulkAsm *a)
{
    uint32_t id = spv_id(a);
    spv_op3(a, &a->vars, SPV_OP_VARIABLE, a->p_fun_u32, id, SPV_SC_FUNCTION);
    return id;
}

/* A fixed size array in Function storage, which is what the short convolution
 * needs for its rolling window.  Dynamic indexing into one is legal and is
 * what makes the kernel a loop rather than sixteen unrolled cases. */
static uint32_t spv_var_ring(VulkAsm *a, uint32_t count, uint32_t *cell_ptr_type)
{
    uint32_t array = spv_id(a);
    uint32_t ptr   = spv_id(a);
    uint32_t id    = spv_id(a);
    spv_op3(a, &a->type, SPV_OP_TYPE_ARRAY, array, a->t_f32, spv_u(a, count));
    spv_op3(a, &a->type, SPV_OP_TYPE_POINTER, ptr, SPV_SC_FUNCTION, array);
    spv_op3(a, &a->vars, SPV_OP_VARIABLE, ptr, id, SPV_SC_FUNCTION);
    *cell_ptr_type = a->p_fun_f32;
    return id;
}

static uint32_t spv_ring_cell(VulkAsm *a, uint32_t ring, uint32_t index)
{
    uint32_t id = spv_id(a);
    spv_op4(a, &a->code, SPV_OP_ACCESS_CHAIN, a->p_fun_f32, id, ring, index);
    return id;
}

static inline uint32_t spv_load_f(VulkAsm *a, uint32_t ptr)
{ uint32_t id = spv_id(a); spv_op3(a, &a->code, SPV_OP_LOAD, a->t_f32, id, ptr); return id; }
static inline uint32_t spv_load_u(VulkAsm *a, uint32_t ptr)
{ uint32_t id = spv_id(a); spv_op3(a, &a->code, SPV_OP_LOAD, a->t_u32, id, ptr); return id; }
static inline void spv_save(VulkAsm *a, uint32_t ptr, uint32_t value)
{ spv_op2(a, &a->code, SPV_OP_STORE, ptr, value); }

/* A cell of a storage binding.  The chain is two steps -- member zero of the
 * struct, then the element -- because the runtime array has to be wrapped in a
 * struct for the BufferBlock decoration to have somewhere to sit. */
static uint32_t spv_cell(VulkAsm *a, int32_t bind, uint32_t index)
{
    uint32_t id = spv_id(a);
    uint32_t w[5];
    w[0] = a->p_buf_f32; w[1] = id; w[2] = a->bind_list[bind];
    w[3] = spv_u(a, 0);  w[4] = index;
    spv_ins(a, &a->code, SPV_OP_ACCESS_CHAIN, w, 5);
    return id;
}

static inline uint32_t spv_get(VulkAsm *a, int32_t bind, uint32_t index)
{ return spv_load_f(a, spv_cell(a, bind, index)); }
static inline void spv_set(VulkAsm *a, int32_t bind, uint32_t index, uint32_t value)
{ spv_save(a, spv_cell(a, bind, index), value); }

/* Push constants are sixteen raw words.  Reading one as an unsigned is a load;
 * reading one as a float is the same load and a bitcast, which is free.  A
 * struct with named typed members would read better and would make the host
 * side a matching struct that has to be kept in step by hand across fourteen
 * kernels; the raw words are checked by the argument order in one place. */
static uint32_t spv_pin_u(VulkAsm *a, uint32_t slot)
{
    uint32_t id = spv_id(a);
    uint32_t w[5];
    w[0] = a->p_push_u32; w[1] = id; w[2] = a->v_push;
    w[3] = spv_u(a, 0);   w[4] = spv_u(a, slot);
    spv_ins(a, &a->code, SPV_OP_ACCESS_CHAIN, w, 5);
    return spv_load_u(a, id);
}

static uint32_t spv_pin_f(VulkAsm *a, uint32_t slot)
{ return spv_un(a, SPV_OP_BITCAST, a->t_f32, spv_pin_u(a, slot)); }

/* Workgroup memory: one array of floats, declared once per kernel that needs
 * one.  Sixty-four floats is a quarter of a kilobyte, well under the sixteen
 * every Vulkan host guarantees. */
static uint32_t spv_share(VulkAsm *a, uint32_t count)
{
    uint32_t array = spv_id(a);
    uint32_t ptr   = spv_id(a);
    spv_op3(a, &a->type, SPV_OP_TYPE_ARRAY, array, a->t_f32, spv_u(a, count));
    spv_op3(a, &a->type, SPV_OP_TYPE_POINTER, ptr, SPV_SC_WORKGROUP, array);
    return spv_global(a, ptr, SPV_SC_WORKGROUP);
}

static uint32_t spv_share_cell(VulkAsm *a, uint32_t share, uint32_t index)
{
    uint32_t id = spv_id(a);
    spv_op4(a, &a->code, SPV_OP_ACCESS_CHAIN, a->p_work_f32, id, share, index);
    return id;
}

/* Every invocation in the workgroup waits here and everything written to
 * workgroup memory before it is visible after it. */
static void spv_barrier(VulkAsm *a)
{
    spv_op3(a, &a->code, SPV_OP_CONTROL_BARRIER,
            spv_u(a, 2), spv_u(a, 2), spv_u(a, 0x108));
}

/* -- the invocation's own numbers ------------------------------------------ */

/* One component of a builtin uvec3.  A whole-vector load and an
 * OpCompositeExtract would do as well; the access chain is one instruction
 * fewer and reads the same. */
static uint32_t spv_axis(VulkAsm *a, uint32_t var, uint32_t axis)
{
    uint32_t ptr = spv_id(a);
    spv_op4(a, &a->code, SPV_OP_ACCESS_CHAIN, a->p_in_u32, ptr, var, spv_u(a, axis));
    return spv_load_u(a, ptr);
}

static uint32_t spv_lane(VulkAsm *a)  { return spv_axis(a, a->v_lid, 0); }  /* 0..63 in the group */
static uint32_t spv_group(VulkAsm *a) { return spv_axis(a, a->v_wid, 0); }
static uint32_t spv_group_y(VulkAsm *a) { return spv_axis(a, a->v_wid, 1); }
static uint32_t spv_groups(VulkAsm *a) { return spv_axis(a, a->v_nwg, 0); }
static uint32_t spv_groups_y(VulkAsm *a) { return spv_axis(a, a->v_nwg, 1); }
static uint32_t spv_index(VulkAsm *a) { return spv_axis(a, a->v_gid, 0); }
static uint32_t spv_stride(VulkAsm *a)
{ return spv_umul(a, spv_groups(a), spv_u(a, VULK_LOCAL)); }

/* -- control flow ---------------------------------------------------------- */

typedef struct VulkIf { uint32_t merge, other; int has_else; } VulkIf;

static void spv_if_open(VulkAsm *a, VulkIf *branch, uint32_t cond)
{
    uint32_t yes = spv_id(a);
    branch->other = spv_id(a);
    branch->merge = spv_id(a);
    branch->has_else = 0;
    spv_op2(a, &a->code, SPV_OP_SELECT_MERGE, branch->merge, 0);
    spv_op3(a, &a->code, SPV_OP_BRANCH_COND, cond, yes, branch->other);
    spv_op1(a, &a->code, SPV_OP_LABEL, yes);
}

static void spv_if_else(VulkAsm *a, VulkIf *branch)
{
    spv_op1(a, &a->code, SPV_OP_BRANCH, branch->merge);
    spv_op1(a, &a->code, SPV_OP_LABEL, branch->other);
    branch->has_else = 1;
}

static void spv_if_shut(VulkAsm *a, VulkIf *branch)
{
    spv_op1(a, &a->code, SPV_OP_BRANCH, branch->merge);
    if (!branch->has_else) {
        spv_op1(a, &a->code, SPV_OP_LABEL, branch->other);
        spv_op1(a, &a->code, SPV_OP_BRANCH, branch->merge);
    }
    spv_op1(a, &a->code, SPV_OP_LABEL, branch->merge);
}

/* A counted loop over an unsigned function variable, in the shape a driver
 * expects: a header block carrying the merge declaration, a separate test
 * block, the body, and a continue block that does the increment and closes
 * the back edge. */
typedef struct VulkFor {
    uint32_t head, test, body, cont, merge, var, limit, step;
} VulkFor;

static void spv_for_open(VulkAsm *a, VulkFor *loop, uint32_t var,
                         uint32_t first, uint32_t limit, uint32_t step)
{
    loop->var   = var;
    loop->limit = limit;
    loop->step  = step;
    loop->head  = spv_id(a);
    loop->test  = spv_id(a);
    loop->body  = spv_id(a);
    loop->cont  = spv_id(a);
    loop->merge = spv_id(a);
    spv_save(a, var, first);
    spv_op1(a, &a->code, SPV_OP_BRANCH, loop->head);
    spv_op1(a, &a->code, SPV_OP_LABEL, loop->head);
    spv_op3(a, &a->code, SPV_OP_LOOP_MERGE, loop->merge, loop->cont, 0);
    spv_op1(a, &a->code, SPV_OP_BRANCH, loop->test);
    spv_op1(a, &a->code, SPV_OP_LABEL, loop->test);
    spv_op3(a, &a->code, SPV_OP_BRANCH_COND,
            spv_ult(a, spv_load_u(a, var), limit), loop->body, loop->merge);
    spv_op1(a, &a->code, SPV_OP_LABEL, loop->body);
}

static void spv_for_shut(VulkAsm *a, VulkFor *loop)
{
    spv_op1(a, &a->code, SPV_OP_BRANCH, loop->cont);
    spv_op1(a, &a->code, SPV_OP_LABEL, loop->cont);
    spv_save(a, loop->var, spv_uadd(a, spv_load_u(a, loop->var), loop->step));
    spv_op1(a, &a->code, SPV_OP_BRANCH, loop->head);
    spv_op1(a, &a->code, SPV_OP_LABEL, loop->merge);
}

/* -- the module ------------------------------------------------------------ */

/* Opens a kernel: the fixed preamble every one of them shares, plus
 * `bind_count` storage buffers at set zero, bindings zero upward. */
static void spv_open(VulkAsm *a, int32_t bind_count, uint32_t local_x)
{
    uint32_t array, push_array, fn_type;
    int32_t  index;

    memset(a, 0, sizeof *a);
    a->next    = 1;
    a->local_x = local_x;
    a->bind_used = bind_count;

    spv_op1(a, &a->head, SPV_OP_CAPABILITY, 1);            /* Shader        */
    a->glsl = spv_id(a);
    {
        uint32_t w[16];
        int32_t  used = 1;
        w[0] = a->glsl;
        used += spv_text("GLSL.std.450", w + 1);
        spv_ins(a, &a->head, SPV_OP_EXT_IMPORT, w, used);
    }
    spv_op2(a, &a->head, SPV_OP_MEMORY_MODEL, 0, 1);       /* Logical/GLSL450 */

    a->t_void  = spv_type_void(a);
    a->t_bool  = spv_type_bool(a);
    a->t_u32   = spv_type_int(a, 32, 0);
    a->t_i32   = spv_type_int(a, 32, 1);
    a->t_f32   = spv_type_float(a, 32);
    a->t_uvec3 = spv_type_vector(a, a->t_u32, 3);

    a->p_fun_u32  = spv_type_pointer(a, SPV_SC_FUNCTION, a->t_u32);
    a->p_fun_f32  = spv_type_pointer(a, SPV_SC_FUNCTION, a->t_f32);
    a->p_work_f32 = spv_type_pointer(a, SPV_SC_WORKGROUP, a->t_f32);
    a->p_in_uvec3 = spv_type_pointer(a, SPV_SC_INPUT, a->t_uvec3);
    a->p_in_u32   = spv_type_pointer(a, SPV_SC_INPUT, a->t_u32);

    /* struct { float cell[]; }, decorated BufferBlock, shared by every binding */
    array      = spv_id(a);
    a->t_store = spv_id(a);
    spv_op2(a, &a->type, SPV_OP_TYPE_RUNTIME, array, a->t_f32);
    spv_op2(a, &a->type, SPV_OP_TYPE_STRUCT, a->t_store, array);
    spv_op3(a, &a->deco, SPV_OP_DECORATE, array, SPV_DEC_ARRAY_STRIDE, 4);
    spv_op2(a, &a->deco, SPV_OP_DECORATE, a->t_store, SPV_DEC_BUFFER_BLOCK);
    spv_op4(a, &a->deco, SPV_OP_MEMBER_DECORATE, a->t_store, 0, SPV_DEC_OFFSET, 0);
    a->p_buf_f32 = spv_type_pointer(a, SPV_SC_UNIFORM, a->t_f32);
    {
        uint32_t p_store = spv_type_pointer(a, SPV_SC_UNIFORM, a->t_store);
        for (index = 0; index < bind_count; ++index) {
            uint32_t id = spv_global(a, p_store, SPV_SC_UNIFORM);
            spv_op3(a, &a->deco, SPV_OP_DECORATE, id, SPV_DEC_SET, 0);
            spv_op3(a, &a->deco, SPV_OP_DECORATE, id, SPV_DEC_BINDING, (uint32_t)index);
            a->bind_list[index] = id;
        }
    }

    /* struct { uint word[16]; } in PushConstant storage */
    push_array = spv_id(a);
    a->t_push  = spv_id(a);
    spv_op3(a, &a->type, SPV_OP_TYPE_ARRAY, push_array, a->t_u32, spv_u(a, VULK_PUSH_WORDS));
    spv_op2(a, &a->type, SPV_OP_TYPE_STRUCT, a->t_push, push_array);
    spv_op3(a, &a->deco, SPV_OP_DECORATE, push_array, SPV_DEC_ARRAY_STRIDE, 4);
    spv_op2(a, &a->deco, SPV_OP_DECORATE, a->t_push, SPV_DEC_BLOCK);
    spv_op4(a, &a->deco, SPV_OP_MEMBER_DECORATE, a->t_push, 0, SPV_DEC_OFFSET, 0);
    a->p_push_u32 = spv_type_pointer(a, SPV_SC_PUSH, a->t_u32);
    {
        uint32_t p_push = spv_type_pointer(a, SPV_SC_PUSH, a->t_push);
        a->v_push = spv_global(a, p_push, SPV_SC_PUSH);
    }

    /* the four builtins, all declared whether or not a kernel reads them */
    {
        struct { uint32_t *slot; uint32_t which; } wanted[4];
        int32_t which;
        wanted[0].slot = &a->v_gid; wanted[0].which = SPV_BUILTIN_GLOBAL_ID;
        wanted[1].slot = &a->v_lid; wanted[1].which = SPV_BUILTIN_LOCAL_ID;
        wanted[2].slot = &a->v_wid; wanted[2].which = SPV_BUILTIN_WORKGROUP_ID;
        wanted[3].slot = &a->v_nwg; wanted[3].which = SPV_BUILTIN_WORKGROUP_COUNT;
        for (which = 0; which < 4; ++which) {
            uint32_t id = spv_global(a, a->p_in_uvec3, SPV_SC_INPUT);
            *wanted[which].slot = id;
            spv_op3(a, &a->deco, SPV_OP_DECORATE, id, SPV_DEC_BUILTIN, wanted[which].which);
        }
    }

    fn_type  = spv_id(a);
    spv_op2(a, &a->type, SPV_OP_TYPE_FUNCTION, fn_type, a->t_void);
    a->t_fn  = fn_type;
    a->entry = spv_id(a);

    {
        uint32_t w[16];
        int32_t  used = 2;
        w[0] = 5;             /* GLCompute                                  */
        w[1] = a->entry;
        used += spv_text("main", w + 2);
        w[used++] = a->v_gid;
        w[used++] = a->v_lid;
        w[used++] = a->v_wid;
        w[used++] = a->v_nwg;
        spv_ins(a, &a->head, SPV_OP_ENTRY_POINT, w, used);
    }
    spv_op5(a, &a->head, SPV_OP_EXEC_MODE, a->entry, 17, local_x, 1, 1);

    /* the function proper; its first label is emitted here so that the
     * variables segment can be spliced straight after it */
    spv_op4(a, &a->code, SPV_OP_FUNCTION, a->t_void, a->entry, 0, fn_type);
    spv_op1(a, &a->code, SPV_OP_LABEL, spv_id(a));
}

/* Closes the module and joins the segments in the order the specification
 * requires.  The caller owns the returned words. */
static uint32_t *spv_shut(VulkAsm *a, size_t *bytes_out)
{
    uint32_t *out;
    int32_t   used = 0;
    int32_t   total;

    spv_op0(a, &a->code, SPV_OP_RETURN);
    spv_op0(a, &a->code, SPV_OP_FUNCTION_END);

    /* the body is [OpFunction, OpLabel] then everything else; the function
     * variables belong between the label and the first real instruction */
    total = 5 + a->head.used + a->deco.used + a->type.used + a->code.used + a->vars.used;
    if (a->fault) return NULL;
    out = (uint32_t *)malloc((size_t)total * sizeof(uint32_t));
    if (!out) return NULL;

    out[used++] = SPV_MAGIC;
    out[used++] = SPV_VERSION;
    out[used++] = 0;            /* generator: this file is not registered   */
    out[used++] = a->next;      /* the id bound                             */
    out[used++] = 0;
    memcpy(out + used, a->head.word, (size_t)a->head.used * sizeof(uint32_t));
    used += a->head.used;
    memcpy(out + used, a->deco.word, (size_t)a->deco.used * sizeof(uint32_t));
    used += a->deco.used;
    memcpy(out + used, a->type.word, (size_t)a->type.used * sizeof(uint32_t));
    used += a->type.used;
    /* OpFunction is five words and OpLabel two */
    memcpy(out + used, a->code.word, 7 * sizeof(uint32_t));
    used += 7;
    memcpy(out + used, a->vars.word, (size_t)a->vars.used * sizeof(uint32_t));
    used += a->vars.used;
    memcpy(out + used, a->code.word + 7, (size_t)(a->code.used - 7) * sizeof(uint32_t));
    used += a->code.used - 7;

    *bytes_out = (size_t)used * sizeof(uint32_t);
    return out;
}

static void spv_free(VulkAsm *a)
{
    free(a->head.word); free(a->deco.word); free(a->type.word);
    free(a->vars.word); free(a->code.word);
    memset(a, 0, sizeof *a);
}

/* ============================================================================
 * part 4 -- the kernels
 *
 * Fourteen compute shaders: six for the text engine's IllBackend and eight for
 * the picture engine's YoloBackend, which is nine operations because
 * `conv_room` is a size rather than a shader.
 *
 * They share three habits, and stating them once saves saying it fourteen
 * times.
 *
 * **A dispatch is a grid stride loop.**  Vulkan guarantees only 65535
 * workgroups in a dimension and a vocabulary projection wants 65536 rows, so
 * no kernel here indexes work by its invocation number alone; each one starts
 * at its own number and strides by the size of the whole dispatch.  That makes
 * the number of workgroups a tuning knob rather than a correctness
 * requirement, and it is what lets the host clamp the dispatch to something
 * the device is comfortable with.
 *
 * **A workgroup is sixty-four invocations, everywhere.**  The floor Vulkan
 * guarantees is 128, so 64 is safe on anything; it is also a whole subgroup on
 * AMD and two on NVIDIA, and one warp on nothing, which is the compromise a
 * file that cannot measure the host has to make.
 *
 * **Reductions go through workgroup memory.**  A subgroup add would be faster
 * on every device that has one, and needs a capability this file would then
 * have to probe for and fall back from.  The tree below is sixty-four floats
 * of shared memory and six rounds, and it is the same six rounds everywhere.
 * ==========================================================================*/

/* silu, as the CPU writes it: v / (1 + exp(-v)). */
static uint32_t spv_silu(VulkAsm *a, uint32_t v)
{
    uint32_t back = spv_call1(a, SPV_GL_EXP, spv_fneg(a, v));
    return spv_fdiv(a, v, spv_fadd(a, spv_f(a, 1.0f), back));
}

static uint32_t spv_sigmoid(VulkAsm *a, uint32_t v)
{
    uint32_t back = spv_call1(a, SPV_GL_EXP, spv_fneg(a, v));
    return spv_fdiv(a, spv_f(a, 1.0f), spv_fadd(a, spv_f(a, 1.0f), back));
}

/* Applies the activation named by a push constant.  Written as a branch rather
 * than three pipelines because the branch is uniform across the dispatch and
 * every device folds it away. */
static uint32_t spv_act(VulkAsm *a, uint32_t value, uint32_t kind)
{
    uint32_t slot = spv_var_f(a);
    VulkIf   one, two;
    spv_save(a, slot, value);
    spv_if_open(a, &one, spv_ueq(a, kind, spv_u(a, 1)));
        spv_save(a, slot, spv_silu(a, value));
    spv_if_else(a, &one);
        spv_if_open(a, &two, spv_ueq(a, kind, spv_u(a, 2)));
            spv_save(a, slot, spv_sigmoid(a, value));
        spv_if_shut(a, &two);
    spv_if_shut(a, &one);
    return spv_load_f(a, slot);
}

/* `bias[index]` when the flag says there is one, and zero otherwise -- written
 * as a branch rather than as a select, because a select evaluates both of its
 * arms and the load would then run off the end of the one-float stand-in that
 * is bound when there is no bias.  Reading past a storage buffer is undefined
 * without `robustBufferAccess`, which this file does not ask a device to
 * enable, so the read has to not happen rather than be harmless. */
static uint32_t spv_bias(VulkAsm *a, int32_t bind, uint32_t index, uint32_t flag)
{
    uint32_t slot = spv_var_f(a);
    VulkIf   have;
    spv_save(a, slot, spv_f(a, 0.0f));
    spv_if_open(a, &have, spv_ueq(a, flag, spv_u(a, 1)));
    spv_save(a, slot, spv_get(a, bind, index));
    spv_if_shut(a, &have);
    return spv_load_f(a, slot);
}

/* The reduction tree.  `share[lane]` goes in, `share[0]` comes out, and every
 * invocation of the workgroup must reach this -- the barrier inside makes it a
 * convergence point, so it never sits under a branch. */
static void spv_fold(VulkAsm *a, uint32_t share, uint32_t first, uint32_t lane, int max_flag)
{
    VulkFor  round;
    uint32_t step = spv_var_u(a);
    uint32_t seat = spv_uadd(a, lane, first);
    spv_barrier(a);
    spv_for_open(a, &round, step, spv_u(a, 0), spv_u(a, 6), spv_u(a, 1));
    {
        uint32_t half = spv_bin(a, SPV_OP_SHIFT_RIGHT, a->t_u32,
                                spv_u(a, VULK_LOCAL / 2), spv_load_u(a, step));
        VulkIf   near;
        spv_if_open(a, &near, spv_ult(a, lane, half));
        {
            uint32_t here = spv_share_cell(a, share, seat);
            uint32_t away = spv_share_cell(a, share, spv_uadd(a, seat, half));
            uint32_t mine = spv_load_f(a, here);
            uint32_t next = spv_load_f(a, away);
            spv_save(a, here, max_flag ? spv_call2(a, SPV_GL_FMAX, mine, next)
                                       : spv_fadd(a, mine, next));
        }
        spv_if_shut(a, &near);
        spv_barrier(a);
    }
    spv_for_shut(a, &round);
}

/* Three loop shapes appear over and over, and they differ only in where an
 * invocation starts and how far it steps.  Naming them is what keeps the
 * kernels below readable, and picking the wrong one is the mistake this
 * comment exists to prevent: `span` spreads work over the whole dispatch,
 * `lane` spreads it over one workgroup, and `unit` gives a workgroup a whole
 * piece of work to itself. */
static uint32_t spv_span_open(VulkAsm *a, VulkFor *loop, uint32_t total)
{
    uint32_t var = spv_var_u(a);
    spv_for_open(a, loop, var, spv_index(a), total, spv_stride(a));
    return spv_load_u(a, var);
}

static uint32_t spv_lane_open(VulkAsm *a, VulkFor *loop, uint32_t total)
{
    uint32_t var = spv_var_u(a);
    spv_for_open(a, loop, var, spv_lane(a), total, spv_u(a, VULK_LOCAL));
    return spv_load_u(a, var);
}

static uint32_t spv_unit_open(VulkAsm *a, VulkFor *loop, uint32_t total)
{
    uint32_t var = spv_var_u(a);
    spv_for_open(a, loop, var, spv_group(a), total, spv_groups(a));
    return spv_load_u(a, var);
}

/* A plain counted loop, one invocation walking the whole range. */
static uint32_t spv_step_open(VulkAsm *a, VulkFor *loop, uint32_t total)
{
    uint32_t var = spv_var_u(a);
    spv_for_open(a, loop, var, spv_u(a, 0), total, spv_u(a, 1));
    return spv_load_u(a, var);
}

/* -- text: dense ---------------------------------------------------------- */

/* dst[token, row] = sum over cols of plane[row, col] * src[token, col].
 *
 * One workgroup to an output cell, the sixty-four invocations striding along
 * the row together.  The alternative -- one invocation to an output cell,
 * looping the whole row by itself -- needs no reduction and no shared memory,
 * and reads the weight matrix with a stride of `cols` floats per invocation,
 * which is the worst access pattern a memory system has.  The reduction is the
 * price of reading the weights in order.
 *
 * The token is the dispatch's second dimension, so a prefill of 512 tokens is
 * 512 times the workgroups rather than 512 dispatches. */
static void vulk_kernel_dense(VulkAsm *a)
{
    VulkFor  token_loop, row_loop, col_loop;
    uint32_t rows, cols, tokens, lane, share, acc, token, row, col, seat;

    spv_open(a, 3, VULK_LOCAL);
    share  = spv_share(a, VULK_LOCAL);
    lane   = spv_lane(a);
    rows   = spv_pin_u(a, 0);
    cols   = spv_pin_u(a, 1);
    tokens = spv_pin_u(a, 2);
    acc    = spv_var_f(a);

    /* the token is the dispatch's second dimension, and strides over it for
     * the same reason the first one does: a prefill may be longer than the
     * workgroup count a device will take */
    seat  = spv_var_u(a);
    spv_for_open(a, &token_loop, seat, spv_group_y(a), tokens, spv_groups_y(a));
    token = spv_load_u(a, seat);
    row   = spv_unit_open(a, &row_loop, rows);
    {
        uint32_t base = spv_umul(a, row, cols);
        uint32_t from = spv_umul(a, token, cols);
        VulkIf   first;
        spv_save(a, acc, spv_f(a, 0.0f));
        col = spv_lane_open(a, &col_loop, cols);
        {
            uint32_t w = spv_get(a, 0, spv_uadd(a, base, col));
            uint32_t x = spv_get(a, 1, spv_uadd(a, from, col));
            spv_save(a, acc, spv_fadd(a, spv_load_f(a, acc), spv_fmul(a, w, x)));
        }
        spv_for_shut(a, &col_loop);
        spv_save(a, spv_share_cell(a, share, lane), spv_load_f(a, acc));
        spv_fold(a, share, spv_u(a, 0), lane, 0);
        spv_if_open(a, &first, spv_ueq(a, lane, spv_u(a, 0)));
        {
            uint32_t out = spv_uadd(a, spv_umul(a, token, rows), row);
            spv_set(a, 2, out, spv_load_f(a, spv_share_cell(a, share, spv_u(a, 0))));
        }
        spv_if_shut(a, &first);
        spv_barrier(a);
    }
    spv_for_shut(a, &row_loop);
    spv_for_shut(a, &token_loop);
}

/* -- text: rms norm -------------------------------------------------------- */

static void vulk_kernel_rmsnorm(VulkAsm *a)
{
    VulkFor  token_loop, sum_loop, out_loop;
    uint32_t tokens, width, eps, lane, share, acc, token;

    spv_open(a, 3, VULK_LOCAL);
    share  = spv_share(a, VULK_LOCAL);
    lane   = spv_lane(a);
    tokens = spv_pin_u(a, 0);
    width  = spv_pin_u(a, 1);
    eps    = spv_pin_f(a, 2);
    acc    = spv_var_f(a);

    token = spv_unit_open(a, &token_loop, tokens);
    {
        uint32_t base = spv_umul(a, token, width);
        uint32_t scale;
        spv_save(a, acc, spv_f(a, 0.0f));
        {
            uint32_t j = spv_lane_open(a, &sum_loop, width);
            uint32_t v = spv_get(a, 0, spv_uadd(a, base, j));
            spv_save(a, acc, spv_fadd(a, spv_load_f(a, acc), spv_fmul(a, v, v)));
            spv_for_shut(a, &sum_loop);
        }
        spv_save(a, spv_share_cell(a, share, lane), spv_load_f(a, acc));
        spv_fold(a, share, spv_u(a, 0), lane, 0);
        {
            uint32_t mass = spv_load_f(a, spv_share_cell(a, share, spv_u(a, 0)));
            uint32_t mean = spv_fdiv(a, mass, spv_u2f(a, width));
            scale = spv_call1(a, SPV_GL_RSQRT, spv_fadd(a, mean, eps));
        }
        spv_barrier(a);
        {
            uint32_t j = spv_lane_open(a, &out_loop, width);
            uint32_t v = spv_get(a, 0, spv_uadd(a, base, j));
            uint32_t g = spv_get(a, 1, j);
            spv_set(a, 2, spv_uadd(a, base, j), spv_fmul(a, spv_fmul(a, v, scale), g));
            spv_for_shut(a, &out_loop);
        }
        spv_barrier(a);
    }
    spv_for_shut(a, &token_loop);
}

/* -- text: swiglu ---------------------------------------------------------- */

static void vulk_kernel_swiglu(VulkAsm *a)
{
    VulkFor  loop;
    uint32_t total, index;
    spv_open(a, 3, VULK_LOCAL);
    total = spv_pin_u(a, 0);
    index = spv_span_open(a, &loop, total);
    {
        uint32_t gate = spv_get(a, 0, index);
        uint32_t lift = spv_get(a, 1, index);
        spv_set(a, 2, index, spv_fmul(a, spv_silu(a, gate), lift));
    }
    spv_for_shut(a, &loop);
}

/* -- text: rope ------------------------------------------------------------ */

/* The table holds one row of `width` floats a token, cosines in the first half
 * and sines in the second, which is the reference's cat(freqs, freqs); a pair
 * is the value at `i` and the value at `i + half`. */
static void vulk_kernel_rope(VulkAsm *a)
{
    VulkFor  loop;
    uint32_t tokens, heads, width, stride, half, total, index;

    spv_open(a, 2, VULK_LOCAL);
    tokens = spv_pin_u(a, 0);
    heads  = spv_pin_u(a, 1);
    width  = spv_pin_u(a, 2);
    stride = spv_pin_u(a, 3);
    half   = spv_udiv(a, width, spv_u(a, 2));
    total  = spv_umul(a, spv_umul(a, tokens, heads), half);

    index = spv_span_open(a, &loop, total);
    {
        uint32_t slot  = spv_umod(a, index, half);
        uint32_t head  = spv_umod(a, spv_udiv(a, index, half), heads);
        uint32_t token = spv_udiv(a, index, spv_umul(a, half, heads));
        uint32_t base  = spv_uadd(a, spv_umul(a, token, stride), spv_umul(a, head, width));
        uint32_t turn  = spv_umul(a, token, width);
        uint32_t lo    = spv_get(a, 0, spv_uadd(a, base, slot));
        uint32_t hi    = spv_get(a, 0, spv_uadd(a, spv_uadd(a, base, half), slot));
        uint32_t cosine = spv_get(a, 1, spv_uadd(a, turn, slot));
        uint32_t sine   = spv_get(a, 1, spv_uadd(a, spv_uadd(a, turn, half), slot));
        spv_set(a, 0, spv_uadd(a, base, slot),
                spv_fsub(a, spv_fmul(a, lo, cosine), spv_fmul(a, hi, sine)));
        spv_set(a, 0, spv_uadd(a, spv_uadd(a, base, half), slot),
                spv_fadd(a, spv_fmul(a, hi, cosine), spv_fmul(a, lo, sine)));
    }
    spv_for_shut(a, &loop);
}

/* -- text: attention ------------------------------------------------------- */

/* One workgroup to a (token, head) pair, and the causal span walked in chunks
 * of sixty-four -- one position an invocation.
 *
 * The CPU version writes the whole row of scores, takes a softmax over it, and
 * then mixes.  That needs `span` floats of scratch a worker, which the seam
 * supplies as `board`; a device would need `tokens * heads * span`, which at a
 * 512 token prefill with 32 heads and a 4096 window is 256 MiB of scratch for
 * a step whose output is 4 MiB.  So this runs the streaming form instead: a
 * running maximum and a running total, both rescaled every time the maximum
 * moves, with the partial mix living in the output row.  It reaches the same
 * answer in one pass over the keys and the values, needs no scratch at all,
 * and ignores `board` entirely.
 *
 * Shared memory is sixty-four scores, sixty-four reduction cells and two
 * scalars -- 520 bytes against the 16 KiB every Vulkan host guarantees. */
#define VULK_ATT_FOLD  VULK_LOCAL
#define VULK_ATT_PEAK  (2 * VULK_LOCAL)
#define VULK_ATT_MASS  (2 * VULK_LOCAL + 1)
#define VULK_ATT_SHARE (2 * VULK_LOCAL + 2)
#define VULK_ATT_FLOOR (-3.0e38f)

static void vulk_kernel_attend(VulkAsm *a)
{
    VulkFor  unit_loop, chunk_loop;
    uint32_t tokens, heads, groups, width, span, base_pos, scale;
    uint32_t lane, share, units, unit;

    spv_open(a, 4, VULK_LOCAL);
    share    = spv_share(a, VULK_ATT_SHARE);
    lane     = spv_lane(a);
    tokens   = spv_pin_u(a, 0);
    heads    = spv_pin_u(a, 1);
    groups   = spv_pin_u(a, 2);
    width    = spv_pin_u(a, 3);
    span     = spv_pin_u(a, 4);
    base_pos = spv_pin_u(a, 5);
    scale    = spv_pin_f(a, 6);
    units    = spv_umul(a, tokens, heads);

    unit = spv_unit_open(a, &unit_loop, units);
    {
        uint32_t token  = spv_udiv(a, unit, heads);
        uint32_t head   = spv_umod(a, unit, heads);
        uint32_t spread = spv_udiv(a, heads, groups);
        uint32_t band   = spv_udiv(a, head, spread);
        uint32_t step   = spv_uadd(a, spv_uadd(a, base_pos, token), spv_u(a, 1));
        uint32_t reach  = spv_pick(a, a->t_u32, spv_ult(a, span, step), span, step);
        uint32_t here   = spv_umul(a, spv_uadd(a, spv_umul(a, token, heads), head), width);
        uint32_t there  = spv_umul(a, spv_umul(a, band, span), width);
        uint32_t last   = spv_usub(a, span, spv_u(a, 1));
        uint32_t peak_cell = spv_share_cell(a, share, spv_u(a, VULK_ATT_PEAK));
        uint32_t mass_cell = spv_share_cell(a, share, spv_u(a, VULK_ATT_MASS));
        VulkFor  zero_loop, tail_loop;
        VulkIf   lead;

        {   /* the output row is the accumulator, so it starts at nothing */
            uint32_t j = spv_lane_open(a, &zero_loop, width);
            spv_set(a, 3, spv_uadd(a, here, j), spv_f(a, 0.0f));
            spv_for_shut(a, &zero_loop);
        }
        spv_if_open(a, &lead, spv_ueq(a, lane, spv_u(a, 0)));
        {
            spv_save(a, peak_cell, spv_f(a, VULK_ATT_FLOOR));
            spv_save(a, mass_cell, spv_f(a, 0.0f));
        }
        spv_if_shut(a, &lead);
        spv_barrier(a);

        {
            uint32_t chunk = spv_var_u(a);
            spv_for_open(a, &chunk_loop, chunk, spv_u(a, 0), reach, spv_u(a, VULK_LOCAL));
            {
                uint32_t start = spv_load_u(a, chunk);
                uint32_t place = spv_uadd(a, start, lane);
                uint32_t was   = spv_load_f(a, peak_cell);
                uint32_t score = spv_var_f(a);
                uint32_t crest, rise, prob, total;
                VulkFor  dot_loop, mix_loop;
                VulkIf   live, note;

                /* the score for this invocation's position, or the floor when
                 * the position is past the causal horizon */
                spv_save(a, score, spv_f(a, VULK_ATT_FLOOR));
                spv_if_open(a, &live, spv_ult(a, place, reach));
                {
                    uint32_t key = spv_uadd(a, there, spv_umul(a, place, width));
                    uint32_t acc = spv_var_f(a);
                    uint32_t j;
                    spv_save(a, acc, spv_f(a, 0.0f));
                    j = spv_step_open(a, &dot_loop, width);
                    spv_save(a, acc, spv_fadd(a, spv_load_f(a, acc),
                             spv_fmul(a, spv_get(a, 0, spv_uadd(a, here, j)),
                                         spv_get(a, 1, spv_uadd(a, key, j)))));
                    spv_for_shut(a, &dot_loop);
                    spv_save(a, score, spv_fmul(a, spv_load_f(a, acc), scale));
                }
                spv_if_shut(a, &live);

                /* the chunk's maximum, folded through the shared cells */
                spv_save(a, spv_share_cell(a, share, spv_uadd(a, lane, spv_u(a, VULK_ATT_FOLD))),
                         spv_load_f(a, score));
                spv_fold(a, share, spv_u(a, VULK_ATT_FOLD), lane, 1);
                crest = spv_load_f(a, spv_share_cell(a, share, spv_u(a, VULK_ATT_FOLD)));
                spv_barrier(a);

                {
                    uint32_t now = spv_call2(a, SPV_GL_FMAX, was, crest);
                    rise = spv_call1(a, SPV_GL_EXP, spv_fsub(a, was, now));
                    prob = spv_call1(a, SPV_GL_EXP, spv_fsub(a, spv_load_f(a, score), now));
                    spv_save(a, spv_share_cell(a, share, lane), prob);
                    spv_save(a, spv_share_cell(a, share,
                                spv_uadd(a, lane, spv_u(a, VULK_ATT_FOLD))), prob);
                    spv_fold(a, share, spv_u(a, VULK_ATT_FOLD), lane, 0);
                    total = spv_load_f(a, spv_share_cell(a, share, spv_u(a, VULK_ATT_FOLD)));
                    spv_barrier(a);
                    spv_if_open(a, &note, spv_ueq(a, lane, spv_u(a, 0)));
                    {
                        spv_save(a, peak_cell, now);
                        spv_save(a, mass_cell,
                                 spv_fadd(a, spv_fmul(a, spv_load_f(a, mass_cell), rise), total));
                    }
                    spv_if_shut(a, &note);
                }

                /* mix this chunk of values into the output row, rescaling what
                 * is already there by however far the maximum moved */
                {
                    uint32_t j = spv_lane_open(a, &mix_loop, width);
                    uint32_t cell = spv_cell(a, 3, spv_uadd(a, here, j));
                    uint32_t acc  = spv_var_f(a);
                    uint32_t slot;
                    VulkFor  over;
                    spv_save(a, acc, spv_fmul(a, spv_load_f(a, cell), rise));
                    slot = spv_step_open(a, &over, spv_u(a, VULK_LOCAL));
                    {
                        uint32_t at   = spv_uadd(a, start, slot);
                        uint32_t safe = spv_pick(a, a->t_u32, spv_ult(a, at, span), at, last);
                        uint32_t hold = spv_load_f(a, spv_share_cell(a, share, slot));
                        uint32_t val  = spv_get(a, 2, spv_uadd(a, there,
                                                spv_uadd(a, spv_umul(a, safe, width), j)));
                        spv_save(a, acc, spv_fadd(a, spv_load_f(a, acc),
                                                  spv_fmul(a, hold, val)));
                    }
                    spv_for_shut(a, &over);
                    spv_save(a, cell, spv_load_f(a, acc));
                    spv_for_shut(a, &mix_loop);
                }
                spv_barrier(a);
            }
            spv_for_shut(a, &chunk_loop);
        }

        /* and the softmax denominator, once */
        {
            uint32_t mass = spv_load_f(a, mass_cell);
            uint32_t back = spv_pick(a, a->t_f32, spv_fgt(a, mass, spv_f(a, 0.0f)),
                                     spv_fdiv(a, spv_f(a, 1.0f), mass), spv_f(a, 0.0f));
            uint32_t j = spv_lane_open(a, &tail_loop, width);
            uint32_t cell = spv_cell(a, 3, spv_uadd(a, here, j));
            spv_save(a, cell, spv_fmul(a, spv_load_f(a, cell), back));
            spv_for_shut(a, &tail_loop);
        }
        spv_barrier(a);
    }
    spv_for_shut(a, &unit_loop);
}

/* -- text: the short convolution ------------------------------------------- */

/* One invocation to a channel, walking the tokens in order with the rolling
 * window in registers, exactly as the CPU does -- the recurrence is along the
 * token axis, so there is nothing to spread there, and a channel is completely
 * independent of every other channel.  `hist` comes in carrying the tail of
 * the previous call and goes out carrying the tail of this one. */
static void vulk_kernel_conv1d(VulkAsm *a)
{
    VulkFor  chan_loop, seed_loop, token_loop, tap_loop, roll_loop, keep_loop;
    uint32_t tokens, dim, width, bias_flag, keep, ring, cell_type, channel;

    spv_open(a, 5, VULK_LOCAL);
    tokens    = spv_pin_u(a, 0);
    dim       = spv_pin_u(a, 1);
    width     = spv_pin_u(a, 2);
    bias_flag = spv_pin_u(a, 3);
    keep      = spv_usub(a, width, spv_u(a, 1));
    ring      = spv_var_ring(a, 16, &cell_type);
    (void)cell_type;

    channel = spv_span_open(a, &chan_loop, dim);
    {
        uint32_t taps = spv_umul(a, channel, width);
        uint32_t past = spv_umul(a, channel, keep);
        uint32_t seed = spv_bias(a, 4, channel, bias_flag);
        uint32_t acc  = spv_var_f(a);
        uint32_t token, slot;

        slot = spv_step_open(a, &seed_loop, keep);
        spv_save(a, spv_ring_cell(a, ring, slot), spv_get(a, 2, spv_uadd(a, past, slot)));
        spv_for_shut(a, &seed_loop);

        token = spv_step_open(a, &token_loop, tokens);
        {
            uint32_t at = spv_uadd(a, spv_umul(a, token, dim), channel);
            spv_save(a, acc, seed);
            spv_save(a, spv_ring_cell(a, ring, keep), spv_get(a, 0, at));
            {
                uint32_t j = spv_step_open(a, &tap_loop, width);
                spv_save(a, acc, spv_fadd(a, spv_load_f(a, acc),
                         spv_fmul(a, spv_get(a, 3, spv_uadd(a, taps, j)),
                                     spv_load_f(a, spv_ring_cell(a, ring, j)))));
                spv_for_shut(a, &tap_loop);
            }
            spv_set(a, 1, at, spv_load_f(a, acc));
            {
                uint32_t j = spv_step_open(a, &roll_loop, keep);
                spv_save(a, spv_ring_cell(a, ring, j),
                         spv_load_f(a, spv_ring_cell(a, ring, spv_uadd(a, j, spv_u(a, 1)))));
                spv_for_shut(a, &roll_loop);
            }
        }
        spv_for_shut(a, &token_loop);

        {
            uint32_t j = spv_step_open(a, &keep_loop, keep);
            spv_set(a, 2, spv_uadd(a, past, j), spv_load_f(a, spv_ring_cell(a, ring, j)));
            spv_for_shut(a, &keep_loop);
        }
    }
    spv_for_shut(a, &chan_loop);
}

/* -- picture: the shared shape arithmetic ---------------------------------- */

/* Every picture kernel is one invocation to an output cell, and every one of
 * them starts by turning a flat index back into (batch, channel, row, column).
 * NCHW means the column moves fastest. */
typedef struct VulkCell { uint32_t x, y, channel, batch; } VulkCell;

static VulkCell spv_unpack(VulkAsm *a, uint32_t index,
                           uint32_t width, uint32_t height, uint32_t channels)
{
    VulkCell cell;
    uint32_t face  = spv_umul(a, width, height);
    uint32_t sheet = spv_umul(a, face, channels);
    cell.x       = spv_umod(a, index, width);
    cell.y       = spv_umod(a, spv_udiv(a, index, width), height);
    cell.channel = spv_umod(a, spv_udiv(a, index, face), channels);
    cell.batch   = spv_udiv(a, index, sheet);
    return cell;
}

/* -- picture: convolution --------------------------------------------------- */

/* out = act(conv(in, weight, bias)), run directly rather than as a lowering
 * and a matrix multiply.
 *
 * The CPU backend lowers a patch matrix and calls the gemm, because a
 * contiguous stream is what a compiler vectorises and a cache wants.  A device
 * wants the opposite: the lowering is nine reads and nine writes of the whole
 * feature map for a 3x3, all of it through memory, where the direct form reads
 * the input nine times out of cache and writes the output once.  That is also
 * why `conv_room` answers zero here -- there is no patch matrix to size, and
 * the TODO entry that asked for this seam said exactly that. */
static void vulk_kernel_yconv(VulkAsm *a)
{
    VulkFor  loop, chan_loop, row_loop, col_loop;
    uint32_t in_c, in_h, in_w, out_c, out_h, out_w;
    uint32_t reach, stride, pad, dilate, groups, act, bias_flag, index;

    spv_open(a, 4, VULK_LOCAL);
    in_c      = spv_pin_u(a, 0);
    in_h      = spv_pin_u(a, 1);
    in_w      = spv_pin_u(a, 2);
    out_c     = spv_pin_u(a, 3);
    out_h     = spv_pin_u(a, 4);
    out_w     = spv_pin_u(a, 5);
    reach     = spv_pin_u(a, 6);
    stride    = spv_pin_u(a, 7);
    pad       = spv_pin_u(a, 8);
    dilate    = spv_pin_u(a, 9);
    groups    = spv_pin_u(a, 10);
    act       = spv_pin_u(a, 11);
    bias_flag = spv_pin_u(a, 12);

    index = spv_span_open(a, &loop, spv_umul(a, spv_umul(a, out_w, out_h),
                                             spv_umul(a, out_c, spv_pin_u(a, 13))));
    {
        VulkCell cell = spv_unpack(a, index, out_w, out_h, out_c);
        uint32_t reach_in  = spv_udiv(a, in_c, groups);
        uint32_t reach_out = spv_udiv(a, out_c, groups);
        uint32_t band      = spv_udiv(a, cell.channel, reach_out);
        uint32_t acc       = spv_var_f(a);
        uint32_t base_y    = spv_u2i(a, spv_usub(a, spv_umul(a, cell.y, stride), pad));
        uint32_t base_x    = spv_u2i(a, spv_usub(a, spv_umul(a, cell.x, stride), pad));
        uint32_t plane     = spv_umul(a, in_h, in_w);
        uint32_t chan;

        spv_save(a, acc, spv_bias(a, 2, cell.channel, bias_flag));
        chan = spv_step_open(a, &chan_loop, reach_in);
        {
            uint32_t from = spv_uadd(a, spv_umul(a, band, reach_in), chan);
            uint32_t face = spv_umul(a, spv_uadd(a, spv_umul(a, cell.batch, in_c), from), plane);
            uint32_t mat  = spv_umul(a, spv_uadd(a, spv_umul(a, cell.channel, reach_in), chan),
                                     spv_umul(a, reach, reach));
            uint32_t ky   = spv_step_open(a, &row_loop, reach);
            {
                uint32_t at_y = spv_iadd(a, base_y, spv_u2i(a, spv_umul(a, ky, dilate)));
                VulkIf   rows;
                spv_if_open(a, &rows, spv_ult(a, spv_i2u(a, at_y), in_h));
                {
                    uint32_t row = spv_uadd(a, face, spv_umul(a, spv_i2u(a, at_y), in_w));
                    uint32_t kx  = spv_step_open(a, &col_loop, reach);
                    {
                        uint32_t at_x = spv_iadd(a, base_x, spv_u2i(a, spv_umul(a, kx, dilate)));
                        VulkIf   cols;
                        spv_if_open(a, &cols, spv_ult(a, spv_i2u(a, at_x), in_w));
                        {
                            uint32_t v = spv_get(a, 0, spv_uadd(a, row, spv_i2u(a, at_x)));
                            uint32_t w = spv_get(a, 1, spv_uadd(a, mat,
                                            spv_uadd(a, spv_umul(a, ky, reach), kx)));
                            spv_save(a, acc, spv_fadd(a, spv_load_f(a, acc), spv_fmul(a, v, w)));
                        }
                        spv_if_shut(a, &cols);
                    }
                    spv_for_shut(a, &col_loop);
                }
                spv_if_shut(a, &rows);
            }
            spv_for_shut(a, &row_loop);
        }
        spv_for_shut(a, &chan_loop);
        spv_set(a, 3, index, spv_act(a, spv_load_f(a, acc), act));
    }
    spv_for_shut(a, &loop);
}

/* -- picture: transposed convolution ---------------------------------------- */

/* Stride equal to the kernel and no padding, which is the only shape the mask
 * prototype head asks for.  The CPU scatters -- every input cell writes into
 * `kernel * kernel` outputs -- and a device cannot, because two invocations
 * would write the same output.  The gather is exact rather than approximate:
 * with stride equal to the kernel, an output cell is written by exactly one
 * input cell, at `out / stride`, through tap `out % stride`. */
static void vulk_kernel_ydeconv(VulkAsm *a)
{
    VulkFor  loop, chan_loop;
    uint32_t in_c, in_h, in_w, out_c, out_h, out_w, reach, stride, act, bias_flag, index;

    spv_open(a, 4, VULK_LOCAL);
    in_c      = spv_pin_u(a, 0);
    in_h      = spv_pin_u(a, 1);
    in_w      = spv_pin_u(a, 2);
    out_c     = spv_pin_u(a, 3);
    out_h     = spv_pin_u(a, 4);
    out_w     = spv_pin_u(a, 5);
    reach     = spv_pin_u(a, 6);
    stride    = spv_pin_u(a, 7);
    act       = spv_pin_u(a, 8);
    bias_flag = spv_pin_u(a, 9);

    index = spv_span_open(a, &loop, spv_umul(a, spv_umul(a, out_w, out_h),
                                             spv_umul(a, out_c, spv_pin_u(a, 10))));
    {
        VulkCell cell = spv_unpack(a, index, out_w, out_h, out_c);
        uint32_t at_y = spv_udiv(a, cell.y, stride);
        uint32_t at_x = spv_udiv(a, cell.x, stride);
        uint32_t ky   = spv_umod(a, cell.y, stride);
        uint32_t kx   = spv_umod(a, cell.x, stride);
        uint32_t acc  = spv_var_f(a);
        uint32_t plane = spv_umul(a, in_h, in_w);
        VulkIf   inside;

        spv_save(a, acc, spv_bias(a, 2, cell.channel, bias_flag));
        spv_if_open(a, &inside, spv_band(a, spv_band(a, spv_ult(a, at_y, in_h),
                                                        spv_ult(a, at_x, in_w)),
                                            spv_band(a, spv_ult(a, ky, reach),
                                                        spv_ult(a, kx, reach))));
        {
            uint32_t chan = spv_step_open(a, &chan_loop, in_c);
            uint32_t face = spv_umul(a, spv_uadd(a, spv_umul(a, cell.batch, in_c), chan), plane);
            /* torch keeps a transposed convolution as [in, out, k, k] */
            uint32_t mat  = spv_umul(a, spv_uadd(a, spv_umul(a, chan, out_c), cell.channel),
                                     spv_umul(a, reach, reach));
            uint32_t v = spv_get(a, 0, spv_uadd(a, face,
                            spv_uadd(a, spv_umul(a, at_y, in_w), at_x)));
            uint32_t w = spv_get(a, 1, spv_uadd(a, mat,
                            spv_uadd(a, spv_umul(a, ky, reach), kx)));
            spv_save(a, acc, spv_fadd(a, spv_load_f(a, acc), spv_fmul(a, v, w)));
            spv_for_shut(a, &chan_loop);
        }
        spv_if_shut(a, &inside);
        spv_set(a, 3, index, spv_act(a, spv_load_f(a, acc), act));
    }
    spv_for_shut(a, &loop);
}

/* -- picture: max pooling --------------------------------------------------- */

static void vulk_kernel_ypool(VulkAsm *a)
{
    VulkFor  loop, row_loop, col_loop;
    uint32_t channels, in_h, in_w, out_h, out_w, reach, stride, pad, index;

    spv_open(a, 2, VULK_LOCAL);
    channels = spv_pin_u(a, 0);
    in_h     = spv_pin_u(a, 1);
    in_w     = spv_pin_u(a, 2);
    out_h    = spv_pin_u(a, 3);
    out_w    = spv_pin_u(a, 4);
    reach    = spv_pin_u(a, 5);
    stride   = spv_pin_u(a, 6);
    pad      = spv_pin_u(a, 7);

    index = spv_span_open(a, &loop, spv_umul(a, spv_umul(a, out_w, out_h), channels));
    {
        uint32_t x    = spv_umod(a, index, out_w);
        uint32_t y    = spv_umod(a, spv_udiv(a, index, out_w), out_h);
        uint32_t face = spv_udiv(a, index, spv_umul(a, out_w, out_h));
        uint32_t base = spv_umul(a, face, spv_umul(a, in_h, in_w));
        uint32_t best = spv_var_f(a);
        uint32_t at_y = spv_u2i(a, spv_usub(a, spv_umul(a, y, stride), pad));
        uint32_t at_x = spv_u2i(a, spv_usub(a, spv_umul(a, x, stride), pad));
        uint32_t ky;

        spv_save(a, best, spv_f(a, -FLT_MAX));
        ky = spv_step_open(a, &row_loop, reach);
        {
            uint32_t row_y = spv_iadd(a, at_y, spv_u2i(a, ky));
            VulkIf   rows;
            spv_if_open(a, &rows, spv_ult(a, spv_i2u(a, row_y), in_h));
            {
                uint32_t row = spv_uadd(a, base, spv_umul(a, spv_i2u(a, row_y), in_w));
                uint32_t kx  = spv_step_open(a, &col_loop, reach);
                {
                    uint32_t col_x = spv_iadd(a, at_x, spv_u2i(a, kx));
                    VulkIf   cols;
                    spv_if_open(a, &cols, spv_ult(a, spv_i2u(a, col_x), in_w));
                    {
                        uint32_t v = spv_get(a, 0, spv_uadd(a, row, spv_i2u(a, col_x)));
                        spv_save(a, best, spv_call2(a, SPV_GL_FMAX, spv_load_f(a, best), v));
                    }
                    spv_if_shut(a, &cols);
                }
                spv_for_shut(a, &col_loop);
            }
            spv_if_shut(a, &rows);
        }
        spv_for_shut(a, &row_loop);
        spv_set(a, 1, index, spv_load_f(a, best));
    }
    spv_for_shut(a, &loop);
}

/* -- picture: nearest neighbour resize -------------------------------------- */

static void vulk_kernel_yresize(VulkAsm *a)
{
    VulkFor  loop;
    uint32_t channels, in_h, in_w, out_h, out_w, scale, index;

    spv_open(a, 2, VULK_LOCAL);
    channels = spv_pin_u(a, 0);
    in_h     = spv_pin_u(a, 1);
    in_w     = spv_pin_u(a, 2);
    out_h    = spv_pin_u(a, 3);
    out_w    = spv_pin_u(a, 4);
    scale    = spv_pin_u(a, 5);

    index = spv_span_open(a, &loop, spv_umul(a, spv_umul(a, out_w, out_h), channels));
    {
        uint32_t x    = spv_umod(a, index, out_w);
        uint32_t y    = spv_umod(a, spv_udiv(a, index, out_w), out_h);
        uint32_t face = spv_udiv(a, index, spv_umul(a, out_w, out_h));
        uint32_t base = spv_umul(a, face, spv_umul(a, in_h, in_w));
        uint32_t from = spv_uadd(a, base,
                          spv_uadd(a, spv_umul(a, spv_udiv(a, y, scale), in_w),
                                      spv_udiv(a, x, scale)));
        spv_set(a, 1, index, spv_get(a, 0, from));
    }
    spv_for_shut(a, &loop);
}

/* -- picture: activation, add, softmax -------------------------------------- */

static void vulk_kernel_yact(VulkAsm *a)
{
    VulkFor  loop;
    uint32_t total, kind, index;
    spv_open(a, 1, VULK_LOCAL);
    total = spv_pin_u(a, 0);
    kind  = spv_pin_u(a, 1);
    index = spv_span_open(a, &loop, total);
    spv_set(a, 0, index, spv_act(a, spv_get(a, 0, index), kind));
    spv_for_shut(a, &loop);
}

static void vulk_kernel_yadd(VulkAsm *a)
{
    VulkFor  loop;
    uint32_t total, index;
    spv_open(a, 2, VULK_LOCAL);
    total = spv_pin_u(a, 0);
    index = spv_span_open(a, &loop, total);
    spv_set(a, 0, index, spv_fadd(a, spv_get(a, 0, index), spv_get(a, 1, index)));
    spv_for_shut(a, &loop);
}

/* Softmax over the last axis.  One invocation to a row rather than one
 * workgroup: the only caller is the distribution focal head, whose rows are
 * sixteen wide and whose row count is the number of anchors, so there are tens
 * of thousands of very short rows and a reduction tree over sixteen values
 * would cost more than the row. */
static void vulk_kernel_ysoftmax(VulkAsm *a)
{
    VulkFor  loop, peak_loop, mass_loop, back_loop;
    uint32_t rows, cols, index;

    spv_open(a, 1, VULK_LOCAL);
    rows = spv_pin_u(a, 0);
    cols = spv_pin_u(a, 1);

    index = spv_span_open(a, &loop, rows);
    {
        uint32_t base = spv_umul(a, index, cols);
        uint32_t peak = spv_var_f(a);
        uint32_t mass = spv_var_f(a);
        uint32_t back;
        uint32_t col;

        spv_save(a, peak, spv_f(a, -FLT_MAX));
        col = spv_step_open(a, &peak_loop, cols);
        spv_save(a, peak, spv_call2(a, SPV_GL_FMAX, spv_load_f(a, peak),
                                    spv_get(a, 0, spv_uadd(a, base, col))));
        spv_for_shut(a, &peak_loop);

        spv_save(a, mass, spv_f(a, 0.0f));
        col = spv_step_open(a, &mass_loop, cols);
        {
            uint32_t v = spv_call1(a, SPV_GL_EXP,
                            spv_fsub(a, spv_get(a, 0, spv_uadd(a, base, col)),
                                        spv_load_f(a, peak)));
            spv_set(a, 0, spv_uadd(a, base, col), v);
            spv_save(a, mass, spv_fadd(a, spv_load_f(a, mass), v));
        }
        spv_for_shut(a, &mass_loop);

        back = spv_pick(a, a->t_f32, spv_fgt(a, spv_load_f(a, mass), spv_f(a, 0.0f)),
                        spv_fdiv(a, spv_f(a, 1.0f), spv_load_f(a, mass)), spv_f(a, 1.0f));
        col = spv_step_open(a, &back_loop, cols);
        spv_set(a, 0, spv_uadd(a, base, col),
                spv_fmul(a, spv_get(a, 0, spv_uadd(a, base, col)), back));
        spv_for_shut(a, &back_loop);
    }
    spv_for_shut(a, &loop);
}

/* -- picture: the general matrix multiply ----------------------------------- */

/* out[row, col] = sum over mid of a[row, mid] * b[mid, col], or b[col, mid]
 * when the swap flag is set.  One invocation to an output cell.  The blocked
 * four-by-eight register tile the CPU carries is a cache argument, and a
 * device's cache is not the thing in the way; what is in the way here is that
 * `b` is read with a stride of `b_step` by every invocation of a row, which is
 * the coalesced direction, so the plain form is already the right one. */
static void vulk_kernel_ygemm(VulkAsm *a)
{
    VulkFor  loop, mid_loop;
    uint32_t rows, mids, cols, a_step, b_step, out_step, swap, index;

    spv_open(a, 3, VULK_LOCAL);
    rows     = spv_pin_u(a, 0);
    mids     = spv_pin_u(a, 1);
    cols     = spv_pin_u(a, 2);
    a_step   = spv_pin_u(a, 3);
    b_step   = spv_pin_u(a, 4);
    out_step = spv_pin_u(a, 5);
    swap     = spv_pin_u(a, 6);

    index = spv_span_open(a, &loop, spv_umul(a, rows, cols));
    {
        uint32_t row = spv_udiv(a, index, cols);
        uint32_t col = spv_umod(a, index, cols);
        uint32_t acc = spv_var_f(a);
        uint32_t from = spv_umul(a, row, a_step);
        uint32_t mid;
        spv_save(a, acc, spv_f(a, 0.0f));
        mid = spv_step_open(a, &mid_loop, mids);
        {
            uint32_t straight = spv_uadd(a, spv_umul(a, mid, b_step), col);
            uint32_t swapped  = spv_uadd(a, spv_umul(a, col, b_step), mid);
            uint32_t at = spv_pick(a, a->t_u32, spv_ueq(a, swap, spv_u(a, 1)),
                                   swapped, straight);
            spv_save(a, acc, spv_fadd(a, spv_load_f(a, acc),
                     spv_fmul(a, spv_get(a, 0, spv_uadd(a, from, mid)),
                                 spv_get(a, 1, at))));
        }
        spv_for_shut(a, &mid_loop);
        spv_set(a, 2, spv_uadd(a, spv_umul(a, row, out_step), col), spv_load_f(a, acc));
    }
    spv_for_shut(a, &loop);
}

/* ============================================================================
 * part 5 -- pipelines
 *
 * A kernel becomes a pipeline once, on the first call that needs it, and the
 * pipeline is kept for the life of the device.  Building all fourteen up front
 * would cost a tenth of a second on a driver that compiles SPIR-V to machine
 * code eagerly, and a run that only ever calls `dense` and `attend` has no use
 * for the picture engine's eight.
 *
 * Every kernel has the same descriptor layout shape -- N storage buffers at
 * set zero, bindings zero upward -- and the same push range, sixty-four bytes
 * at offset zero, so the only thing that differs between two pipelines is the
 * module and the number of bindings.  One descriptor set is allocated per
 * pipeline and rewritten before each dispatch, which is safe because every
 * operation waits on its fence before returning and nothing else is in flight.
 * ==========================================================================*/

typedef enum VulkKernel {
    VULK_K_DENSE = 0,
    VULK_K_RMSNORM,
    VULK_K_SWIGLU,
    VULK_K_ROPE,
    VULK_K_ATTEND,
    VULK_K_CONV1D,
    VULK_K_YCONV,
    VULK_K_YDECONV,
    VULK_K_YPOOL,
    VULK_K_YRESIZE,
    VULK_K_YACT,
    VULK_K_YADD,
    VULK_K_YSOFTMAX,
    VULK_K_YGEMM,
    VULK_K_COUNT
} VulkKernel;

typedef struct VulkPipe {
    VulkObject set_layout;
    VulkObject layout;
    VulkObject module;
    VulkObject pipeline;
    VulkObject pool;
    VulkObject set;
    int32_t    binds;
    int        ready;
} VulkPipe;

typedef struct VulkPlan {
    const char *name;
    int32_t     binds;
    void      (*build)(VulkAsm *);
} VulkPlan;

static const VulkPlan vulk_plan_list[VULK_K_COUNT] = {
    { "dense",   3, vulk_kernel_dense },
    { "rmsnorm", 3, vulk_kernel_rmsnorm },
    { "swiglu",  3, vulk_kernel_swiglu },
    { "rope",    2, vulk_kernel_rope },
    { "attend",  4, vulk_kernel_attend },
    { "conv1d",  5, vulk_kernel_conv1d },
    { "yconv",   4, vulk_kernel_yconv },
    { "ydeconv", 4, vulk_kernel_ydeconv },
    { "ypool",   2, vulk_kernel_ypool },
    { "yresize", 2, vulk_kernel_yresize },
    { "yact",    1, vulk_kernel_yact },
    { "yadd",    2, vulk_kernel_yadd },
    { "ysoftmax",1, vulk_kernel_ysoftmax },
    { "ygemm",   3, vulk_kernel_ygemm }
};

static VulkPipe vulk_pipe_list[VULK_K_COUNT];

static void vulk_pipe_drop(VulkDev *dev, VulkPipe *pipe)
{
    if (!pipe->ready) return;
    if (pipe->pipeline)  dev->api.pipeline_drop(dev->device, pipe->pipeline, NULL);
    if (pipe->pool)      dev->api.pool_drop(dev->device, pipe->pool, NULL);
    if (pipe->layout)    dev->api.layout_drop(dev->device, pipe->layout, NULL);
    if (pipe->set_layout)dev->api.set_layout_drop(dev->device, pipe->set_layout, NULL);
    if (pipe->module)    dev->api.module_drop(dev->device, pipe->module, NULL);
    memset(pipe, 0, sizeof *pipe);
}

static int vulk_pipe_make(VulkDev *dev, VulkKernel which)
{
    VulkPipe         *pipe = &vulk_pipe_list[which];
    const VulkPlan   *plan = &vulk_plan_list[which];
    VulkAsm           build;
    uint32_t         *words;
    size_t            bytes = 0;
    VulkBindingInfo   binding[VULK_BIND_MAX];
    VulkSetLayoutInfo set_want;
    VulkPushRange     push;
    VulkLayoutInfo    layout_want;
    VulkModuleInfo    module_want;
    VulkComputeInfo   pipe_want;
    VulkPoolSize      pool_size;
    VulkPoolInfo      pool_want;
    VulkSetAllocInfo  set_alloc;
    int32_t           index;

    if (pipe->ready) return 1;
    pipe->binds = plan->binds;

    plan->build(&build);
    words = spv_shut(&build, &bytes);
    spv_free(&build);
    if (!words) return 0;

    memset(&module_want, 0, sizeof module_want);
    module_want.type  = VULK_S_SHADER_MODULE;
    module_want.bytes = bytes;
    module_want.words = words;
    if (dev->api.module_make(dev->device, &module_want, NULL, &pipe->module) != VULK_SUCCESS) {
        free(words);
        vulk_say(1, "%s: the driver rejected the assembled module", plan->name);
        return 0;
    }
    free(words);

    memset(binding, 0, sizeof binding);
    for (index = 0; index < plan->binds; ++index) {
        binding[index].binding = (uint32_t)index;
        binding[index].kind    = VULK_DESC_STORAGE_BUFFER;
        binding[index].count   = 1;
        binding[index].stages  = VULK_STAGE_BIT_COMPUTE;
    }
    memset(&set_want, 0, sizeof set_want);
    set_want.type          = VULK_S_DESCRIPTOR_SET_LAYOUT;
    set_want.binding_count = (uint32_t)plan->binds;
    set_want.binding_list  = binding;
    if (dev->api.set_layout_make(dev->device, &set_want, NULL, &pipe->set_layout) != VULK_SUCCESS)
        { vulk_pipe_drop(dev, pipe); return 0; }

    push.stages = VULK_STAGE_BIT_COMPUTE;
    push.offset = 0;
    push.size   = VULK_PUSH_WORDS * 4;
    memset(&layout_want, 0, sizeof layout_want);
    layout_want.type       = VULK_S_PIPELINE_LAYOUT;
    layout_want.set_count  = 1;
    layout_want.set_list   = &pipe->set_layout;
    layout_want.push_count = 1;
    layout_want.push_list  = &push;
    if (dev->api.layout_make(dev->device, &layout_want, NULL, &pipe->layout) != VULK_SUCCESS)
        { vulk_pipe_drop(dev, pipe); return 0; }

    memset(&pipe_want, 0, sizeof pipe_want);
    pipe_want.type         = VULK_S_COMPUTE_PIPELINE;
    pipe_want.stage.type   = VULK_S_PIPELINE_SHADER_STAGE;
    pipe_want.stage.stage  = VULK_STAGE_BIT_COMPUTE;
    pipe_want.stage.module = pipe->module;
    pipe_want.stage.entry  = "main";
    pipe_want.layout       = pipe->layout;
    pipe_want.parent_index = -1;
    if (dev->api.compute_make(dev->device, 0, 1, &pipe_want, NULL,
                              &pipe->pipeline) != VULK_SUCCESS) {
        vulk_say(1, "%s: the driver would not build a pipeline", plan->name);
        vulk_pipe_drop(dev, pipe);
        return 0;
    }

    pool_size.kind  = VULK_DESC_STORAGE_BUFFER;
    pool_size.count = (uint32_t)plan->binds;
    memset(&pool_want, 0, sizeof pool_want);
    pool_want.type       = VULK_S_DESCRIPTOR_POOL;
    pool_want.max_sets   = 1;
    pool_want.size_count = 1;
    pool_want.size_list  = &pool_size;
    if (dev->api.pool_make(dev->device, &pool_want, NULL, &pipe->pool) != VULK_SUCCESS)
        { vulk_pipe_drop(dev, pipe); return 0; }

    memset(&set_alloc, 0, sizeof set_alloc);
    set_alloc.type        = VULK_S_DESCRIPTOR_SET_ALLOC;
    set_alloc.pool        = pipe->pool;
    set_alloc.count       = 1;
    set_alloc.layout_list = &pipe->set_layout;
    if (dev->api.set_alloc(dev->device, &set_alloc, &pipe->set) != VULK_SUCCESS)
        { vulk_pipe_drop(dev, pipe); return 0; }

    pipe->ready = 1;
    vulk_say(3, "built %s", plan->name);
    return 1;
}

typedef struct VulkCall {
    const VulkSlab *bind_list[VULK_BIND_MAX];
    uint32_t        push_list[VULK_PUSH_WORDS];
    uint32_t        groups_x;
    uint32_t        groups_y;
} VulkCall;

static void vulk_call_open(VulkCall *call)
{
    memset(call, 0, sizeof *call);
    call->groups_x = 1;
    call->groups_y = 1;
}

static inline void vulk_push_u(VulkCall *call, int32_t slot, uint32_t value)
{ if (slot >= 0 && slot < VULK_PUSH_WORDS) call->push_list[slot] = value; }

static inline void vulk_push_f(VulkCall *call, int32_t slot, float value)
{ uint32_t bits; memcpy(&bits, &value, sizeof bits); vulk_push_u(call, slot, bits); }

/* How many workgroups cover `total` cells, capped at what the device will take
 * in one dimension.  The cap is not a truncation: every kernel here is a grid
 * stride loop, so a dispatch smaller than the work simply goes round more
 * times. */
static uint32_t vulk_grid(const VulkDev *dev, size_t total)
{
    size_t want = (total + VULK_LOCAL - 1) / VULK_LOCAL;
    if (want < 1) want = 1;
    if (want > dev->group_limit) want = dev->group_limit;
    return (uint32_t)want;
}

/* Records one dispatch into the open command buffer. */
static int vulk_run(VulkDev *dev, VulkKernel which, const VulkCall *call)
{
    VulkPipe      *pipe = &vulk_pipe_list[which];
    VulkWriteSet   write[VULK_BIND_MAX];
    VulkBufferBind bound[VULK_BIND_MAX];
    int32_t        index;

    if (!vulk_pipe_make(dev, which)) return 0;
    memset(write, 0, sizeof write);
    for (index = 0; index < pipe->binds; ++index) {
        const VulkSlab *slab = call->bind_list[index];
        if (!slab || !slab->buffer) return 0;
        bound[index].buffer = slab->buffer;
        bound[index].offset = 0;
        bound[index].range  = VULK_WHOLE_SIZE;
        write[index].type    = VULK_S_WRITE_DESCRIPTOR_SET;
        write[index].set     = pipe->set;
        write[index].binding = (uint32_t)index;
        write[index].count   = 1;
        write[index].kind    = VULK_DESC_STORAGE_BUFFER;
        write[index].buffers = &bound[index];
    }
    dev->api.set_write(dev->device, (uint32_t)pipe->binds, write, 0, NULL);
    dev->api.cmd_bind_pipeline(dev->cmd, VULK_PIPELINE_COMPUTE, pipe->pipeline);
    dev->api.cmd_bind_sets(dev->cmd, VULK_PIPELINE_COMPUTE, pipe->layout, 0, 1,
                           &pipe->set, 0, NULL);
    dev->api.cmd_push(dev->cmd, pipe->layout, VULK_STAGE_BIT_COMPUTE, 0,
                      VULK_PUSH_WORDS * 4, call->push_list);
    dev->api.cmd_dispatch(dev->cmd, call->groups_x, call->groups_y, 1);
    return 1;
}

/* ============================================================================
 * part 6 -- residence
 *
 * Weights do not change after a model is loaded, so the copy on the device can
 * outlive the call that put it there.  Everything else -- activations, the key
 * and value cache, a feature map -- is written by the host between calls and
 * has to travel every time, because neither seam says when.
 *
 * The cache is keyed on the host pointer **and** the byte count, and that pair
 * is the whole of the correctness argument: the same address with a different
 * length is a different tensor and misses.  What it cannot catch is a caller
 * that writes through a pointer it has already handed over as a weight, which
 * is why this is only ever used for a plane the model owns and never for
 * anything a session touches.  `ill_model_load` is done writing before the
 * first forward pass, and yolo's graph is immutable from the moment it is
 * read; a caller that broke either would see stale numbers rather than a
 * crash, so the rule is stated here rather than enforced.
 *
 * Eviction is least recently used against a budget, default three quarters of
 * a gibibyte, which is chosen to be smaller than the memory on any device
 * worth using and large enough for a 2.6B model's attention and feed forward
 * planes at f32.  A miss costs an upload, not an error.
 * ==========================================================================*/

#define VULK_BUDGET_DEFAULT ((size_t)768 << 20)

/* A storage binding may not span more than `maxStorageBufferRange` bytes, and
 * the floor Vulkan guarantees for that is 128 MiB.  Real desktop drivers
 * report two to four gibibytes and SwiftShader one, so nothing measured here
 * comes near it -- the vocabulary plane of the 2.6B checkpoint at f32 is
 * 0.977 GiB, which fits under SwiftShader's limit by twenty-three thousandths.
 * A driver that reports the floor would not, and reading past the limit is
 * undefined rather than an error, so it is checked here and refused by name.
 * Splitting a plane into bands and dispatching one a band is the fix and is
 * TODO item 54; the refusal is honest in the meantime, which silently wrong
 * numbers would not be. */
static int vulk_fits(VulkDev *dev, size_t bytes, const char *what)
{
    if (dev->storage_limit == 0 || bytes <= (size_t)dev->storage_limit) return 1;
    vulk_say(1, "%s wants %.1f MiB in one binding and this device allows %.1f",
             what, (double)bytes / 1048576.0, (double)dev->storage_limit / 1048576.0);
    return 0;
}

static void vulk_hold_drop(VulkDev *dev, int32_t slot)
{
    VulkHold *hold = &dev->hold_list[slot];
    if (dev->held >= hold->bytes) dev->held -= hold->bytes;
    vulk_slab_drop(dev, &hold->slab);
    dev->hold_list[slot] = dev->hold_list[--dev->hold_used];
}

static void vulk_hold_clear(VulkDev *dev)
{
    while (dev->hold_used > 0) vulk_hold_drop(dev, dev->hold_used - 1);
    dev->held = 0;
}

/* The pin is the fix for a bug that is easy to write and hard to see: an
 * operation wanting two weights -- the short convolution's taps and its bias,
 * a convolution's sheet and its bias -- looks both up before it dispatches,
 * and the second lookup can be the miss that evicts the first.  The caller is
 * then holding a pointer to a slab whose buffer has been destroyed and whose
 * slot has been overwritten by whatever entry was last in the list.  A pinned
 * entry is not a candidate for eviction, and `vulk_op` releases every pin once
 * the dispatch it was setting up has completed.
 *
 * An operation that fails between its first lookup and its dispatch leaves its
 * pins set until the next one completes.  That costs a few entries their turn
 * at eviction, for one operation, on a path that is already reporting a
 * failure, and is not worth a second mechanism to avoid. */
static void vulk_hold_free(VulkDev *dev)
{
    int32_t index;
    for (index = 0; index < dev->hold_used; ++index) dev->hold_list[index].pin = 0;
}

/* The oldest entry that may be evicted, or -1 when every one is in use. */
static int32_t vulk_hold_stale(VulkDev *dev)
{
    int32_t index, old = -1;
    for (index = 0; index < dev->hold_used; ++index) {
        if (dev->hold_list[index].pin) continue;
        if (old < 0 || dev->hold_list[index].seen < dev->hold_list[old].seen) old = index;
    }
    return old;
}

static void vulk_hold_trim(VulkDev *dev, size_t room)
{
    while (dev->hold_used > 0 && dev->held + room > dev->budget) {
        int32_t old = vulk_hold_stale(dev);
        if (old < 0) break;
        vulk_hold_drop(dev, old);
    }
}

/* Finds the device copy of `bytes` bytes at `key`, uploading it through
 * `fill` if it is not there.  `fill` writes the f32 form into a buffer the
 * caller sizes -- it is where a q4 plane is widened -- and is not called at
 * all on a hit, which is what makes the widening a load time cost rather than
 * a per token one. */
static VulkSlab *vulk_hold_find(VulkDev *dev, const void *key, size_t bytes,
                                void (*fill)(void *room, const void *from), const void *from)
{
    int32_t   index;
    VulkHold *hold;
    void     *room = NULL;
    VulkSlab  stage;

    for (index = 0; index < dev->hold_used; ++index) {
        if (dev->hold_list[index].key != key) continue;
        if (dev->hold_list[index].bytes != bytes) continue;
        dev->hold_list[index].seen = ++dev->hold_tick;
        dev->hold_list[index].pin  = 1;
        return &dev->hold_list[index].slab;
    }
    if (!vulk_fits(dev, bytes, "a weight")) return NULL;
    if (dev->hold_used >= VULK_HOLD_MAX) {
        int32_t old = vulk_hold_stale(dev);
        if (old < 0) return NULL;
        vulk_hold_drop(dev, old);
    }
    vulk_hold_trim(dev, bytes);

    hold = &dev->hold_list[dev->hold_used];
    memset(hold, 0, sizeof *hold);
    if (!vulk_slab_make(dev, &hold->slab, bytes, 0)) return NULL;

    /* the widening buffer: the mapping itself when the device is unified, and
     * a heap block otherwise, because a staging slab has to be filled before
     * the copy is recorded */
    if (hold->slab.mapped) {
        room = hold->slab.mapped;
        fill(room, from);
        memset(&stage, 0, sizeof stage);
    } else {
        room = malloc(bytes);
        if (!room) { vulk_slab_drop(dev, &hold->slab); return NULL; }
        fill(room, from);
        if (!vulk_send_open(dev)) { free(room); vulk_slab_drop(dev, &hold->slab); return NULL; }
        if (!vulk_write(dev, &hold->slab, room, bytes, &stage)) {
            free(room); vulk_slab_drop(dev, &hold->slab); return NULL;
        }
        if (!vulk_send_wait(dev)) {
            free(room); vulk_slab_drop(dev, &stage); vulk_slab_drop(dev, &hold->slab);
            return NULL;
        }
        free(room);
        vulk_slab_drop(dev, &stage);
    }

    hold->key   = key;
    hold->bytes = bytes;
    hold->seen  = ++dev->hold_tick;
    hold->pin   = 1;
    dev->held  += bytes;
    dev->hold_used++;
    return &hold->slab;
}

/* The plain case: host memory that is already f32 and needs no widening. */
static void vulk_fill_copy(void *room, const void *from)
{
    const struct { const void *src; size_t bytes; } *say = from;
    memcpy(room, say->src, say->bytes);
}

/* ============================================================================
 * part 7 -- opening and closing the device
 *
 * Instance, device, queue, command pool, command buffer, fence.  No layers and
 * no extensions are asked for: validation is a developer's tool that belongs
 * in the environment (`VK_INSTANCE_LAYERS`) rather than in a shipped
 * engine, and nothing here needs anything past Vulkan 1.0.
 *
 * The device is chosen by preference -- a discrete card first, then an
 * integrated one, then a software one -- unless the caller names a substring
 * of a device name, in which case the first match wins.  The preference is
 * wrong exactly once, on a laptop where the integrated part shares the memory
 * the model already sits in, and naming the device is how that is said.
 * ==========================================================================*/

#define VULK_LOOK(field, name) \
    do { \
        *(void **)&dev->api.field = look(owner, name); \
        if (!dev->api.field) { vulk_say(1, "missing entry point %s", name); return 0; } \
    } while (0)

static int vulk_api_instance(VulkDev *dev, VulkGetProc look, VulkHandle owner)
{
    VULK_LOOK(phys_list,    "vkEnumeratePhysicalDevices");
    VULK_LOOK(phys_props,   "vkGetPhysicalDeviceProperties");
    VULK_LOOK(phys_queues,  "vkGetPhysicalDeviceQueueFamilyProperties");
    VULK_LOOK(phys_memory,  "vkGetPhysicalDeviceMemoryProperties");
    VULK_LOOK(device_make,  "vkCreateDevice");
    VULK_LOOK(device_proc,  "vkGetDeviceProcAddr");
    VULK_LOOK(instance_drop,"vkDestroyInstance");
    return 1;
}

static int vulk_api_device(VulkDev *dev, VulkGetProc look, VulkHandle owner)
{
    VULK_LOOK(device_drop,     "vkDestroyDevice");
    VULK_LOOK(device_idle,     "vkDeviceWaitIdle");
    VULK_LOOK(queue_get,       "vkGetDeviceQueue");
    VULK_LOOK(queue_submit,    "vkQueueSubmit");
    VULK_LOOK(buffer_make,     "vkCreateBuffer");
    VULK_LOOK(buffer_drop,     "vkDestroyBuffer");
    VULK_LOOK(buffer_needs,    "vkGetBufferMemoryRequirements");
    VULK_LOOK(memory_make,     "vkAllocateMemory");
    VULK_LOOK(memory_drop,     "vkFreeMemory");
    VULK_LOOK(buffer_bind,     "vkBindBufferMemory");
    VULK_LOOK(memory_map,      "vkMapMemory");
    VULK_LOOK(memory_unmap,    "vkUnmapMemory");
    VULK_LOOK(set_layout_make, "vkCreateDescriptorSetLayout");
    VULK_LOOK(set_layout_drop, "vkDestroyDescriptorSetLayout");
    VULK_LOOK(pool_make,       "vkCreateDescriptorPool");
    VULK_LOOK(pool_drop,       "vkDestroyDescriptorPool");
    VULK_LOOK(set_alloc,       "vkAllocateDescriptorSets");
    VULK_LOOK(set_write,       "vkUpdateDescriptorSets");
    VULK_LOOK(layout_make,     "vkCreatePipelineLayout");
    VULK_LOOK(layout_drop,     "vkDestroyPipelineLayout");
    VULK_LOOK(module_make,     "vkCreateShaderModule");
    VULK_LOOK(module_drop,     "vkDestroyShaderModule");
    VULK_LOOK(compute_make,    "vkCreateComputePipelines");
    VULK_LOOK(pipeline_drop,   "vkDestroyPipeline");
    VULK_LOOK(cmd_pool_make,   "vkCreateCommandPool");
    VULK_LOOK(cmd_pool_drop,   "vkDestroyCommandPool");
    VULK_LOOK(cmd_alloc,       "vkAllocateCommandBuffers");
    VULK_LOOK(cmd_begin,       "vkBeginCommandBuffer");
    VULK_LOOK(cmd_end,         "vkEndCommandBuffer");
    VULK_LOOK(cmd_reset,       "vkResetCommandBuffer");
    VULK_LOOK(cmd_bind_pipeline,"vkCmdBindPipeline");
    VULK_LOOK(cmd_bind_sets,   "vkCmdBindDescriptorSets");
    VULK_LOOK(cmd_push,        "vkCmdPushConstants");
    VULK_LOOK(cmd_dispatch,    "vkCmdDispatch");
    VULK_LOOK(cmd_barrier,     "vkCmdPipelineBarrier");
    VULK_LOOK(cmd_copy,        "vkCmdCopyBuffer");
    VULK_LOOK(fence_make,      "vkCreateFence");
    VULK_LOOK(fence_drop,      "vkDestroyFence");
    VULK_LOOK(fence_reset,     "vkResetFences");
    VULK_LOOK(fence_wait,      "vkWaitForFences");
    return 1;
}

#undef VULK_LOOK

/* A device's standing, for the preference order: higher is better. */
static int vulk_device_rank(uint32_t kind)
{
    switch (kind) {
        case 2:  return 4;   /* discrete                                     */
        case 1:  return 3;   /* integrated                                   */
        case 3:  return 2;   /* virtual                                      */
        case 4:  return 1;   /* cpu                                          */
        default: return 0;
    }
}

static int vulk_family_find(VulkDev *dev, VulkHandle phys, uint32_t *family_out)
{
    VulkQueueProps list[16];
    uint32_t       count = 16, index;
    memset(list, 0, sizeof list);
    dev->api.phys_queues(phys, &count, list);
    if (count > 16) count = 16;
    /* a compute-only family first: on a card that has one it is the queue the
     * driver does not also schedule the desktop on */
    for (index = 0; index < count; ++index)
        if ((list[index].flags & VULK_QUEUE_COMPUTE) &&
            !(list[index].flags & VULK_QUEUE_GRAPHICS) && list[index].count > 0) {
            *family_out = index; return 1;
        }
    for (index = 0; index < count; ++index)
        if ((list[index].flags & VULK_QUEUE_COMPUTE) && list[index].count > 0) {
            *family_out = index; return 1;
        }
    return 0;
}

void vulk_memory_budget_set(size_t bytes)
{
    vulk_dev.budget = bytes ? bytes : VULK_BUDGET_DEFAULT;
    if (vulk_dev.open_flag) vulk_hold_trim(&vulk_dev, 0);
}

size_t vulk_memory_used(void) { return vulk_dev.used; }

void vulk_report(uint64_t *sent_out, uint64_t *moved_out)
{
    if (sent_out)  *sent_out  = vulk_dev.sent;
    if (moved_out) *moved_out = vulk_dev.moved;
}
int    vulk_ready(void)       { return vulk_dev.open_flag; }

const char *vulk_device_name(void)
{
    return vulk_dev.open_flag ? vulk_dev.name : "none";
}

VulkStatus vulk_open(const char *want)
{
    VulkDev         *dev = &vulk_dev;
    VulkGetProc      look;
    VulkAppInfo      app;
    VulkInstanceInfo instance_want;
    VulkHandle       phys_list[16];
    uint32_t         phys_count = 16, index;
    int32_t          best = -1, best_rank = -1;
    float            priority = 1.0f;
    VulkQueueInfo    queue_want;
    VulkDeviceInfo   device_want;
    VulkCmdPoolInfo  pool_want;
    VulkCmdAllocInfo cmd_want;
    VulkFenceInfo    fence_want;

    if (dev->open_flag) return VULK_OK;
    if (dev->state != VULK_OK) return dev->state;   /* a failure is sticky */
    if (!dev->budget) dev->budget = VULK_BUDGET_DEFAULT;

    dev->lib = vulk_lib_open();
    if (!dev->lib) { dev->state = VULK_NO_LOADER; return dev->state; }
    look = (VulkGetProc)vulk_lib_find(dev->lib, "vkGetInstanceProcAddr");
    if (!look) { vulk_close(); dev->state = VULK_NO_LOADER; return dev->state; }
    *(void **)&dev->api.instance_make = look(NULL, "vkCreateInstance");
    if (!dev->api.instance_make) { vulk_close(); dev->state = VULK_NO_LOADER; return dev->state; }

    memset(&app, 0, sizeof app);
    app.type        = VULK_S_APPLICATION;
    app.app_name    = "inferliq";
    app.engine_name = "inferliq";
    app.api_version = (1u << 22);            /* 1.0.0                       */
    memset(&instance_want, 0, sizeof instance_want);
    instance_want.type = VULK_S_INSTANCE;
    instance_want.app  = &app;
    if (dev->api.instance_make(&instance_want, NULL, &dev->instance) != VULK_SUCCESS) {
        vulk_close(); dev->state = VULK_NO_DEVICE; return dev->state;
    }
    if (!vulk_api_instance(dev, look, dev->instance)) {
        vulk_close(); dev->state = VULK_NO_LOADER; return dev->state;
    }

    memset(phys_list, 0, sizeof phys_list);
    if (dev->api.phys_list(dev->instance, &phys_count, phys_list) < 0 || phys_count == 0) {
        vulk_close(); dev->state = VULK_NO_DEVICE; return dev->state;
    }
    if (phys_count > 16) phys_count = 16;
    for (index = 0; index < phys_count; ++index) {
        VulkPhysProps props;
        uint32_t      family;
        int           rank;
        memset(&props, 0, sizeof props);
        dev->api.phys_props(phys_list[index], &props);
        props.device_name[sizeof props.device_name - 1] = 0;
        if (!vulk_family_find(dev, phys_list[index], &family)) continue;
        rank = vulk_device_rank(props.device_kind);
        if (want && *want) {
            if (!strstr(props.device_name, want)) continue;
            rank = 1000;
        }
        if (rank > best_rank) {
            best_rank = rank;
            best      = (int32_t)index;
            dev->family = family;
            dev->kind   = props.device_kind;
            memcpy(dev->name, props.device_name, sizeof dev->name);
            dev->group_limit   = props.limits.max_group_count[0];
            dev->shared_limit  = props.limits.max_shared_bytes;
            dev->storage_limit = props.limits.max_storage_range;
        }
    }
    if (best < 0) { vulk_close(); dev->state = VULK_NO_DEVICE; return dev->state; }
    dev->phys = phys_list[best];
    if (dev->group_limit < 1) dev->group_limit = 65535;
    if (dev->group_limit > 65535) dev->group_limit = 65535;

    memset(&queue_want, 0, sizeof queue_want);
    queue_want.type          = VULK_S_DEVICE_QUEUE;
    queue_want.family        = dev->family;
    queue_want.count         = 1;
    queue_want.priority_list = &priority;
    memset(&device_want, 0, sizeof device_want);
    device_want.type        = VULK_S_DEVICE;
    device_want.queue_count = 1;
    device_want.queue_list  = &queue_want;
    if (dev->api.device_make(dev->phys, &device_want, NULL, &dev->device) != VULK_SUCCESS) {
        vulk_close(); dev->state = VULK_NO_DEVICE; return dev->state;
    }
    if (!vulk_api_device(dev, (VulkGetProc)dev->api.device_proc, dev->device)) {
        vulk_close(); dev->state = VULK_NO_LOADER; return dev->state;
    }
    dev->api.queue_get(dev->device, dev->family, 0, &dev->queue);
    if (!vulk_memory_pick(dev)) { vulk_close(); dev->state = VULK_NO_MEMORY; return dev->state; }

    memset(&pool_want, 0, sizeof pool_want);
    pool_want.type   = VULK_S_COMMAND_POOL;
    pool_want.flags  = 0x02;          /* buffers may be reset individually   */
    pool_want.family = dev->family;
    if (dev->api.cmd_pool_make(dev->device, &pool_want, NULL, &dev->cmd_pool) != VULK_SUCCESS) {
        vulk_close(); dev->state = VULK_NO_DEVICE; return dev->state;
    }
    memset(&cmd_want, 0, sizeof cmd_want);
    cmd_want.type  = VULK_S_COMMAND_BUFFER_ALLOC;
    cmd_want.pool  = dev->cmd_pool;
    cmd_want.level = 0;
    cmd_want.count = 1;
    if (dev->api.cmd_alloc(dev->device, &cmd_want, &dev->cmd) != VULK_SUCCESS) {
        vulk_close(); dev->state = VULK_NO_DEVICE; return dev->state;
    }
    memset(&fence_want, 0, sizeof fence_want);
    fence_want.type = VULK_S_FENCE;
    if (dev->api.fence_make(dev->device, &fence_want, NULL, &dev->fence) != VULK_SUCCESS) {
        vulk_close(); dev->state = VULK_NO_DEVICE; return dev->state;
    }

    dev->open_flag = 1;
    dev->state     = VULK_OK;
    vulk_say(2, "device %s, %s memory, %u workgroups a dispatch",
             dev->name, dev->unified_flag ? "unified" : "staged", dev->group_limit);
    return VULK_OK;
}

void vulk_close(void)
{
    VulkDev *dev = &vulk_dev;
    int32_t  index;
    if (dev->device) {
        vulk_say(2, "%lu submissions, %.1f MiB moved, %.1f MiB resident",
                 (unsigned long)dev->sent,
                 (double)dev->moved / 1048576.0, (double)dev->used / 1048576.0);
        if (dev->api.device_idle) dev->api.device_idle(dev->device);
        for (index = 0; index < VULK_K_COUNT; ++index) vulk_pipe_drop(dev, &vulk_pipe_list[index]);
        vulk_hold_clear(dev);
        vulk_scratch_clear(dev);
        if (dev->fence)    dev->api.fence_drop(dev->device, dev->fence, NULL);
        if (dev->cmd_pool) dev->api.cmd_pool_drop(dev->device, dev->cmd_pool, NULL);
        dev->api.device_drop(dev->device, NULL);
    }
    if (dev->instance && dev->api.instance_drop) dev->api.instance_drop(dev->instance, NULL);
    if (dev->lib) vulk_lib_close(dev->lib);
    {
        size_t budget = dev->budget;
        VulkStatus state = dev->state;
        memset(dev, 0, sizeof *dev);
        dev->budget = budget;
        dev->state  = state;
    }
    memset(vulk_pipe_list, 0, sizeof vulk_pipe_list);
}

/* ============================================================================
 * part 8 -- one operation, end to end
 *
 * Every entry in both tables has the same shape: decide where each binding's
 * bytes live, upload what the device has not got, dispatch, and bring the
 * answers back.  `vulk_op` is that shape written once.  A binding either names
 * a slab that is already resident -- a weight -- or names host memory that
 * travels, in one direction or both.
 * ==========================================================================*/

typedef struct VulkBind {
    const void *up;      /* host bytes to send, or null                     */
    void       *down;    /* where to put the answer, or null                */
    size_t      bytes;
    VulkSlab   *hold;    /* a resident slab, when this binding is a weight   */
} VulkBind;

static int vulk_op(VulkDev *dev, VulkKernel which, VulkCall *call,
                   VulkBind *bind_list, int32_t bind_count)
{
    VulkSlab own[VULK_BIND_MAX];
    VulkSlab up[VULK_BIND_MAX];
    VulkSlab down[VULK_BIND_MAX];
    int32_t  index;
    int      good = 1;

    memset(own, 0, sizeof own);
    memset(up, 0, sizeof up);
    memset(down, 0, sizeof down);

    for (index = 0; index < bind_count; ++index) {
        if (bind_list[index].hold) { call->bind_list[index] = bind_list[index].hold; continue; }
        if (!vulk_fits(dev, bind_list[index].bytes, "an operand")) { good = 0; break; }
        if (!vulk_scratch_take(dev, &own[index], bind_list[index].bytes)) { good = 0; break; }
        call->bind_list[index] = &own[index];
    }
    if (good) good = vulk_send_open(dev);
    if (good) {
        for (index = 0; index < bind_count; ++index) {
            if (!bind_list[index].up || bind_list[index].hold) continue;
            if (!vulk_write(dev, &own[index], bind_list[index].up,
                            bind_list[index].bytes, &up[index])) { good = 0; break; }
        }
    }
    if (good) { vulk_send_fence(dev); good = vulk_run(dev, which, call); }
    if (good) {
        vulk_send_fence(dev);
        for (index = 0; index < bind_count; ++index) {
            if (!bind_list[index].down || bind_list[index].hold) continue;
            if (!vulk_read_stage(dev, &own[index], bind_list[index].bytes, &down[index]))
                { good = 0; break; }
        }
    }
    if (good) good = vulk_send_wait(dev);
    if (good) {
        for (index = 0; index < bind_count; ++index) {
            if (!bind_list[index].down || bind_list[index].hold) continue;
            vulk_read_take(dev, &own[index], &down[index],
                           bind_list[index].down, bind_list[index].bytes);
        }
    }
    for (index = 0; index < bind_count; ++index) {
        vulk_slab_drop(dev, &up[index]);
        vulk_slab_drop(dev, &down[index]);
        vulk_scratch_give(dev, &own[index]);
    }
    vulk_hold_free(dev);
    return good;
}

/* Host memory that is already f32 and only needs a copy into the device's. */
typedef struct VulkSpan { const void *src; size_t bytes; } VulkSpan;

static VulkSlab *vulk_hold_span(VulkDev *dev, const void *key, size_t bytes)
{
    VulkSpan span;
    span.src   = key;
    span.bytes = bytes;
    return vulk_hold_find(dev, key, bytes, vulk_fill_copy, &span);
}

#ifndef VULK_NO_TEXT

/* ============================================================================
 * part 9 -- the IllBackend table
 *
 * Six operations.  Each one is a shape decision and then a call to `vulk_op`;
 * the arithmetic is all in part 4.
 *
 * Two things are worth saying about what the table does *not* do.
 *
 * `width` answers one rather than the device's real parallelism, because the
 * engine multiplies it by the window to size `board`, the per-worker score
 * scratch -- and this backend's attention never reads `board`, so a truthful
 * answer would allocate a megabyte of host memory that nothing touches.  It is
 * also what the command line prints as the thread count, and a device backend
 * genuinely has no host threads.
 *
 * `dense` ignores `pad`.  Quantised planes are widened on the way to the
 * device, so there is no quantised activation to pack, and the seam's
 * activation scratch goes unused.  Both are honest consequences of the
 * widening decision rather than gaps.
 * ==========================================================================*/

typedef struct VulkText { int32_t asked; } VulkText;
static VulkText vulk_text;

static IllResult vulk_ill_setup(IllBackend *self, int32_t threads)
{
    VulkStatus code;
    (void)threads;
    code = vulk_open(getenv("ILL_VULKAN_DEVICE"));
    if (code != VULK_OK) {
        ill_note(1, "vulkan: %s", vulk_status_text(code));
        return ILL_STATE;
    }
    self->inner = &vulk_text;
    ill_note(2, "vulkan: %s", vulk_device_name());
    return ILL_OK;
}

/* Releasing a model gives back what was cached for it, and leaves the device
 * standing: it is process-wide and the picture engine may still be on it, so
 * only `vulk_close` takes it down.  The cache is not per model, so this also
 * drops whatever the picture engine had cached -- that costs it an upload and
 * nothing else, which is the right trade against two caches and a way to tell
 * whose plane is whose. */
static void vulk_ill_close(IllBackend *self)
{
    self->inner = NULL;
    vulk_hold_clear(&vulk_dev);
    vulk_scratch_clear(&vulk_dev);
}

static int32_t vulk_ill_width(const IllBackend *self) { (void)self; return 1; }

/* A weight plane, widened to f32 on the way across.  `ill_plane_row` is the
 * engine's own widening, so q8 and q4 reach the device with exactly the values
 * the CPU dot product would have reconstructed from them. */
static void vulk_fill_plane(void *room, const void *from)
{
    const IllPlane *plane = *(const IllPlane *const *)from;
    int32_t         row;
    for (row = 0; row < plane->rows; ++row)
        ill_plane_row(plane, row, (float *)room + (size_t)row * plane->cols);
}

static void vulk_ill_dense(IllBackend *self, const IllPlane *plane, const float *src,
                           float *dst, int32_t tokens, IllPad *pad)
{
    VulkDev  *dev = &vulk_dev;
    VulkSlab *sheet;
    VulkCall  call;
    VulkBind  bind[3];
    size_t    cells = (size_t)plane->rows * (size_t)plane->cols;
    (void)self; (void)pad;

    sheet = vulk_hold_find(dev, plane->cells, cells * sizeof(float),
                           vulk_fill_plane, &plane);
    if (!sheet) {
        ill_note(1, "vulkan: no room for a %dx%d plane", plane->rows, plane->cols);
        return;
    }

    memset(bind, 0, sizeof bind);
    bind[0].hold  = sheet;
    bind[1].up    = src;
    bind[1].bytes = (size_t)tokens * (size_t)plane->cols * sizeof(float);
    bind[2].down  = dst;
    bind[2].bytes = (size_t)tokens * (size_t)plane->rows * sizeof(float);

    vulk_call_open(&call);
    vulk_push_u(&call, 0, (uint32_t)plane->rows);
    vulk_push_u(&call, 1, (uint32_t)plane->cols);
    vulk_push_u(&call, 2, (uint32_t)tokens);
    call.groups_x = (uint32_t)plane->rows;
    if (call.groups_x > dev->group_limit) call.groups_x = dev->group_limit;
    call.groups_y = (uint32_t)tokens;
    if (call.groups_y > dev->group_limit) call.groups_y = dev->group_limit;
    if (!vulk_op(dev, VULK_K_DENSE, &call, bind, 3))
        ill_note(1, "vulkan: dense did not complete");
}

static void vulk_ill_rmsnorm(IllBackend *self, const float *src, const float *gain,
                             float *dst, int32_t tokens, int32_t width, float eps)
{
    VulkDev  *dev = &vulk_dev;
    VulkCall  call;
    VulkBind  bind[3];
    VulkSlab *held = vulk_hold_span(dev, gain, (size_t)width * sizeof(float));
    size_t    cells = (size_t)tokens * (size_t)width * sizeof(float);
    (void)self;
    if (!held) return;

    memset(bind, 0, sizeof bind);
    bind[0].up = src;   bind[0].bytes = cells;
    bind[1].hold = held;
    bind[2].down = dst; bind[2].bytes = cells;

    vulk_call_open(&call);
    vulk_push_u(&call, 0, (uint32_t)tokens);
    vulk_push_u(&call, 1, (uint32_t)width);
    vulk_push_f(&call, 2, eps);
    call.groups_x = (uint32_t)tokens;
    if (call.groups_x > dev->group_limit) call.groups_x = dev->group_limit;
    if (!vulk_op(dev, VULK_K_RMSNORM, &call, bind, 3))
        ill_note(1, "vulkan: rmsnorm did not complete");
}

static void vulk_ill_swiglu(IllBackend *self, const float *gate, const float *lift,
                            float *dst, int32_t tokens, int32_t width)
{
    VulkDev *dev = &vulk_dev;
    VulkCall call;
    VulkBind bind[3];
    size_t   total = (size_t)tokens * (size_t)width;
    (void)self;

    memset(bind, 0, sizeof bind);
    bind[0].up = gate; bind[0].bytes = total * sizeof(float);
    bind[1].up = lift; bind[1].bytes = total * sizeof(float);
    bind[2].down = dst; bind[2].bytes = total * sizeof(float);

    vulk_call_open(&call);
    vulk_push_u(&call, 0, (uint32_t)total);
    call.groups_x = vulk_grid(dev, total);
    if (!vulk_op(dev, VULK_K_SWIGLU, &call, bind, 3))
        ill_note(1, "vulkan: swiglu did not complete");
}

static void vulk_ill_rope(IllBackend *self, float *cells, const float *table,
                          int32_t tokens, int32_t heads, int32_t width, int32_t stride)
{
    VulkDev *dev = &vulk_dev;
    VulkCall call;
    VulkBind bind[2];
    size_t   room  = (size_t)tokens * (size_t)stride * sizeof(float);
    size_t   total = (size_t)tokens * (size_t)heads * (size_t)(width / 2);
    (void)self;
    if (width < 2) return;

    memset(bind, 0, sizeof bind);
    bind[0].up = cells; bind[0].down = cells; bind[0].bytes = room;
    bind[1].up = table; bind[1].bytes = (size_t)tokens * (size_t)width * sizeof(float);

    vulk_call_open(&call);
    vulk_push_u(&call, 0, (uint32_t)tokens);
    vulk_push_u(&call, 1, (uint32_t)heads);
    vulk_push_u(&call, 2, (uint32_t)width);
    vulk_push_u(&call, 3, (uint32_t)stride);
    call.groups_x = vulk_grid(dev, total);
    if (!vulk_op(dev, VULK_K_ROPE, &call, bind, 2))
        ill_note(1, "vulkan: rope did not complete");
}

static void vulk_ill_attend(IllBackend *self, const IllHeedJob *job)
{
    VulkDev *dev = &vulk_dev;
    VulkCall call;
    VulkBind bind[4];
    size_t   rows  = (size_t)job->tokens * (size_t)job->heads * (size_t)job->width;
    size_t   cache = (size_t)job->groups * (size_t)job->span * (size_t)job->width;
    (void)self;

    memset(bind, 0, sizeof bind);
    bind[0].up = job->query; bind[0].bytes = rows * sizeof(float);
    bind[1].up = job->keys;  bind[1].bytes = cache * sizeof(float);
    bind[2].up = job->vals;  bind[2].bytes = cache * sizeof(float);
    bind[3].down = job->value; bind[3].bytes = rows * sizeof(float);

    vulk_call_open(&call);
    vulk_push_u(&call, 0, (uint32_t)job->tokens);
    vulk_push_u(&call, 1, (uint32_t)job->heads);
    vulk_push_u(&call, 2, (uint32_t)job->groups);
    vulk_push_u(&call, 3, (uint32_t)job->width);
    vulk_push_u(&call, 4, (uint32_t)job->span);
    vulk_push_u(&call, 5, (uint32_t)job->base);
    vulk_push_f(&call, 6, job->scale);
    call.groups_x = (uint32_t)(job->tokens * job->heads);
    if (call.groups_x > dev->group_limit) call.groups_x = dev->group_limit;
    if (!vulk_op(dev, VULK_K_ATTEND, &call, bind, 4))
        ill_note(1, "vulkan: attend did not complete");
}

static void vulk_ill_conv1d(IllBackend *self, const IllFlowJob *job)
{
    VulkDev  *dev = &vulk_dev;
    VulkCall  call;
    VulkBind  bind[5];
    VulkSlab *taps, *bias = NULL;
    float     spare = 0.0f;
    size_t    cells = (size_t)job->tokens * (size_t)job->dim * sizeof(float);
    size_t    keep  = (size_t)job->dim * (size_t)(job->width - 1) * sizeof(float);
    (void)self;

    if (job->width > 16) {
        ill_note(1, "vulkan: conv1d window %d is wider than 16", job->width);
        return;
    }
    taps = vulk_hold_span(dev, job->taps, (size_t)job->dim * (size_t)job->width * sizeof(float));
    if (!taps) return;
    if (job->bias) {
        bias = vulk_hold_span(dev, job->bias, (size_t)job->dim * sizeof(float));
        if (!bias) return;
    }

    memset(bind, 0, sizeof bind);
    bind[0].up = job->src;  bind[0].bytes = cells;
    bind[1].down = job->dst; bind[1].bytes = cells;
    bind[2].up = job->hist; bind[2].down = job->hist; bind[2].bytes = keep;
    bind[3].hold = taps;
    if (bias) bind[4].hold = bias;
    else { bind[4].up = &spare; bind[4].bytes = sizeof spare; }

    vulk_call_open(&call);
    vulk_push_u(&call, 0, (uint32_t)job->tokens);
    vulk_push_u(&call, 1, (uint32_t)job->dim);
    vulk_push_u(&call, 2, (uint32_t)job->width);
    vulk_push_u(&call, 3, job->bias ? 1u : 0u);
    call.groups_x = vulk_grid(dev, (size_t)job->dim);
    if (!vulk_op(dev, VULK_K_CONV1D, &call, bind, 5))
        ill_note(1, "vulkan: conv1d did not complete");
}

static IllBackend vulk_ill_table = {
    "vulkan",
    vulk_ill_setup, vulk_ill_close, vulk_ill_width,
    vulk_ill_dense, vulk_ill_rmsnorm, vulk_ill_swiglu,
    vulk_ill_rope, vulk_ill_attend, vulk_ill_conv1d,
    NULL
};

IllResult vulk_join(void) { return ill_backend_join(&vulk_ill_table); }

#endif /* VULK_NO_TEXT */

#ifndef VULK_NO_PICTURE

/* ============================================================================
 * part 10 -- the YoloBackend table
 *
 * Nine entries, eight of them shaders.  `conv_room` is the ninth and answers
 * zero: the CPU backend lowers a patch matrix and multiplies, and asks for
 * room to lower into, while this one convolves directly and has nothing to
 * lower.  The TODO entry that asked for this seam said that `conv_room` has to
 * answer for the backend that will run the convolution rather than for the CPU
 * one, and this is what that meant.
 *
 * A weight sheet is cached by its host pointer; a feature map is not, and that
 * is where the cost of this backend sits.  `yolo26n` over a 640 picture is
 * about two hundred operations, and every one of them sends its input and
 * fetches its output, so a plane in the neck travels four times -- out of the
 * convolution that made it, into the concatenation, out again, into the next
 * convolution.  The seam has no way to say "leave it there".  Until it does,
 * this is a correctness result and a way to find out what the arithmetic costs
 * on a device, not a faster detector.
 * ==========================================================================*/

static size_t yolo_plane_cells(const YoloPlane *plane)
{
    return (size_t)plane->batch_size * (size_t)plane->channel_count
         * (size_t)plane->height * (size_t)plane->width;
}

static size_t vulk_yolo_room(const YoloPlane *in, const YoloPlane *out,
                             int kernel_size, int group_count)
{
    (void)in; (void)out; (void)kernel_size; (void)group_count;
    return 0;
}

static void vulk_yolo_conv(const YoloBackend *backend,
                           const YoloPlane *in, YoloPlane *out,
                           const float *weight_sheet, const float *bias_list,
                           int kernel_size, int stride_size, int pad_size,
                           int dilation_size, int group_count,
                           YoloActKind act_kind, float *room_list)
{
    VulkDev  *dev = &vulk_dev;
    VulkCall  call;
    VulkBind  bind[4];
    VulkSlab *sheet, *bias = NULL;
    float     spare = 0.0f;
    size_t    taps = (size_t)out->channel_count
                   * (size_t)(in->channel_count / group_count)
                   * (size_t)kernel_size * (size_t)kernel_size;
    (void)backend; (void)room_list;

    sheet = vulk_hold_span(dev, weight_sheet, taps * sizeof(float));
    if (!sheet) return;
    if (bias_list) {
        bias = vulk_hold_span(dev, bias_list, (size_t)out->channel_count * sizeof(float));
        if (!bias) return;
    }

    memset(bind, 0, sizeof bind);
    bind[0].up = in->cell_list;  bind[0].bytes = yolo_plane_cells(in) * sizeof(float);
    bind[1].hold = sheet;
    if (bias) bind[2].hold = bias;
    else { bind[2].up = &spare; bind[2].bytes = sizeof spare; }
    bind[3].down = out->cell_list; bind[3].bytes = yolo_plane_cells(out) * sizeof(float);

    vulk_call_open(&call);
    vulk_push_u(&call, 0,  (uint32_t)in->channel_count);
    vulk_push_u(&call, 1,  (uint32_t)in->height);
    vulk_push_u(&call, 2,  (uint32_t)in->width);
    vulk_push_u(&call, 3,  (uint32_t)out->channel_count);
    vulk_push_u(&call, 4,  (uint32_t)out->height);
    vulk_push_u(&call, 5,  (uint32_t)out->width);
    vulk_push_u(&call, 6,  (uint32_t)kernel_size);
    vulk_push_u(&call, 7,  (uint32_t)stride_size);
    vulk_push_u(&call, 8,  (uint32_t)pad_size);
    vulk_push_u(&call, 9,  (uint32_t)dilation_size);
    vulk_push_u(&call, 10, (uint32_t)group_count);
    vulk_push_u(&call, 11, (uint32_t)act_kind);
    vulk_push_u(&call, 12, bias_list ? 1u : 0u);
    vulk_push_u(&call, 13, (uint32_t)out->batch_size);
    call.groups_x = vulk_grid(dev, yolo_plane_cells(out));
    (void)vulk_op(dev, VULK_K_YCONV, &call, bind, 4);
}

static void vulk_yolo_deconv(const YoloBackend *backend,
                             const YoloPlane *in, YoloPlane *out,
                             const float *weight_sheet, const float *bias_list,
                             int kernel_size, int stride_size, YoloActKind act_kind)
{
    VulkDev  *dev = &vulk_dev;
    VulkCall  call;
    VulkBind  bind[4];
    VulkSlab *sheet, *bias = NULL;
    float     spare = 0.0f;
    size_t    taps = (size_t)in->channel_count * (size_t)out->channel_count
                   * (size_t)kernel_size * (size_t)kernel_size;
    (void)backend;

    sheet = vulk_hold_span(dev, weight_sheet, taps * sizeof(float));
    if (!sheet) return;
    if (bias_list) {
        bias = vulk_hold_span(dev, bias_list, (size_t)out->channel_count * sizeof(float));
        if (!bias) return;
    }

    memset(bind, 0, sizeof bind);
    bind[0].up = in->cell_list;  bind[0].bytes = yolo_plane_cells(in) * sizeof(float);
    bind[1].hold = sheet;
    if (bias) bind[2].hold = bias;
    else { bind[2].up = &spare; bind[2].bytes = sizeof spare; }
    bind[3].down = out->cell_list; bind[3].bytes = yolo_plane_cells(out) * sizeof(float);

    vulk_call_open(&call);
    vulk_push_u(&call, 0,  (uint32_t)in->channel_count);
    vulk_push_u(&call, 1,  (uint32_t)in->height);
    vulk_push_u(&call, 2,  (uint32_t)in->width);
    vulk_push_u(&call, 3,  (uint32_t)out->channel_count);
    vulk_push_u(&call, 4,  (uint32_t)out->height);
    vulk_push_u(&call, 5,  (uint32_t)out->width);
    vulk_push_u(&call, 6,  (uint32_t)kernel_size);
    vulk_push_u(&call, 7,  (uint32_t)stride_size);
    vulk_push_u(&call, 8,  (uint32_t)act_kind);
    vulk_push_u(&call, 9,  bias_list ? 1u : 0u);
    vulk_push_u(&call, 10, (uint32_t)out->batch_size);
    call.groups_x = vulk_grid(dev, yolo_plane_cells(out));
    (void)vulk_op(dev, VULK_K_YDECONV, &call, bind, 4);
}

static void vulk_yolo_pool(const YoloBackend *backend,
                           const YoloPlane *in, YoloPlane *out,
                           int kernel_size, int stride_size, int pad_size)
{
    VulkDev *dev = &vulk_dev;
    VulkCall call;
    VulkBind bind[2];
    (void)backend;

    memset(bind, 0, sizeof bind);
    bind[0].up = in->cell_list;  bind[0].bytes = yolo_plane_cells(in) * sizeof(float);
    bind[1].down = out->cell_list; bind[1].bytes = yolo_plane_cells(out) * sizeof(float);

    vulk_call_open(&call);
    vulk_push_u(&call, 0, (uint32_t)(in->batch_size * in->channel_count));
    vulk_push_u(&call, 1, (uint32_t)in->height);
    vulk_push_u(&call, 2, (uint32_t)in->width);
    vulk_push_u(&call, 3, (uint32_t)out->height);
    vulk_push_u(&call, 4, (uint32_t)out->width);
    vulk_push_u(&call, 5, (uint32_t)kernel_size);
    vulk_push_u(&call, 6, (uint32_t)stride_size);
    vulk_push_u(&call, 7, (uint32_t)pad_size);
    call.groups_x = vulk_grid(dev, yolo_plane_cells(out));
    (void)vulk_op(dev, VULK_K_YPOOL, &call, bind, 2);
}

static void vulk_yolo_resize(const YoloBackend *backend,
                             const YoloPlane *in, YoloPlane *out, int scale_size)
{
    VulkDev *dev = &vulk_dev;
    VulkCall call;
    VulkBind bind[2];
    (void)backend;

    memset(bind, 0, sizeof bind);
    bind[0].up = in->cell_list;  bind[0].bytes = yolo_plane_cells(in) * sizeof(float);
    bind[1].down = out->cell_list; bind[1].bytes = yolo_plane_cells(out) * sizeof(float);

    vulk_call_open(&call);
    vulk_push_u(&call, 0, (uint32_t)(in->batch_size * in->channel_count));
    vulk_push_u(&call, 1, (uint32_t)in->height);
    vulk_push_u(&call, 2, (uint32_t)in->width);
    vulk_push_u(&call, 3, (uint32_t)out->height);
    vulk_push_u(&call, 4, (uint32_t)out->width);
    vulk_push_u(&call, 5, (uint32_t)scale_size);
    call.groups_x = vulk_grid(dev, yolo_plane_cells(out));
    (void)vulk_op(dev, VULK_K_YRESIZE, &call, bind, 2);
}

static void vulk_yolo_act(const YoloBackend *backend, float *cell_list,
                          size_t cell_count, YoloActKind act_kind)
{
    VulkDev *dev = &vulk_dev;
    VulkCall call;
    VulkBind bind[1];
    (void)backend;
    if (act_kind == YOLO_ACT_NONE || cell_count == 0) return;

    memset(bind, 0, sizeof bind);
    bind[0].up = cell_list; bind[0].down = cell_list;
    bind[0].bytes = cell_count * sizeof(float);

    vulk_call_open(&call);
    vulk_push_u(&call, 0, (uint32_t)cell_count);
    vulk_push_u(&call, 1, (uint32_t)act_kind);
    call.groups_x = vulk_grid(dev, cell_count);
    (void)vulk_op(dev, VULK_K_YACT, &call, bind, 1);
}

static void vulk_yolo_add(const YoloBackend *backend, float *a_list,
                          const float *b_list, size_t cell_count)
{
    VulkDev *dev = &vulk_dev;
    VulkCall call;
    VulkBind bind[2];
    (void)backend;
    if (cell_count == 0) return;

    memset(bind, 0, sizeof bind);
    bind[0].up = a_list; bind[0].down = a_list; bind[0].bytes = cell_count * sizeof(float);
    bind[1].up = b_list; bind[1].bytes = cell_count * sizeof(float);

    vulk_call_open(&call);
    vulk_push_u(&call, 0, (uint32_t)cell_count);
    call.groups_x = vulk_grid(dev, cell_count);
    (void)vulk_op(dev, VULK_K_YADD, &call, bind, 2);
}

static void vulk_yolo_gemm(const YoloBackend *backend,
                           const float *a_list, const float *b_list, float *out_list,
                           int row_count, int mid_count, int col_count,
                           int a_step, int b_step, int out_step, int b_swap_flag)
{
    VulkDev *dev = &vulk_dev;
    VulkCall call;
    VulkBind bind[3];
    size_t   a_bytes, b_bytes, out_bytes;
    (void)backend;
    if (row_count <= 0 || col_count <= 0 || mid_count <= 0) return;

    a_bytes   = ((size_t)(row_count - 1) * (size_t)a_step + (size_t)mid_count) * sizeof(float);
    b_bytes   = b_swap_flag
              ? ((size_t)(col_count - 1) * (size_t)b_step + (size_t)mid_count) * sizeof(float)
              : ((size_t)(mid_count - 1) * (size_t)b_step + (size_t)col_count) * sizeof(float);
    out_bytes = ((size_t)(row_count - 1) * (size_t)out_step + (size_t)col_count) * sizeof(float);

    memset(bind, 0, sizeof bind);
    bind[0].up = a_list; bind[0].bytes = a_bytes;
    bind[1].up = b_list; bind[1].bytes = b_bytes;
    bind[2].down = out_list; bind[2].bytes = out_bytes;

    vulk_call_open(&call);
    vulk_push_u(&call, 0, (uint32_t)row_count);
    vulk_push_u(&call, 1, (uint32_t)mid_count);
    vulk_push_u(&call, 2, (uint32_t)col_count);
    vulk_push_u(&call, 3, (uint32_t)a_step);
    vulk_push_u(&call, 4, (uint32_t)b_step);
    vulk_push_u(&call, 5, (uint32_t)out_step);
    vulk_push_u(&call, 6, b_swap_flag ? 1u : 0u);
    call.groups_x = vulk_grid(dev, (size_t)row_count * (size_t)col_count);
    (void)vulk_op(dev, VULK_K_YGEMM, &call, bind, 3);
}

static void vulk_yolo_softmax(const YoloBackend *backend, float *cell_list,
                              int row_count, int col_count)
{
    VulkDev *dev = &vulk_dev;
    VulkCall call;
    VulkBind bind[1];
    (void)backend;
    if (row_count <= 0 || col_count <= 0) return;

    memset(bind, 0, sizeof bind);
    bind[0].up = cell_list; bind[0].down = cell_list;
    bind[0].bytes = (size_t)row_count * (size_t)col_count * sizeof(float);

    vulk_call_open(&call);
    vulk_push_u(&call, 0, (uint32_t)row_count);
    vulk_push_u(&call, 1, (uint32_t)col_count);
    call.groups_x = vulk_grid(dev, (size_t)row_count);
    (void)vulk_op(dev, VULK_K_YSOFTMAX, &call, bind, 1);
}

static const YoloBackend vulk_yolo_table = {
    "vulkan",
    vulk_yolo_conv,
    vulk_yolo_room,
    vulk_yolo_deconv,
    vulk_yolo_pool,
    vulk_yolo_resize,
    vulk_yolo_act,
    vulk_yolo_gemm,
    vulk_yolo_add,
    vulk_yolo_softmax
};

/* Null until a device is up, because `yolo_model_backend_set` has nowhere to
 * report a failure and a model pointed at a table with no device behind it
 * would fail one operation at a time instead of once, here. */
const YoloBackend *yolo_backend_vulkan(void)
{
    if (vulk_open(getenv("ILL_VULKAN_DEVICE")) != VULK_OK) return NULL;
    return &vulk_yolo_table;
}

#endif /* VULK_NO_PICTURE */

/* ============================================================================
 * part 11 -- the self test
 *
 * Built with -DVULK_MAIN, this runs every operation of both tables twice --
 * once through the CPU backend and once through this one -- over the same
 * pseudo-random input, and reports the largest disagreement.
 *
 * The comparison is to a tolerance rather than bit for bit, and that is a
 * decision rather than a shortcut.  A device adds in a different order: the
 * dense kernel folds sixty-four partial sums through a tree where the CPU
 * walks the row, and no reassociation of floating point addition is exact.
 * `AGENTS.md` asks that a change which moves output says so, and this is a new
 * backend rather than a change to an existing one -- the CPU table is
 * untouched and still holds bit for bit against itself.  What is claimed here
 * is that the two agree to about a part in ten thousand of the magnitude
 * involved, which is what a reordered sum costs and no more.
 * ==========================================================================*/


#if defined(VULK_MAIN) || defined(VULK_PROBE)

/* The comparison is written once and reported through a callback, so that the
 * standalone runner below and `test/test.c` share it rather than each carrying
 * a copy that could drift from the other. */
typedef void (*VulkTell)(void *ctx, const char *name, double worst, int good);

typedef struct VulkProbe {
    VulkTell tell;
    void    *ctx;
    int32_t  run;
    int32_t  bad;
} VulkProbe;

static uint32_t vulk_probe_seed = 0x9e3779b9u;

static float vulk_probe_next(void)
{
    vulk_probe_seed = vulk_probe_seed * 1664525u + 1013904223u;
    return (float)((double)(vulk_probe_seed >> 8) / 8388608.0 - 1.0);
}

static void vulk_probe_fill(float *room, size_t count)
{
    size_t index;
    for (index = 0; index < count; ++index) room[index] = vulk_probe_next();
}

/* The disagreement that matters is relative to the size of the numbers rather
 * than absolute: a dot product over two hundred terms has a magnitude of tens
 * and an absolute error many times a single product's. */
static void vulk_probe_same(VulkProbe *probe, const char *name,
                            const float *want, const float *got, size_t count, float tol)
{
    double worst = 0.0, scale = 1e-6;
    size_t index;
    for (index = 0; index < count; ++index) {
        double size = fabs((double)want[index]);
        double gap  = fabs((double)want[index] - (double)got[index]);
        if (size > scale) scale = size;
        if (gap > worst)  worst = gap;
    }
    ++probe->run;
    if (worst / scale > (double)tol) ++probe->bad;
    probe->tell(probe->ctx, name, worst / scale, worst / scale <= (double)tol);
}

#ifndef VULK_NO_TEXT

/* SPAN and `base` are chosen so the causal horizon is several chunks wide: a
 * window that fits in one pass of sixty-four never rescales, and a check that
 * never rescales does not test the streaming softmax at all. */
#define VULK_PROBE_TOKENS 3
#define VULK_PROBE_COLS   200
#define VULK_PROBE_ROWS   70
#define VULK_PROBE_WIDTH  64
#define VULK_PROBE_HEADS  4
#define VULK_PROBE_GROUPS 2
#define VULK_PROBE_SPAN   260
#define VULK_PROBE_BASE   200
#define VULK_PROBE_DIM    130
#define VULK_PROBE_WINDOW 4

static void vulk_probe_text(VulkProbe *probe)
{
    enum { TOKENS = VULK_PROBE_TOKENS, COLS = VULK_PROBE_COLS, ROWS = VULK_PROBE_ROWS,
           WIDTH = VULK_PROBE_WIDTH, HEADS = VULK_PROBE_HEADS, GROUPS = VULK_PROBE_GROUPS,
           SPAN = VULK_PROBE_SPAN, DIM = VULK_PROBE_DIM, WINDOW = VULK_PROBE_WINDOW,
           WIDE = TOKENS * COLS > TOKENS * ROWS ? TOKENS * COLS : TOKENS * ROWS };
    IllBackend *cpu  = &ill_backend_cpu;
    IllBackend *card = &vulk_ill_table;
    static float sheet[ROWS * COLS];
    static float src[TOKENS * COLS];
    static float want[WIDE], got[WIDE];
    static float gain[COLS];
    IllPlane plane;

    vulk_probe_fill(sheet, ROWS * COLS);
    vulk_probe_fill(src, TOKENS * COLS);
    vulk_probe_fill(gain, COLS);

    memset(&plane, 0, sizeof plane);
    plane.cells = sheet;
    plane.type  = ILL_TYPE_F32;
    plane.rows  = ROWS;
    plane.cols  = COLS;
    cpu->dense(cpu, &plane, src, want, TOKENS, NULL);
    card->dense(card, &plane, src, got, TOKENS, NULL);
    vulk_probe_same(probe, "dense matches the cpu", want, got, TOKENS * ROWS, 1e-5f);

    /* the same plane again: the second call must take the cached upload and
     * must reach the same answer from it */
    memset(got, 0, sizeof got);
    card->dense(card, &plane, src, got, TOKENS, NULL);
    vulk_probe_same(probe, "a cached plane gives the same answer",
                    want, got, TOKENS * ROWS, 1e-5f);

    /* and a narrower plane at the same address is a different tensor: a cache
     * keyed on the pointer alone would hand back the wider one */
    {
        static float small[TOKENS * ROWS];
        IllPlane narrow = plane;
        narrow.cols = COLS / 2;
        cpu->dense(cpu, &narrow, src, want, TOKENS, NULL);
        card->dense(card, &narrow, src, small, TOKENS, NULL);
        vulk_probe_same(probe, "a reshaped plane is not a cache hit",
                        want, small, TOKENS * ROWS, 1e-5f);
        /* and the original shape still reads correctly afterwards */
        cpu->dense(cpu, &plane, src, want, TOKENS, NULL);
        card->dense(card, &plane, src, got, TOKENS, NULL);
        vulk_probe_same(probe, "the wider plane survives the narrow one",
                        want, got, TOKENS * ROWS, 1e-5f);
    }

    cpu->rmsnorm(cpu, src, gain, want, TOKENS, COLS, 1e-5f);
    card->rmsnorm(card, src, gain, got, TOKENS, COLS, 1e-5f);
    vulk_probe_same(probe, "rms norm matches the cpu", want, got, TOKENS * COLS, 1e-5f);

    cpu->swiglu(cpu, src, gain, want, 1, COLS);
    card->swiglu(card, src, gain, got, 1, COLS);
    vulk_probe_same(probe, "swiglu matches the cpu", want, got, COLS, 1e-5f);

    {
        static float cells_a[TOKENS * HEADS * WIDTH];
        static float cells_b[TOKENS * HEADS * WIDTH];
        static float table[TOKENS * WIDTH];
        vulk_probe_fill(cells_a, TOKENS * HEADS * WIDTH);
        vulk_probe_fill(table, TOKENS * WIDTH);
        memcpy(cells_b, cells_a, sizeof cells_a);
        cpu->rope(cpu, cells_a, table, TOKENS, HEADS, WIDTH, HEADS * WIDTH);
        card->rope(card, cells_b, table, TOKENS, HEADS, WIDTH, HEADS * WIDTH);
        vulk_probe_same(probe, "rope matches the cpu",
                        cells_a, cells_b, TOKENS * HEADS * WIDTH, 1e-5f);
    }

    {
        static float query[TOKENS * HEADS * WIDTH];
        static float keys[GROUPS * SPAN * WIDTH];
        static float vals[GROUPS * SPAN * WIDTH];
        static float mix_a[TOKENS * HEADS * WIDTH];
        static float mix_b[TOKENS * HEADS * WIDTH];
        static float board[8 * SPAN];
        IllHeedJob job;
        vulk_probe_fill(query, TOKENS * HEADS * WIDTH);
        vulk_probe_fill(keys, GROUPS * SPAN * WIDTH);
        vulk_probe_fill(vals, GROUPS * SPAN * WIDTH);
        memset(&job, 0, sizeof job);
        job.query = query; job.keys = keys; job.vals = vals; job.board = board;
        job.tokens = TOKENS; job.heads = HEADS; job.groups = GROUPS;
        job.width = WIDTH; job.span = SPAN; job.base = VULK_PROBE_BASE;
        job.scale = 1.0f / sqrtf((float)WIDTH);
        job.value = mix_a; cpu->attend(cpu, &job);
        job.value = mix_b; card->attend(card, &job);
        vulk_probe_same(probe, "attention matches the cpu over many chunks",
                        mix_a, mix_b, TOKENS * HEADS * WIDTH, 1e-4f);
    }

    {
        static float flow[TOKENS * DIM];
        static float out_a[TOKENS * DIM], out_b[TOKENS * DIM];
        static float hist_a[DIM * (WINDOW - 1)], hist_b[DIM * (WINDOW - 1)];
        static float taps[DIM * WINDOW], bias[DIM];
        IllFlowJob job;
        vulk_probe_fill(flow, TOKENS * DIM);
        vulk_probe_fill(taps, DIM * WINDOW);
        vulk_probe_fill(bias, DIM);
        static float hist_0[DIM * (WINDOW - 1)];
        vulk_probe_fill(hist_0, DIM * (WINDOW - 1));
        memcpy(hist_a, hist_0, sizeof hist_0);
        memcpy(hist_b, hist_0, sizeof hist_0);
        memset(&job, 0, sizeof job);
        job.src = flow; job.taps = taps; job.bias = bias;
        job.tokens = TOKENS; job.dim = DIM; job.width = WINDOW;
        job.dst = out_a; job.hist = hist_a; cpu->conv1d(cpu, &job);
        job.dst = out_b; job.hist = hist_b; card->conv1d(card, &job);
        vulk_probe_same(probe, "the short convolution matches the cpu",
                        out_a, out_b, TOKENS * DIM, 1e-5f);
        vulk_probe_same(probe, "and hands back the same rolling window",
                        hist_a, hist_b, DIM * (WINDOW - 1), 1e-5f);

        /* The same call again with a budget too small to hold both of its
         * weights.  The short convolution looks up its taps and then its bias,
         * so without a pin the second lookup evicts the first and the dispatch
         * binds a buffer that has been destroyed.  The budget is restored
         * afterwards; it is deliberately absurd, because the point is to make
         * the eviction certain rather than likely. */
        {
            size_t keep = vulk_dev.budget;
            vulk_hold_clear(&vulk_dev);
            vulk_memory_budget_set((size_t)DIM * WINDOW * sizeof(float));
            memcpy(hist_b, hist_0, sizeof hist_0);
            memset(out_b, 0, sizeof out_b);
            job.dst = out_b; job.hist = hist_b; card->conv1d(card, &job);
            vulk_memory_budget_set(keep);
            vulk_hold_clear(&vulk_dev);
        }
        vulk_probe_same(probe, "a weight in use is not evicted by the next one",
                        out_a, out_b, TOKENS * DIM, 1e-5f);

        /* and with no bias at all, which is the path where the kernel must not
         * read the one-float stand-in that gets bound in its place */
        job.bias = NULL;
        memcpy(hist_a, hist_0, sizeof hist_0);
        memcpy(hist_b, hist_0, sizeof hist_0);
        job.dst = out_a; job.hist = hist_a; cpu->conv1d(cpu, &job);
        job.dst = out_b; job.hist = hist_b; card->conv1d(card, &job);
        vulk_probe_same(probe, "the short convolution with no bias matches the cpu",
                        out_a, out_b, TOKENS * DIM, 1e-5f);
    }

    /* A quantised plane, against the widening the engine itself does.  This is
     * deliberately not a comparison against the CPU's own q8 dense: that path
     * also quantises the activations and this one does not, so the two are not
     * the same arithmetic.  What is being checked is that the device is handed
     * exactly the values `ill_plane_row` reconstructs. */
    {
        static int8_t quant[ROWS * COLS];
        static float  steps[ROWS * ((COLS + 31) / 32)];
        static float  wide[COLS];
        int32_t       blocks = (COLS + 31) / 32;
        int32_t       row, token, col;
        for (row = 0; row < ROWS; ++row)
            ill_q8_pack(sheet + (size_t)row * COLS, COLS,
                        quant + (size_t)row * COLS, steps + (size_t)row * blocks);
        plane.cells  = quant;
        plane.steps  = steps;
        plane.type   = ILL_TYPE_Q8;
        plane.blocks = blocks;
        for (token = 0; token < TOKENS; ++token)
            for (row = 0; row < ROWS; ++row) {
                double total = 0.0;
                ill_plane_row(&plane, row, wide);
                for (col = 0; col < COLS; ++col)
                    total += (double)wide[col] * (double)src[(size_t)token * COLS + col];
                want[(size_t)token * ROWS + row] = (float)total;
            }
        card->dense(card, &plane, src, got, TOKENS, NULL);
        vulk_probe_same(probe, "a q8 plane widens to the same numbers",
                        want, got, TOKENS * ROWS, 1e-5f);
    }
}

#endif /* VULK_NO_TEXT */

#ifndef VULK_NO_PICTURE

static void vulk_probe_picture(VulkProbe *probe)
{
    enum { IN_C = 6, OUT_C = 8, SIDE = 9, WIDE = OUT_C * SIDE * SIDE * 4 };
    const YoloBackend *cpu  = yolo_backend_cpu();
    const YoloBackend *card = &vulk_yolo_table;
    static float in_cells[IN_C * SIDE * SIDE];
    static float out_a[WIDE], out_b[WIDE];
    static float sheet[OUT_C * IN_C * 9];
    static float bias[OUT_C];
    YoloPlane in, out;

    vulk_probe_fill(in_cells, IN_C * SIDE * SIDE);
    vulk_probe_fill(sheet, OUT_C * IN_C * 9);
    vulk_probe_fill(bias, OUT_C);

    in.batch_size = 1; in.channel_count = IN_C; in.height = SIDE; in.width = SIDE;
    in.cell_list = in_cells;
    out.batch_size = 1; out.channel_count = OUT_C; out.height = SIDE; out.width = SIDE;

    /* a 3x3 at stride one with a pad, which is most of the backbone */
    {
        size_t room = cpu->conv_room(&in, &out, 3, 1);
        float *scratch = (float *)malloc(room ? room : 4);
        out.cell_list = out_a;
        cpu->conv_run(cpu, &in, &out, sheet, bias, 3, 1, 1, 1, 1, YOLO_ACT_SILU, scratch);
        free(scratch);
        out.cell_list = out_b;
        card->conv_run(card, &in, &out, sheet, bias, 3, 1, 1, 1, 1, YOLO_ACT_SILU, NULL);
        vulk_probe_same(probe, "a padded 3x3 convolution matches the cpu",
                        out_a, out_b, OUT_C * SIDE * SIDE, 1e-5f);
    }
    /* the same convolution with no bias, which binds a one-float stand-in the
     * kernel must never read */
    {
        size_t room = cpu->conv_room(&in, &out, 3, 1);
        float *scratch = (float *)malloc(room ? room : 4);
        out.cell_list = out_a;
        cpu->conv_run(cpu, &in, &out, sheet, NULL, 3, 1, 1, 1, 1, YOLO_ACT_SILU, scratch);
        free(scratch);
        out.cell_list = out_b;
        card->conv_run(card, &in, &out, sheet, NULL, 3, 1, 1, 1, 1, YOLO_ACT_SILU, NULL);
        vulk_probe_same(probe, "a convolution with no bias matches the cpu",
                        out_a, out_b, OUT_C * SIDE * SIDE, 1e-5f);
    }

    probe->tell(probe->ctx, "conv_room asks for nothing to lower into",
                0.0, cpu->conv_room(&in, &out, 3, 1) > 0 && card->conv_room(&in, &out, 3, 1) == 0);
    ++probe->run;
    if (!(cpu->conv_room(&in, &out, 3, 1) > 0 && card->conv_room(&in, &out, 3, 1) == 0))
        ++probe->bad;

    /* stride two, which halves the map */
    {
        YoloPlane half = out;
        size_t room;
        float *scratch;
        half.height = (SIDE + 2 - 3) / 2 + 1;
        half.width  = half.height;
        room = cpu->conv_room(&in, &half, 3, 1);
        scratch = (float *)malloc(room ? room : 4);
        half.cell_list = out_a;
        cpu->conv_run(cpu, &in, &half, sheet, bias, 3, 2, 1, 1, 1, YOLO_ACT_NONE, scratch);
        free(scratch);
        half.cell_list = out_b;
        card->conv_run(card, &in, &half, sheet, bias, 3, 2, 1, 1, 1, YOLO_ACT_NONE, NULL);
        vulk_probe_same(probe, "a strided convolution matches the cpu", out_a, out_b,
                        (size_t)OUT_C * (size_t)half.height * (size_t)half.width, 1e-5f);
    }

    /* depthwise, where every output channel sees one input channel */
    {
        static float deep[IN_C * 9];
        YoloPlane deep_out = in;
        vulk_probe_fill(deep, IN_C * 9);
        deep_out.cell_list = out_a;
        cpu->conv_run(cpu, &in, &deep_out, deep, bias, 3, 1, 1, 1, IN_C, YOLO_ACT_SILU, NULL);
        deep_out.cell_list = out_b;
        card->conv_run(card, &in, &deep_out, deep, bias, 3, 1, 1, 1, IN_C, YOLO_ACT_SILU, NULL);
        vulk_probe_same(probe, "a depthwise convolution matches the cpu",
                        out_a, out_b, IN_C * SIDE * SIDE, 1e-5f);
    }

    /* two groups, where the weight sheet is read a group at a time */
    {
        static float split[OUT_C * (IN_C / 2) * 9];
        size_t room;
        float *scratch;
        vulk_probe_fill(split, OUT_C * (IN_C / 2) * 9);
        room = cpu->conv_room(&in, &out, 3, 2);
        scratch = (float *)malloc(room ? room : 4);
        out.cell_list = out_a;
        cpu->conv_run(cpu, &in, &out, split, bias, 3, 1, 1, 1, 2, YOLO_ACT_NONE, scratch);
        free(scratch);
        out.cell_list = out_b;
        card->conv_run(card, &in, &out, split, bias, 3, 1, 1, 1, 2, YOLO_ACT_NONE, NULL);
        vulk_probe_same(probe, "a grouped convolution matches the cpu",
                        out_a, out_b, OUT_C * SIDE * SIDE, 1e-5f);
    }

    /* the transposed convolution the mask prototype head asks for */
    {
        static float taps[IN_C * OUT_C * 4];
        YoloPlane up = out;
        vulk_probe_fill(taps, IN_C * OUT_C * 4);
        up.height = SIDE * 2; up.width = SIDE * 2;
        up.cell_list = out_a;
        cpu->deconv_run(cpu, &in, &up, taps, bias, 2, 2, YOLO_ACT_NONE);
        up.cell_list = out_b;
        card->deconv_run(card, &in, &up, taps, bias, 2, 2, YOLO_ACT_NONE);
        vulk_probe_same(probe, "a 2x2 transposed convolution matches the cpu",
                        out_a, out_b, OUT_C * SIDE * 2 * SIDE * 2, 1e-5f);
        up.cell_list = out_a;
        cpu->deconv_run(cpu, &in, &up, taps, NULL, 2, 2, YOLO_ACT_NONE);
        up.cell_list = out_b;
        card->deconv_run(card, &in, &up, taps, NULL, 2, 2, YOLO_ACT_NONE);
        vulk_probe_same(probe, "and matches it with no bias too",
                        out_a, out_b, OUT_C * SIDE * 2 * SIDE * 2, 1e-5f);
    }

    /* the 5x5 pooling the spatial pyramid uses, pad and all */
    {
        YoloPlane pooled = in;
        pooled.cell_list = out_a; cpu->pool_run(cpu, &in, &pooled, 5, 1, 2);
        pooled.cell_list = out_b; card->pool_run(card, &in, &pooled, 5, 1, 2);
        vulk_probe_same(probe, "max pooling matches the cpu",
                        out_a, out_b, IN_C * SIDE * SIDE, 1e-6f);
    }

    /* the nearest neighbour the neck climbs with */
    {
        YoloPlane wide = in;
        wide.height = SIDE * 2; wide.width = SIDE * 2;
        wide.cell_list = out_a; cpu->resize_run(cpu, &in, &wide, 2);
        wide.cell_list = out_b; card->resize_run(card, &in, &wide, 2);
        vulk_probe_same(probe, "nearest neighbour resize matches the cpu",
                        out_a, out_b, IN_C * SIDE * 2 * SIDE * 2, 1e-6f);
    }

    {
        size_t cells = IN_C * SIDE * SIDE;
        memcpy(out_a, in_cells, cells * sizeof(float));
        memcpy(out_b, in_cells, cells * sizeof(float));
        cpu->act_run(cpu, out_a, cells, YOLO_ACT_SILU);
        card->act_run(card, out_b, cells, YOLO_ACT_SILU);
        vulk_probe_same(probe, "silu matches the cpu", out_a, out_b, cells, 1e-6f);

        memcpy(out_a, in_cells, cells * sizeof(float));
        memcpy(out_b, in_cells, cells * sizeof(float));
        cpu->act_run(cpu, out_a, cells, YOLO_ACT_SIGMOID);
        card->act_run(card, out_b, cells, YOLO_ACT_SIGMOID);
        vulk_probe_same(probe, "sigmoid matches the cpu", out_a, out_b, cells, 1e-6f);

        memcpy(out_a, in_cells, cells * sizeof(float));
        memcpy(out_b, in_cells, cells * sizeof(float));
        cpu->add_run(cpu, out_a, in_cells, cells);
        card->add_run(card, out_b, in_cells, cells);
        vulk_probe_same(probe, "add matches the cpu", out_a, out_b, cells, 1e-6f);
    }

    {
        enum { ROWS = 7, MIDS = 40, COLS = 11 };
        static float left[ROWS * MIDS], right[MIDS * COLS], swapped[COLS * MIDS];
        vulk_probe_fill(left, ROWS * MIDS);
        vulk_probe_fill(right, MIDS * COLS);
        vulk_probe_fill(swapped, COLS * MIDS);
        cpu->gemm_run(cpu, left, right, out_a, ROWS, MIDS, COLS, MIDS, COLS, COLS, 0);
        card->gemm_run(card, left, right, out_b, ROWS, MIDS, COLS, MIDS, COLS, COLS, 0);
        vulk_probe_same(probe, "gemm matches the cpu", out_a, out_b, ROWS * COLS, 1e-5f);
        cpu->gemm_run(cpu, left, swapped, out_a, ROWS, MIDS, COLS, MIDS, MIDS, COLS, 1);
        card->gemm_run(card, left, swapped, out_b, ROWS, MIDS, COLS, MIDS, MIDS, COLS, 1);
        vulk_probe_same(probe, "gemm with b swapped matches the cpu",
                        out_a, out_b, ROWS * COLS, 1e-5f);
    }

    {
        enum { ROWS = 40, COLS = 16 };
        static float rows_a[ROWS * COLS], rows_b[ROWS * COLS];
        vulk_probe_fill(rows_a, ROWS * COLS);
        memcpy(rows_b, rows_a, sizeof rows_a);
        cpu->softmax_run(cpu, rows_a, ROWS, COLS);
        card->softmax_run(card, rows_b, ROWS, COLS);
        vulk_probe_same(probe, "softmax matches the cpu", rows_a, rows_b, ROWS * COLS, 1e-5f);
    }
}

#endif /* VULK_NO_PICTURE */

/* Runs every comparison there is, and answers the number that failed.  A host
 * with no device is not a failure: `run` comes back zero and the caller says
 * what it wants to about that. */
static int32_t vulk_probe_all(VulkTell tell, void *ctx, int32_t *run_out)
{
    VulkProbe probe;
    memset(&probe, 0, sizeof probe);
    probe.tell = tell;
    probe.ctx  = ctx;
    vulk_probe_seed = 0x9e3779b9u;
    if (vulk_open(NULL) == VULK_OK) {
#ifndef VULK_NO_TEXT
        vulk_ill_table.inner = &vulk_text;
        vulk_probe_text(&probe);
#endif
#ifndef VULK_NO_PICTURE
        vulk_probe_picture(&probe);
#endif
    }
    if (run_out) *run_out = probe.run;
    return probe.bad;
}

#endif /* VULK_MAIN || VULK_PROBE */

#ifdef VULK_MAIN

static void vulk_main_tell(void *ctx, const char *name, double worst, int good)
{
    (void)ctx;
    printf("  %-4s %-48s %.2g\n", good ? "ok" : "FAIL", name, worst);
}

int main(int count, char **args)
{
    VulkStatus code;
    int32_t    run = 0, bad;

    code = vulk_open(count > 1 ? args[1] : NULL);
    printf("vulkan: %s\n", vulk_status_text(code));
    if (code != VULK_OK) return 1;
    printf("device: %s, %s memory, %u workgroups a dispatch, %u bytes shared\n",
           vulk_device_name(), vulk_dev.unified_flag ? "unified" : "staged",
           vulk_dev.group_limit, vulk_dev.shared_limit);

    bad = vulk_probe_all(vulk_main_tell, NULL, &run);
    {
        uint64_t sent = 0, moved = 0;
        vulk_report(&sent, &moved);
        printf("\n%d of %d checks pass -- %lu submissions, %.1f MiB moved,"
               " %.1f MiB on the device\n",
               run - bad, run, (unsigned long)sent,
               (double)moved / 1048576.0, (double)vulk_memory_used() / 1048576.0);
    }
    vulk_close();
    return bad ? 1 : 0;
}

#endif /* VULK_MAIN */

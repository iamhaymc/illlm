/* ============================================================================
 * app/main.c -- command line front end for the Liquid inference engine.
 *
 * Commands
 *   info      report what a checkpoint contains and how it would be loaded
 *   tokens    encode text to ids, or decode ids back to text
 *   generate  continue a prompt and stream the result
 *   chat      hold a turn by turn conversation
 *   logits    dump raw logits, the hook test/test.py compares against
 *   bench     measure prefill and decode throughput
 *   perplexity  score a text, so an accuracy trade has a number
 *
 * Every command takes --model PATH, the folder holding config.json, the
 * safetensors shards, and tokenizer.json.  `app_main help` lists the rest.
 * ==========================================================================*/

#include "core.c"

/* -- shared options -------------------------------------------------------- */

typedef struct AppOpts {
    const char *model_path;
    const char *backend_name;
    const char *prompt_text;
    const char *system_text;
    const char *token_text;
    const char *out_path;
    IllType     weight_type;
    int32_t     thread_count;
    int32_t     context_span;
    int32_t     batch_span;
    int32_t     reply_span;
    int32_t     every_logit;
    int32_t     stream_mode;
    int32_t     prefill_span;
    int32_t     raw_prompt;
    int32_t     show_stats;
    int32_t     quiet_load;
    IllTuning   tuning;
} AppOpts;

static void app_opts_init(AppOpts *opts)
{
    memset(opts, 0, sizeof(*opts));
    opts->weight_type  = ILL_TYPE_KEEP;
    opts->context_span = 4096;
    opts->batch_span   = 256;
    opts->reply_span   = 256;
    opts->prefill_span = 1;
    opts->show_stats   = 1;
    ill_tuning_init(&opts->tuning);
}

static void app_help(void)
{
    printf(
"illlm " ILL_VERSION_TEXT " -- Liquid inference engine\n"
"\n"
"usage: app_main <command> --model PATH [options]\n"
"\n"
"commands\n"
"  info                  describe the checkpoint and the load plan\n"
"  tokens                encode --prompt, or decode --tokens\n"
"  generate              continue --prompt and stream the completion\n"
"  chat                  interactive conversation on stdin\n"
"  logits                write logits for --prompt or --tokens\n"
"  bench                 time prefill and decode\n"
"  perplexity            score stdin or --prompt, in nats and perplexity\n"
"  help                  this page\n"
"\n"
"model options\n"
"  --model PATH          checkpoint folder (required)\n"
"  --quant q8|none       repack weights to q8 at load (default none)\n"
"  --threads N           worker threads (default: host cpu count)\n"
"  --ctx N               context window in tokens (default 4096)\n"
"  --batch N             prefill chunk in tokens (default 256)\n"
"  --backend NAME        compute backend (default cpu)\n"
"  --quiet               suppress load progress on stderr\n"
"\n"
"prompt options\n"
"  --prompt TEXT         prompt text\n"
"  --system TEXT         system message for chat and generate\n"
"  --tokens a,b,c        raw token ids instead of text\n"
"  --raw                 skip the chat template, feed the prompt verbatim\n"
"  --max-tokens N        reply length cap (default 256)\n"
"\n"
"sampling options\n"
"  --temp F              temperature, 0 for greedy (default 0.7)\n"
"  --top-p F             nucleus mass (default 0.95)\n"
"  --top-k N             candidate cap (default off)\n"
"  --min-p F             floor relative to the peak (default off)\n"
"  --repeat F            repetition penalty (default 1.0)\n"
"  --repeat-span N       penalty history depth (default 64)\n"
"  --seed N              sampler seed\n"
"\n"
"logits options\n"
"  --every               emit logits for every token, not just the last\n"
"  --stream              feed one token at a time after --prefill, one row each\n"
"  --prefill N           tokens fed as a single batch before streaming (default 1)\n"
"  --out PATH            write raw float32 to PATH instead of stdout\n");
}

static int app_opts_read(AppOpts *opts, int argc, char **argv, int from)
{
    int index;
    for (index = from; index < argc; ++index) {
        const char *flag = argv[index];
        const char *next = index + 1 < argc ? argv[index + 1] : NULL;
#define APP_TAKE(name) (!strcmp(flag, name) && next && (++index, 1))
        if      (APP_TAKE("--model"))       opts->model_path = next;
        else if (APP_TAKE("--backend"))     opts->backend_name = next;
        else if (APP_TAKE("--prompt"))      opts->prompt_text = next;
        else if (APP_TAKE("--system"))      opts->system_text = next;
        else if (APP_TAKE("--tokens"))      opts->token_text = next;
        else if (APP_TAKE("--out"))         opts->out_path = next;
        else if (APP_TAKE("--threads"))     opts->thread_count = atoi(next);
        else if (APP_TAKE("--ctx"))         opts->context_span = atoi(next);
        else if (APP_TAKE("--batch"))       opts->batch_span = atoi(next);
        else if (APP_TAKE("--max-tokens"))  opts->reply_span = atoi(next);
        else if (APP_TAKE("--temp"))        opts->tuning.temperature = (float)atof(next);
        else if (APP_TAKE("--top-p"))       opts->tuning.top_p = (float)atof(next);
        else if (APP_TAKE("--min-p"))       opts->tuning.min_p = (float)atof(next);
        else if (APP_TAKE("--top-k"))       opts->tuning.top_k = atoi(next);
        else if (APP_TAKE("--repeat"))      opts->tuning.repeat_penalty = (float)atof(next);
        else if (APP_TAKE("--repeat-span")) opts->tuning.repeat_span = atoi(next);
        else if (APP_TAKE("--seed"))        opts->tuning.seed = strtoull(next, NULL, 10);
        else if (APP_TAKE("--quant"))       opts->weight_type =
                                              !strcmp(next, "q8") ? ILL_TYPE_Q8 : ILL_TYPE_KEEP;
        else if (APP_TAKE("--prefill"))     opts->prefill_span = atoi(next);
        else if (!strcmp(flag, "--every"))  opts->every_logit = 1;
        else if (!strcmp(flag, "--stream")) opts->stream_mode = 1;
        else if (!strcmp(flag, "--raw"))    opts->raw_prompt = 1;
        else if (!strcmp(flag, "--quiet"))  { opts->quiet_load = 1; opts->show_stats = 0; }
        else {
            fprintf(stderr, "unknown option: %s\n", flag);
            return 0;
        }
#undef APP_TAKE
    }
    if (!opts->model_path) {
        fprintf(stderr, "missing --model PATH\n");
        return 0;
    }
    return 1;
}

static IllResult app_model_open(const AppOpts *opts, IllModel **model)
{
    IllPlan plan;
    ill_plan_init(&plan);
    plan.model_path   = opts->model_path;
    plan.backend_name = opts->backend_name;
    plan.weight_type  = opts->weight_type;
    plan.thread_count = opts->thread_count;
    plan.batch_span   = opts->batch_span;
    plan.quiet_load   = opts->quiet_load;
    return ill_model_load(model, &plan);
}

/* -- prompt assembly ------------------------------------------------------- */

typedef struct AppFeed {
    int32_t *tokens;
    int32_t  count;
} AppFeed;

static void app_feed_free(AppFeed *feed) { ill_block_free(feed->tokens); feed->tokens = NULL; }

/* Parses "12,7,3" into ids. */
static int app_feed_ids(AppFeed *feed, const char *text)
{
    const char *scan = text;
    int32_t     room = 8, fill = 0;
    int32_t    *slot = (int32_t *)ill_block_make((size_t)room * sizeof(int32_t));
    if (!slot) return 0;
    while (*scan) {
        char *stop;
        long  cell;
        while (*scan == ',' || *scan == ' ') ++scan;
        if (!*scan) break;
        cell = strtol(scan, &stop, 10);
        if (stop == scan) break;
        if (fill == room) {
            int32_t *grown = (int32_t *)ill_block_make((size_t)room * 2 * sizeof(int32_t));
            if (!grown) { ill_block_free(slot); return 0; }
            memcpy(grown, slot, (size_t)fill * sizeof(int32_t));
            ill_block_free(slot);
            slot = grown;
            room *= 2;
        }
        slot[fill++] = (int32_t)cell;
        scan = stop;
    }
    feed->tokens = slot;
    feed->count  = fill;
    return 1;
}

static IllResult app_feed_text(AppFeed *feed, const IllVocab *vocab,
                               const char *text, int32_t add_begin)
{
    int32_t need = 0;
    IllResult code = ill_vocab_encode(vocab, text, strlen(text), add_begin, NULL, 0, &need);
    if (code != ILL_OK && code != ILL_LIMIT) return code;
    feed->tokens = (int32_t *)ill_block_make((size_t)(need + 1) * sizeof(int32_t));
    if (!feed->tokens) return ILL_ALLOC;
    code = ill_vocab_encode(vocab, text, strlen(text), add_begin, feed->tokens, need + 1, &need);
    feed->count = need;
    return code;
}

/* Builds the token feed a command should run, honouring --tokens and --raw. */
static IllResult app_feed_make(AppFeed *feed, const AppOpts *opts, const IllModel *model,
                               const char *body)
{
    const IllVocab *vocab = ill_model_vocab(model);
    memset(feed, 0, sizeof(*feed));

    if (opts->token_text) return app_feed_ids(feed, opts->token_text) ? ILL_OK : ILL_ALLOC;
    if (!vocab) {
        fprintf(stderr, "this checkpoint has no readable tokenizer; use --tokens\n");
        return ILL_VOCAB;
    }
    if (!body) body = "";
    if (opts->raw_prompt) return app_feed_text(feed, vocab, body, 1);

    {
        IllTurn turns[2];
        int32_t count = 0;
        char   *text;
        size_t  need = 0;
        IllResult code;
        if (opts->system_text) {
            turns[count].role = "system";
            turns[count].text = opts->system_text;
            ++count;
        }
        turns[count].role = "user";
        turns[count].text = body;
        ++count;
        ill_vocab_prompt(vocab, turns, count, 1, NULL, 0, &need);
        text = (char *)ill_block_make(need + 1);
        if (!text) return ILL_ALLOC;
        code = ill_vocab_prompt(vocab, turns, count, 1, text, need + 1, &need);
        if (code == ILL_OK) code = app_feed_text(feed, vocab, text, 1);
        ill_block_free(text);
        return code;
    }
}

/* -- streaming -------------------------------------------------------------
 *
 * A token's bytes can stop in the middle of a character.  Written straight
 * out, the terminal draws a replacement character and the next token corrects
 * it, so text in any multi-byte language flickers as it streams.  `AppTail`
 * holds the unfinished bytes back until the token that completes them arrives.
 *
 * It is a tail, not a buffer: everything that is whole goes out at once, and
 * at most three bytes are ever held.  If those three are never completed --
 * because the model emitted a byte that is not valid UTF-8, which a byte level
 * tokenizer is entitled to do -- `app_tail_flush` releases them as they are.
 * Nothing is dropped and nothing waits for a continuation that is not coming.
 * ------------------------------------------------------------------------*/

typedef struct AppTail {
    char    hold[4];   /* bytes of a sequence still waiting to be finished    */
    int32_t held;
} AppTail;

static void app_tail_open(AppTail *tail) { tail->held = 0; }

static void app_tail_show(AppTail *tail, const IllVocab *vocab, int32_t token)
{
    char    text[512 + 4];
    int32_t span = 0, whole;
    if (ill_vocab_decode(vocab, token, text + tail->held,
                         (int32_t)sizeof(text) - tail->held, &span) != ILL_OK) return;
    if (tail->held) {
        memcpy(text, tail->hold, (size_t)tail->held);
        span += tail->held;
        tail->held = 0;
    }
    whole = span - ill_utf8_hold(text, span);
    if (whole > 0) { fwrite(text, 1, (size_t)whole, stdout); fflush(stdout); }
    tail->held = span - whole;
    if (tail->held) memcpy(tail->hold, text + whole, (size_t)tail->held);
}

/* Releases whatever is still held, valid or not.  Called when a reply ends,
 * so a run never swallows its own last bytes. */
static void app_tail_flush(AppTail *tail)
{
    if (tail->held) { fwrite(tail->hold, 1, (size_t)tail->held, stdout); fflush(stdout); }
    tail->held = 0;
}

typedef struct AppRun {
    int32_t made;      /* tokens produced */
    double  fill_secs; /* prefill seconds */
    double  step_secs; /* decode seconds  */
    int32_t read;      /* prompt tokens   */
} AppRun;

static IllResult app_run_loop(IllModel *model, IllState *state, IllSampler *sampler,
                              const AppFeed *feed, int32_t reply_span, int show,
                              AppRun *run)
{
    const IllVocab *vocab = ill_model_vocab(model);
    AppTail   tail;
    IllBatch  batch;
    float    *logits = NULL;
    IllResult code;
    double    mark;
    int32_t   step, next;

    app_tail_open(&tail);
    memset(run, 0, sizeof(*run));
    run->read = feed->count;

    batch.tokens = feed->tokens;
    batch.count  = feed->count;
    batch.every  = 0;
    mark = ill_clock_now();
    code = ill_model_apply(model, state, &batch, &logits);
    run->fill_secs = ill_clock_now() - mark;
    if (code != ILL_OK) return code;

    mark = ill_clock_now();
    for (step = 0; step < reply_span; ++step) {
        next = ill_sampler_pick(sampler, logits);
        if (next < 0) break;
        if (vocab && ill_vocab_stop(vocab, next)) break;
        ill_sampler_note(sampler, next);
        ++run->made;
        if (show && vocab) app_tail_show(&tail, vocab, next);
        if (ill_state_fill(state) >= ill_state_span(state)) break;
        batch.tokens = &next;
        batch.count  = 1;
        code = ill_model_apply(model, state, &batch, &logits);
        if (code != ILL_OK) { if (show && vocab) app_tail_flush(&tail); return code; }
    }
    run->step_secs = ill_clock_now() - mark;
    if (show && vocab) app_tail_flush(&tail);
    return ILL_OK;
}

static void app_run_note(const AppRun *run)
{
    double fill_rate = run->fill_secs > 0.0 ? (double)run->read / run->fill_secs : 0.0;
    double step_rate = run->step_secs > 0.0 ? (double)run->made / run->step_secs : 0.0;
    fprintf(stderr,
            "\n[prefill %d tok in %.2fs, %.1f tok/s | decode %d tok in %.2fs, %.1f tok/s]\n",
            run->read, run->fill_secs, fill_rate, run->made, run->step_secs, step_rate);
}

/* -- commands -------------------------------------------------------------- */

static int app_do_info(const AppOpts *opts)
{
    IllModel *model = NULL;
    const IllArch *arch;
    const IllVocab *vocab;
    IllResult code = app_model_open(opts, &model);
    int32_t index;

    if (code != ILL_OK) {
        fprintf(stderr, "load failed: %s\n", ill_result_text(code));
        return 1;
    }
    arch  = ill_model_arch(model);
    vocab = ill_model_vocab(model);

    printf("path          %s\n", opts->model_path);
    printf("layers        %d (%d attention, %d convolution)\n",
           arch->layer_count, arch->attn_count, arch->conv_count);
    printf("model_dim     %d\n", arch->model_dim);
    printf("inner_dim     %d\n", arch->inner_dim);
    printf("heads         %d query / %d key-value, head_dim %d\n",
           arch->head_count, arch->group_count, arch->head_dim);
    printf("conv_width    %d%s\n", arch->conv_width, arch->conv_bias ? " (biased)" : "");
    printf("vocab_size    %d\n", arch->vocab_size);
    printf("token_span    %d\n", arch->token_span);
    printf("norm_eps      %g\n", (double)arch->norm_eps);
    printf("rope_base     %g\n", (double)arch->rope_base);
    printf("bound_embed   %s\n", arch->bound_embed ? "yes" : "no");
    printf("tokenizer     %s\n", vocab ? "loaded" : "absent");
    if (vocab) printf("stop_tokens   %d\n", ill_vocab_stops(vocab));
    printf("weights       %.2f GiB resident\n", (double)ill_model_bytes(model) / 1073741824.0);
    printf("backend       %s / %s, %d threads\n", ill_model_backend(model),
           ill_backend_simd(), ill_model_threads(model));
    printf("layout        ");
    for (index = 0; index < arch->layer_count; ++index)
        putchar(arch->layer_kind[index] == ILL_LAYER_ATTN ? 'A' : 'c');
    putchar('\n');

    ill_model_free(model);
    return 0;
}

static int app_do_tokens(const AppOpts *opts)
{
    IllModel *model = NULL;
    const IllVocab *vocab;
    IllResult code = app_model_open(opts, &model);
    AppFeed feed;
    int32_t index;

    if (code != ILL_OK) {
        fprintf(stderr, "load failed: %s\n", ill_result_text(code));
        return 1;
    }
    vocab = ill_model_vocab(model);
    if (!vocab) { fprintf(stderr, "no tokenizer\n"); ill_model_free(model); return 1; }

    if (opts->token_text) {
        if (!app_feed_ids(&feed, opts->token_text)) { ill_model_free(model); return 1; }
        AppTail tail;
        app_tail_open(&tail);
        for (index = 0; index < feed.count; ++index)
            app_tail_show(&tail, vocab, feed.tokens[index]);
        app_tail_flush(&tail);
        putchar('\n');
        app_feed_free(&feed);
        ill_model_free(model);
        return 0;
    }

    code = app_feed_text(&feed, vocab, opts->prompt_text ? opts->prompt_text : "", 0);
    if (code != ILL_OK) {
        fprintf(stderr, "encode failed: %s\n", ill_result_text(code));
        ill_model_free(model);
        return 1;
    }
    for (index = 0; index < feed.count; ++index)
        printf("%s%d", index ? "," : "", feed.tokens[index]);
    putchar('\n');
    for (index = 0; index < feed.count; ++index)
        printf("%6d  %s\n", feed.tokens[index], ill_vocab_name(vocab, feed.tokens[index]));
    app_feed_free(&feed);
    ill_model_free(model);
    return 0;
}

static int app_do_generate(const AppOpts *opts)
{
    IllModel   *model = NULL;
    IllState   *state = NULL;
    IllSampler *sampler = NULL;
    AppFeed     feed;
    AppRun      run;
    IllResult   code = app_model_open(opts, &model);
    int         exit_code = 1;

    if (code != ILL_OK) { fprintf(stderr, "load failed: %s\n", ill_result_text(code)); return 1; }
    memset(&feed, 0, sizeof(feed));

    code = app_feed_make(&feed, opts, model, opts->prompt_text);
    if (code != ILL_OK) { fprintf(stderr, "prompt failed: %s\n", ill_result_text(code)); goto done; }
    code = ill_state_make(&state, model, opts->context_span);
    if (code != ILL_OK) { fprintf(stderr, "state failed: %s\n", ill_result_text(code)); goto done; }
    code = ill_sampler_make(&sampler, &opts->tuning, ill_model_arch(model)->vocab_size);
    if (code != ILL_OK) { fprintf(stderr, "sampler failed: %s\n", ill_result_text(code)); goto done; }

    code = app_run_loop(model, state, sampler, &feed, opts->reply_span, 1, &run);
    if (code != ILL_OK) { fprintf(stderr, "\nrun failed: %s\n", ill_result_text(code)); goto done; }
    putchar('\n');
    if (opts->show_stats) app_run_note(&run);
    exit_code = 0;

done:
    app_feed_free(&feed);
    ill_sampler_free(sampler);
    ill_state_free(state);
    ill_model_free(model);
    return exit_code;
}

static int app_do_chat(const AppOpts *opts)
{
    IllModel   *model = NULL;
    IllState   *state = NULL;
    IllSampler *sampler = NULL;
    const IllVocab *vocab;
    IllResult   code = app_model_open(opts, &model);
    char        line[8192];
    int         opened = 0;
    int         exit_code = 1;

    if (code != ILL_OK) { fprintf(stderr, "load failed: %s\n", ill_result_text(code)); return 1; }
    vocab = ill_model_vocab(model);
    if (!vocab) { fprintf(stderr, "chat needs a tokenizer\n"); goto done; }

    code = ill_state_make(&state, model, opts->context_span);
    if (code != ILL_OK) { fprintf(stderr, "state failed: %s\n", ill_result_text(code)); goto done; }
    code = ill_sampler_make(&sampler, &opts->tuning, ill_model_arch(model)->vocab_size);
    if (code != ILL_OK) { fprintf(stderr, "sampler failed: %s\n", ill_result_text(code)); goto done; }

    printf("chat ready -- blank line or ctrl-d exits, /reset clears the window\n");
    for (;;) {
        AppFeed feed;
        AppRun  run;
        IllTurn turns[2];
        int32_t count = 0;
        char   *text;
        size_t  need = 0;

        fputs("\n> ", stdout);
        fflush(stdout);
        if (!fgets(line, (int)sizeof(line), stdin)) break;
        line[strcspn(line, "\r\n")] = '\0';
        if (!line[0]) break;
        if (!strcmp(line, "/reset")) {
            ill_state_reset(state);
            ill_sampler_wipe(sampler);
            opened = 0;
            printf("[window cleared]\n");
            continue;
        }

        if (!opened && opts->system_text) {
            turns[count].role = "system";
            turns[count].text = opts->system_text;
            ++count;
        }
        turns[count].role = "user";
        turns[count].text = line;
        ++count;

        ill_vocab_prompt(vocab, turns, count, 1, NULL, 0, &need);
        text = (char *)ill_block_make(need + 1);
        if (!text) break;
        ill_vocab_prompt(vocab, turns, count, 1, text, need + 1, &need);

        memset(&feed, 0, sizeof(feed));
        code = app_feed_text(&feed, vocab, text, !opened);
        ill_block_free(text);
        if (code != ILL_OK) { fprintf(stderr, "encode failed: %s\n", ill_result_text(code)); break; }
        opened = 1;

        code = app_run_loop(model, state, sampler, &feed, opts->reply_span, 1, &run);
        app_feed_free(&feed);
        if (code != ILL_OK) { fprintf(stderr, "\nrun failed: %s\n", ill_result_text(code)); break; }
        putchar('\n');
        if (opts->show_stats) app_run_note(&run);
    }
    exit_code = 0;

done:
    ill_sampler_free(sampler);
    ill_state_free(state);
    ill_model_free(model);
    return exit_code;
}

static int app_do_logits(const AppOpts *opts)
{
    IllModel *model = NULL;
    IllState *state = NULL;
    AppFeed   feed;
    IllBatch  batch;
    float    *logits = NULL;
    float    *stream = NULL;
    IllResult code = app_model_open(opts, &model);
    int32_t   rows = 0, vocab_size, index, slot;
    int       exit_code = 1;

    if (code != ILL_OK) { fprintf(stderr, "load failed: %s\n", ill_result_text(code)); return 1; }
    memset(&feed, 0, sizeof(feed));

    code = app_feed_make(&feed, opts, model, opts->prompt_text);
    if (code != ILL_OK) { fprintf(stderr, "prompt failed: %s\n", ill_result_text(code)); goto done; }
    code = ill_state_make(&state, model, opts->context_span);
    if (code != ILL_OK) { fprintf(stderr, "state failed: %s\n", ill_result_text(code)); goto done; }

    vocab_size = ill_model_arch(model)->vocab_size;

    if (opts->stream_mode) {
        /* Prefill once, then advance a token at a time, so the key/value and
         * stride caches carry the sequence rather than a single wide batch. */
        int32_t lead = ILL_MIN(ILL_MAX(opts->prefill_span, 1), feed.count);
        int32_t at;
        rows = feed.count - lead + 1;
        stream = (float *)ill_block_make((size_t)rows * vocab_size * sizeof(float));
        if (!stream) { code = ILL_ALLOC; goto done; }
        batch.tokens = feed.tokens; batch.count = lead; batch.every = 0;
        code = ill_model_apply(model, state, &batch, &logits);
        if (code != ILL_OK) { fprintf(stderr, "apply failed: %s\n", ill_result_text(code)); goto done; }
        memcpy(stream, logits, (size_t)vocab_size * sizeof(float));
        for (at = lead; at < feed.count; ++at) {
            batch.tokens = feed.tokens + at; batch.count = 1; batch.every = 0;
            code = ill_model_apply(model, state, &batch, &logits);
            if (code != ILL_OK) { fprintf(stderr, "apply failed: %s\n", ill_result_text(code)); goto done; }
            memcpy(stream + (size_t)(at - lead + 1) * vocab_size, logits,
                   (size_t)vocab_size * sizeof(float));
        }
        logits = stream;
    } else {
        batch.tokens = feed.tokens;
        batch.count  = feed.count;
        batch.every  = opts->every_logit;
        code = ill_model_apply(model, state, &batch, &logits);
        if (code != ILL_OK) { fprintf(stderr, "apply failed: %s\n", ill_result_text(code)); goto done; }
        rows = opts->every_logit ? feed.count : 1;
    }

    if (opts->out_path) {
        FILE *sink = fopen(opts->out_path, "wb");
        if (!sink) { fprintf(stderr, "cannot write %s\n", opts->out_path); goto done; }
        fwrite(logits, sizeof(float), (size_t)rows * vocab_size, sink);
        fclose(sink);
        fprintf(stderr, "wrote %d x %d logits to %s\n", rows, vocab_size, opts->out_path);
    } else {
        for (index = 0; index < rows; ++index) {
            for (slot = 0; slot < vocab_size; ++slot)
                printf("%s%.6f", slot ? " " : "", (double)logits[(size_t)index * vocab_size + slot]);
            putchar('\n');
        }
    }
    exit_code = 0;

done:
    ill_block_free(stream);
    app_feed_free(&feed);
    ill_state_free(state);
    ill_model_free(model);
    return exit_code;
}

/* -- perplexity ------------------------------------------------------------
 *
 * How surprised the model is by a text it did not write, in one number.  The
 * engine had no such number: `--quant q8` was reported as a correlation
 * against the reference, which says the two agree with each other and not what
 * either is worth, so nothing that trades accuracy for speed could be landed
 * on evidence.
 *
 * The score is the mean negative log likelihood the model assigns to each
 * token given everything before it, and perplexity is its exponent.  Every
 * token but the first is scored, in one left to right pass, so the context a
 * token is judged on is the whole text up to it rather than a window.  The
 * pass is cut into chunks only because asking for every row of a long text at
 * once would allocate the vocabulary once per token.
 * ------------------------------------------------------------------------*/

/* Reads all of a stream into one heap block.  Returns NULL on failure. */
static char *app_slurp(FILE *source)
{
    size_t room = 65536, fill = 0;
    char  *text = (char *)ill_block_make(room);
    if (!text) return NULL;
    for (;;) {
        size_t got = fread(text + fill, 1, room - fill - 1, source);
        fill += got;
        if (fill + 1 < room) break;
        {
            char *grown = (char *)ill_block_make(room * 2);
            if (!grown) { ill_block_free(text); return NULL; }
            memcpy(grown, text, fill);
            ill_block_free(text);
            text = grown;
            room *= 2;
        }
    }
    text[fill] = '\0';
    return text;
}

static int app_do_perplexity(const AppOpts *opts)
{
    IllModel *model = NULL;
    IllState *state = NULL;
    char     *body  = NULL;
    AppFeed   feed;
    IllResult code  = app_model_open(opts, &model);
    int32_t   vocab_size, chunk, done, scored = 0;
    double    total = 0.0;
    int       exit_code = 1;

    if (code != ILL_OK) { fprintf(stderr, "load failed: %s\n", ill_result_text(code)); return 1; }
    memset(&feed, 0, sizeof(feed));
    vocab_size = ill_model_arch(model)->vocab_size;

    /* The text is scored as it stands.  Wrapping it in the chat template would
     * score the template's own tokens as well, which is not what the number is
     * for, so `perplexity` does not shape a prompt even without `--raw`. */
    if (opts->token_text) {
        if (!app_feed_ids(&feed, opts->token_text)) { code = ILL_ALLOC; goto done; }
    } else {
        const IllVocab *vocab = ill_model_vocab(model);
        if (!vocab) {
            fprintf(stderr, "this checkpoint has no readable tokenizer; use --tokens\n");
            goto done;
        }
        body = opts->prompt_text ? NULL : app_slurp(stdin);
        if (!opts->prompt_text && !body) { code = ILL_ALLOC; goto done; }
        code = app_feed_text(&feed, vocab, opts->prompt_text ? opts->prompt_text : body, 1);
        if (code != ILL_OK) { fprintf(stderr, "encode failed: %s\n", ill_result_text(code)); goto done; }
    }

    if (feed.count < 2) {
        fprintf(stderr, "perplexity needs at least two tokens, got %d\n", feed.count);
        goto done;
    }
    if (feed.count > opts->context_span) {
        fprintf(stderr, "text is %d tokens and the window is %d; raise --ctx\n",
                feed.count, opts->context_span);
        goto done;
    }

    code = ill_state_make(&state, model, opts->context_span);
    if (code != ILL_OK) { fprintf(stderr, "state failed: %s\n", ill_result_text(code)); goto done; }

    chunk = opts->batch_span > 0 ? opts->batch_span : 256;
    for (done = 0; done < feed.count; ) {
        int32_t span = ILL_MIN(chunk, feed.count - done);
        float  *rows = NULL;
        IllBatch batch;
        int32_t at;
        batch.tokens = feed.tokens + done;
        batch.count  = span;
        batch.every  = 1;
        code = ill_model_apply(model, state, &batch, &rows);
        if (code != ILL_OK) { fprintf(stderr, "apply failed: %s\n", ill_result_text(code)); goto done; }
        /* Row `at` predicts the token after it, so the last token of the text
         * has no row to be scored against and the last row of a chunk is
         * scored against the first token of the next one. */
        for (at = 0; at < span; ++at) {
            const float *row = rows + (size_t)at * vocab_size;
            int32_t next = done + at + 1;
            if (next >= feed.count) break;
            total += ill_row_logsum(row, vocab_size) - (double)row[feed.tokens[next]];
            ++scored;
        }
        done += span;
    }

    printf("tokens    %d scored of %d\n", scored, feed.count);
    printf("weights   %s\n", opts->weight_type == ILL_TYPE_Q8 ? "q8" : "as stored");
    printf("nll       %.4f nats a token\n", total / (double)scored);
    printf("bits      %.4f a token\n", total / (double)scored / log(2.0));
    printf("perplexity %.4f\n", exp(total / (double)scored));
    exit_code = 0;

done:
    ill_block_free(body);
    app_feed_free(&feed);
    ill_state_free(state);
    ill_model_free(model);
    return exit_code;
}

static int app_do_bench(const AppOpts *opts)
{
    IllModel *model = NULL;
    IllState *state = NULL;
    IllResult code = app_model_open(opts, &model);
    int32_t  *feed = NULL;
    int32_t   fill = opts->batch_span > 0 ? opts->batch_span : 256;
    int32_t   steps = opts->reply_span > 0 ? opts->reply_span : 64;
    int32_t   index, vocab_size;
    IllBatch  batch;
    float    *logits = NULL;
    double    mark, fill_secs, step_secs;
    int       exit_code = 1;

    if (code != ILL_OK) { fprintf(stderr, "load failed: %s\n", ill_result_text(code)); return 1; }
    vocab_size = ill_model_arch(model)->vocab_size;
    if (fill + steps > opts->context_span) fill = opts->context_span - steps;
    if (fill < 1) fill = 1;

    feed = (int32_t *)ill_block_make((size_t)fill * sizeof(int32_t));
    if (!feed) goto done;
    for (index = 0; index < fill; ++index) feed[index] = (index * 7919 + 13) % vocab_size;

    code = ill_state_make(&state, model, opts->context_span);
    if (code != ILL_OK) { fprintf(stderr, "state failed: %s\n", ill_result_text(code)); goto done; }

    batch.tokens = feed; batch.count = fill; batch.every = 0;
    mark = ill_clock_now();
    code = ill_model_apply(model, state, &batch, &logits);
    fill_secs = ill_clock_now() - mark;
    if (code != ILL_OK) { fprintf(stderr, "prefill failed: %s\n", ill_result_text(code)); goto done; }

    mark = ill_clock_now();
    for (index = 0; index < steps; ++index) {
        int32_t next = (index * 104729 + 7) % vocab_size;
        batch.tokens = &next; batch.count = 1; batch.every = 0;
        code = ill_model_apply(model, state, &batch, &logits);
        if (code != ILL_OK) { fprintf(stderr, "decode failed: %s\n", ill_result_text(code)); goto done; }
    }
    step_secs = ill_clock_now() - mark;

    printf("weights   %.2f GiB (%s)\n", (double)ill_model_bytes(model) / 1073741824.0,
           opts->weight_type == ILL_TYPE_Q8 ? "q8" : "as stored");
    printf("state     %.2f GiB for %d tokens\n",
           (double)ill_state_bytes(state) / 1073741824.0, opts->context_span);
    printf("backend   %s / %s, %d threads\n", ill_model_backend(model),
           ill_backend_simd(), ill_model_threads(model));
    printf("prefill   %d tok in %.3f s -> %.1f tok/s\n", fill, fill_secs,
           fill_secs > 0.0 ? fill / fill_secs : 0.0);
    printf("decode    %d tok in %.3f s -> %.1f tok/s\n", steps, step_secs,
           step_secs > 0.0 ? steps / step_secs : 0.0);
    exit_code = 0;

done:
    ill_block_free(feed);
    ill_state_free(state);
    ill_model_free(model);
    return exit_code;
}

/* -- entry ----------------------------------------------------------------- */

int main(int argc, char **argv)
{
    AppOpts opts;
    const char *verb;

    if (argc < 2) { app_help(); return 1; }
    verb = argv[1];
    if (!strcmp(verb, "help") || !strcmp(verb, "--help") || !strcmp(verb, "-h")) {
        app_help();
        return 0;
    }

    app_opts_init(&opts);
    if (!app_opts_read(&opts, argc, argv, 2)) return 1;

    if (!strcmp(verb, "info"))     return app_do_info(&opts);
    if (!strcmp(verb, "tokens"))   return app_do_tokens(&opts);
    if (!strcmp(verb, "generate")) return app_do_generate(&opts);
    if (!strcmp(verb, "chat"))     return app_do_chat(&opts);
    if (!strcmp(verb, "logits"))   return app_do_logits(&opts);
    if (!strcmp(verb, "bench"))    return app_do_bench(&opts);
    if (!strcmp(verb, "perplexity")) return app_do_perplexity(&opts);

    fprintf(stderr, "unknown command: %s\n", verb);
    app_help();
    return 1;
}

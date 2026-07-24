# Parallelism in this codebase

This document explains every mechanism in this repository (and its related helper
scripts) that runs work concurrently, why each one exists, and the specific bugs it was
built to avoid. It complements `PROJECT_OVERVIEW.md` (architecture) and `IFBO_METHOD.md`
(optimizer algorithm) — read those first if you need the surrounding context. It reflects
the code on branch `dev-bilstm-only` as of 2026-07-24.

There are three independent layers of concurrency in this project, and they interact:

1. **Trial-level parallelism** — running several HPO trials (configs) at once, one per
   GPU. Threads, coordinated by `IfboOptimizer`.
2. **Data-loading parallelism** — PyTorch `DataLoader` worker subprocesses that prefetch
   and collate batches off the main training thread.
3. **Process-level parallelism** — separate OS processes launched by the user (e.g. two
   `make run` invocations, or a training run alongside `ifbo_predict.py`) that never
   share Python state and only touch shared *files* on disk.

Only (1) is genuinely new/interesting code; (2) is mostly PyTorch defaults tuned
carefully to survive (1); (3) is just "don't corrupt shared files," handled with file
locks.

---

## 1. Trial-level parallelism: `num_parallel_trials`

### Where

`automl/core/optimizers/ifbo/optimizer.py` — `_perform_ifbo_parallel`, `_step_on_device`,
`_step`. Controlled by the `num_parallel_trials` runtime-config key (CLI:
`--num-parallel-trials`, default `1` = fully sequential, current `runconfig.yml` sets it
to `2`).

Only the ifBO optimizer supports this. `smac`, `random`, and `rl_freeze_thaw` all train
one configuration at a time.

### What it does

ifBO's freeze-thaw loop normally does, one trial at a time: build FT-PFN context from
history → pick the single best candidate to extend → train it for a few more epochs →
repeat. With `num_parallel_trials = N > 1`, `_perform_ifbo_parallel` instead:

1. Builds the FT-PFN context **once** for the round.
2. Greedily selects up to `N` *distinct* candidates against that same context
   (candidates 2..N in a round are picked slightly "stale" relative to candidate 1 —
   accepted as standard batch-BO staleness, not a bug).
3. Submits all `N` `_step_on_device` calls to a `concurrent.futures.ThreadPoolExecutor`
   with `max_workers=N`, each one pinned to a device from `self.devices` (round-robin if
   `N` exceeds the visible device count).
4. Blocks (`fut.result()` for every future) until the whole round finishes, then loops.

This is **threads**, not processes — Python's GIL is not a bottleneck here because each
trial spends the overwhelming majority of its time inside PyTorch/CUDA kernels, which
release the GIL. The GIL only serializes the small amount of pure-Python bookkeeping
(config sampling, bookkeeping, logging), which is cheap by comparison. Using threads
(instead of `multiprocessing`) also means every trial shares the same process memory —
no need to pickle configs/results across a process boundary, and caches (see §2) can be
shared for free.

### Device assignment (`self.devices`)

`Optimizer._resolve_devices` (`automl/core/optimizers/base_optimizer.py`) builds the
device pool once at optimizer construction:

- If `device` is set explicitly in the runtime config, it's the sole device — parallel
  trials would all round-robin onto that one device, which still works but gives no
  speedup (just interleaving on the same GPU/CPU).
- If `device == "auto"` (the default) and CUDA is available, it expands to **every**
  visible `cuda:i`, so `num_parallel_trials` trials naturally map one-to-one onto GPUs
  when `N <= device_count`.
- Otherwise it falls back to the single best non-CUDA device (`mps` or `cpu`).

Each worker thread calls `torch.cuda.set_device(device)` itself
(`_step_on_device`) before training, because PyTorch's "current CUDA device" is
**thread-local**, not inherited from the thread that created the pool — without this,
implicit current-device ops inside the trainer (e.g. `torch.cuda.empty_cache()`) would
silently all target `cuda:0` regardless of which GPU a trial is actually using.

### Shared mutable state and locking

Everything a concurrent trial can touch that isn't local to that trial's own call stack
is guarded:

| State | Guard | Why |
|---|---|---|
| `self.history`, `self.trial_no`, `self.best_val_error`, `self.highest_budget_seen`, "best" checkpoint write | `self._state_lock` (`threading.Lock`) | Multiple trials finishing around the same time must not race on read-modify-write of shared counters/incumbent tracking. A no-op when only one trial runs, so it's always safe to hold. |
| Per-config trainer checkpoint file (freeze-thaw resume state) | one `threading.Lock` per `config_id`, created lazily under `_state_lock` via `_checkpoint_lock_for` | Two *different* configs training concurrently must not block each other, but two trials that happen to share a config hash (shouldn't normally happen, but is possible after freeze-thaw restarts) must not corrupt the same checkpoint file. |
| `history.log.jsonl` append | `filelock.FileLock` on a `.lock` sidecar file (`_append_trial_to_jsonl`) | Survives both concurrent **threads** in this process and concurrent **processes** (see §3) appending to the same file — `threading.Lock` alone wouldn't cover the multi-process case. |
| RNG draw for a trial's seed (`self._rng.randint(...)`) | `self._state_lock` | `self._rng` is one shared `random.Random` instance; concurrent `.randint()` calls on it aren't safe. |
| `set_seed(seed)` (reseeds global torch/numpy/random RNGs) | `self._state_lock` | Only the reseed call itself is locked — cheap. `approach.prepare()` right after it is deliberately **not** locked even though it reads RNG state (default weight init), because `prepare()` also does the expensive CPU-bound work (tokenizing the corpus, building the model, PCA-projecting pretrained embeddings). Serializing that behind a lock would starve every GPU but one, defeating the purpose of running trials concurrently. |

The `set_seed`/`prepare()` trade-off means that with `num_parallel_trials > 1`, exact
"same seed → same result" reproducibility no longer holds (a concurrently-running
trial's `set_seed` can land between this trial's seeding and its own weight init). This
is an accepted trade-off, documented inline in `base_optimizer.py`: ifBO already treats
observed accuracy as noisy regardless of exact reproducibility.

### The `DataLoader(num_workers>0)` + `filelock` + fork interaction

This is the trickiest bug this codebase works around, so it's worth spelling out in
full (see the long comment block in `ifbo/optimizer.py` around `_effective_num_workers`):

- `torch.utils.data.DataLoader(num_workers=k>0)` spawns `k` worker subprocesses. On
  Linux, the default multiprocessing start method is `"fork"` — i.e. `os.fork()`.
- `filelock` (Python 3.12+) actively **refuses** to let a fork happen while *any*
  `FileLock` in the process is mid-acquire/release, to avoid the child inheriting a
  half-modified file descriptor table.
- With several ifBO trials running concurrently on separate threads, one thread can
  easily be inside `_append_trial_to_jsonl`'s `FileLock` at the exact moment another
  thread's `DataLoader` tries to fork its workers — raising `RuntimeError: os.fork() was
  called ... filelock is changing descriptor ownership`.
- Rather than just reducing the worker count (which still forks, just less often — the
  race is still there), the fix forces `num_workers` to **0** whenever
  `num_parallel_trials > 1` (`self._effective_num_workers`). Data loading then happens
  synchronously in each trial's own thread, sidestepping forking — and this entire class
  of bug — completely.
- This is cheap to accept because tokenization already happens once, upfront, in
  `TextSequenceDataset.__init__` (see §2's encoding cache) rather than per-batch, so the
  throughput lost from having no `DataLoader` worker processes is small.

### Thread-local tokenizer cache

`sequence_dl.py`'s `_load_tokenizer` used to be a single `functools.lru_cache`-wrapped
tokenizer shared by the whole process — safe only when calls are sequential. A fast
(Rust-backed) HuggingFace tokenizer mutates its own truncation/padding config in place
on every call (`max_length` varies per trial's sampled `max_seq_length`), so two trials
tokenizing concurrently on separate threads raced on that shared mutable state and
crashed with `RuntimeError: Already borrowed`.

The fix: `_tokenizer_cache = threading.local()`, so each thread gets and keeps its own
private tokenizer instance — one load per thread, reused across every trial that thread
ever runs, with no cross-thread mutation race.

### Prewarming shared caches

`_prewarm_shared_resources`, called once at optimizer construction when
`num_parallel_trials > 1`, loads the dataset and (for `sequence-dl`) the pretrained word
embedding matrix *before* the first parallel round starts. Both are lazily cached behind
a plain check-then-set (not lock-protected) the first time they're touched; without this
prewarm, the very first parallel batch of trials would race to populate that cache and
redundantly repeat expensive work (parsing the full dataset, downloading/loading a
pretrained transformer). Note this does **not** cover the tokenizer cache above — that
one is thread-local by design, so warming it on the main thread wouldn't help any worker
thread; each one loads its own on first use instead.

### Per-round GPU memory cleanup

Both the sequential and parallel ifBO loops run `gc.collect()` +
`torch.cuda.empty_cache()` / `torch.mps.empty_cache()` after each unit of work — but the
parallel loop does it **once per round** (after all `N` concurrent trials in the round
finish), not once per individual trial, since doing it mid-round would fight over device
state with trials still running on other threads.

---

## 2. Data-loading parallelism: `DataLoader(num_workers=...)`

### Where

`automl/core/approaches/sequence_dl.py` — `train_loader` / `val_loader` construction.
Controlled by the `num_workers` runtime-config key (CLI `--num-workers`, default `2`).

### What it does

Standard PyTorch `DataLoader` worker-process parallelism: with `num_workers=k>0`, `k`
subprocesses independently call `Dataset.__getitem__` and run `collate_fn`, feeding
batches back to the main process through a queue, so CPU-side batch prep overlaps with
GPU compute in the main thread instead of blocking it. `persistent_workers=True` is set
whenever `num_workers>0` so worker processes survive across epochs instead of being
torn down and respawned each time. `pin_memory` is enabled whenever training on CUDA,
so the host→device copy can be asynchronous.

As covered in §1, this is **forced to 0** whenever `num_parallel_trials > 1`, to avoid
the fork/filelock race. In the current `runconfig.yml` (`num_parallel_trials: 2`), the
configured `num_workers: 1` is therefore overridden to 0 at runtime — data loading runs
inline on each trial's own thread.

### Why the impact is small

`TextSequenceDataset` (same file) stores every tokenized sequence back-to-back in one
contiguous `int32` tensor + offsets, rather than as a Python list of per-sample tensors —
specifically to avoid the classic `DataLoader`-multiprocessing pitfall where touching
many individual Python objects' refcounts inside a forked worker process forces the OS to
copy-on-write pages that were otherwise shared with the parent, silently multiplying
memory usage by roughly `num_workers`. The same rationale applies to the label tensor.
Combined with the fact that tokenization is a one-time upfront cost (see below), losing
worker processes when trials run in parallel is a small, deliberate trade-off rather than
a real bottleneck.

### The tokenization cache (not parallelism per se, but load-bearing for it)

`_encode_texts_cached` (module-level `dict[str, dict[str, list[int]]]`, keyed by
tokenizer path then by raw text, guarded by `_full_encoding_cache_lock`, a plain
`threading.Lock`) caches the full, untruncated token ids for every text ever seen. ifBO
resamples train/val from the same fixed underlying pool of texts every trial, so after
the first trial nearly every text is already cached — subsequent trials only pay for a
dict lookup, then per-sample truncation to that trial's own `max_seq_length`. This is
what makes forcing `num_workers=0` under parallel trials cheap: the expensive part
(tokenizing raw text) isn't happening per-batch or per-worker anyway.

`_load_pretrained_word_embeddings` (`functools.lru_cache(maxsize=1)`) and its downstream
`_pretrained_svd` / `_pretrained_embedding_init` (`lru_cache`-wrapped) follow the same
"compute once, shared across every trial and thread" pattern for the BiLSTM's pretrained
embedding warm-start — safe under multiple threads because `lru_cache` itself is
thread-safe (guarded internally by a lock), and the underlying computation is
side-effect-free.

---

## 3. Process-level parallelism (outside a single `python -m automl` run)

### Where

The `Makefile`'s `run-all` target, and running independent scripts (`run_config.py`,
`ifbo_predict.py`, `train_top_k_from_history.py`, etc.) by hand.

### What it does — and deliberately does *not* do

`make run-all` iterates `DATASETS` (`amazon ag_news imdb dbpedia`) in a shell `for` loop
and runs `python -m automl` **sequentially**, one dataset fully finishing before the next
starts (`|| exit $$?` stops the whole loop on the first failure). There is no
`make -j`/background-`&` parallel-dataset mode — this is intentional, since each run
already saturates all visible GPUs on its own via `num_parallel_trials` (§1), so running
two datasets' optimizers at once would just make them contend for the same devices.

If a user *does* launch two independent runs by hand (e.g. two terminals, or one dataset
run plus a `run_incumbant.py`/`ifbo_predict.py` invocation reading the same output
directory), those are separate OS processes with no shared Python state at all — the
only thing that can go wrong is two processes writing to the same file at once. That's
exactly what `FileLock` on `history.log.jsonl` (§1) protects against; it works across
both threads *and* processes for that reason. Other outputs are per-run directories
already keyed by dataset/`runtime_id`
(`results/<dataset>/...`), so independent runs on different datasets don't collide by
construction.

---

## 4. Summary: what to change, and what happens if you do

| Knob | Effect | Trade-off |
|---|---|---|
| `num_parallel_trials: 1` (default) | Fully sequential ifBO, one trial at a time. `num_workers` used as configured. | Simplest, fully reproducible per-seed, slowest wall-clock. |
| `num_parallel_trials: N > 1` | `N` trials train concurrently, one per GPU (round-robin if `N > device_count`). | `DataLoader` workers forced to 0 (§1). Exact seed reproducibility lost (§1). Needs `N` GPUs' worth of memory headroom at once. |
| `num_workers: k` | `DataLoader` uses `k` prefetch subprocesses — only takes effect when `num_parallel_trials == 1`. | Higher `k` overlaps CPU batch-prep with GPU compute better, at the cost of `k` extra processes' memory (mitigated by the contiguous-tensor dataset layout, §2). |
| `device: auto` vs explicit | `auto` expands to all visible CUDA devices (enables real speedup from `num_parallel_trials`); an explicit device pins every trial to it (parallel trials still work, just interleave on one device instead of running truly concurrently). | |

Only `ifbo` currently benefits from `num_parallel_trials` — it's read but has no effect
under `smac`, `random`, or `rl_freeze_thaw`, which are single-trial-at-a-time by
construction.

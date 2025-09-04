import logging
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Optional, Sequence
import os

from wisent_guard.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet

try:
    from wisent_guard.core.download_full_benchmarks import FullBenchmarkDownloader  
except Exception: 
    FullBenchmarkDownloader = None  

# --- Allowed tasks (must exist; fail fast with a clear error) --------------------
try:
    from wisent_guard.parameters.task_config import ALLOWED_TASKS 
except Exception as exc:  
    raise ImportError(
        "Failed to import ALLOWED_TASKS from wisent_guard.parameters.task_config. "
        "Ensure your project is installed and PYTHONPATH is set correctly."
    ) from exc



logger = logging.getLogger(__name__) 


# --- Resolvers for CORE_BENCHMARKS and UNAVAILABLE_BENCHMARKS --------------------
def _resolve_benchmark_catalogs() -> tuple[dict[str, dict[str, Any]], Sequence[str]]:
    """Resolve CORE_BENCHMARKS and UNAVAILABLE_BENCHMARKS with fallbacks.

    Returns:
        tuple[dict[str, dict[str, Any]], Sequence[str]]: (CORE_BENCHMARKS, UNAVAILABLE_BENCHMARKS)
    """
    # 1) Primary import path
    try:
        from wisent_guard.core.lm_harness_integration.only_benchmarks import CORE_BENCHMARKS 
        try:
            # Prefer canonical source for unavailable set if available
            if FullBenchmarkDownloader is not None:
                UNAVAILABLE_BENCHMARKS = getattr(FullBenchmarkDownloader, "UNAVAILABLE_BENCHMARKS", set())
            else:
                UNAVAILABLE_BENCHMARKS = set()
        except Exception:  
            UNAVAILABLE_BENCHMARKS = set()
        return CORE_BENCHMARKS, UNAVAILABLE_BENCHMARKS
    except Exception:  
        pass

    # 2) Minimal fallback — keep the pipeline usable without full deps
    logger.warning(
        "Could not import CORE_BENCHMARKS; using minimal fallback list (reduced coverage)."
    )
    CORE_BENCHMARKS = {
        "truthfulqa_mc1": {"task": "truthfulqa_mc1", "tags": ["hallucination"], "priority": "high"},
        "hellaswag": {"task": "hellaswag", "tags": ["reasoning"], "priority": "high"},
        "mmlu": {"task": "mmlu", "tags": ["knowledge"], "priority": "high"},
    }
    UNAVAILABLE_BENCHMARKS = set()
    return CORE_BENCHMARKS, UNAVAILABLE_BENCHMARKS


CORE_BENCHMARKS, UNAVAILABLE_BENCHMARKS = _resolve_benchmark_catalogs()

AVAILABLE_BENCHMARKS: dict[str, dict[str, Any]] = {
    name: cfg
    for name, cfg in CORE_BENCHMARKS.items()
    if name not in UNAVAILABLE_BENCHMARKS and name in ALLOWED_TASKS
}

DEFAULT_TRAIN_CAP = 1000
DEFAULT_TEST_CAP = 200
SLOW_GROUP_TASKS: set[str] = {"livecodebench"}

@dataclass(frozen=True)
class Caps:
    train: int
    test: int

@dataclass
class PrepState:
    qa_pairs: list[dict[str, Any]]
    test_source: list[Any]
    group_processed: bool
    group_qa_format: bool
    task_data: Any
    train_docs: list[Any]
    skip_qa_display: bool
    used_cache: bool
    all_cached_pairs: list[dict[str, Any]]

def _log(msg: str, verbose: bool) -> None:
    """Route messages through logging; fall back to silent if not verbose.

    Args:
        msg: Message to emit.
        verbose: Whether to emit user-facing progress logs.
    """
    if verbose:
        logger.info(msg)

def _get_cache_path(task_name: str, cache_dir: str) -> str:
    """Return the filesystem path of the cached benchmark pickle.

    Args:
        task_name: Benchmark/task identifier.
        cache_dir: Root directory for cache files.

    Returns:
        Absolute path to the cache file (under ``<cache_dir>/data/<task>.pkl``).
    """
    # Match the structure used by FullBenchmarkDownloader which saves to data/ subdirectory.
    return os.path.join(cache_dir, "data", f"{task_name}.pkl")


def _is_benchmark_cached(task_name: str, cache_dir: str) -> bool:
    """Check whether a benchmark cache file exists.

    Args:
        task_name: Benchmark/task identifier.
        cache_dir: Root directory for cache files.

    Returns:
        True if the cache file exists; False otherwise.
    """
    return os.path.exists(_get_cache_path(task_name, cache_dir))


def _load_cached_benchmark(task_name: str, cache_dir: str) -> Optional[list[dict[str, Any]]]:
    """Load cached benchmark data from disk.

    Args:
        task_name: Benchmark/task identifier.
        cache_dir: Root directory for cache files.

    Returns:
        The cached list of contrastive samples, or None if missing / failed to load.
    """
    cache_path = _get_cache_path(task_name, cache_dir)
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:  
        logger.warning("Failed to load cached data at %s: %s", cache_path, exc)
        return None


def _get_actual_task_name(benchmark_name: str) -> str:
    """Map a benchmark label to its lm-eval-harness task name.

    Args:
        benchmark_name: A human-friendly benchmark name (key in AVAILABLE_BENCHMARKS).

    Returns:
        The corresponding lm-eval task name (falls back to the input name).
    """
    if benchmark_name in AVAILABLE_BENCHMARKS:
        return AVAILABLE_BENCHMARKS[benchmark_name].get("task", benchmark_name)
    return benchmark_name


def _convert_cached_data_to_qa_pairs(
    cached_data: list[dict[str, Any]], limit: Optional[int] = None
) -> list[dict[str, Any]]:
    """Convert stored contrastive samples to QA-pair format.

    The cached format is expected to be:
      ``{"context": str, "good_response": str, "bad_response": str, "metadata": dict}``.

    Args:
        cached_data: list of serialized contrastive samples.
        limit: Optional maximum number of items to return.

    Returns:
        list of QA-pair dicts with keys: ``question``, ``correct_answer``,
        ``incorrect_answer``, ``metadata``.
    """
    qa_pairs: list[dict[str, Any]] = []
    for sample in cached_data:
        qa_pairs.append(
            {
                "question": sample.get("context", ""),
                "correct_answer": sample.get("good_response", ""),
                "incorrect_answer": sample.get("bad_response", ""),
                "metadata": sample.get("metadata", {}),
            }
        )
        if limit and len(qa_pairs) >= limit:
            break
    return qa_pairs


def _save_benchmark_to_cache(
    task_name: str, cache_dir: str, verbose: bool = False
) -> bool:
    """Download and persist a benchmark into the cache directory.

    This uses the ``FullBenchmarkDownloader`` if available. If the downloader
    is unavailable or the task is not in ``AVAILABLE_BENCHMARKS``, this function
    returns ``False``.

    Args:
        task_name: Benchmark/task identifier.
        cache_dir: Root cache directory (a ``data/`` subfolder will be used).
        verbose: When True, emits INFO logs for progress.

    Returns:
        True if the benchmark was cached successfully; False otherwise.
    """

    if FullBenchmarkDownloader is None:
        _log("Warning: Download infrastructure not available; cannot cache.", verbose)
        return False

    if task_name not in AVAILABLE_BENCHMARKS:
        _log(f"Warning: {task_name!r} not in AVAILABLE_BENCHMARKS; skipping cache.", verbose)
        return False

    # Ensure the standard layout exists: <cache_dir>/data/
    os.makedirs(os.path.join(cache_dir, "data"), exist_ok=True)

    try:
        _log(f"Caching benchmark {task_name!r} ...")
        downloader = FullBenchmarkDownloader(download_dir=cache_dir)
        benchmark_config = AVAILABLE_BENCHMARKS[task_name]

        # Force refresh to ensure cache is written
        result_path = downloader.download_complete_benchmark(
            task_name,
            benchmark_config,
            force=True,
        )
        if result_path:
            _log(f"Successfully cached {task_name!r} at {result_path}", verbose)
            return True

        _log(f"Failed to cache {task_name!r}: downloader returned no path.", verbose)
        return False
    except Exception as exc:  # noqa: BLE001
        _log(f"Error caching {task_name!r}: {exc}", verbose)
        return False

def _effective_caps(training_limit: Optional[int], testing_limit: Optional[int]) -> Caps:
    """Resolve explicit caps or use defaults."""
    return Caps(
        train=training_limit if training_limit is not None else DEFAULT_TRAIN_CAP,
        test=testing_limit if testing_limit is not None else DEFAULT_TEST_CAP,
    )

def _derived_total_limit(
    base_limit: Optional[int], train_lim: Optional[int], test_lim: Optional[int], split_ratio: float
) -> Optional[int]:
    """Compute upstream fetch size when only one side has a cap."""
    if train_lim is None and test_lim is None:
        return base_limit
    if train_lim is not None and test_lim is not None:
        return train_lim + test_lim
    if train_lim is not None:
        return int(train_lim / max(split_ratio, 1e-6) * 1.2) + 1
    return int(test_lim / max(1 - split_ratio, 1e-6) * 1.2) + 1

def _split_and_cap(
    items: list[dict[str, Any]],
    split_ratio: float,
    caps: Caps,
    seed: int,
    verbose: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically shuffle, split, and cap two partitions."""
    import random as _random
    rng = _random.Random(seed)
    data = list(items)
    rng.shuffle(data)
    pivot = int(len(data) * split_ratio)
    train, test = data[:pivot], data[pivot:]

    if len(train) > caps.train:
        train = train[:caps.train]
        _log(f"Training data limited to {caps.train} samples", verbose)
    if len(test) > caps.test:
        test = test[:caps.test]
        _log(f"Test data limited to {caps.test} samples", verbose)

    _log(f"Data split: {len(train)} training, {len(test)} test", verbose)
    return train, test

def _synthetic_mode(from_synth: bool, synth_pairs: Any, verbose: bool) -> Optional[PrepState]:
    if from_synth and synth_pairs:
        _log("\nSynthetic pair mode:", verbose)
        _log(f"• Training pairs: {len(synth_pairs.pairs)}", verbose)
        _log(f"• Task name: {synth_pairs.name}", verbose)
        return PrepState([], [], True, True, None, [], False, False, [])
    return None

def _cross_benchmark_mode(
    cross_mode: bool, train_pairs: Any, eval_pairs: Any, verbose: bool
) -> Optional[PrepState]:
    if cross_mode and train_pairs and eval_pairs:
        _log("\nCross-benchmark evaluation mode:", verbose)
        _log(f"• Training pairs: {len(train_pairs.pairs)}", verbose)
        _log(f"• Evaluation pairs: {len(eval_pairs.pairs)}", verbose)
        return PrepState([], [], True, True, None, [], True, False, [])
    return None

def _preloaded_mode(preloaded: Optional[Sequence[dict[str, Any]]], verbose: bool) -> Optional[PrepState]:
    if preloaded:
        _log(f"\nUsing pre-loaded mixed dataset with {len(preloaded)} QA pairs...", verbose)
        pairs = list(preloaded)
        return PrepState(pairs, list(preloaded), True, True, None, [], False, False, [])
    return None

def _load_from_csv_json(
    from_csv: bool,
    from_json: bool,
    task_name: str,
    question_col: str,
    correct_col: str,
    incorrect_col: str,
    limit: Optional[int],
    caps: Caps,
    split_ratio: float,
    seed: int,
    verbose: bool,
) -> Optional[PrepState]:
    if not (from_csv or from_json):
        return None

    _log(f"\nLoading data from {'CSV' if from_csv else 'JSON'} file...", verbose)
    if from_csv:
        pair_set = ContrastivePairSet.from_csv_file(
            name="csv_data",
            csv_path=task_name,
            question_col=question_col,
            correct_col=correct_col,
            incorrect_col=incorrect_col,
            limit=limit,
        )
    else:
        pair_set = ContrastivePairSet.from_json_file(name="json_data", json_path=task_name, limit=limit)

    raw_pairs: list[dict[str, Any]] = []
    for pair in pair_set.pairs:
        if hasattr(pair, "question"):
            raw_pairs.append(
                {"question": pair.question, "correct_answer": pair.correct_answer, "incorrect_answer": pair.incorrect_answer}
            )

    qa_pairs, test_source = _split_and_cap(raw_pairs, split_ratio, caps, seed, verbose)
    return PrepState(qa_pairs, test_source, True, True, None, [], False, False, [])

def _try_cached(
    task_name: str, cache_dir: str, limit: Optional[int], use_cached: bool, force_download: bool, verbose: bool
) -> tuple[bool, list[dict[str, Any]]]:
    if use_cached and not force_download and _is_benchmark_cached(task_name, cache_dir):
        _log(f"Found cached benchmark data for {task_name}", verbose)
        cached = _load_cached_benchmark(task_name, cache_dir)
        if cached:
            _log(f"Loaded {len(cached)} cached contrastive pairs", verbose)
            qa_pairs = _convert_cached_data_to_qa_pairs(cached, limit)
            _log(
                f"Converted to {len(qa_pairs)} QA pairs" + (f"\nLimited to {limit} pairs as requested"
                if limit and len(qa_pairs) >= limit else ""),
                verbose,
            )
            return True, qa_pairs
        _log("Failed to load cached data, falling back to fresh download", verbose)
    return False, []

def _is_group_task(task_name: str, verbose: bool) -> tuple[bool, list[str]]:
    if task_name in SLOW_GROUP_TASKS:
        _log(f"Skipping group task check for {task_name} (known to be slow)", verbose)
        return False, []
    try:
        import threading
        from lm_eval import evaluator

        _log(f"Checking if '{task_name}' is a group task...", verbose)
        task_dict, exc = {}, None

        def load():
            nonlocal task_dict, exc
            try:
                task_dict = evaluator.get_task_dict([task_name])
            except Exception as e: 
                exc = e

        t = threading.Thread(target=load, daemon=True)
        t.start()
        t.join(timeout=60)
        if t.is_alive():
            raise TimeoutError(f"Task loading timed out for {task_name}")
        if exc:
            raise exc
        expanded = list(task_dict.keys())
        return (len(expanded) > 1), expanded
    except TimeoutError as e:
        _log(f"Task loading timed out for '{task_name}': {e}\nProceeding with standard task loading...", verbose)
        return False, []
    except Exception as e: 
        if "Group task" in str(e):
            raise
        _log(f"Could not check if '{task_name}' is a group task: {e}\n🔄 Proceeding with standard task loading...", verbose)
        return False, []

def _process_group_task(
    expanded: list[str],
    model: Any,
    shots: int,
    limit: Optional[int],
    split_ratio: float,
    seed: int,
    caps: Caps,
    verbose: bool,
) -> PrepState:
    _log(
        f"Detected GROUP task with {len(expanded)} subtasks: {expanded[:5]}{'...' if len(expanded) > 5 else ''}\n"
        "Extracting samples from all subtasks...",
        verbose,
    )
    all_pairs: list[dict[str, Any]] = []
    for sub in expanded:
        try:
            _log(f"Loading subtask: {sub}", verbose)
            subtask_data = model.load_lm_eval_task(sub, shots=shots, limit=limit)
            sub_train_docs, _ = model.split_task_data(subtask_data, split_ratio=split_ratio, random_seed=seed)
            sub_pairs = ContrastivePairSet.extract_qa_pairs_from_task_docs(sub, subtask_data, sub_train_docs)
            _log(f"Extracted {len(sub_pairs)} pairs from {sub}", verbose)
            for p in sub_pairs:
                p["source_subtask"] = sub
            all_pairs.extend(sub_pairs)
        except Exception as e:  # noqa: BLE001
            _log(f"Failed to load subtask {sub}: {e}", verbose)
            continue
    if not all_pairs:
        raise ValueError("No QA pairs could be extracted from any subtasks.")

    qa_pairs, test_source = _split_and_cap(all_pairs, split_ratio, caps, seed, verbose=False)

    from collections import Counter
    _log(f"Combined dataset: {len(qa_pairs)} total QA pairs from {len(expanded)} subtasks", verbose)
    for sub, cnt in Counter(p.get("source_subtask", 'unknown') for p in qa_pairs).most_common():
        _log(f"• {sub}: {cnt} pairs", verbose)

    return PrepState(qa_pairs, test_source, True, True, None, [], False, False, [])

def _load_single_task(
    *,
    model: Any,
    task_name: str,
    shots: int,
    split_ratio: float,
    seed: int,
    limit: Optional[int],
    training_limit: Optional[int],
    testing_limit: Optional[int],
    cache_benchmark: bool,
    cache_dir: str,
    force_download: bool,
    livecodebench_version: str,
    verbose: bool,
) -> PrepState:
    actual = _get_actual_task_name(task_name)
    if actual != task_name:
        _log(f"Resolving benchmark '{task_name}' to task '{actual}'", verbose)

    total_limit = _derived_total_limit(limit, training_limit, testing_limit, split_ratio)
    extra: dict[str, Any] = {"release_version": livecodebench_version} if actual == "livecodebench" else {}

    task_data = model.load_lm_eval_task(actual, shots=shots, limit=total_limit, **extra)

    # Some tasks expose their own loader; otherwise split via model
    if hasattr(task_data, "get_name") and hasattr(task_data, "load_data"):
        train_docs = task_data.load_data(limit=training_limit)
        test_docs = task_data.load_data(limit=testing_limit)
    else:
        train_docs, test_docs = model.split_task_data(task_data, split_ratio=split_ratio, random_seed=seed)

    # Enforce explicit caps if requested
    if training_limit is not None and len(train_docs) > training_limit:
        _log(f"Training data limited to {training_limit} samples (from {len(train_docs)})", verbose)
        train_docs = train_docs[:training_limit]
    if testing_limit is not None and len(test_docs) > testing_limit:
        _log(f"Test data limited to {testing_limit} samples (from {len(test_docs)})", verbose)
        test_docs = test_docs[:testing_limit]

    _log(f"Data split: {len(train_docs)} training docs, {len(test_docs)} test docs", verbose)

    # Extract QA pairs (prefer managed caches if configured)
    if cache_benchmark:
        try:
            from wisent_guard.core.managed_cached_benchmarks import get_managed_cache
            managed_cache = get_managed_cache(cache_dir)
            cache_limit = _derived_total_limit(limit, training_limit, testing_limit, split_ratio)
            cached_samples = managed_cache.get_task_samples(
                task_name=task_name, limit=cache_limit or 1000, force_fresh=force_download
            )
            qa_pairs = [s["normalized"] for s in cached_samples]
            _log(f"Using managed cache: {len(qa_pairs)} samples loaded efficiently", verbose)
        except Exception as e: 
            _log(f"Managed cache failed, trying Full Benchmark Downloader cache: {e}", verbose)
            full_benchmark_file = Path(cache_dir) / "data" / f"{task_name}.pkl"
            qa_pairs = []
            if full_benchmark_file.exists():
                try:
                    with open(full_benchmark_file, "rb") as f:
                        bench = pickle.load(f)
                    qa_pairs = bench if isinstance(bench, list) else bench.get("contrastive_pairs", [])
                    _log(f"Using Full Benchmark Downloader cache: {len(qa_pairs)} samples loaded", verbose)
                except Exception as load_error:  # noqa: BLE001
                    _log(f"Full Benchmark Downloader cache failed: {load_error}", verbose)
            if not qa_pairs:
                _log("All cache methods failed, falling back to traditional extraction", verbose)
                qa_pairs = ContrastivePairSet.extract_qa_pairs_from_task_docs(task_name, task_data, train_docs)
    else:
        qa_pairs = ContrastivePairSet.extract_qa_pairs_from_task_docs(task_name, task_data, train_docs)

    return PrepState(qa_pairs, test_docs, False, False, task_data, train_docs, False, False, [])

def _maybe_persist_cache(
    cache_benchmark: bool,
    used_cache: bool,
    task_name: str,
    cache_dir: str,
    verbose: bool,
) -> None:
    if cache_benchmark and not used_cache:
        _log("Caching benchmark data for future use...", verbose)
        ok = _save_benchmark_to_cache(task_name, cache_dir, verbose)
        _log("Benchmark cached successfully!" if ok else "Failed to cache benchmark", verbose)

def prepare_dataset(
    *,
    model,
    task_name: str,
    shots: int,
    split_ratio: float,
    limit: Optional[int],
    training_limit: Optional[int],
    testing_limit: Optional[int],
    seed: int,
    from_csv: bool,
    from_json: bool,
    question_col: str,
    correct_col: str,
    incorrect_col: str,
    preloaded_qa_pairs: Optional[list[dict[str, Any]]],
    use_cached: bool,
    force_download: bool,
    cache_dir: str,
    cache_benchmark: bool,
    cross_benchmark_mode: bool,
    train_contrastive_pairs: Optional[Any],
    eval_contrastive_pairs: Optional[Any],
    from_synthetic: bool,
    synthetic_contrastive_pairs: Optional[Any],
    livecodebench_version: str,
    verbose: bool,
) -> dict[str, Any]:
    """Prepare training QA pairs plus test source for the pipeline.

    Returns:
        dict[str, Any]: Keys:
            qa_pairs: list of training QA dicts
            test_qa_pairs_source: test docs or QA pairs
            group_task_processed: bool
            group_task_qa_format: bool
            task_data: optional lm-eval task object
            train_docs: optional training docs
            skip_qa_display: bool
    """
    caps = _effective_caps(training_limit, testing_limit)

    # 1) Early-exit modes (each returns a fully formed state)
    for maybe in (
        _synthetic_mode(from_synthetic, synthetic_contrastive_pairs, verbose),
        _cross_benchmark_mode(cross_benchmark_mode, train_contrastive_pairs, eval_contrastive_pairs, verbose),
        _preloaded_mode(preloaded_qa_pairs, verbose),
        _load_from_csv_json(
            from_csv, from_json, task_name, question_col, correct_col, incorrect_col,
            limit, caps, split_ratio, seed, verbose
        ),
    ):
        if maybe is not None:
            return {
                "qa_pairs": maybe.qa_pairs,
                "test_qa_pairs_source": maybe.test_source,
                "group_task_processed": maybe.group_processed,
                "group_task_qa_format": maybe.group_qa_format,
                "task_data": maybe.task_data,
                "train_docs": maybe.train_docs,
                "skip_qa_display": maybe.skip_qa_display,
            }

    # 2) Benchmark path (cache → group/single → persist cache)
    _log(f"Loading task data for {task_name}...", verbose)
    used_cache, cached_pairs = _try_cached(task_name, cache_dir, limit, use_cached, force_download, verbose)

    if used_cache:
        # Split already-cached QA pairs
        qa_pairs, test_source = _split_and_cap(cached_pairs, split_ratio, caps, seed, verbose)
        return {
            "qa_pairs": qa_pairs,
            "test_qa_pairs_source": test_source,
            "group_task_processed": False,
            "group_task_qa_format": True,
            "task_data": None,
            "train_docs": [],
            "skip_qa_display": False,
        }

    is_group, expanded = _is_group_task(task_name, verbose)
    if is_group:
        state = _process_group_task(expanded, model, shots, limit, split_ratio, seed, caps, verbose)
        _maybe_persist_cache(cache_benchmark, used_cache, task_name, cache_dir, verbose)
        return {
            "qa_pairs": state.qa_pairs,
            "test_qa_pairs_source": state.test_source,
            "group_task_processed": True,
            "group_task_qa_format": True,
            "task_data": None,
            "train_docs": [],
            "skip_qa_display": False,
        }

    # Single task
    state = _load_single_task(
        model=model,
        task_name=task_name,
        shots=shots,
        split_ratio=split_ratio,
        seed=seed,
        limit=limit,
        training_limit=training_limit,
        testing_limit=testing_limit,
        cache_benchmark=cache_benchmark,
        cache_dir=cache_dir,
        force_download=force_download,
        livecodebench_version=livecodebench_version,
        verbose=verbose,
    )
    _maybe_persist_cache(cache_benchmark, used_cache, task_name, cache_dir, limit, verbose)

    return {
        "qa_pairs": state.qa_pairs,
        "test_qa_pairs_source": state.test_source,
        "group_task_processed": state.group_processed,
        "group_task_qa_format": state.group_qa_format,
        "task_data": state.task_data,
        "train_docs": state.train_docs,
        "skip_qa_display": state.skip_qa_display,
    }
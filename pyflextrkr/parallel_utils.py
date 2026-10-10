"""
Helpers for the post-processing drivers that run many independent tasks (one per data window) on a Dask cluster
without the usual problems of delayed graphs with thousands of tasks:

- at most max_inflight tasks are submitted at a time (no scheduler overload, no giant graph),
- the read-only inputs are scattered to the workers once,
- every task ends with a garbage collection and malloc_trim, so a worker does not grow by the memory glibc keeps
  from freed work arrays (seen as "Unmanaged memory use is high" / "Pausing worker" with many frames),
- a failed task is retried, then reported and left out (the caller decides how many failures are acceptable),
- run_parallel == 0 runs the same worker function serially, with identical results.
"""
import gc
import ctypes
import time
import logging
import traceback


def release_memory():
    """Garbage collection and return of the freed heap to the OS (glibc); harmless where malloc_trim does not exist."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _run_task(func, task, static):
    """Worker-side wrapper: call func(task, static), then release memory. Exceptions are returned, not raised,
    so that a failing task does not poison the worker or the as_completed loop."""
    try:
        result = func(task, static)
        err = None
    except Exception:                                   # noqa: BLE001 - reported by the driver
        result, err = None, traceback.format_exc()
    release_memory()
    return result, err


def _total_threads(client):
    try:
        return max(1, sum(client.nthreads().values()))
    except Exception:                                   # noqa: BLE001
        return 1


def run_bounded(func, tasks, static=None, run_parallel=0, max_inflight=None, retries=1, label="task", log_every=0.1,
                consume=None):
    """
    Run func(task, static) for every task.

    Args:
        func: callable(task, static) -> result. Must be importable by the workers.
        tasks: list
            Small, picklable task descriptions.
        static: object, optional
            Read-only data shared by all tasks (sent to the workers once).
        run_parallel: int
            0 = serial, >= 1 = use the current Dask client (distributed.get_client()).
        max_inflight: int, optional
            Maximum number of submitted tasks; default 2 x the number of worker threads.
        retries: int
            How many times a failed task is submitted again.
        label: string
            Name used in the log messages.
        log_every: float
            Progress is logged every this fraction of the tasks.
        consume: callable(index, result), optional
            Called with each result as soon as it arrives instead of keeping it (for large results that are
            reduced on the fly; the returned list then holds None for finished tasks).

    Returns:
        results: list
            Result of every task in task order, None for tasks that failed all attempts.
        failed: list of int
            Indices of the tasks that failed.
    """
    logger = logging.getLogger(__name__)
    ntasks = len(tasks)
    results = [None] * ntasks
    failed = []
    if ntasks == 0:
        return results, failed
    t0 = time.time()
    step = max(1, int(ntasks * log_every))

    def report(ndone):
        if ndone % step == 0 or ndone == ntasks:
            elapsed = time.time() - t0
            eta = elapsed / ndone * (ntasks - ndone)
            logger.info(f"{label}: {ndone}/{ntasks} done, {elapsed / 60:.1f} min elapsed, about {eta / 60:.1f} min left, "
                        f"{len(failed)} failed")

    if run_parallel == 0:
        for i, task in enumerate(tasks):
            for attempt in range(retries + 1):
                result, err = _run_task(func, task, static)
                if err is None:
                    if consume is not None:
                        consume(i, result)
                    else:
                        results[i] = result
                    break
                logger.warning(f"{label} {i} failed (attempt {attempt + 1}):\n{err}")
            else:
                failed.append(i)
            report(i + 1)
        return results, failed

    from distributed import get_client, as_completed
    client = get_client()
    if max_inflight is None:
        max_inflight = 2 * _total_threads(client)
    # scatter a list holding the object with hash=False: scattering a dict scatters its entries under their own
    # names, and equal-looking objects would be merged into one key
    static_future = client.scatter([static], broadcast=True, hash=False)[0] if static is not None else None

    ac = as_completed([])
    pending = {}                       # future key -> (task index, attempt)
    queue = list(range(ntasks))[::-1]  # pop() takes the first task
    attempts = [0] * ntasks

    def submit(i):
        fut = client.submit(_run_task, func, tasks[i], static_future, pure=False)
        pending[fut.key] = i
        ac.add(fut)

    while queue and len(pending) < max_inflight:
        submit(queue.pop())
    ndone = 0
    for fut in ac:
        i = pending.pop(fut.key)
        try:
            result, err = fut.result()
        except Exception:                               # noqa: BLE001 - e.g. a worker that died
            result, err = None, traceback.format_exc()
        fut.release()
        if err is None:
            if consume is not None:
                consume(i, result)
            else:
                results[i] = result
            ndone += 1
            report(ndone + len(failed))
        elif attempts[i] < retries:
            attempts[i] += 1
            logger.warning(f"{label} {i} failed, submitting it again:\n{err}")
            submit(i)
        else:
            logger.warning(f"{label} {i} failed for good:\n{err}")
            failed.append(i)
            report(ndone + len(failed))
        while queue and len(pending) < max_inflight:
            submit(queue.pop())
    return results, failed

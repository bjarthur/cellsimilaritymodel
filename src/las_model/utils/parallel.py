"""Run independent tasks across processes, with a progress bar."""
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm


def progress_file():
    """
    Where progress bars are drawn. run_simulations.py --progress hands the child a copy of
    its terminal and names the descriptor in PROGRESS_FD, so bars reach the terminal while
    stdout and stderr go to the log. Otherwise bars go to stderr.
    """
    fd = os.environ.get('PROGRESS_FD')
    if fd:
        try:
            return os.fdopen(int(fd), 'w', closefd=False)
        except (ValueError, OSError):
            pass
    return sys.stderr


def run_pool(worker, tasks, desc=None, num_workers=None):
    """
    Apply worker to every task across num_workers forked processes (default: one per core)
    and return the results in task order.

    Runs serially when num_workers is 1, or when called from inside a worker process, so
    pools never nest. A tqdm bar counts completed tasks; it is drawn only when its output is
    a terminal (see progress_file), and TQDM_DISABLE=1 switches it off everywhere.
    """
    tasks = list(tasks)
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, len(tasks))
    if mp.parent_process() is not None:
        num_workers = 1
    print(f"{desc or 'tasks'}: {len(tasks)} tasks on {num_workers} worker(s)")

    # disable=None draws the bar only when its output is a terminal. tqdm's TQDM_DISABLE
    # environment switch only applies to arguments not passed explicitly, so leave it out then.
    bar_options = {} if os.environ.get('TQDM_DISABLE') else {'disable': None}

    results = [None] * len(tasks)
    with tqdm(total=len(tasks), desc=desc, file=progress_file(), **bar_options) as bar:
        if num_workers <= 1:
            for i, task in enumerate(tasks):
                results[i] = worker(task)
                bar.update()
        else:
            ctx = mp.get_context('fork')
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
                futures = {executor.submit(worker, task): i for i, task in enumerate(tasks)}
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
                    bar.update()
    return results

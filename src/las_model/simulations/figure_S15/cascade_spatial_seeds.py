# 3 Step Cascade Spatial Simulation: relatedness curves for different random seeds
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import numpy as np
from las_model.utils import gridfunc as gf
from las_model.utils.config import PROJECT_DIR
from las_model.utils.output import save_experiment

# Experiment metadata
metadata = {
    'experiment_name': 'cascade_spatial_seeds',
    'experiment_directory': 'cascade',
    'created': datetime.now().isoformat(),
    'seeds': list(range(1000, 1008)),
    'maxCells': 2**10,
    'gridSize': 101,
    'Tcc': 1000,
    'varTcc': 10,
    'circuit': 'cascade',
    'PprodA': 10**-1,
    'kcatA': 10**-2,
    'kcatB': 10**-2,
    'maxRadius': 9,
    'relatedness_t': 10000,
}

def _simulate_single_seed(task_args):
    """Worker task: run one seeded grid and return its relatedness curve."""
    seed, meta = task_args

    print(f"Simulating grid for seed = {seed}")

    # Pin random seed. gridfunc draws from its module-level rng, and each worker is its own 
    # process, so this only affects the grid built below. The seed itself is the swept 
    # variable here, so it is used directly rather than drawn from a master rng. 
    gf.rng = np.random.default_rng(seed=seed)

    # Initiate, seed and run grid
    grid = gf.Grid(meta['gridSize'], meta['gridSize'], meta['maxCells'])
    grid.seed(meta['circuit'], [meta['PprodA'], meta['kcatA'], meta['kcatB']], meta['Tcc'], meta['varTcc'])
    grid.run()

    # Compute relatedness curve 
    return grid.calcCollectiveLocalRelatedness(meta['maxRadius'], meta['relatedness_t'])

if __name__ == '__main__':
    tasks = [(seed, metadata) for seed in metadata['seeds']]

    num_workers = min(os.cpu_count() or 4, len(tasks))
    print(f"Running {len(tasks)} seeded grids using {num_workers} parallel workers...")

    ctx = mp.get_context('fork')
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        relatedness = list(executor.map(_simulate_single_seed, tasks))

    # Relatedness curves stay a list with one (cells, radii) array per seed: the number of 
    # cells alive at relatedness_t differs between runs, so they cannot be stacked 

    # Save results 
    exp_dir = save_experiment(
        experiment_name=metadata['experiment_name'],
        data=[metadata['seeds'], {'relatedness': relatedness}],
        metadata=metadata,
        base_dir=PROJECT_DIR / metadata['experiment_directory']
    )
    print(f"Experiment saved to {exp_dir}")

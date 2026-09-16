# Saturated Production sweep PprodA
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import numpy as np
from las_model.utils import motiffunc as mf
from las_model.utils.config import PROJECT_DIR
from las_model.utils.output import save_experiment
from las_model.utils.analyze import calculate_division_differences

# Experiment metadata
metadata = {
    'experiment_name': 'satprod_PprodAsweep',
    'experiment_directory': 'satprod',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'Tcc': 1000,
    'varTcc': 0,
    'circuit': 'prodsat',
    'PprodAs': list(np.logspace(-3, 2, 6)),
    'kcatA': 10**-1,
}

def _simulate_single_pprod(task_args):
    """Worker task simulating one parameter value."""
    PprodAindex, PprodA, meta, seed = task_args
    task_rng = np.random.default_rng(seed)

    print(f"Simulating for PprodAindex={PprodAindex}, PprodA={PprodA}")

    motherCell = mf.Cell(meta['Tcc'], meta['varTcc'], task_rng)
    motherCell.parameterize(meta['circuit'], [PprodA, meta['kcatA']])
    motherCell.equilibrate(meta['nCells_equilibrium'])
    motherCell.run(meta['nCells'])

    divStates = motherCell.getMotherStates()
    dsis, drnd, vardsis, vardrnd, normvar = calculate_division_differences(divStates, task_rng)

    return {
        'mother_As': divStates[0],
        'mother_Bs': divStates[1],
        'dsis': dsis,
        'drnd': drnd,
        'vardsis': vardsis,
        'vardrnd': vardrnd,
        'normvar': normvar,
    }

if __name__ == '__main__':
    # Generate statistically independent, deterministic seeds for each parameter
    master_rng = np.random.default_rng(seed=metadata['seed'])
    seeds = master_rng.integers(0, 2**63 - 1, size=len(metadata['PprodAs']))

    tasks = [
        (idx, pprodA, metadata, seeds[idx])
        for idx, pprodA in enumerate(metadata['PprodAs'])
    ]

    num_workers = min(os.cpu_count() or 4, len(tasks))
    print(f"Running sweep across {len(tasks)} parameters using {num_workers} parallel workers...")

    ctx = mp.get_context('fork')
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        sweep_results = list(executor.map(_simulate_single_pprod, tasks))

    # Stack results along axis 0 (maintains exact same shape as sequential version)
    results = {
        k: np.stack([res[k] for res in sweep_results], axis=0)
        for k in sweep_results[0].keys()
    }

    exp_dir = save_experiment(
        experiment_name=metadata['experiment_name'],
        data=[metadata['PprodAs'], results],
        metadata=metadata,
        base_dir=PROJECT_DIR / metadata['experiment_directory'],
    )
    print(f"Experiment saved to {exp_dir}")


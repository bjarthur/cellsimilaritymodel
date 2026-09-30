# Fixed Reactant: sweep kcatA and PprodB 
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime 
import numpy as np 
from las_model.utils.fixedreactant import simulate_fixed_reactant
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_division_differences
from las_model.utils.output import save_experiment 

# Experiment metadata
metadata = {
    'experiment_name': 'fixedreactant_sweep_kcatA',
    'experiment_directory': 'fixed_reactant',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'Tcc': 1000,
    'PprodA': 10**-1,
    'kcatAs': list(np.logspace(-2,2,5)),
    'PprodBs': list(np.logspace(-3,3,31)),
    'Km': 10**3,
}

def _simulate_single_kcatA_pprodB(task_args):
    """Worker task simulating one (kcatA, PprodB) pair."""
    kcatA, PprodB, meta, seed = task_args
    task_rng = np.random.default_rng(seed)

    print(f"Running simulation for kcatA={kcatA}, PprodB={PprodB}")

    means, variances, motherMolecules = simulate_fixed_reactant(
        meta['PprodA'], PprodB, kcatA, meta['Km'], meta['Tcc'], meta['nCells'], task_rng)

    # Division differences from the mother states 
    dsis, drnd, vardsis, vardrnd, normvar = calculate_division_differences(motherMolecules, task_rng)

    return {
        'means': means,
        'variances': variances,
        'dsis': dsis,
        'drnd': drnd,
        'vardsis': vardsis,
        'vardrnd': vardrnd,
        'normvar': normvar,
    }

if __name__ == '__main__':
    # Pin random seed and generate statistically independent seeds per task
    nKcatA, nPprodB = len(metadata['kcatAs']), len(metadata['PprodBs'])
    master_rng = np.random.default_rng(seed=metadata['seed'])
    seeds = master_rng.integers(0, 2**63 - 1, size=nKcatA * nPprodB)

    # kcatA outer, PprodB inner, so the flat task order maps onto the (kcatA, PprodB) grid below
    tasks = [
        (kcatA, PprodB, metadata, seeds[i * nPprodB + j])
        for i, kcatA in enumerate(metadata['kcatAs'])
        for j, PprodB in enumerate(metadata['PprodBs'])
    ]

    num_workers = min(os.cpu_count() or 4, len(tasks))
    print(f"Running sweep across {len(tasks)} (kcatA, PprodB) conditions using {num_workers} parallel workers...")

    ctx = mp.get_context('fork')
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        sweep_results = list(executor.map(_simulate_single_kcatA_pprodB, tasks))

    # Stack results into a (kcatA, PprodB, ...) grid 
    results = {
        k: np.stack([res[k] for res in sweep_results], axis=0)
        for k in sweep_results[0].keys()
    }
    results = {k: v.reshape(nKcatA, nPprodB, *v.shape[1:]) for k, v in results.items()}

    # Save results 
    exp_dir = save_experiment(
        experiment_name=metadata['experiment_name'],
        data=[[metadata['kcatAs'], metadata['PprodBs']], results],
        metadata=metadata,
        base_dir=PROJECT_DIR / metadata['experiment_directory']
    )
    print(f"Experiment saved to {exp_dir}")

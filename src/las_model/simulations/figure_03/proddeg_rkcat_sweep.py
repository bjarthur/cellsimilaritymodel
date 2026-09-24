# Production and Degradation: Rkcat sweep 
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime 
import numpy as np
from las_model.utils.cell import Cell
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_offspring_similarity_time
from las_model.utils.output import save_experiment 

# Experiment metadata
metadata = {
    'experiment_name': 'proddeg_rkcat_sweep',
    'experiment_directory': 'prodanddeg',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 100,
    'nCells_equilibrium': 10,
    'nCycles': 10,
    'Tcc': 1000,
    'varTcc': 0,
    'circuit': 'proddeg',
    'PprodA': 10**-1,
    'kcatAs': list(np.logspace(-1.3,0,20)),
    'PprodB': 10**-1
}

# Calculate Rkcat values 
kcatBs = 2*np.array(metadata['kcatAs'])-0.1
K_Ms = (np.array(metadata['kcatAs']) * metadata['PprodA'] - kcatBs * metadata['PprodB'] /2) * metadata['Tcc']**2

sweep_values = {
    'kcatAs': metadata['kcatAs'],
    'kcatBs': kcatBs,
    'K_Ms': K_Ms
}

def _simulate_single_kcat(task_args):
    """Worker simulating one kcatA condition."""
    i, kcatA, kcatB, K_M, meta, seed = task_args
    task_rng = np.random.default_rng(seed)

    print(f"Simulating for kcatA = {kcatA}")

    motherCell = Cell(meta['Tcc'], meta['varTcc'], task_rng)
    motherCell.parameterize(
        meta['circuit'],
        [meta['PprodA'], meta['PprodB'], kcatA, kcatB, K_M]
    )
    motherCell.equilibrate(meta['nCells_equilibrium'])
    motherCell.run(meta['nCells'])

    # num_workers=1 prevents nested multiprocessing inside workers
    dsis, drnd, vardsis, vardrnd, normvar = calculate_offspring_similarity_time(
        motherCell, meta, task_rng, num_workers=1
    )

    return {
        'dsis': dsis,
        'drnd': drnd,
        'vardsis': vardsis,
        'vardrnd': vardrnd,
        'normvar': normvar,
    }

if __name__ == '__main__':
    master_rng = np.random.default_rng(seed=metadata['seed'])
    seeds = master_rng.integers(0, 2**63 - 1, size=len(metadata['kcatAs']))

    tasks = [
        (i, metadata['kcatAs'][i], kcatBs[i], K_Ms[i], metadata, seeds[i])
        for i in range(len(metadata['kcatAs']))
    ]

    num_workers = min(os.cpu_count() or 4, len(tasks))
    print(f"Running sweep across {len(tasks)} conditions using {num_workers} parallel workers...")

    ctx = mp.get_context('fork')
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        sweep_results = list(executor.map(_simulate_single_kcat, tasks))

    results = {
        k: np.stack([res[k] for res in sweep_results], axis=0)
        for k in sweep_results[0].keys()
    }

    # Save results 
    exp_dir = save_experiment(
        experiment_name=metadata['experiment_name'],
        data=[sweep_values, results],
        metadata=metadata,
        base_dir=PROJECT_DIR / metadata['experiment_directory']
    )
    print(f"Experiment saved to {exp_dir}")

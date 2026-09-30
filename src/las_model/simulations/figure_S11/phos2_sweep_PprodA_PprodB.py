#TODO: Run simulation for full ranges of PprodA and PprodB values 

# Phosphorylation Motif 2: Sweep PprodA and PprodB
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime 
import numpy as np 
from las_model.utils import motiffunc as mf
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_division_differences
from las_model.utils.output import save_experiment 

# Experiment metadata 
metadata = {
    'experiment_name': 'phos_bi_sweep_PprodA_PprodB',
    'experiment_directory': 'phos_bi',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'Tcc': 1000,
    'varTcc': 0,
    'circuit': 'phos2',
    'PprodAs': list(np.logspace(-3,2,31)),
    'PprodBs': list(np.logspace(-3,2,31)),
    'kk': 10**-2,
    'krevk': 10**-2,
    'k1': 10**-4,
    'krev1': 10**-2,
    'k2': 10**-4,
    'krev2': 10**-2,
    'kt': 10**-2,
    'kp':10**-2,
}

def _simulate_single_pprodA_pprodB(task_args):
    """Worker task simulating one (PprodA, PprodB) pair."""
    PprodA, PprodB, meta, seed = task_args
    task_rng = np.random.default_rng(seed)

    print(f"Running simulation for PprodA={PprodA}, PprodB={PprodB}")

    motherCell = mf.Cell(meta['Tcc'], meta['varTcc'], task_rng)
    motherCell.parameterize(
        meta['circuit'],
        [
            PprodA,
            PprodB,
            meta['kk'],
            meta['krevk'],
            meta['k1'],
            meta['krev1'],
            meta['kt'],
            meta['k2'],
            meta['krev2'],
            meta['kp']
        ]
    )
    motherCell.equilibrate(meta['nCells_equilibrium'])
    motherCell.run(meta['nCells'])

    # Get mother states and calculate division differences
    divStates = motherCell.getMotherStates()
    dsis, drnd, vardsis, vardrnd, normvar = calculate_division_differences(divStates, task_rng)

    # Get molecule amounts 
    molecules = motherCell.getMolecules()
    means = np.mean(molecules, axis=1)
    variances = np.var(molecules, axis=1)

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
    nPprodA, nPprodB = len(metadata['PprodAs']), len(metadata['PprodBs'])
    master_rng = np.random.default_rng(seed=metadata['seed'])
    seeds = master_rng.integers(0, 2**63 - 1, size=nPprodA * nPprodB)

    # PprodA outer, PprodB inner, so the flat task order maps onto the (PprodA, PprodB) grid below
    tasks = [
        (PprodA, PprodB, metadata, seeds[i * nPprodB + j])
        for i, PprodA in enumerate(metadata['PprodAs'])
        for j, PprodB in enumerate(metadata['PprodBs'])
    ]

    num_workers = min(os.cpu_count() or 4, len(tasks))
    print(f"Running sweep across {len(tasks)} (PprodA, PprodB) conditions using {num_workers} parallel workers...")

    ctx = mp.get_context('fork')
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        sweep_results = list(executor.map(_simulate_single_pprodA_pprodB, tasks))

    # Stack results into a (PprodA, PprodB, ...) grid 
    results = {
        k: np.stack([res[k] for res in sweep_results], axis=0)
        for k in sweep_results[0].keys()
    }
    results = {k: v.reshape(nPprodA, nPprodB, *v.shape[1:]) for k, v in results.items()}

    # Save results 
    exp_dir = save_experiment(
        experiment_name=metadata['experiment_name'],
        data=[metadata['PprodAs'], metadata['PprodBs'], results],
        metadata=metadata,
        base_dir=PROJECT_DIR / metadata['experiment_directory']
    )
    print(f"Experiment saved to {exp_dir}")

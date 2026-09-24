#TODO: run this script 

# Saturated Production: 2D Sweep kcatA, Tcc
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime 
import numpy as np 
from las_model.utils.cell import Cell
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_division_differences
from las_model.utils.output import save_experiment 

# Experiment metadata
metadata = {
    'experiment_name': 'prodsat_sweep_kcatA_Tcc',
    'experiment_directory': 'satprod',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'Tccs': list(np.logspace(2,4,5)),
    'varTcc': 0,
    'circuit': 'prodsat',
    'PprodA': 10**-1,
    'kcats': list(np.logspace(-4,0,9)),
}

def _simulate_single_tcc_kcat(task_args):
    """Worker task simulating one (Tcc, kcatA) pair."""
    Tcc, kcatA, meta, seed = task_args
    task_rng = np.random.default_rng(seed)

    print(f"Running simulation for Tcc={Tcc}, kcatA={kcatA}")

    motherCell = Cell(Tcc, meta['varTcc'], task_rng)
    motherCell.parameterize(meta['circuit'], [meta['PprodA'], kcatA])
    motherCell.equilibrate(meta['nCells_equilibrium'])
    motherCell.run(meta['nCells'])

    # Get mother states and calculate division differences 
    divStates = motherCell.getMotherStates()
    dsis, drnd, vardsis, vardrnd, normvar = calculate_division_differences(divStates, task_rng)

    return {
        'dsis': dsis,
        'drnd': drnd,
        'vardsis': vardsis,
        'vardrnd': vardrnd,
        'normvar': normvar,
    }

if __name__ == '__main__':
    # Pin random seed and generate statistically independent seeds per task
    nTcc, nKcat = len(metadata['Tccs']), len(metadata['kcats'])
    master_rng = np.random.default_rng(seed=metadata['seed'])
    seeds = master_rng.integers(0, 2**63 - 1, size=nTcc * nKcat)

    # Tcc outer, kcatA inner, so the flat task order maps onto the (Tcc, kcatA) grid below
    tasks = [
        (Tcc, kcatA, metadata, seeds[i * nKcat + j])
        for i, Tcc in enumerate(metadata['Tccs'])
        for j, kcatA in enumerate(metadata['kcats'])
    ]

    num_workers = min(os.cpu_count() or 4, len(tasks))
    print(f"Running sweep across {len(tasks)} (Tcc, kcatA) conditions using {num_workers} parallel workers...")

    ctx = mp.get_context('fork')
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        sweep_results = list(executor.map(_simulate_single_tcc_kcat, tasks))

    # Stack results into a (Tcc, kcatA, ...) grid 
    results = {
        k: np.stack([res[k] for res in sweep_results], axis=0)
        for k in sweep_results[0].keys()
    }
    results = {k: v.reshape(nTcc, nKcat, *v.shape[1:]) for k, v in results.items()}

    # Save results 
    exp_dir = save_experiment(
        experiment_name=metadata['experiment_name'],
        data=[metadata['kcats'], metadata['Tccs'], results],
        metadata=metadata,
        base_dir=PROJECT_DIR / metadata['experiment_directory']
    )
    print(f"Experiment saved to {exp_dir}")

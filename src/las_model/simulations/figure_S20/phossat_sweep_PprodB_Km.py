#TODO: Run simulation for all values of PprodB and Km 

# Saturated Phosphorylation: Sweep PprodB and Km
from datetime import datetime 
import numpy as np 
from las_model.utils import motiffunc as mf
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_division_differences
from las_model.utils.output import save_experiment 
from las_model.utils.parallel import run_pool

# Experiment metadata
metadata = {
    'experiment_name': 'phossat_sweep_PprodB_Km',
    'experiment_directory': 'phos_sat',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'Tcc': 1000,
    'varTcc': 0,
    'circuit': 'phos_sat',
    'PprodA': 10**-1,
    'PprodBs': list(np.logspace(-2,3,31)),
    'Kms': list(np.logspace(0,4,25)),
    'PprodC': 10**-1,
    'ka': 10**-2,
    'kc': 10**-2,
}

def _simulate_single_pprodB_km(task_args):
    """Worker task simulating one (PprodB, Km) pair."""
    PprodB, Km, meta, seed = task_args
    task_rng = np.random.default_rng(seed)

    print(f"Running simulation for PprodB={PprodB}, Km={Km}")

    motherCell = mf.Cell(meta['Tcc'], meta['varTcc'], task_rng)
    motherCell.parameterize(
        meta['circuit'],
        [
            meta['PprodA'],
            meta['ka'],
            Km,
            PprodB,
            meta['PprodC'],
            meta['kc'],
            Km
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
    nPprodB, nKm = len(metadata['PprodBs']), len(metadata['Kms'])
    master_rng = np.random.default_rng(seed=metadata['seed'])
    seeds = master_rng.integers(0, 2**63 - 1, size=nPprodB * nKm)

    # PprodB outer, Km inner, so the flat task order maps onto the (PprodB, Km) grid below
    tasks = [
        (PprodB, Km, metadata, seeds[i * nKm + j])
        for i, PprodB in enumerate(metadata['PprodBs'])
        for j, Km in enumerate(metadata['Kms'])
    ]

    sweep_results = run_pool(_simulate_single_pprodB_km, tasks, desc=metadata['experiment_name'])

    # Stack results into a (PprodB, Km, ...) grid 
    results = {
        k: np.stack([res[k] for res in sweep_results], axis=0)
        for k in sweep_results[0].keys()
    }
    results = {k: v.reshape(nPprodB, nKm, *v.shape[1:]) for k, v in results.items()}

    # Save results 
    exp_dir = save_experiment(
        experiment_name=metadata['experiment_name'],
        data=[metadata['PprodBs'], metadata['Kms'], results],
        metadata=metadata,
        base_dir=PROJECT_DIR / metadata['experiment_directory']
    )
    print(f"Experiment saved to {exp_dir}")

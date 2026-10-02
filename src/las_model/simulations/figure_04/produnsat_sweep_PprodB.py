# Unsaturated production sweep substrate production rate 
from datetime import datetime 
import numpy as np
from las_model.utils import motiffunc as mf
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_division_differences
from las_model.utils.output import save_experiment 
from las_model.utils.parallel import run_pool

# Experiment metadata
metadata = {
    'experiment_name': 'produnsat_sweep_PprodB',
    'experiment_directory': 'production',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'Tcc': 1000,
    'varTcc': 0,
    'circuit': 'produnsat',
    'PprodA': 10**-1,
    'kcatA': 10**-1,
    'PprodBs': list(np.logspace(-2,4,31)),
    'Km': 10**3
}

def _simulate_single_pprodB(task_args):
    """Worker task simulating one PprodB value."""
    PprodB, meta, seed = task_args
    task_rng = np.random.default_rng(seed)

    print(f"Running simulation for PprodB={PprodB}")

    # Initialize and run mother cell 
    motherCell = mf.Cell(meta['Tcc'], meta['varTcc'], task_rng)
    motherCell.parameterize(meta['circuit'], [PprodB, meta['PprodA'], meta['kcatA'], meta['Km']])
    motherCell.equilibrate(meta['nCells_equilibrium'])
    motherCell.run(meta['nCells'])

    # Get molecule amounts 
    molecules = motherCell.getMolecules()
    means = np.array([np.mean(molecules[0]), np.mean(molecules[1]), np.mean(molecules[2])])
    variances = np.array([np.var(molecules[0]), np.var(molecules[1]), np.var(molecules[2])])

    # Get mother states and calculate division differences 
    divStates = motherCell.getMotherStates()
    dsis, drnd, vardsis, vardrnd, normvar = calculate_division_differences(divStates, task_rng)

    return {
        'means': means,
        'variances': variances,
        'dsis': dsis,
        'drnd': drnd,
        'vardrnd': vardrnd,
        'vardsis': vardsis,
        'normvar': normvar,
    }

if __name__ == '__main__':
    # Pin random seed and generate statistically independent seeds per task
    master_rng = np.random.default_rng(seed=metadata['seed'])
    seeds = master_rng.integers(0, 2**63 - 1, size=len(metadata['PprodBs']))

    tasks = [
        (PprodB, metadata, seeds[idx])
        for idx, PprodB in enumerate(metadata['PprodBs'])
    ]

    sweep_results = run_pool(_simulate_single_pprodB, tasks, desc=metadata['experiment_name'])

    # Stack results along axis 0
    results = {
        k: np.stack([res[k] for res in sweep_results], axis=0)
        for k in sweep_results[0].keys()
    }

    # Save results 
    exp_dir = save_experiment(
        experiment_name=metadata['experiment_name'],
        data=[metadata['PprodBs'], results],
        metadata=metadata,
        base_dir=PROJECT_DIR / metadata['experiment_directory']
    )
    print(f"Experiment saved to {exp_dir}")

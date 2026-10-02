# Fixed Reactant: sweep PprodA and PprodB
from datetime import datetime 
import numpy as np 
from las_model.utils.fixedreactant import simulate_fixed_reactant
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_division_differences
from las_model.utils.output import save_experiment 
from las_model.utils.parallel import run_pool

# Experiment metadata
metadata = {
    'experiment_name': 'fixedreactant_sweep_PprodA',
    'experiment_directory': 'fixed_reactant',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'Tcc': 1000,
    'PprodAs': list(np.logspace(-2,2,5)),
    'kcatA': 10**-1,
    'PprodBs': list(np.logspace(-3,3,31)),
    'Km': 10**3,
}

def _simulate_single_pprodA_pprodB(task_args):
    """Worker task simulating one (PprodA, PprodB) pair."""
    PprodA, PprodB, meta, seed = task_args
    task_rng = np.random.default_rng(seed)

    print(f"Running simulation for PprodA={PprodA}, PprodB={PprodB}")

    means, variances, motherMolecules = simulate_fixed_reactant(
        PprodA, PprodB, meta['kcatA'], meta['Km'], meta['Tcc'], meta['nCells'], task_rng)

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
    nPprodA, nPprodB = len(metadata['PprodAs']), len(metadata['PprodBs'])
    master_rng = np.random.default_rng(seed=metadata['seed'])
    seeds = master_rng.integers(0, 2**63 - 1, size=nPprodA * nPprodB)

    # PprodA outer, PprodB inner, so the flat task order maps onto the (PprodA, PprodB) grid below
    tasks = [
        (PprodA, PprodB, metadata, seeds[i * nPprodB + j])
        for i, PprodA in enumerate(metadata['PprodAs'])
        for j, PprodB in enumerate(metadata['PprodBs'])
    ]

    sweep_results = run_pool(_simulate_single_pprodA_pprodB, tasks, desc=metadata['experiment_name'])

    # Stack results into a (PprodA, PprodB, ...) grid 
    results = {
        k: np.stack([res[k] for res in sweep_results], axis=0)
        for k in sweep_results[0].keys()
    }
    results = {k: v.reshape(nPprodA, nPprodB, *v.shape[1:]) for k, v in results.items()}

    # Save results 
    exp_dir = save_experiment(
        experiment_name=metadata['experiment_name'],
        data=[[metadata['PprodAs'], metadata['PprodBs']], results],
        metadata=metadata,
        base_dir=PROJECT_DIR / metadata['experiment_directory']
    )
    print(f"Experiment saved to {exp_dir}")

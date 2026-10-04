# Burst Size Sweep, repeated for each total A production rate
from datetime import datetime
import numpy as np
from las_model.utils import motiffunc as mf
from las_model.utils.config import PROJECT_DIR
from las_model.utils.output import save_experiment
from las_model.utils.parallel import run_pool
from las_model.utils.analyze import calculate_division_differences

# Experiment metadata
metadata = {
    'experiment_name': 'burstSize_prodA',
    'experiment_directory': 'burstSize',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCycles_equilibrium': 10,
    'Tcc': 1000,
    'circuit': 'prodsat_burst',
    'burstSizes': list(np.linspace(1,20,20)),
    'prodA_std_exponents': [0, -1, -2],   # prodA_std = 10**exponent
    'kcatA': 10**-2
}

def _simulate_single_burst(task_args):
    """Worker task simulating one (prodA_std, burstSize) condition."""
    prodA, burstSize, meta, seed = task_args
    task_rng = np.random.default_rng(seed)

    print(f"Simulating burst size {burstSize} with prodA {prodA}")

    motherCell = mf.Cell(meta['Tcc'],0,task_rng)
    motherCell.parameterize(meta['circuit'],[prodA,meta['kcatA'],burstSize])
    motherCell.equilibrate(meta['nCycles_equilibrium'])
    motherCell.run(meta['nCells'])

    molecules = motherCell.getMolecules()
    divStates = motherCell.getMotherStates()
    _, _, _, _, normvar = calculate_division_differences(divStates, task_rng)

    return np.mean(molecules[0]), np.mean(molecules[1]), normvar[0], normvar[1]

if __name__ == '__main__':
    burstSizes = np.array(metadata['burstSizes'])
    nBursts = len(burstSizes)

    # Generate statistically independent, deterministic seeds for each condition
    master_rng = np.random.default_rng(seed=metadata['seed'])
    seeds = master_rng.integers(0, 2**63 - 1, size=len(metadata['prodA_std_exponents']) * nBursts)

    # Exponent outer, burst size inner, so the flat task order maps onto the loop below
    tasks = [
        (10**exponent / burstSize, burstSize, metadata, seeds[e * nBursts + i])
        for e, exponent in enumerate(metadata['prodA_std_exponents'])
        for i, burstSize in enumerate(burstSizes)
    ]

    sweep_results = run_pool(_simulate_single_burst, tasks, desc=metadata['experiment_name'])

    # Regroup the flat results per production rate and save each as its own experiment
    for e, exponent in enumerate(metadata['prodA_std_exponents']):
        prodA_std = 10**exponent
        prodAs = prodA_std / burstSizes

        Aeqs, Beqs, normvarAs, normvarBs = (np.array(x) for x in zip(*sweep_results[e * nBursts:(e + 1) * nBursts]))

        run_metadata = {
            **metadata,
            'experiment_name': f"{metadata['experiment_name']}-{-exponent}",
            'prodA_std': prodA_std,
        }

        exp_dir = save_experiment(
            experiment_name=run_metadata['experiment_name'],
            data=[burstSizes, prodAs, Aeqs, Beqs, normvarAs, normvarBs],
            metadata=run_metadata,
            base_dir=PROJECT_DIR / metadata['experiment_directory']
        )
        print(f"Saved experiment to {exp_dir}")

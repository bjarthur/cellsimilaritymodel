# Saturated Production: effect of burst size on LAS duration 
from datetime import datetime 
import numpy as np 
from las_model.utils.cell import Cell
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_offspring_similarity_time
from las_model.utils.output import save_experiment 

# Experiment metadata 
metadata = {
    'experiment_name': 'satprod_burst_time_sweep',
    'experiment_directory': 'satprod',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'nCycles': 10,
    'Tcc': 1000,
    'varTcc': 0,
    'circuit': 'prodsat_burst',
    'PprodA': 10**-2,
    'kcatA': 10**-1,
    'burstSizes': [1, 5, 10, 20],
}

# Pin random seed 
rng = np.random.default_rng(seed=metadata['seed'])

# Accumulate results 
results = {
    'vardsis': [],
    'vardrnd': [],
    'normvar': [],
}

# === Iterate over burst sizes and simulate mother cells ==========
for burstSize in metadata['burstSizes']:

    print(f"Simulating for burst size = {burstSize}")

    # Bursts of burstSize molecules at PprodA / burstSize, so the mean production is the same for every burst size 
    motherCell = Cell(metadata['Tcc'], metadata['varTcc'], rng)
    motherCell.parameterize(metadata['circuit'], [metadata['PprodA'] / burstSize, metadata['kcatA'], burstSize])
    motherCell.equilibrate(metadata['nCells_equilibrium'])
    motherCell.run(metadata['nCells'])

    # Offspring similarity over time 
    dsis, drnd, vardsis, vardrnd, normvar = calculate_offspring_similarity_time(motherCell, metadata, rng)

    results['vardsis'].append(vardsis)
    results['vardrnd'].append(vardrnd)
    results['normvar'].append(normvar)

# Stack results 
results = {k: np.stack(v, axis=0) for k, v in results.items()}

# Save results 
exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data=[metadata['burstSizes'], results],
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory']
)
print(f"Experiment saved to {exp_dir}")

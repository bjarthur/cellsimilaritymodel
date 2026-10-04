# Saturated Production: scramble inherited A
from datetime import datetime
import numpy as np
from las_model.utils import motiffunc as mf
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import simulate_offspring_scramble_time, calculate_offspring_differences
from las_model.utils.output import save_experiment

# Experiment metadata
metadata = {
    'experiment_name': 'satprod_scramble_A',
    'experiment_directory': 'satprod',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'nCycles': 10,
    'Tcc': 1000,
    'varTcc': 0,
    'circuit': 'prodsat',
    'PprodA': 10**-2,
    'kcatA': 10**-2,
    'scrambled_molecules': [0],   # A only
}

# Pin random seed
rng = np.random.default_rng(seed=metadata['seed'])

# Initialize and run Mother Cell
motherCell = mf.Cell(metadata['Tcc'],metadata['varTcc'],rng)
motherCell.parameterize(metadata['circuit'],[metadata['PprodA'],metadata['kcatA']])
motherCell.equilibrate(metadata['nCells_equilibrium'])
motherCell.run(metadata['nCells'])

# Simulate inherited and scrambled offspring, then pairwise differences over time
sis1, sis2, rnd1, sis1scr, sis2scr = simulate_offspring_scramble_time(motherCell,metadata,rng)
_, _, vardsis_inherited, vardrnd_inherited, normvar_inherited = calculate_offspring_differences(sis1, sis2, rnd1)
_, _, vardsis_scrambled, vardrnd_scrambled, normvar_scrambled = calculate_offspring_differences(sis1scr, sis2scr, rnd1)

results = {
    'inherited': {
        'vardsis': vardsis_inherited,
        'vardrnd': vardrnd_inherited,
        'normvar': normvar_inherited,
    },
    'scrambled': {
        'vardsis': vardsis_scrambled,
        'vardrnd': vardrnd_scrambled,
        'normvar': normvar_scrambled,
    },
}

# Save results
exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data = results,
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory']
)
print(f"Experiment saved to {exp_dir}")

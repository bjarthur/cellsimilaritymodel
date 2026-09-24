# Single Molecule, vary Tcc screen 
import numpy as np 
from las_model.utils.analyze import calculate_division_differences
from datetime import datetime
from las_model.utils.cell import Cell
from las_model.utils.config import PROJECT_DIR
from las_model.utils.output import save_experiment

# Experiment metadata
metadata = {
    'experiment_name': 'single_var_cell_cycle_time',
    'experiment_directory': 'varTcc',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'Tcc': 1000,
    'varTccs': list(np.logspace(0,3,11)),
    'circuit': 'single',
    'PprodA': 10**-1,
}

# Pin random seed 
rng = np.random.default_rng(seed=metadata['seed'])

# Accumulate results 
results = {
    'dsis': [],
    'drnd': [],
    'vardsis': [],
    'vardrnd': [],
    'normvar': [],
}

# === Iterate over varTcc values and simulate mother cells ==========================================
for varTcc in metadata['varTccs']:

    print(f"Simulating for varTcc: {varTcc}")

    motherCell = Cell(metadata['Tcc'],varTcc,rng)
    motherCell.parameterize(metadata['circuit'],[metadata['PprodA']])
    motherCell.equilibrate(metadata['nCells_equilibrium'])

    # Run mother cell 
    motherCell.run(metadata['nCells'])

    # Get mother states and calculate division differences
    divStates = motherCell.getMotherStates()
    dsis, drnd, vardsis, vardrnd, normvar = calculate_division_differences(divStates, rng)

    # Append to results 
    results['dsis'].append(dsis)
    results['drnd'].append(drnd)
    results['vardsis'].append(vardsis)
    results['vardrnd'].append(vardrnd)
    results['normvar'].append(normvar)

# Stack results 
results = {key: np.stack(v, axis=0) for key, v in results.items()}

# Save results 
exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data = [metadata['varTccs'], results],
    metadata = metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory'],
)
print(f"Experiment saved to: {exp_dir}")

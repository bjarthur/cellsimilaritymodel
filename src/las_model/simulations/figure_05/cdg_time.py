# TODO: run this script, then update the path to the new cdG data in the plotting script

# c-di-GMP circuit model 
from datetime import datetime 
import numpy as np
from las_model.utils.cell import Cell
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_offspring_similarity_time
from las_model.utils.output import save_experiment

# Experiment metadata
metadata = {
    'experiment_name': 'cdg_time',
    'experiment_directory': 'cdg',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'nCycles': 10,
    'Tcc': 1000,
    'varTcc': 0,
    'circuit': 'cdg',
    'PprodA': 10**-1,
    'kcatA': 10**-1,
    'PprodB': 10**-1,
    'kcatB': 10**-1,
    'KM': 5000,
    'kprod': 10**-1,
}

# Pin random seed 
rng = np.random.default_rng(seed=metadata['seed'])

# Accumulate results
results = {
    'dsis': [],
    'drnd': [],
    'vardrnd': [],
    'vardsis': [],
    'normvar': []
}

# Run simulation 
motherCell = Cell(metadata['Tcc'],metadata['varTcc'],rng)
motherCell.parameterize(metadata['circuit'],[metadata['PprodA'],metadata['PprodB'],metadata['kcatA'],metadata['kcatB'],metadata['KM'],metadata['kprod']])
motherCell.equilibrate(metadata['nCells_equilibrium'])
motherCell.run(metadata['nCells'])

# Get mother states and calculate division differences 
divStates = motherCell.getMotherStates()
dsis, drnd, vardsis, vardrnd, normvar = calculate_offspring_similarity_time(divStates,rng)

# Store results
results['dsis'].append(dsis)
results['drnd'].append(drnd)
results['vardrnd'].append(vardrnd)
results['vardsis'].append(vardsis)
results['normvar'].append(normvar)

# Stack results 
results = {k: np.stack(v,axis=0) for k, v in results.items()}

# Save results 
exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data = [results],
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory']
)
print(f"Experiment saved to f{exp_dir}")

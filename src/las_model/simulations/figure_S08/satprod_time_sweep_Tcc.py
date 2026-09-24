#TODO: run this for full sweep of Tcc values

# Saturated Production: Dynamic, Sweep Tcc values 
from datetime import datetime 
import numpy as np
from las_model.utils.cell import Cell
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_offspring_similarity_time 
from las_model.utils.output import save_experiment 

# Experiment metadata
metadata = {
    'experiment_name': 'satprod_time_Tccsweep',
    'experiment_directory': 'satprod/time_Tccsweep',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'nCycles': 10,
    'Tccs': list((500,1000,2000,5000)),
    'varTcc': 0,
    'circuit': 'prodsat',
    'PprodA': 10**-1,
    'kcatA': 10**-1
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

# === Iterate over kcatA values and simulate cells 
for Tcc in metadata['Tccs']:

    print(f"Simulating for Tcc = {Tcc}")

    # Initialize and run Mother Cell
    motherCell = Cell(Tcc,metadata['varTcc'],rng)
    motherCell.parameterize(metadata['circuit'],[metadata['PprodA'],metadata['kcatA']])
    motherCell.equilibrate(metadata['nCells_equilibrium'])
    motherCell.run(metadata['nCells'])

    # Calculate offspring similarity 
    dsis, drnd, vardsis, vardrnd, normvar = calculate_offspring_similarity_time(motherCell,metadata,rng)

    # append results 
    results['dsis'].append(dsis)
    results['drnd'].append(drnd)
    results['vardsis'].append(vardsis)
    results['vardrnd'].append(vardrnd)
    results['normvar'].append(normvar)

# Stack results 
results = {k: np.stack(v,axis=0) for k, v in results.items()}

# Save results 
exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data = [metadata['PprodAs'],results],
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory']
)
print(f"Experiment saved to f{exp_dir}")

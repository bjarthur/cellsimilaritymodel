#TODO: run simulation 

# Unsaturated Production: Order Analysis with varying PprodB 
import numpy as np 
from datetime import datetime 
from las_model.utils.cell import Cell
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calcOrder, calculate_division_differences
from las_model.utils.output import save_experiment 

# Experiment metadata 
metadata = {
    'experiment_name': 'produnsat_order_sweep_PprodB',
    'experiment_directory': 'orderAnalysis',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'Tcc': 1000,
    'varTcc': 0,
    'circuit': 'produnsat',
    'PprodA': 10**-1,
    'PprodBs': list(np.logspace(-2,3,31)),
    'kcatA': 10**-1,
    'Km': 10**3
}

# Pin random seed 
rng = np.random.default_rng(seed=metadata['seed'])

# Aggregate results 
results = {
    'dsis': [],
    'drnd': [],
    'vardsis': [],
    'vardrnd': [],
    'normvar': [],
    'order': [],
}

for i, PprodB in enumerate(metadata['PprodBs']):

    print(f"Running simulation {i}/{len(metadata['PprodBs'])} for PprodB={PprodB}")

    # Initialize and run mother cell 
    motherCell = Cell(metadata['Tcc'],metadata['varTcc'],rng)
    motherCell.parameterize(metadata['circuit'],[PprodB,metadata['PprodA'],metadata['kcatA'],metadata['Km']])
    motherCell.equilibrate(metadata['nCells_equilibrium'])
    motherCell.run(metadata['nCells'])

    # Get mother states and calculate division differences
    divStates = motherCell.getMotherStates()
    dsis, drnd, vardsis, vardrnd, normvar = calculate_division_differences(divStates,rng)

    # Get molecule amounts 
    molecules = motherCell.getMolecules()

    # calculate order at each time step 
    order = calcOrder(molecules[1],metadata['kcatA'],metadata['Km'],molecules[0])

    # Append results 
    results['dsis'].append(dsis)
    results['drnd'].append(drnd)
    results['vardsis'].append(vardsis)
    results['vardrnd'].append(vardrnd)
    results['normvar'].append(normvar)
    results['order'].append(order)

# Stack results 
results = {k: np.stack(v,axis=0) for k, v in results.items()}

exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data=[metadata['PprodBs'],results],
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory'],
)
print(f"Experiment saved to {exp_dir}")

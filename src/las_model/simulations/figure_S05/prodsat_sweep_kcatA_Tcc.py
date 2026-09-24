#TODO: run this script 

# Saturated Production: 2D Sweep kcatA, Tcc
import numpy as np 
from datetime import datetime 
from las_model.utils.cell import Cell
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_division_differences
from las_model.utils.output import save_experiment 

# Experiment metadata
metadata = {
    'experiment_name': 'prodsat_sweep_kcatA_Tcc',
    'experiment_directory': 'satprod',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'Tccs': list(np.logspace(2,4,5)),
    'varTcc': 0,
    'circuit': 'prodsat',
    'PprodA': 10**-1,
    'kcats': list(np.logspace(-4,0,9)),
}

# Pin random seed 
rng = np.random.default_rng(seed=metadata['seed'])

results = None
for i, Tcc in enumerate(metadata['Tccs']):
    for j, kcatA in enumerate(metadata['kcats']):

        print(f"Running simulation for Tcc={Tcc}, kcatA={kcatA}")

        motherCell = Cell(Tcc,metadata['varTcc'],rng)
        motherCell.parameterize(metadata['circuit'],[metadata['PprodA'],kcatA])
        motherCell.equilibrate(metadata['nCells_equilibrium'])
        motherCell.run(metadata['nCells'])

        # Get mother states and calculate division differences 
        divStates = motherCell.getMotherStates()
        dsis, drnd, vardsis, vardrnd, normvar = calculate_division_differences(divStates,rng)
    
        if results is None:
            nVars = dsis.shape[0]
            results = {
                'dsis': np.zeros((len(metadata['Tccs']), len(metadata['kcats']), nVars, metadata['nCells'])),
                'drnd': np.zeros((len(metadata['Tccs']), len(metadata['kcats']), nVars, metadata['nCells'])),
                'vardsis': np.zeros((len(metadata['Tccs']), len(metadata['kcats']), nVars)),
                'vardrnd': np.zeros((len(metadata['Tccs']), len(metadata['kcats']), nVars)),
                'normvar': np.zeros((len(metadata['Tccs']), len(metadata['kcats']), nVars)),
            }

        results['dsis'][i, j] = dsis
        results['drnd'][i, j] = drnd
        results['vardsis'][i, j] = vardsis
        results['vardrnd'][i, j] = vardrnd
        results['normvar'][i, j] = normvar


# Save results 
exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data = [metadata['kcats'], metadata['Tccs'], results],
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory']
)
print(f"Experiment saved to f{exp_dir}")

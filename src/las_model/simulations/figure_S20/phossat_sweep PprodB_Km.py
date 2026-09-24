#TODO: Run simulation for all values of PprodB and Km 

# Saturated Phosphorylation: Sweep PprodB and Km
import numpy as np 
from datetime import datetime 
from las_model.utils.cell import Cell
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_division_differences
from las_model.utils.output import save_experiment 

# Experiment metadata
metadata = {
    'experiment_name': 'phossat_sweep_PprodA_PprodB',
    'experiment_directory': 'phos_sat',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'Tcc': 1000,
    'varTcc': 0,
    'circuit': 'phos_sat',
    'PprodA': 10**-1,
    'PprodBs': list(np.logspace(-2,3,31)),
    'Kms': list(np.logspace(0,4,25)),
    'PprodC': 10**-1,
    'ka': 10**-2,
    'kc': 10**-2,
}

# Pin random seed 
rng = np.random.default_rng(seed=metadata['seed'])

results = None 
for i, PprodB in enumerate(metadata['PprodBs']):
    for j, Km in enumerate(metadata['Kms']):

        print(f"Running simulation {j}/{len(metadata['Kms'])} of group {i}/{len(metadata['PprodBs'])}")

        motherCell = Cell(metadata['Tcc'],metadata['varTcc'],rng)
        motherCell.parameterize(
            metadata['circuit'],
            [
                metadata['PprodA'],
                metadata['ka'],
                Km,
                PprodB,
                metadata['PprodC'],
                metadata['kc'],
                Km
            ]
        )
        motherCell.equilibrate(metadata['nCells_equilibrium'])
        motherCell.run(metadata['nCells'])

        # Get mother states and calculate division differences
        divStates = motherCell.getMotherStates()
        dsis, drnd, vardsis, vardrnd, normvar = calculate_division_differences(divStates,rng)

        # Get molecule amounts 
        molecules = motherCell.getMolecules()
        means = np.mean(molecules,axis=1)
        variances = np.var(molecules,axis=1)

        if results is None:
            nVars = dsis.shape[0]
            results = {
                'means': np.zeros((len(metadata['PprodBs']), len(metadata['Kms']), nVars)),
                'variances': np.zeros((len(metadata['PprodBs']), len(metadata['Kms']), nVars)),
                'dsis': np.zeros((len(metadata['PprodBs']), len(metadata['Kms']), nVars, metadata['nCells'])),
                'drnd': np.zeros((len(metadata['PprodBs']), len(metadata['Kms']), nVars, metadata['nCells'])),
                'vardsis': np.zeros((len(metadata['PprodBs']), len(metadata['Kms']), nVars)),
                'vardrnd': np.zeros((len(metadata['PprodBs']), len(metadata['Kms']), nVars)),
                'normvar': np.zeros((len(metadata['PprodBs']), len(metadata['Kms']), nVars)),
            }

        results['means'][i, j] = means
        results['variances'][i, j] = variances
        results['dsis'][i, j] = dsis
        results['drnd'][i, j] = drnd
        results['vardsis'][i, j] = vardsis
        results['vardrnd'][i, j] = vardrnd
        results['normvar'][i, j] = normvar

# Save results 
exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data = [metadata['PprodBs'], metadata['Kms'], results],
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory']
)
print(f"Experiment saved to f{exp_dir}")

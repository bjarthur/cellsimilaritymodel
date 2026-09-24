#TODO: run simulation for all values of PprodA and PprodB 

# Phosphorylation Monocycle: Sweep PprodA and PprodB
import numpy as np 
from datetime import datetime 
from las_model.utils.cell import Cell
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_division_differences
from las_model.utils.output import save_experiment 

# Experiment metadata
metadata = {
    'experiment_name': 'phoscycle_sweep_PprodA_PprodB',
    'experiment_directory': 'phos_cycle',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'Tcc': 1000,
    'varTcc': 0,
    'circuit': 'phos_cycle',
    'PprodAs': list(np.logspace(-3,2,31)),
    'PprodBs': list(np.logspace(-3,2,31)),
    'PprodC': 10**-1,
    'ka': 10**-1,
    'kc': 10**-1,
}

# Pin random seed 
rng = np.random.default_rng(seed=metadata['seed'])

results = None 
for i, PprodA in enumerate(metadata['PprodAs']):
    for j, PprodB in enumerate(metadata['PprodBs']):

        print(f"Running simulation {j}/{len(metadata['PprodBs'])} of group {i}/{len(metadata['PprodAs'])} for PprodA={PprodA}, PprodB={PprodB}")

        motherCell = Cell(metadata['Tcc'],metadata['varTcc'],rng)
        motherCell.parameterize(
            metadata['circuit'],
            [
                PprodA,
                PprodB,
                metadata['PprodC'],
                metadata['ka'],
                metadata['kc']
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
                'means': np.zeros((len(metadata['PprodAs']), len(metadata['PprodBs']), nVars)),
                'variances': np.zeros((len(metadata['PprodAs']), len(metadata['PprodBs']), nVars)),
                'dsis': np.zeros((len(metadata['PprodAs']), len(metadata['PprodBs']), nVars, metadata['nCells'])),
                'drnd': np.zeros((len(metadata['PprodAs']), len(metadata['PprodBs']), nVars, metadata['nCells'])),
                'vardsis': np.zeros((len(metadata['PprodAs']), len(metadata['PprodBs']), nVars)),
                'vardrnd': np.zeros((len(metadata['PprodAs']), len(metadata['PprodBs']), nVars)),
                'normvar': np.zeros((len(metadata['PprodAs']), len(metadata['PprodBs']), nVars)),
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
    data = [metadata['PprodAs'], metadata['PprodBs'], results],
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory']
)
print(f"Experiment saved to f{exp_dir}")

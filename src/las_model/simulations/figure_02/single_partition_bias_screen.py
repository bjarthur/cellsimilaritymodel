# Assymetric partitioning simulation 
import numpy as np 
from datetime import datetime 
from las_model.utils.cell import Cell
from las_model.utils.config import PROJECT_DIR
from las_model.utils.output import save_experiment 

# Experiment metadata
metadata = {
    'experiment_name': 'single_partition_bias_screen',
    'experiment_directory': 'asymmetric',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'Tcc': 1000,
    'Tcc_var': 0,
    'circuit': 'single',
    'PprodA': 10**-1,
    'division': 'asymmetric',
    'biases': list(np.linspace(0.55,1,10)),
}

# Pin random seed 
rng = np.random.default_rng(seed=metadata['seed'])

# Set arrays for saving data 
vardsis = []
vardrnd = []
normvar = []

# === Iterate over biases and simulate mother cells ==========================================
for bias in metadata['biases']:

    print(f"Simulating bias: {bias}")

    motherCell = Cell(metadata['Tcc'],metadata['Tcc_var'],rng)
    motherCell.parameterize(metadata['circuit'],[metadata['PprodA']])
    motherCell.equilibrate(metadata['nCells_equilibrium'])

    # Run simulation 
    motherCell.run(metadata['nCells'],metadata['division'],bias)

    # Save mother cell state 
    divStates = motherCell.getMotherStates()

    dsis = np.zeros([6,metadata['nCells']])
    drnd = np.zeros_like(dsis)

    for k in range(metadata['nCells']):

        # Choose bias direction 
        cell1_bias = rng.integers(2)
        cell2_bias = rng.integers(2)

        # Get mother cell state 
        motherCellState = divStates[:,k].astype('int')
        # print(f'mother cell state: {motherCellState}')
        
        # Partition sister cells 
        if cell1_bias==0:
            cell1 = (motherCellState * bias).astype('int')
        else:
            cell1 = (motherCellState * (1-bias)).astype('int')

        # Get random cell state 
        rndCellIndx = rng.integers(0,metadata['nCells'])
        rndCellState = divStates[:,rndCellIndx].astype('int')

        # Partition random cells 
        if cell2_bias == 0:
            cell2 = (rndCellState * bias).astype('int')
        else:
            cell2 = (rndCellState * (1-bias)).astype('int')

        # Compute pairwise differences
        dsis[:,k] = motherCellState - 2*cell1
        drnd[:,k] = cell1 - cell2

    # Compute variance of pairwise differences
    vardsis.append(np.var(dsis, axis=1))
    vardrnd.append(np.var(drnd, axis=1))
    normvar.append(1-vardsis[-1][0]/vardrnd[-1][0])

exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data=[metadata['biases'],vardsis,vardrnd,normvar],
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory'],
)
print(f"Experiment saved to {exp_dir}")

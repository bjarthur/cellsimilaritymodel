# Burst Size Sweep 
from datetime import datetime 
import numpy as np 
from las_model.utils.cell import Cell
from las_model.utils.config import PROJECT_DIR
from las_model.utils.output import save_experiment 

# Experiment metadata 
metadata = {
    'experiment_name': 'burstSize_prodA-0',
    'experiment_directory': 'burstSize',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCycles_equilibrium': 10,
    'Tcc': 1000,
    'circuit': 'prodsat_burst',
    'burstSizes': list(np.linspace(1,20,20)),
    'prodA_std': 10**0,
    # 'prodA_std': 10**-2,
    'kcatA': 10**-2
}

# Pin random seed 
rng = np.random.default_rng(seed=metadata['seed'])

# Set burst size array 
burstSizes = np.array(metadata['burstSizes'])
prodA_std = metadata['prodA_std']
prodAs = prodA_std / burstSizes
kcatA = metadata['kcatA']

# Initialize arrays to store simulation results
Aeqs = np.zeros([len(burstSizes)])
Beqs = np.zeros_like(Aeqs)
normvarAs = np.zeros_like(Aeqs)
normvarBs = np.zeros_like(Aeqs)

# === Iterate over burst sizes and simulate mother cells =========
for i in range(len(burstSizes)):

    # Set burst size and PprodA value
    burstSize = burstSizes[i]
    prodA = prodAs[i]

    print(f"Simulating burst size {burstSize} with prodA {prodA}")

    motherCell = Cell(metadata['Tcc'],0,rng)
    motherCell.parameterize(metadata['circuit'],[prodA,kcatA,burstSize])
    motherCell.equilibrate(metadata['nCycles_equilibrium'])

    # Run simulation 
    motherCell.run(metadata['nCells'])

    # Get molecules 
    molecules = motherCell.getMolecules()

    Aeqs[i] = np.mean(molecules[0])
    Beqs[i] = np.mean(molecules[1])
    
    divStates = motherCell.getMotherStates()

    dsis = np.zeros([metadata['nCells'],6])
    drnd = np.zeros([metadata['nCells'],6])

    for k in range(metadata['nCells']):
        cell1 = rng.binomial(divStates[:,k].astype('int'),0.5)
        cell2 = rng.binomial(divStates[:,rng.integers(0,metadata['nCells'])].astype('int'),0.5)
        
        dsis[k] = divStates[:,k] - 2*cell1
        drnd[k] = cell1 - cell2
    
    normvarAs[i] = 1-np.var(dsis[:,0],axis=0)/np.var(drnd[:,0],axis=0)
    normvarBs[i] = 1-np.var(dsis[:,1],axis=0)/np.var(drnd[:,1],axis=0)

    print(f"normvarAs[i]: {normvarAs[i]}, normvarBs[i]: {normvarBs[i]}")

# Save simulation results to a pickle file using the save_experiment utility
exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data=[burstSizes, prodAs, Aeqs, Beqs, normvarAs, normvarBs],
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory']
)
print(f"Saved experiment to {exp_dir}")

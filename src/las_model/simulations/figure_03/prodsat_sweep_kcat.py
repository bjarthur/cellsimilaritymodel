# Saturated Production Sweep kcat 
import numpy as np 
from datetime import datetime 
from las_model.utils.cell import Cell
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_division_differences
from las_model.utils.output import save_experiment 

# Experiment metadata
metadata = {
    'experiment_name': 'prodsat_sweep_kcat_low',
    'experiment_directory': 'satprod/prodsat_sweep_kcat_low/',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'Tcc': 1000,
    'varTcc': 0,
    'circuit': 'prodsat',
    'PprodA': 10**-2,
    'kcats': list(np.logspace(-4,0,9)),
}

# Pin random seed 
rng = np.random.default_rng(metadata['seed'])

# Accumulate results 
results = {
    'dsis': [],
    'drnd': [],
    'vardsis': [],
    'vardrnd': [],
    'normvar': [],
}

# === Iterate over kcatA values and simulate cells ==========
for kcatA in metadata['kcats']:

    print(f"Simulating for kcatA = {kcatA}")

    motherCell = Cell(metadata['Tcc'],metadata['varTcc'],rng)
    motherCell.parameterize(metadata['circuit'],[metadata['PprodA'],kcatA])
    motherCell.equilibrate(metadata['nCells_equilibrium'])

    motherCell.run(metadata['nCells'])

    # Get mother states and calculate division differences 
    divStates = motherCell.getMotherStates()
    dsis, drnd, vardsis, vardrnd, normvar = calculate_division_differences(divStates,rng)

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
    data = [metadata['kcats'],results],
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory']
)
print(f"Experiment saved to f{exp_dir}")

# nCells = 1000
# rng = np.random.default_rng(seed=1000)

# Tcc = 1000
# PprodA = 10**-2
# kcats = np.logspace(-4,0,9)

# motherCells = []
# divStates = np.zeros([len(kcats),5,nCells])

# for i in range(len(kcats)):

#     motherCell = Cell(Tcc,0)
#     motherCell.parameterize('prodsat',[PprodA,kcats[i]])
#     motherCell.run(nCells)
    
#     motherCells.append(motherCell)
    
#     divStates[i] = motherCell.getMotherStates()
    
# dsis = np.zeros([len(kcats),nCells,5])
# drnd = np.zeros([len(kcats),nCells,5])

# for j in range(len(kcats)):
#     for i in range(nCells):
#         cell1 = rng.binomial(divStates[j,:,i].astype('int'),0.5)
#         cell2 = rng.binomial(divStates[j,:,rng.integers(0,nCells)].astype('int'),0.5)
        
#         dsis[j,i] = divStates[j,:,i] - 2*cell1
#         drnd[j,i] = cell1 - cell2

# with open(PROJECT_DIR / 'analyticalData/motifs_prodsat_kcatsweep.pickle','wb') as f:
#     pickle.dump([kcats,motherCells,dsis,drnd],f,pickle.HIGHEST_PROTOCOL)

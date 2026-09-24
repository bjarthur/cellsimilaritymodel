# TODO: Run this for full sweep of PprodA and PprodB' values

# Reversible Binding: Sweep PprodA and PprodB
import numpy as np 
from datetime import datetime 
from las_model.utils.cell import Cell
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_division_differences
from las_model.utils.output import save_experiment 

# Experiment metadata
metadata = {
    'experiment_name': 'revbind_sweep_PprodA_PprodB',
    'experiment_directory': 'binding_rev',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'Tcc': 1000,
    'varTcc': 0,
    'circuit': 'revbind',
    'PprodAs': list(np.logspace(-3,2,31)),
    'PprodBs': list(np.logspace(-3,2,31)),
    'k1': 10**-3,
    'k2': 10**-5,
}

# Pin random seed 
rng = np.random.default_rng(seed=metadata['seed'])

results = None 
for i, PprodA in enumerate(metadata['PprodAs']):
    for j, PprodB in enumerate(metadata['PprodBs']):

        print(f"Running simulation for PprodA={PprodA}, PprodB={PprodB}")

        motherCell = Cell(metadata['Tcc'],metadata['varTcc'],rng)
        motherCell.parameterize(metadata['circuit'],[PprodA,PprodB,metadata['k1'],metadata['k2']])
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


# Tcc = 1000
# nCells = 2000
# rng = np.random.default_rng(seed=1000)

# prodAs = np.logspace(-3,2,31)
# prodBs = np.logspace(-3,2,31)

# prodAindex = int(sys.argv[1])
# prodA = prodAs[prodAindex]
# prodBindex = int(sys.argv[2])
# prodB = prodBs[prodBindex]

# k1 = 10**-3
# k2 = 10**-5

# motherCell = Cell(Tcc,0)
# motherCell.parameterize('revbind',[prodA,prodB,k1,k2])
# motherCell.run(nCells)

# Aeq = np.mean(motherCell.A/motherCell.V)
# Beq = np.mean(motherCell.B/motherCell.V)
# Ceq = np.mean(motherCell.C/motherCell.V)

# varA = np.var(motherCell.A/motherCell.V)
# varB = np.var(motherCell.B/motherCell.V)
# varC = np.var(motherCell.C/motherCell.V)

# divStates = motherCell.getMotherStates()

# dsis = np.zeros([nCells,6])
# drnd = np.zeros([nCells,6])

# for k in range(nCells):
#     cell1 = rng.binomial(divStates[:,k].astype('int'),0.5)
#     cell2 = rng.binomial(divStates[:,rng.integers(0,nCells)].astype('int'),0.5)
    
#     dsis[k] = divStates[:,k] - 2*cell1
#     drnd[k] = cell1 - cell2

# normvarA = 1-np.var(dsis[:,0],axis=0)/np.var(drnd[:,0],axis=0)
# normvarB = 1-np.var(dsis[:,1],axis=0)/np.var(drnd[:,1],axis=0)
# normvarC = 1-np.var(dsis[:,2],axis=0)/np.var(drnd[:,2],axis=0)

# with open(PROJECT_DIR / 'binding_rev/revbind4revbind4_prodA_%.2i_prodB_%.2i.pickle' % (prodAindex,prodBindex),'wb') as f:
#     pickle.dump([prodA,prodB,Aeq,Beq,Ceq,varA,varB,varC,normvarA,normvarB,normvarC],f,pickle.HIGHEST_PROTOCOL)



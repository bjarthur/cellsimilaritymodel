#TODO: Run simulation 

# Cascade 5 Time Sweep
import numpy as np 
from datetime import datetime 
from las_model.utils.cell import Cell
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_offspring_similarity_time 
from las_model.utils.output import save_experiment 

# Experiment metadata 
metadata = {
    'experiment_name': 'cascade5_time',
    'experiment_directory': 'cascade',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 20, 
    'nCycles': 20, 
    'Tcc': 1000,
    'varTcc': 0,
    'circuit': 'cascade6',
    'PprodA': 5*10**-3,
    'kcatA': 5*10**-3,
    'kcatB': 5*10**-3,
    'kcatC': 5*10**-3,
    'kcatD': 5*10**-3,
    'kcatE': 0,
}

# Pin random seed 
rng = np.random.default_rng(seed=metadata['seed'])

# Initialize and run mother cell 
motherCell = Cell(metadata['Tcc'],metadata['varTcc'],rng)
motherCell.parameterize(metadata['circuit'],[metadata['PprodA'],metadata['kcatA'],metadata['kcatB'],metadata['kcatC'],metadata['kcatD'],metadata['kcatE']])
motherCell.equilibrate(metadata['nCells_equilibrium'])
motherCell.run(metadata['nCells'])

# Get mother states and calculate division differences 
divStates = motherCell.getMotherStates()
dsis, drnd, vardsis, vardrnd, normvar = calculate_offspring_similarity_time(motherCell,metadata,rng)

results = {
    'dsis': dsis,
    'drnd': drnd,
    'vardsis': vardsis,
    'vardrnd': vardrnd,
    'normvar': normvar
}

# Save results 
exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data = results, 
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory']
)
print(f"Experiment saved to f{exp_dir}")


# nCells = 1000
# nCycles = 20
# rng = np.random.default_rng(seed=1000)

# PprodA = 5*10**-3
# kcatA = 5*10**-3
# kcatB = 5*10**-3
# kcatC = 5*10**-3
# kcatD = 5*10**-3
# kcatE = 0
# Tcc = 1000

# motherCell = Cell(Tcc,0)
# motherCell.parameterize('cascade6',[PprodA,kcatA,kcatB,kcatC,kcatD,kcatE])
# motherCell.equilibrate()
# motherCell.run(nCells)

# Aeq = np.mean(motherCell.A/motherCell.V)
# Beq = np.mean(motherCell.B/motherCell.V)
# Ceq = np.mean(motherCell.C/motherCell.V)
# Deq = np.mean(motherCell.D/motherCell.V)
# Eeq = np.mean(motherCell.E/motherCell.V)
# Feq = np.mean(motherCell.F/motherCell.V)
# means = [Aeq,Beq,Ceq,Deq,Eeq,Feq]

# Avar = np.var(motherCell.A/motherCell.V)
# Bvar = np.var(motherCell.B/motherCell.V)
# Cvar = np.var(motherCell.C/motherCell.V)
# Dvar = np.var(motherCell.D/motherCell.V)
# Evar = np.var(motherCell.E/motherCell.V)
# Fvar = np.var(motherCell.F/motherCell.V)
# variances = [Avar,Bvar,Cvar,Dvar,Evar,Fvar]

# divStates = motherCell.getMotherStates()
# concentrations = np.zeros([3,nCells,6,int(nCycles*Tcc/10+1)])

# dsis = np.zeros([nCells,6])
# drnd = np.zeros_like(dsis)

# for k in range(nCells):
#     sis1state = rng.binomial(divStates[:,k].astype('int'),0.5)
#     sis2state = divStates[:,k] - sis1state
#     rnd1state = rng.binomial(divStates[:,rng.integers(0,nCells)].astype('int'),0.5)
    
#     sis1 = Cell(Tcc,0)
#     sis1.inherit(motherCell,sis1state)
#     sis1.run(nCycles)
#     concentrations[0,k] = sis1.getMolecules()
    
#     sis2 = Cell(Tcc,0)
#     sis2.inherit(motherCell,sis2state)
#     sis2.run(nCycles)
#     concentrations[1,k] = sis2.getMolecules()
    
#     rnd1 = Cell(Tcc,0)
#     rnd1.inherit(motherCell,rnd1state)
#     rnd1.run(nCycles)
#     concentrations[2,k] = rnd1.getMolecules()

# vardsis = np.var(concentrations[0] - concentrations[1],axis=0)
# vardrnd = np.var(concentrations[0] - concentrations[2],axis=0)
# normvar = 1-vardsis/vardrnd

# with open(PROJECT_DIR / 'cascade/cascade5_time.pickle','wb') as f:
#     pickle.dump([means,variances,normvar],f,pickle.HIGHEST_PROTOCOL)


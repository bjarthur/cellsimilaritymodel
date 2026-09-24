# Sat prod effect of bursting on LAS duration experiment  
import pickle
import numpy as np 
from las_model.utils.cell import Cell
from las_model.utils.config import PROJECT_DIR

# Pin random seed 
rng = np.random.default_rng(seed=1000)

# Initialize cell parameters 
Tcc = 1000
circuit = 'prodsat'
prodA = 10**-2
kcatA = 10**-1
burstSizes = [1,5,10,20]

# Set simulation parameters 
nCells = 1000
nCycles_equilibrate = 10
nCycles = 10

vardsis = []
vardrnd = []
normvar = []

# ======== Iterate over burst sizes and simulate mother cells ================
for burstSize in burstSizes:
    motherCell = Cell(Tcc,0)
    motherCell.parameterize('prodsat_burst',[prodA/burstSize,kcatA,burstSize])
    motherCell.equilibrate(nCycles_equilibrate)

    # Run simulation 
    motherCell.run(nCells)

    # Save mother cell state 
    divStates = motherCell.getMotherStates()

    # Initialize molecules arrays 
    molecules = np.zeros([3,nCells,6,int(nCycles*Tcc/10+1)])

    for i in range(nCells):

        print(f"Burst size: {burstSize}, simulating cell pair {i}")

        sis1state = rng.binomial(divStates[:,i].astype('int'),0.5)
        sis2state = (divStates[:,i] - sis1state).astype('int')
        rnd1state = rng.binomial(divStates[:,rng.integers(0,nCells)].astype('int'),0.5)
        
        sis1 = Cell(Tcc,0)
        sis1.inherit(motherCell,sis1state)
        sis1.run(nCycles)
        molecules[0,i] = sis1.molecules

        sis2 = Cell(Tcc,0)
        sis2.inherit(motherCell,sis2state)
        sis2.run(nCycles)
        molecules[1,i] = sis2.molecules

        rnd1 = Cell(Tcc,0)
        rnd1.inherit(motherCell,rnd1state)
        rnd1.run(nCycles)
        molecules[2,i] = rnd1.molecules

    # compute variance of pairwise differences 
    vardsis.append(np.var(molecules[0] - molecules[1],axis=0)) 
    vardrnd.append(np.var(molecules[0] - molecules[2],axis=0)) 
    normvar.append(1-vardsis[-1][0:2]/vardrnd[-1][0:2]) 

with open(PROJECT_DIR / 'satprod_burst_time/satprod_burst_time_sweep.pickle','wb') as f:
    pickle.dump([burstSizes,vardsis,vardrnd,normvar],f,pickle.HIGHEST_PROTOCOL)
    
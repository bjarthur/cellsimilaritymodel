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
burstSize = 10
prodA_with_bursting = prodA/burstSize 

# Set simulation parameters 
nCells = 1000
nCycles_equilibrate = 10
nCycles = 10

# ================== Non-burst Cell ==========================

# Initialize mother Cell 
motherCell_no_burst = Cell(Tcc,0)
motherCell_no_burst.parameterize(circuit,[prodA,kcatA])
motherCell_no_burst.equilibrate(nCycles_equilibrate)

# Run simulation 
motherCell_no_burst.run(nCells)

# Save mother cell state 
divStates_no_burst = motherCell_no_burst.getMotherStates()

# ================== Bursting Cell ===========================
motherCell_burst = Cell(Tcc,0)
motherCell_burst.parameterize('prodsat_burst',[prodA_with_bursting,kcatA,burstSize])
motherCell_burst.equilibrate(nCycles_equilibrate)

# Run simulation 
motherCell_burst.run(nCells)

# Save mother cell state 
divStates_burst = motherCell_burst.getMotherStates()


# Initialize molecules arrays 
molecules_no_burst = np.zeros([3,nCells,6,int(nCycles*Tcc/10+1)])
molecules_burst = np.zeros_like(molecules_no_burst)

for i in range(nCells):

    print(f"simulating cell pair {i}")

    sis1state = rng.binomial(divStates_no_burst[:,i].astype('int'),0.5)
    sis2state = (divStates_no_burst[:,i] - sis1state).astype('int')
    rnd1state = rng.binomial(divStates_no_burst[:,rng.integers(0,nCells)].astype('int'),0.5)
    
    sis1 = Cell(Tcc,0)
    sis1.inherit(motherCell_no_burst,sis1state)
    sis1.run(nCycles)
    molecules_no_burst[0,i] = sis1.molecules

    sis2 = Cell(Tcc,0)
    sis2.inherit(motherCell_no_burst,sis2state)
    sis2.run(nCycles)
    molecules_no_burst[1,i] = sis2.molecules

    rnd1 = Cell(Tcc,0)
    rnd1.inherit(motherCell_no_burst,rnd1state)
    rnd1.run(nCycles)
    molecules_no_burst[2,i] = rnd1.molecules

    # ================== Bursting Pairs =================================
    sis1state = rng.binomial(divStates_burst[:,i].astype('int'),0.5)
    sis2state = (divStates_burst[:,i] - sis1state).astype('int')
    rnd1state = rng.binomial(divStates_burst[:,rng.integers(0,nCells)].astype('int'),0.5)
    
    sis1 = Cell(Tcc,0)
    sis1.inherit(motherCell_burst,sis1state)
    sis1.run(nCycles)
    molecules_burst[0,i] = sis1.molecules

    sis2 = Cell(Tcc,0)
    sis2.inherit(motherCell_burst,sis2state)
    sis2.run(nCycles)
    molecules_burst[1,i] = sis2.molecules

    rnd1 = Cell(Tcc,0)
    rnd1.inherit(motherCell_burst,rnd1state)
    rnd1.run(nCycles)
    molecules_burst[2,i] = rnd1.molecules

# compute variance of pairwise differences 
vardsis_no_burst = np.var(molecules_no_burst[0] - molecules_no_burst[1],axis=0)
vardrnd_no_burst = np.var(molecules_no_burst[0] - molecules_no_burst[2],axis=0)
normvar_no_burst = 1-vardsis_no_burst[0:2]/vardrnd_no_burst[0:2]

vardsis_burst = np.var(molecules_burst[0] - molecules_burst[1],axis=0)
vardrnd_burst = np.var(molecules_burst[0] - molecules_burst[2],axis=0)
normvar_burst = 1-vardsis_burst[0:2]/vardrnd_burst[0:2]

with open(PROJECT_DIR / 'satprod_burst_time/satprod_burst_time.pickle','wb') as f:
    pickle.dump([vardsis_no_burst,vardrnd_no_burst,normvar_no_burst,vardsis_burst,vardrnd_burst,normvar_burst],f,pickle.HIGHEST_PROTOCOL)
    
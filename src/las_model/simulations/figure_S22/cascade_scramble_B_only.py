# Sat prod scrambled A experiment
import pickle
import numpy as np 
from las_model.utils.cell import Cell
from las_model.utils.config import PROJECT_DIR

# Pin random seed 
rng = np.random.default_rng(seed=1000)

# Initialize cell parameters 
Tcc = 1000
circuit = 'cascade'
prodA = 10**-3
kcatA = 10**-2
kcatB = 10**-2

# Set simulation parameters 
nCells = 1000
nCycles = 10

# Initialize storage arrays 
motherCells = []
divStates = np.zeros([6,nCells])

# Initialize mother Cell 
motherCell = Cell(Tcc,0)
motherCell.parameterize(circuit,[prodA,kcatA,kcatB])
motherCell.equilibrate(10)

# Run simulation 
motherCell.run(nCells)

# Save mother cell state 
motherCells.append(motherCell)
divStates = motherCell.getMotherStates()

# Initialize storage array for offspring cells
molecules_inherited = np.zeros([3,nCells,6,int(nCycles*Tcc/10+1)])
molecules_scrambled = np.zeros_like(molecules_inherited)

for i in range(nCells):
    print(f"Running offspring pair {i}...")
    sis1state = rng.binomial(divStates[:,i].astype('int'),0.5)
    sis2state = (divStates[:,i] - sis1state).astype('int')
    rnd1state = rng.binomial(divStates[:,rng.integers(0,nCells)].astype('int'),0.5)
    # print(f"""
    #       For offspring simulation {i}
    #       sister cell 1 is [A: {sis1state[0]}, B: {sis1state[1]}],
    #       sister cell 2 is [A: {sis2state[0]}, B: {sis2state[1]}],
    #       random cell 1 is [A: {rnd1state[0]}, B: {rnd1state[1]}]
    #       """)

    sis1 = Cell(Tcc,0)
    sis1.inherit(motherCell,sis1state)
    sis1.run(nCycles)
    molecules_inherited[0,i] = sis1.molecules

    sis2 = Cell(Tcc,0)
    sis2.inherit(motherCell,sis2state)
    sis2.run(nCycles)
    molecules_inherited[1,i] = sis2.molecules

    rnd1 = Cell(Tcc,0)
    rnd1.inherit(motherCell,rnd1state)
    rnd1.run(nCycles)
    molecules_inherited[2,i] = rnd1.molecules

    # Scramble inherited B only 
    sis1state_new = sis1state
    newMother_1 = rng.integers(0,nCells)
    newB_1 = rng.binomial(divStates[1,newMother_1].astype(int),0.5)
    sis1state_new[1] = newB_1

    sis2state_new = sis2state
    newMother_2 = rng.integers(0,nCells)
    newB_2 = rng.binomial(divStates[1,newMother_2].astype(int),0.5)
    sis2state_new[1] = newB_2

#     print(f"""For offspring simulation {i}
# Mother cell state is [A: {divStates[0,i]}, B: {divStates[1,i]}]
# Original sister 1 state is [A: {sis1state[0]}, B: {sis1state[1]}]
# New A is from mother cell {newMother_1} and is {newA_1}
# """)
    
    sis1_new = Cell(Tcc,0)
    sis1_new.inherit(motherCell,sis1state_new)
    sis1_new.run(nCycles)
    molecules_scrambled[0,i] = sis1_new.molecules

    sis2_new = Cell(Tcc,0)
    sis2_new.inherit(motherCell,sis2state_new)
    sis2_new.run(nCycles)
    molecules_scrambled[1,i] = sis2_new.molecules
    
    molecules_scrambled[2,i] = rnd1.molecules


vardsis_inherited = np.var(molecules_inherited[0] - molecules_inherited[1],axis=0)
vardrnd_inherited = np.var(molecules_inherited[0] - molecules_inherited[2],axis=0)
normvar_inherited = 1-vardsis_inherited[0:3]/vardrnd_inherited[0:3]

vardsis_scrambled = np.var(molecules_scrambled[0] - molecules_scrambled[1],axis=0)
vardrnd_scrambled = np.var(molecules_scrambled[0] - molecules_scrambled[2],axis=0)
normvar_scrambled = 1-vardsis_scrambled[0:3]/vardrnd_scrambled[0:3]

# save to file
with open(PROJECT_DIR / 'cascade_scramble/cascade_scramble_B_only.pickle','wb') as f:
    pickle.dump([motherCells,divStates,molecules_inherited,molecules_scrambled,vardrnd_inherited,vardsis_inherited,vardrnd_scrambled,vardsis_scrambled,normvar_inherited,normvar_scrambled],f,pickle.HIGHEST_PROTOCOL)

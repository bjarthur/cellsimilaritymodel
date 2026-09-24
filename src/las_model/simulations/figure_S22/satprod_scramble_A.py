# Sat prod scrambled A experiment
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
kcatA = 10**-2

# Set simulation parameters 
nCells = 1000
nCycles = 10

# Initialize storage arrays 
motherCells = []
divStates = np.zeros([6,nCells])

# Initialize mother Cell 
motherCell = Cell(Tcc,0)
motherCell.parameterize(circuit,[prodA,kcatA])
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

    # define inherited states for scrambled cells
    sis1state_new = sis1state
    newMother_1 = rng.integers(0,nCells)
    newA_1 = rng.binomial(divStates[0,newMother_1].astype(int),0.5)
    sis1state_new[0] = newA_1

    sis2state_new = sis2state
    newMother_2 = rng.integers(0,nCells)
    newA_2 = rng.binomial(divStates[0,newMother_2].astype(int),0.5)
    sis2state_new[0] = newA_2

    print(f"""For offspring simulation {i}
Mother cell state is [A: {divStates[0,i]}, B: {divStates[1,i]}]
Original sister 1 state is [A: {sis1state[0]}, B: {sis1state[1]}]
New A is from mother cell {newMother_1} and is {newA_1}
""")
    
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
normvar_inherited = 1-vardsis_inherited[0:2]/vardrnd_inherited[0:2]

vardsis_scrambled = np.var(molecules_scrambled[0] - molecules_scrambled[1],axis=0)
vardrnd_scrambled = np.var(molecules_scrambled[0] - molecules_scrambled[2],axis=0)
normvar_scrambled = 1-vardsis_scrambled[0:2]/vardrnd_scrambled[0:2]

# save to file
# print(f"vardsis\nA:{vardsis[0]}\nB: {vardsis[1]}")
import matplotlib.pyplot as plt 

plt.figure(figsize=(16,8))
plt.subplot(2,4,1)
plt.plot(vardsis_inherited[0],color='r')
plt.plot(vardrnd_inherited[0],color='grey')

plt.subplot(2,4,2)
plt.plot(vardsis_inherited[1],color='b')
plt.plot(vardrnd_inherited[1],color='grey')

plt.subplot(2,4,3)
plt.plot(normvar_inherited[0],color='r')
plt.plot(normvar_inherited[1],color='b')

plt.subplot(2,4,5)
plt.plot(vardsis_scrambled[0],color='r')
plt.plot(vardrnd_scrambled[0],color='grey')

plt.subplot(2,4,6)
plt.plot(vardsis_scrambled[1],color='b')
plt.plot(vardrnd_scrambled[1],color='grey')

plt.subplot(2,4,7)
plt.plot(normvar_scrambled[0],color='r')
plt.plot(normvar_scrambled[1],color='b')

plt.subplot(2,4,8)
plt.plot(normvar_inherited[1], color='b', label='inherited A')
plt.plot(normvar_scrambled[1],color='black',label='scrambled A')

plt.show()


# with open(PROJECT_DIR / 'analyticalData/motifs_prodsat.pickle','wb') as f:
#     pickle.dump([prodAs,motherCells,dsis,drnd],f,pickle.HIGHEST_PROTOCOL)

# import matplotlib.pyplot as plt 

# plt.figure(figsize=(10,6))
# plt.subplot(2,3,1)
# plt.hist(divStates[0,0],density=1)
# plt.xlabel('Mother A')

# plt.subplot(2,3,4)
# plt.hist(divStates[0,1])
# plt.xlabel('Mother B')

# plt.subplot(2,3,2)
# plt.hist(cell1[0,:,0],density=1)
# plt.xlabel('Daughter A')

# plt.subplot(2,3,5)
# plt.hist(cell1[0,:,1],density=1)
# plt.xlabel('Daughter B')

# plt.subplot(2,3,3)
# plt.hist(dsis[0,:,0],density=1,color='r')
# plt.hist(drnd[0,:,0],density=1,color='grey')
# plt.xlabel('$\Delta_A$')

# plt.subplot(2,3,6)
# plt.hist(dsis[0,:,0],density=1,color='b')
# plt.hist(drnd[0,:,1],density=1,color='grey')
# plt.xlabel('$\Delta_B$')

# plt.tight_layout()
# plt.show()
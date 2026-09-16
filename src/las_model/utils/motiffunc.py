# Cell Class

import numpy as np 
import math
from las_model.utils.gillespie_numba import run_cycle_numba, pack_params, to_scalar, step_reaction

class Cell:
    def __init__(self,Tcc,varTcc,rng):
        self.Tcc = Tcc
        self.varTcc = varTcc
        self.rng = rng 
        self.divTime = self.rng.normal(self.Tcc,self.varTcc)
        self.divTimes = np.array([self.divTime])
        self.prodA = 0
        self.prodB = 0
        self.prodC = 0
        self.k1 = 0
        self.k2 = 0 
        self.k3 = 0
        self.k4 = 0
        self.k5 = 0
        self.k6 = 0
        self.k7 = 0
        self.k8 = 0
        self.burstSize = 1
        self.arrSize = int(1e7)
        self.t = np.array([0])
        self.V = np.array([1])
        self.A = np.array([0])
        self.B = np.array([0])
        self.C = np.array([0])
        self.D = np.array([0])
        self.E = np.array([0])
        self.F = np.array([0])
        self._init_buffers()

    def _init_buffers(self):
        self.t_array = np.empty(self.arrSize)
        self.V_array = np.empty(self.arrSize)
        self.A_array = np.empty(self.arrSize)
        self.B_array = np.empty(self.arrSize)
        self.C_array = np.empty(self.arrSize)
        self.D_array = np.empty(self.arrSize)
        self.E_array = np.empty(self.arrSize)
        self.F_array = np.empty(self.arrSize)

    def __getstate__(self):
        """Exclude pre-allocated buffer arrays from being pickled."""
        state = self.__dict__.copy()
        buffers = ['t_array', 'V_array', 'A_array', 'B_array', 'C_array', 'D_array', 'E_array', 'F_array']
        for key in buffers:
            state.pop(key, None)
        return state

    #def __setstate__(self, state):
    #    """Restore instance state and re-initialize buffer arrays when loaded."""
    #    self.__dict__.update(state)
    #    self._init_buffers()  # Omit this line if buffers are not needed after unpickling
        
    def parameterize(self,circuit,params):
        self.circuit = circuit
        if circuit == 'single':
            self.prodA = params[0]
            self.A[0] = self.prodA * self.Tcc
        
        elif circuit == 'pos_fb':
            self.prodA = params[0]
            self.k1 = params[1]
            self.k2 = params[2]
            self.k3 = params[3]
            
            self.A[0] = self.prodA + self.k1 * self.Tcc
            
        elif circuit == 'neg_fb':
            self.k1 = params[0]
            self.k2 = params[1]
            self.k3 = params[2]
            
            self.A[0] = self.prodA * self.Tcc - self.k1 * self.Tcc
        
        elif circuit == 'bind':
            self.prodA = params[0]
            self.prodB = params[1]
            self.k1 = params[2]
            
            self.A[0] = self.prodA * self.Tcc
            self.B[0] = self.prodB * self.Tcc
        
        elif circuit == 'bind2':
            self.prodA = params[0]
            self.prodB = params[1]
            self.k1 = params[2]/(self.prodA*self.prodB*self.Tcc**2)
            
            self.A[0] = self.prodA * self.Tcc
            self.B[0] = self.prodB * self.Tcc
        
        elif circuit == 'revbind':
            self.prodA = params[0]
            self.prodB = params[1]
            self.k1 = params[2]/(self.prodA*self.prodB*self.Tcc**2)
            self.k2 = params[3]/(np.min((self.prodA,self.prodB))*self.Tcc)
        
            self.A[0] = self.prodA * self.Tcc
            self.B[0] = self.prodB * self.Tcc
            self.C[0] = 0
        
        elif circuit == 'prodsat':
            self.prodA = params[0]
            self.k1 = params[1]
            
            self.A[0] = self.prodA * self.Tcc
            self.B[0] = 3/2 * self.prodA * self.k1 * self.Tcc**2
        
        elif circuit == 'prodsat_burst':
            self.prodA = params[0]
            self.k1 = params[1]
            self.burstSize = params[2]
            
            self.A[0] = self.prodA * self.burstSize * self.Tcc
            self.B[0] = 3/2 * self.prodA * self.burstSize * self.k1 * self.Tcc**2
            
        elif circuit == 'produnsat':
            self.prodA = params[0]
            self.prodB = params[1]
            self.k1 = params[2]
            self.k2 = params[3]
            
            Ai = int(self.prodB * self.Tcc)
            Bi = int(self.prodA * self.Tcc)
            reactionFrac = self.k1/2*(Ai+Bi+self.k2-np.sqrt((Ai+Bi+self.k2)**2-4*Ai*Bi))/(self.k1*Ai)
            
            self.A[0] = int(Bi * reactionFrac)
            self.B[0] = Ai
            self.C[0] = Bi - self.A[0]
            
        elif circuit == 'prod_fixedB':
            self.prodA = params[0]
            self.prodB = params[1]
            self.k1 = params[2]
            self.k2 = params[3]
        
            self.A[0] = int(self.prodA *self.Tcc)
            self.B[0] = int(self.prodB * self.Tcc)
            self.C[0] = int(4* self.k1 * self.Tcc * self.prodA * self.Tcc * self.prodB * self.Tcc / (self.k2+self.prodA * self.Tcc * self.prodB * self.Tcc))
        
        elif circuit =='cascade':
            self.prodA = params[0]
            self.k1 = params[1]
            self.k2 = params[2]
            
            self.A[0] = self.prodA * self.Tcc
            self.B[0] = 3/2 * self.prodA * self.k1 * self.Tcc**2
            self.C[0] = 3/2 * 3/2 * self.prodA * self.k1 * self.Tcc**3
            
        elif circuit == 'proddeg':
            self.prodA = params[0]
            self.prodB = params[1]
            self.k1 = params[2]
            self.k2 = params[3]
            self.k3 = params[4]
            
            self.A[0] = self.prodA * self.Tcc
            self.B[0] = self.prodB * self.Tcc
            self.C[0] = self.k3
            
        elif circuit == 'phos':
            self.prodA = params[0]
            self.prodB = params[1]
            self.k1 = params[2]
            self.k2 = params[3]
        
        elif circuit == 'phos_int':
            self.prodA = params[0]
            self.prodB = params[1]
            self.k1 = params[2]
            self.k2 = params[3]
            self.k3 = params[4]
            
            self.A[0] = self.prodA * self.Tcc
            self.C[0] = self.prodB * self.Tcc
        
        elif circuit == 'phos_cycle':
            self.prodA = params[0]
            self.prodB = params[1]
            self.prodC = params[2]
            self.k1 = params[3]/(self.prodA*self.prodB*self.Tcc**2)
            self.k2 = params[4]/(self.prodC*self.prodB*self.Tcc**2)
            
            self.A[0] = self.prodA * self.Tcc
            self.B[0] = self.prodB * self.Tcc // 2
            self.C[0] = self.prodC * self.Tcc
            self.D[0] = self.prodB * self.Tcc -  self.B[0]
        
        elif circuit == 'phos_sat':
            self.prodA = params[0]
            self.k1 = params[1]
            self.k2 = params[2]
            self.prodB = params[3]
            self.prodC = params[4]
            self.k3 = params[5]
            self.k4 = params[6]
            
            self.A[0] = self.prodA * self.Tcc
            self.B[0] = self.prodB * self.Tcc // 2
            self.C[0] = self.prodC * self.Tcc
            self.D[0] = self.prodB * self.Tcc -  self.B[0]
        
        elif circuit == 'phos2':
            self.prodA = params[0]
            self.prodB = params[1]
            self.k1 = params[2]
            self.k2 = params[3]
            self.k3 = params[4]
            self.k4 = params[5]
            self.k5 = params[6]
            self.k6 = params[7]
            self.k7 = params[8]
            self.k8 = params[9]
            
            self.A[0] = self.prodA * self.Tcc
            self.C[0] = self.prodB * self.Tcc
        
        elif circuit == 'diffTF':
            self.prodA = params[0]
            self.prodB = params[1]
            self.k1 = params[2]/(self.prodA*self.prodB*self.Tcc**2)
            self.k2 = params[3]
            
            self.A[0] = self.prodA * self.Tcc / 2
            self.B[0] = self.prodB * self.Tcc / 2
            self.C[0] = self.prodA * self.Tcc / 2
            self.D[0] = self.C[0] * self.k2 * self.Tcc
            
        elif circuit == 'cdg':
            self.prodA = params[0]
            self.prodB = params[1]
            self.k1 = params[2]
            self.k2 = params[3]
            self.k3 = params[4]
            self.k4 = params[5]
            
            self.A[0] = self.prodA * self.Tcc
            self.B[0] = self.prodB * self.Tcc
            self.C[0] = self.k3
            self.D[0] = self.C[0] * self.k4 * self.Tcc
        
        elif circuit == 'cascade6':
            self.prodA = params[0]
            self.k1 = params[1]
            self.k2 = params[2]
            self.k3 = params[3]
            self.k4 = params[4]
            self.k5 = params[5]
            
            self.A[0] = self.prodA * self.Tcc
            self.B[0] = self.A[0] * (3/2*self.k1*self.Tcc)
            self.C[0] = self.B[0] * (3/2*self.k2*self.Tcc)
            self.D[0] = self.C[0] * (3/2*self.k3*self.Tcc)
            self.E[0] = self.D[0] * (3/2*self.k4*self.Tcc)
            self.F[0] = self.E[0] * (3/2*self.k5*self.Tcc)
        
        elif circuit =='crazyCascade':
            self.prodA = params[0]      # PprodA
            self.k1 = params[1]         # kcatA
            self.prodB = params[2]      # PprodC
            self.k2 = params[3]         # kcatC
            self.k3 = params[4]         # KM,C
            self.k4 = params[5]         # kcatB
            self.prodC = params[6]      # PprodE
            self.k5 = params[7]         # kcatE
            self.k6 = params[8]         # KM,E
            self.k7 = params[9]         # kcatD
        
        else: # TCS
            self.prodA = params[0]
            self.prodB = params[1]
            self.k1 = params[2]
            self.k2 = params[3]
            self.k3 = params[4]
            self.k4 = params[5]

        
    
        
    def equilibrate(self,nCycles,partition='binomial',bias=0):
        
        # create array to store mother states during equilibration cycles 
        self.motherStates = np.zeros([8,nCycles])
        
        # create array to store molecule amounts during equilibration 
        self.molecules = np.zeros([6,int(nCycles*self.Tcc/10+1)])
        
        # set partition bias if not binomial
        if partition == 'asymmetric':
            self.partitionBias = bias
        
        # run equilibration cycles 
        for i in range(nCycles):
            self.cellCycle(partition,i)
        
        self.sampleCycle()
        print(f"Setting array size as: {self.arrSize}")
    

    def inherit(self,motherCell,motherState):
        self.circuit = motherCell.circuit
        if self.arrSize != motherCell.arrSize:
            self.arrSize = motherCell.arrSize
            self._init_buffers()
        self.prodA = motherCell.prodA
        self.prodB = motherCell.prodB
        self.prodC = motherCell.prodC
        self.k1 = motherCell.k1
        self.k2 = motherCell.k2
        self.k3 = motherCell.k3
        self.k4 = motherCell.k4 
        self.k5 = motherCell.k5 
        self.k6 = motherCell.k6
        self.k7 = motherCell.k7
        self.k8 = motherCell.k8
        self.burstSize = motherCell.burstSize
        
        self.A[0] = motherState[0]
        self.B[0] = motherState[1]
        self.C[0] = motherState[2]
        self.D[0] = motherState[3]
        self.E[0] = motherState[4]
        self.F[0] = motherState[5]

    def run(self,nCycles,partition='binomial',bias=0):
        
        # reset time 
        self.t = 0
        
        # create array to store mother states during equilibration cycles 
        self.motherStates = np.zeros([8,nCycles])
        
        # create array to store molecule amounts over time 
        self.molecules = np.zeros([6,int(nCycles*self.Tcc/10+1)])
        
        if partition == 'asymmetric':
            self.partitionBias = bias
        
        for i in range(nCycles):
            self.cellCycle(partition,i)
            
            

    def cellCycle(self,partition,cycleIndex):
        # print(f"Running cycle {cycleIndex}")
        self.runCycle(cycleIndex)

        # print(f"After cycle {cycleIndex}, time is {self.motherStates[0]}")
        
        # store downsampled molecules amounts 
        # self.molcules[cycleIndex
        
        # set time to last divTime
        self.t = self.motherStates[0,cycleIndex]
        
        
        # reset volume to 1
        self.V = 1
        
        if partition == 'binomial':
            
            if self.circuit=='prod_fixedB':
                self.A = self.prodA*self.Tcc
            else:
                self.A = self.rng.binomial(self.motherStates[2,cycleIndex],0.5)
            
            self.B = self.rng.binomial(self.motherStates[3,cycleIndex],0.5)
            self.C = self.rng.binomial(self.motherStates[4,cycleIndex],0.5)
            self.D = self.rng.binomial(self.motherStates[5,cycleIndex],0.5)
            self.E = self.rng.binomial(self.motherStates[6,cycleIndex],0.5)
            self.F = self.rng.binomial(self.motherStates[7,cycleIndex],0.5)
        elif partition == 'perfect':
            self.A = np.array([self.A[-1]//2])
            self.B = np.array([self.B[-1]//2])
            self.C = np.array([self.C[-1]//2])
            self.D = np.array([self.D[-1]//2])
            self.E = np.array([self.E[-1]//2])
            self.F = np.array([self.F[-1]//2])
        elif partition == 'correlated':
            coef = self.rng.normal(0.5,0.1)
            self.A = np.array([int(self.A[-1]*coef)])
            self.B = np.array([int(self.B[-1]*coef)])
            self.C = np.array([int(self.C[-1]*coef)])
            self.D = np.array([int(self.D[-1]*coef)])
            self.E = np.array([int(self.E[-1]*coef)])
            self.F = np.array([int(self.F[-1]*coef)])
        elif partition =='asymmetric':
            if self.rng.integers(2) == 0:
                coef = self.partitionBias
            else:
                coef = 1-self.partitionBias
           
            self.A = self.motherStates[2,cycleIndex] * coef
            self.B = self.motherStates[3,cycleIndex] * coef
            self.C = self.motherStates[4,cycleIndex] * coef
            self.D = self.motherStates[5,cycleIndex] * coef
            self.E = self.motherStates[6,cycleIndex] * coef
            self.F = self.motherStates[7,cycleIndex] * coef
        else:
            print('invalid partition')
            return


    def updateDivTimes(self):
        self.divTime = self.rng.normal(self.Tcc,self.varTcc)

        # check if divTime is negative
        while self.divTime < 0:
            self.divTime = self.rng.normal(self.Tcc,self.varTcc)

        self.divTimes = np.concatenate((self.divTimes,np.array([self.divTimes[-1] + self.divTime])))
        # print(f"updated div times to {self.divTime}")

    def sampleCycle(self):
        growthRate = 1/self.divTime
        params = pack_params(self)
        n, _, _, _, _, _, _, _, _, overflow = run_cycle_numba(
            self.circuit,
            to_scalar(self.A), to_scalar(self.B), to_scalar(self.C),
            to_scalar(self.D), to_scalar(self.E), to_scalar(self.F),
            to_scalar(self.V), to_scalar(self.t),
            growthRate, params, self.rng,
            self.t_array, self.V_array, self.A_array, self.B_array,
            self.C_array, self.D_array, self.E_array, self.F_array,
            len(self.t_array)
        )
        print(f"Sample cycle n is {n}, setting self.arrSize to {int(n*5)}")
        self.arrSize = int(n * 5)
        
        if self.arrSize < 1e4:
            self.arrSize = int(1e5)

        self._init_buffers()

        print(f"Setting self.arrSize to: {self.arrSize}")

    def runCycle(self,cycleIndex):

        self.updateDivTimes()
        growthRate = 1/self.divTime
        params = pack_params(self)

        n, _, _, _, _, _, _, _, _, overflow = run_cycle_numba(
            self.circuit,
            to_scalar(self.A), to_scalar(self.B), to_scalar(self.C),
            to_scalar(self.D), to_scalar(self.E), to_scalar(self.F),
            to_scalar(self.V), to_scalar(self.t),
            growthRate, params, self.rng,
            self.t_array, self.V_array, self.A_array, self.B_array,
            self.C_array, self.D_array, self.E_array, self.F_array,
            len(self.t_array)
        )
        while overflow:
            self.arrSize *= 2
            self._init_buffers()
            n, _, _, _, _, _, _, _, _, overflow = run_cycle_numba(
                self.circuit,
                to_scalar(self.A), to_scalar(self.B), to_scalar(self.C),
                to_scalar(self.D), to_scalar(self.E), to_scalar(self.F),
                to_scalar(self.V), to_scalar(self.t),
                growthRate, params, self.rng,
                self.t_array, self.V_array, self.A_array, self.B_array,
                self.C_array, self.D_array, self.E_array, self.F_array,
                len(self.t_array)
            )
        
        # print('Array size: %i, cycle size: %i' % (self.arrSize,n))
        
        # print(f"At the end of cycle {cycleIndex}, t={self.t_array[n-1]}, V={self.V_array[n-1]}, A={self.A_array[n-1]}, B={self.B_array[n-1]}")

        # update mother state
        self.motherStates[0,cycleIndex] = self.t_array[n-1]
        self.motherStates[1,cycleIndex] = self.V_array[n-1]
        self.motherStates[2,cycleIndex] = self.A_array[n-1]
        self.motherStates[3,cycleIndex] = self.B_array[n-1]
        self.motherStates[4,cycleIndex] = self.C_array[n-1]
        self.motherStates[5,cycleIndex] = self.D_array[n-1]
        self.motherStates[6,cycleIndex] = self.E_array[n-1]
        self.motherStates[7,cycleIndex] = self.F_array[n-1]
        
        # update downsampled molecule tracker
        times = np.linspace(t_array[0],t_array[n-1],int(self.Tcc/10)+1)
        pos = np.searchsorted(t_array[:n], times)
        pos = np.clip(pos, 1, n - 1)
        left = t_array[pos - 1]
        right = t_array[pos]

        indices = np.where((times - left) <= (right - times), pos - 1, pos)
        
        # print(indices)
        
        molecules = np.zeros([6,len(indices)])
        molecules[0] = self.A_array[indices]/self.V_array[indices]
        molecules[1] = self.B_array[indices]/self.V_array[indices]
        molecules[2] = self.C_array[indices]/self.V_array[indices]
        molecules[3] = self.D_array[indices]/self.V_array[indices]
        molecules[4] = self.E_array[indices]/self.V_array[indices]
        molecules[5] = self.F_array[indices]/self.V_array[indices]
        
        startIndex = cycleIndex*int(self.Tcc/10)
        endIndex = (cycleIndex+1)*int(self.Tcc/10)+1
        
        # print('start index: %i, end index: %i' % (startIndex,endIndex))
        
        self.molecules[:,startIndex:endIndex] = molecules
        
    def reaction(self,A,B,C,D,E,F,V):
        params = pack_params(self)
        return step_reaction(self.circuit, A, B, C, D, E, F, V, params, self.rng)

    def getMotherStates(self):
        
        return self.motherStates[2::]
    
    def getMotherStates2(self):
        motherStates = np.zeros([5,len(self.divTimes)-1])
        
        divIndices = np.where(self.V > 2)[0]
        
        motherStates[0] = self.A[divIndices]
        motherStates[1] = self.B[divIndices]
        motherStates[2] = self.C[divIndices]
        motherStates[3] = self.D[divIndices]
        motherStates[4] = self.E[divIndices]
        
        return motherStates

    def getIntegerTimes(self):
        print(f"Getting integer times, self.motherStates: {self.motherStates[0]}")
        times = self.motherStates[0].astype(int)
        print(f"Times are: {times}")

        print(f"self.A: {self.A}")

        return times

        # t_repeat = np.repeat(self.t[:,np.newaxis],len(times),axis=1)
        
        # return np.argmin(abs(np.subtract(t_repeat,times)),axis=0)
        
    def getMolecules(self):
       
        return self.molecules

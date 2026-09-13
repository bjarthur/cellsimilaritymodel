# Cell Class

import numpy as np 
# rng = np.random.default_rng(seed=1000)

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
        
        self.t_array[0] = self.t
        self.V_array[0] = self.V
        self.A_array[0] = self.A
        self.B_array[0] = self.B
        self.C_array[0] = self.C
        self.D_array[0] = self.D
        self.E_array[0] = self.E
        self.F_array[0] = self.F
        
        n = 1
        while self.V_array[n-1] < 2:
            
            # update arrays 
            self.V_array[n] = self.V_array[n-1]
            self.A_array[n] = self.A_array[n-1]
            self.B_array[n] = self.B_array[n-1]
            self.C_array[n] = self.C_array[n-1]
            self.D_array[n] = self.D_array[n-1]
            self.E_array[n] = self.E_array[n-1]
            self.F_array[n] = self.F_array[n-1]
            
            # calculate reaction for time step 
            self.A_array[n],self.B_array[n],self.C_array[n],self.D_array[n],self.E_array[n],self.F_array[n],tau = self.reaction(self.A_array[n],self.B_array[n],self.C_array[n],self.D_array[n],self.E_array[n],self.F_array[n],self.V_array[n])
            
            # calculate cell growth 
            self.V_array[n] = self.V_array[n] + tau*growthRate
            
            # update time 
            self.t_array[n] = self.t_array[n-1] + tau
            
            # update counter
            n = n+1
            
        print(f"Sample cycle n is {n}, setting self.arrSize to {int(n*5)}")
        self.arrSize = int(n * 5)
        
        if self.arrSize < 1e4:
            self.arrSize = int(1e5)

        self._init_buffers()

        print(f"Setting self.arrSize to: {self.arrSize}")

    def runCycle(self,cycleIndex):

        self.updateDivTimes()
        growthRate = 1/self.divTime
        
        self.t_array[0] = self.t
        self.V_array[0] = self.V
        self.A_array[0] = self.A
        self.B_array[0] = self.B
        self.C_array[0] = self.C
        self.D_array[0] = self.D
        self.E_array[0] = self.E
        self.F_array[0] = self.F
        
        n = 1
        while self.V_array[n-1] < 2:
            
            # update arrays 
            self.V_array[n] = self.V_array[n-1]
            self.A_array[n] = self.A_array[n-1]
            self.B_array[n] = self.B_array[n-1]
            self.C_array[n] = self.C_array[n-1]
            self.D_array[n] = self.D_array[n-1]
            self.E_array[n] = self.E_array[n-1]
            self.F_array[n] = self.F_array[n-1]
            
            # calculate reaction for time step 
            self.A_array[n],self.B_array[n],self.C_array[n],self.D_array[n],self.E_array[n],self.F_array[n],tau = self.reaction(self.A_array[n],self.B_array[n],self.C_array[n],self.D_array[n],self.E_array[n],self.F_array[n],self.V_array[n])
            
            # calculate cell growth 
            self.V_array[n] = self.V_array[n] + tau*growthRate
            
            # update time 
            self.t_array[n] = self.t_array[n-1] + tau
            
            # update counter
            n = n+1
        
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
        times = np.linspace(self.t_array[0],self.t_array[n-1],int(self.Tcc/10)+1)
        t_repeat = np.repeat(self.t_array[:n,np.newaxis],len(times),axis=1)
        
        indices = np.argmin(abs(np.subtract(t_repeat,times)),axis=0)
        
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
        
        if self.circuit == 'single':
            # calculate probabilities
            prodA = self.prodA
            
            Rtot = prodA
            
            # generate random numbers
            r1 = self.rng.uniform()
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            A = A + 1
        
        elif self.circuit == 'pos_fb':
        
            # calculate probabilities
            
            prodA = self.prodA + self.k1*A**self.k3/(self.k2**self.k3+A**self.k3)
            
            Rtot = prodA
            
            # generate random numbers
            r1 = self.rng.uniform()
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            A = A + 1
        
        elif self.circuit == 'neg_fb':
            
            # calculate probabilities
            
            prodA = self.k1/(1+(A/self.k2)**self.k3)
            
            Rtot = prodA
            
            # generate random numbers
            r1 = self.rng.uniform()
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            A = A + 1    
        
        elif self.circuit == 'bind':
            
            # calculate probabilities
            prodA = self.prodA
            prodB = self.prodB
            prodC = self.k1 * A * B
            
            Rtot = prodA + prodB + prodC
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A = A + 1
            elif r2 < prodA + prodB:
                B = B + 1
            else:
                A = A - 1
                B = B - 1
                C = C + 1
                
        elif self.circuit == 'bind2':
             
             # calculate probabilities
             prodA = self.prodA
             prodB = self.prodB
             prodC = self.k1 * A * B
             
             Rtot = prodA + prodB + prodC
             
             # generate random numbers
             r1 = self.rng.uniform()
             r2 = self.rng.uniform() * Rtot
             
             # calculate time step
             tau = 1/Rtot*np.log(1/r1)
             
             # pick reaction 
             if r2 < prodA:
                 A = A + 1
             elif r2 < prodA + prodB:
                 B = B + 1
             else:
                 A = A - 1
                 B = B - 1
                 C = C + 1
                 
        elif self.circuit == 'revbind':
            
            # calculate probabilities
            prodA = self.prodA
            prodB = self.prodB
            prodC = self.k1 * A * B
            revC = self.k2* C
            
            Rtot = prodA + prodB + prodC + revC
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A = A + 1
            elif r2 < prodA + prodB:
                B = B + 1
            elif r2< prodA + prodB + prodC:
                A = A - 1
                B = B - 1
                C = C + 1
            else:
                A = A + 1
                B = B + 1
                C = C - 1
        
        elif self.circuit == 'prodsat':
            
            # calculate probabilities
            prodA = self.prodA
            prodB = self.k1 * A
            
            Rtot = self.prodA + prodB
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A = A + 1
            else:
                B = B + 1
                
        elif self.circuit == 'prodsat_burst':
            
            # calculate probabilities
            prodA = self.prodA
            prodB = self.k1 * A
            
            Rtot = self.prodA + prodB
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A = A + self.burstSize
            else:
                B = B + 1
        
        elif self.circuit == 'produnsat':
            
            prodA = self.prodA
            prodB = self.prodB
            prodC = self.k1/2 * (self.k2+A+B-np.sqrt((self.k2+A+B)**2-4*A*B))
            
            Rtot = prodA + prodB + prodC
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A = A + 1
            elif r2 < prodA + prodB:
                B = B + 1
            else:
                A = A - 1
                C = C + 1
        
        elif self.circuit == 'prod_fixedB':
            
            A = self.prodA * self.Tcc * V
            
            prodB = self.prodB
            prodC = self.k1/2 * (self.k2+A+B-np.sqrt((self.k2+A+B)**2-4*A*B))
            
            Rtot = prodB + prodC
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodB:
                B = B + 1
            else:
                A = A - 1
                C = C + 1
        
            # update A to maintain constant concentration 
            A = self.prodA * self.Tcc * V
        
        elif self.circuit == 'cascade':
            
            prodA = self.prodA
            prodB = self.k1 * A
            prodC = self.k2 * B
            
            Rtot = prodA + prodB + prodC
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A = A + 1
            elif r2 < prodA + prodB:
                B = B + 1
            else:
                C = C + 1
        
        elif self.circuit == 'proddeg':
            
            prodA = self.prodA
            prodB = self.prodB
            prodC = self.k1*A
            degC = self.k2/2*(self.k3+B+C-np.sqrt((self.k3+B+C)**2-4*B*C))
            
            Rtot = prodA + prodB + prodC + degC
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A = A + 1
            elif r2 < prodA + prodB:
                B = B + 1
            elif r2 < prodA + prodB + prodC:
                C = C + 1
            else:
                C = C - 1
                
        elif self.circuit == 'phos':
            
            prodA = self.prodA
            prodC = self.prodB
            prodB = self.k1*A
            prodD = self.k2*B*C
            
            Rtot = prodA + prodB + prodC + prodD
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A = A + 1
            elif r2 < prodA + prodB:
                A = A - 1
                B = B + 1
            elif r2 < prodA + prodB + prodC:
                C = C + 1
            else:
                A = A + 1
                B = B - 1
                C = C - 1
                D = D + 1
                
        elif self.circuit == 'phos_sat':
            
            Bp = D
            
            ka = self.k1
            kma = self.k2
            kc = self.k3
            kmc = self.k4
            
            prodA = self.prodA
            prodB = self.prodB
            formB =  kc/2*(kmc+C+Bp-np.sqrt((kmc+C+Bp)**2-4*C*Bp))
            formBp = ka/2*(kma+A+B -np.sqrt((kma+A+B )**2-4*A*B ))
            prodC = self.prodC
            
            
            Rtot = prodA + prodB + formB + formBp + prodC
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction
            if r2 < prodA:
                A = A + 1
            elif r2 < prodA + prodB:
                B = B + 1
            elif r2 < prodA + prodB + formB:
                B = B + 1
                D = D - 1
            elif r2 < prodA + prodB + formB + formBp: 
                B = B - 1
                D = D + 1
            else:
                C = C + 1
                
        elif self.circuit == 'phos_int':
            
            A = A
            Ap = B
            B = C
            ApB = D
            
            prodA = self.prodA
            prodAp = self.k1*A
            prodB = self.prodB
            prodApB = self.k2*Ap*B
            prodBp = self.k3*ApB
            
            Rtot = prodA + prodAp + prodB + prodApB + prodBp
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A = A + 1
            elif r2 < prodA + prodAp:
                A = A - 1
                B = B + 1
            elif r2 < prodA + prodAp + prodB:
                C = C + 1
            elif r2 < prodA + prodAp + prodB + prodApB: 
                B = B - 1
                C = C - 1
                D = D + 1
            else:
                D = D - 1
                E = E + 1
                A = A + 1
            
        elif self.circuit == 'phos2':
            
            A_unphos = A
            Ap = B
            B_unphos = C
            Bp = D
            ApB = E
            ABp = F
            
            kk = self.k1
            krevk = self.k2
            k1 = self.k3
            krev1 = self.k4
            kt = self.k5
            k2 = self.k6
            krev2 = self.k7
            kp = self.k8
            
            prodA = self.prodA
            prodB = self.prodB
            a_phos = kk*A_unphos
            a_dephos = krevk*Ap
            apb_bind = k1*Ap*B_unphos
            apb_unbind = krev1*ApB
            tphos = kt*ApB
            abp_bind = k2*A_unphos*Bp
            abp_unbind = krev2 * ABp
            dephos = kp * ABp
            
            Rtot = prodA + prodB + a_phos + a_dephos + apb_bind + apb_unbind + tphos + abp_bind + abp_unbind + dephos
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                # Generate unphosphorylated A 
                A = A + 1
            elif r2 < prodA + prodB:
                # Generate unphosphorylated B 
                C = C + 1
            elif r2 < prodA + prodB + a_phos: 
                # autophosphorylation of A 
                A = A - 1
                B = B + 1
            elif r2 < prodA + prodB + a_phos + a_dephos: 
                # dephosphorylation of A 
                A = A + 1
                B = B - 1
            elif r2 < prodA + prodB + a_phos + a_dephos + apb_bind:
                # Binding of Phosphorylated A and unphosphorylated B 
                B = B - 1
                C = C - 1
                E = E + 1
            elif r2 < prodA + prodB + a_phos + a_dephos + apb_bind + apb_unbind:
                # Unbinding of Phosphorylated A and unphosphorylated B
                B = B + 1
                C = C + 1
                E = E - 1
            elif r2 < prodA + prodB + a_phos + a_dephos + apb_bind + apb_unbind + tphos:
                # Transphosphorylation of B by A 
                E = E - 1
                A = A + 1
                D = D + 1
            elif r2 < prodA + prodB + a_phos + a_dephos + apb_bind + apb_unbind + tphos + abp_bind:
                # Binding of unphosphorylated A and phosphorylated B
                A = A - 1
                D = D - 1
                F = F + 1
            elif r2 < prodA + prodB + a_phos + a_dephos + apb_bind + apb_unbind + tphos + abp_bind + abp_unbind: 
                # Unbinding of unphosphorylated A and phosphorylated B
                A = A + 1
                D = D + 1
                F = F - 1
            else:
                # Dephosphorylation of B when bound to unphosphorylated A 
                A = A + 1
                C = C + 1
                F = F - 1
        
        elif self.circuit == 'phos_cycle':
            
            Bp = D
            
            ka = self.k1
            kc = self.k3
            
            prodA = self.prodA
            prodB = self.prodB
            formBp = ka*A*B 
            formB = kc*C*Bp
            prodC = self.prodC
            
            Rtot = prodA + prodB + formB + formBp + prodC
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction
            if r2 < prodA:
                A = A + 1
            elif r2 < prodA + prodB:
                B = B + 1
            elif r2 < prodA + prodB + formB:
                B = B + 1
                D = D - 1
            elif r2 < prodA + prodB + formB + formBp: 
                B = B - 1
                D = D + 1
            else:
                C = C + 1
        
        elif self.circuit == 'diffTF':
            
            # calculate probabilities
            prodA = self.prodA
            prodB = self.prodB
            prodC = self.k1 * A * B
            prodD = self.k2 * C
            
            Rtot = prodA + prodB + prodC + prodD
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A = A + 1
            elif r2 < prodA + prodB:
                B = B + 1
            elif r2 < prodA + prodB + prodC:
                A = A - 1
                B = B - 1
                C = C + 1
            else:
                D = D + 1
    
        elif self.circuit == 'cdg':
            
            prodA = self.prodA
            prodB = self.prodB
            prodC = self.k1*A
            degC = self.k2/2*(self.k3+B+C-np.sqrt((self.k3+B+C)**2-4*B*C))
            prodD = self.k4 * C
            
            Rtot = prodA + prodB + prodC + degC + prodD
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A = A + 1
            elif r2 < prodA + prodB:
                B = B + 1
            elif r2 < prodA + prodB + prodC:
                C = C + 1
            elif r2 < prodA + prodB + prodC + degC:
                C = C - 1
            else:
                D = D + 1
        
        elif self.circuit == 'cascade6':
           
            # calculate reaction probabilities
            prodA = self.prodA
            prodB = self.k1*A
            prodC = self.k2*B
            prodD = self.k3*C
            prodE = self.k4*D
            prodF = self.k5*E
            
            # sum reaction probabilities 
            Rtot = prodA + prodB + prodC + prodD + prodE + prodF
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A = A + 1
            elif r2 < prodA + prodB:
                B = B + 1
            elif r2 < prodA + prodB + prodC:
                C = C + 1
            elif r2 < prodA + prodB + prodC + prodD:
                D = D + 1
            elif r2 < prodA + prodB + prodC + prodD + prodE:
                E = E + 1
            else:
                F = F + 1
        
        elif self.circuit=='crazyCascade': 
            
            # calculate reaction probabilities 
            prodA = self.prodA
            prodB = self.k1*A
            prodC = self.prodB
            degB = self.k2/2*(self.k3+B+C-np.sqrt((self.k3+B+C)**2-4*B*C))
            prodD = self.k4*B
            prodE = self.prodC
            degD = self.k5/2*(self.k6+D+E-np.sqrt((self.k6+D+E)**2-4*D*E))
            prodF = self.k7*D
            
            # sum reaction probabilities 
            Rtot = prodA + prodB + prodC + degB + prodD + prodE + degD + prodF
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A = A + 1
            elif r2 < prodA + prodB:
                B = B + 1
            elif r2 < prodA + prodB + prodC:
                C = C + 1
            elif r2 < prodA + prodB + prodC + degB:
                B = B - 1
            elif r2 < prodA + prodB + prodC + degB + prodD:
                D = D + 1
            elif r2 < prodA + prodB + prodC + degB + prodD + prodE: 
                E = E + 1
            elif r2 < prodA + prodB + prodC + degB + prodD + prodE + degD:
                D = D - 1
            else:
                F = F + 1 
        
        else:
            # TCS
            
            A = A
            Ap = B
            B = C
            ApB = D
            Bp = E
            
            prodA = self.prodA
            prodAp = self.k1*A
            prodB = self.prodB
            prodApB = self.k2*Ap*B
            prodBp = self.k3*ApB
            prodP = self.k4 * Bp
            
            Rtot = prodA + prodAp + prodB + prodApB + prodBp + prodP
            
            # generate random numbers
            r1 = self.rng.uniform()
            r2 = self.rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A = A + 1
            elif r2 < prodA + prodAp:
                A = A - 1
                B = B + 1
            elif r2 < prodA + prodAp + prodB:
                C = C + 1
            elif r2 < prodA + prodAp + prodB + prodApB: 
                B = B - 1
                C = C - 1
                D = D + 1
            elif r2 < prodA + prodAp + prodB + prodApB + prodBp:
                D = D - 1
                E = E + 1
                A = A + 1
            else:
                F = F + 1
        
        
        return A,B,C,D,E,F,tau

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
        
    
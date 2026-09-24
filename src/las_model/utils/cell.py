"""
A single cell running a circuit through successive cell cycles with the numba
Gillespie kernel.  Used directly for lineage (motif) simulations and by
gridfunc.Grid for spatial simulations.

Each completed cycle records the end-of-cycle state (motherStates) and the
trajectory downsampled to Tcc/10 (molecules, as concentrations).  Division
splits the end-of-cycle counts complementarily: the cell keeps one share and
partition() returns the other for a daughter.
"""

import numpy as np
from las_model.utils.gillespie_numba import run_cycle_numba, pack_params, to_scalar
from las_model.utils.parameterize import parameterize_cell


class Cell:
    def __init__(self,Tcc,varTcc,rng,t0=0):
        self.Tcc = Tcc
        self.varTcc = varTcc
        self.rng = rng 
        self.divTime = self.rng.normal(self.Tcc,self.varTcc)
        self.divTimes = np.array([t0])
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
        self.t = np.array([t0])
        self.V = np.array([1])
        self.A = np.array([0])
        self.B = np.array([0])
        self.C = np.array([0])
        self.D = np.array([0])
        self.E = np.array([0])
        self.F = np.array([0])
        self._reset_history()
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

    def _reset_history(self):
        self._states = []      # per cycle: [t, V, A, B, C, D, E, F] at the end of the cycle
        self._times = []       # per cycle: times of the Tcc/10 downsampled trajectory
        self._molecules = []   # per cycle: (6, samples) concentrations of A..F

    def __getstate__(self):
        """Exclude pre-allocated buffer arrays from being pickled."""
        state = self.__dict__.copy()
        buffers = ['t_array', 'V_array', 'A_array', 'B_array', 'C_array', 'D_array', 'E_array', 'F_array']
        for key in buffers:
            state.pop(key, None)
        return state

    @property
    def motherStates(self):
        """(8, nCycles): t, V, A, B, C, D, E, F at the end of each completed cycle."""
        if not self._states:
            return np.zeros([8,0])
        return np.array(self._states).T

    def _stack_samples(self, blocks, width):
        """Lay per-cycle sample blocks end to end; consecutive blocks share one sample."""
        nCycles = len(blocks)
        step = int(self.Tcc/10)
        out = np.zeros([width, int(nCycles*self.Tcc/10+1)]) if width else np.zeros(int(nCycles*self.Tcc/10+1))
        for i, block in enumerate(blocks):
            out[..., i*step:(i+1)*step+1] = block
        return out

    @property
    def molecules(self):
        """(6, nCycles*Tcc/10+1): concentrations of A..F downsampled to Tcc/10."""
        return self._stack_samples(self._molecules, 6)

    @property
    def sampleTimes(self):
        """Times of the columns of molecules."""
        return self._stack_samples(self._times, 0)

    def parameterize(self,circuit,params):
        self.circuit = circuit
        parameterize_cell(self, circuit, params)

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

    def equilibrate(self,nCycles,partition='binomial',bias=0):
        self._reset_history()
        
        # set partition bias if not binomial
        if partition == 'asymmetric':
            self.partitionBias = bias
        
        # run equilibration cycles 
        for i in range(nCycles):
            self.cellCycle(partition)
        
        self.sampleCycle()
        print(f"Setting array size as: {self.arrSize}")

    def run(self,nCycles,partition='binomial',bias=0):
        
        # reset time 
        self.t = 0
        self._reset_history()
        
        if partition == 'asymmetric':
            self.partitionBias = bias
        
        for i in range(nCycles):
            self.cellCycle(partition)

    def cellCycle(self,partition='binomial'):
        self.runCycle()
        self.partition(partition)

    def partition(self,partition='binomial'):
        """Split the end-of-cycle counts between this cell and a daughter.

        This cell keeps its share (A..F) and its time is set to the end of the
        cycle; the daughter's share is returned as [A, B, C, D, E, F].
        """
        motherState = self._states[-1]
        
        # set time to end of cycle
        self.t = motherState[0]
        
        # reset volume to 1
        self.V = 1
        
        if partition == 'binomial':
            
            if self.circuit=='prod_fixedB':
                self.A = self.prodA*self.Tcc
            else:
                self.A = self.rng.binomial(motherState[2],0.5)
            
            self.B = self.rng.binomial(motherState[3],0.5)
            self.C = self.rng.binomial(motherState[4],0.5)
            self.D = self.rng.binomial(motherState[5],0.5)
            self.E = self.rng.binomial(motherState[6],0.5)
            self.F = self.rng.binomial(motherState[7],0.5)
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
           
            self.A = motherState[2] * coef
            self.B = motherState[3] * coef
            self.C = motherState[4] * coef
            self.D = motherState[5] * coef
            self.E = motherState[6] * coef
            self.F = motherState[7] * coef
        else:
            raise ValueError(f"unknown partition '{partition}'")

        mine = np.array([to_scalar(x) for x in (self.A,self.B,self.C,self.D,self.E,self.F)])
        return np.array(motherState[2:8]) - mine

    def updateDivTimes(self):
        self.divTime = self.rng.normal(self.Tcc,self.varTcc)

        # check if divTime is negative
        while self.divTime < 0:
            self.divTime = self.rng.normal(self.Tcc,self.varTcc)

        self.divTimes = np.concatenate((self.divTimes,np.array([self.divTimes[-1] + self.divTime])))

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

    def runCycle(self):
        """Simulate one cell cycle from the current state and record it."""

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

        # record end-of-cycle state
        self._states.append([
            self.t_array[n-1], self.V_array[n-1],
            self.A_array[n-1], self.B_array[n-1], self.C_array[n-1],
            self.D_array[n-1], self.E_array[n-1], self.F_array[n-1],
        ])
        
        # record downsampled molecule concentrations
        times = np.linspace(self.t_array[0],self.t_array[n-1],int(self.Tcc/10)+1)
        pos = np.searchsorted(self.t_array[:n], times)
        pos = np.clip(pos, 1, n - 1)
        left = self.t_array[pos - 1]
        right = self.t_array[pos]

        indices = np.where((times - left) <= (right - times), pos - 1, pos)
        
        molecules = np.zeros([6,len(indices)])
        molecules[0] = self.A_array[indices]/self.V_array[indices]
        molecules[1] = self.B_array[indices]/self.V_array[indices]
        molecules[2] = self.C_array[indices]/self.V_array[indices]
        molecules[3] = self.D_array[indices]/self.V_array[indices]
        molecules[4] = self.E_array[indices]/self.V_array[indices]
        molecules[5] = self.F_array[indices]/self.V_array[indices]
        
        self._times.append(self.t_array[indices].copy())
        self._molecules.append(molecules)

    def getMotherStates(self):
        
        return self.motherStates[2::]
    
    def getMolecules(self):
        return self.molecules

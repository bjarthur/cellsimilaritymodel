# Grid Cells Function file 

import numpy as np
import matplotlib.animation as animation
from matplotlib import pyplot as plt 
import copy, cv2, cmapy
from datetime import datetime
from scipy import stats 

rng = np.random.default_rng(seed=1000)

class Grid:
    def __init__(self,ysize,xsize,maxCells):
        self.xsize = xsize
        self.ysize = ysize
        self.maxCells = maxCells
        self.timepoints = np.zeros(maxCells)
        self.data = np.zeros([maxCells,ysize,xsize])
        self.Cells = []
        
    def seed(self,circuit,params,Tcc,varTcc):
        starterCell = Cell(1,self.ysize//2,self.xsize//2,Tcc,varTcc,0)
        starterCell.parameterize(circuit,params)
        starterCell.lineage = [starterCell.ID]
        starterCell.runCycle()
        self.data[0,starterCell.yloc,starterCell.xloc] = starterCell.ID
        self.Cells.append(starterCell)
    
    def getCellNums(self,frameNum):
        return self.data[frameNum][np.nonzero(self.data[frameNum])].astype('int')
    
    def run(self):
        # iterate timepoints 
        i = 1
        while len(self.Cells) < self.maxCells:
            
            # get cells in previous timepoint
            cellNums = self.getCellNums(i-1)
            
            # copy existing cells to new frame 
            for j in range(len(cellNums)):
                motherCell = self.Cells[cellNums[j]-1]
                self.data[i,motherCell.yloc,motherCell.xloc] = motherCell.ID
                
            # get which cell divides next 
            divTime, divCellIndices = self.calcNextEvent()
            
            for j in range(len(divCellIndices)):
            
                # replicate cell
                motherCell = self.Cells[divCellIndices[j]]
                
                # replication location 
                repDir = self.getRepDir(motherCell,i)
                repLoc = self.getRepLoc(motherCell,repDir)
                
                # if location is full, move cells 
                if self.data[i,repLoc[0],repLoc[1]] != 0:
                    self.moveCells(repDir,motherCell,i)
                
                # replicate cell 
                daughterCell = Cell(len(self.Cells)+1,repLoc[0],repLoc[1],motherCell.Tcc,motherCell.varTcc,divTime)
                daughterCell.inherit(motherCell)
                
                # update daughter cell 
                daughterCell.updateLineage()
                daughterCell.cellCycle()
                
                motherCell.updateLineage()
                motherCell.cellCycle()
                
                self.data[i,daughterCell.yloc,daughterCell.xloc] = daughterCell.ID
                self.Cells.append(daughterCell)
                
            self.timepoints[i] = divTime
            i += 1

        self.timepoints = np.trim_zeros(self.timepoints,trim='b')

    def calcNextEvent(self):
        nextDivTimes = np.zeros(len(self.Cells))
        for i in range(len(nextDivTimes)):
            nextDivTimes[i] = self.Cells[i].divTimes[-1]
        return np.min(nextDivTimes),np.where(nextDivTimes==np.amin(nextDivTimes))[0]

    def calcCollectiveLocalRelatedness(self,maxradius,timepoint):
        frame = np.argmin(abs(self.timepoints-timepoint))
        
        radii = np.linspace(1,maxradius,maxradius)
        relatedness = np.zeros([np.count_nonzero(self.data[frame]),len(radii)])
        
        for i in range(len(relatedness)):
            for j in range(len(relatedness[i])):
                relatedness[i,j] = self.calcNeighborRelatedness(i+1, timepoint, radii[j])

        return relatedness

    def calcNeighborRelatedness(self,cellNum,timepoint,radius):
        frame = np.argmin(abs(self.timepoints-timepoint))
        
        # get location of cell num 
        cellLoc = np.where(self.data[frame] == cellNum)
        
        # get locations within radius
        neighborLocs = self.calcNeighborLocs(cellLoc,radius)
        
        # check for cells 
        neighborCells = self.calcNeighborCells(neighborLocs,frame)
        
        neighborCousinNums = np.zeros(len(neighborCells))
        for i in range(len(neighborCousinNums)):
            neighborCousinNums[i] = self.Cells[cellNum-1].calcCousinNum(self.Cells[int(neighborCells[i]-1)],timepoint)
    
        return np.mean(neighborCousinNums)
    
    def calcNeighborCells(self,locs,frame):
        neighborCellIDs = []
        for i in range(len(locs)):
            if self.data[frame,locs[i][0],locs[i][1]] != 0:
                neighborCellIDs.append(self.data[frame,locs[i][0],locs[i][1]])
        return neighborCellIDs

    def calcNeighborLocs(self,cellLoc,radius):
        neighborLocs = []
        for i in np.arange(cellLoc[0]-radius,cellLoc[0]+radius+1):
            for j in np.arange(cellLoc[1]-radius,cellLoc[1]+radius+1):
                if self.calcDistance(cellLoc,[i,j]) <= radius:
                    neighborLocs.append([int(i),int(j)])

        return neighborLocs

    def calcDistance(self,loc1,loc2):
        return np.sqrt((loc1[0]-loc2[0])**2 + (loc1[1]-loc2[1])**2)

    def cousinMap(self,cellNum,frame):
        cousinMap = np.zeros_like(self.data[-1])
        
        dataIndex = np.argmin(abs(self.timepoints-frame))
        
        for i in range(len(cousinMap)):
            for j in range(len(cousinMap[i])):
                if self.data[dataIndex,i,j] != 0:
                    cousinMap[i,j] = self.Cells[cellNum-1].calcCousinNum(self.Cells[int(self.data[dataIndex,i,j])-1],frame)
                else:
                    cousinMap[i,j] = -2   
        return cousinMap                                                                   

    def cousinVideo(self,cellNum,scale,downSample,filePrefix):
        vid = cv2.VideoWriter(filePrefix + '_cousinvid_cellNum_' + str(cellNum) + '.avi',cv2.VideoWriter_fourcc(*'MJPG'),500/downSample,(self.ysize*scale,self.xsize*scale),1)
        
        for i in range(0,int(self.timepoints[-1]),downSample):
            cousinMap = self.cousinMap(cellNum,i)
            
            scaledIm = (np.ceil(np.log2(len(self.Cells)))-(cousinMap+1)) / np.ceil(np.log2(len(self.Cells)))
            im8 = np.multiply(scaledIm,255).astype('uint8')
            frame = cv2.applyColorMap(im8,cmapy.cmap('viridis'))
            frame[np.where(cousinMap==-2)] = [0,0,0]
        
            frame = cv2.resize(frame,[self.ysize*scale,self.xsize*scale],interpolation=cv2.INTER_NEAREST)
            cv2.putText(frame,'t=%5.i' % i,(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,[255,255,255],4,cv2.LINE_AA)
            
            vid.write(frame)
    
        vid.release()
        cv2.destroyAllWindows()

    def molVideo(self,filePrefix,scale,downSample,molecule):
        vid = cv2.VideoWriter(filePrefix + '_' + molecule + '_vid.avi',cv2.VideoWriter_fourcc(*'MJPG'),500/downSample,(self.ysize*scale,self.xsize*scale),1)
        
        # scale video
        cell0vals = []
        if molecule == 'A':
            cell0vals = self.Cells[0].A/self.Cells[0].V
        elif molecule == 'B':
            cell0vals = self.Cells[0].B/self.Cells[0].V
        elif molecule == 'C':
            cell0vals = self.Cells[0].C/self.Cells[0].V
        elif molecule == 'D':
            cell0vals = self.Cells[0].D/self.Cells[0].V
        else:
            cell0vals = self.Cells[0].E/self.Cells[0].V
        vmin = np.min(cell0vals)
        vmax = np.max(cell0vals)
        
        for i in range(0,int(self.timepoints[-1]),downSample):
            
            frame = self.getFrame(i,molecule)
            
            scaledIm = (frame-vmin)/(vmax-vmin)
            im8 = np.multiply(scaledIm,255).astype('uint8')
            
            frame = cv2.applyColorMap(im8,cmapy.cmap('inferno'))
            dataIndex = np.where(self.timepoints - i > 0)[0][0]-1
            frame[np.where(self.data[dataIndex]==0)] = [0,0,0]
            
            frame = cv2.resize(frame,[self.ysize*scale,self.xsize*scale],interpolation=cv2.INTER_NEAREST)
            cv2.putText(frame,'t=%5.i' % i,(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,[255,255,255],4,cv2.LINE_AA)
            
            vid.write(frame)
    
        vid.release()
        cv2.destroyAllWindows()                          

    def Avideo(self,filePrefix,scale,downSample):
        vid = cv2.VideoWriter(filePrefix + '_Avid.avi',cv2.VideoWriter_fourcc(*'MJPG'),1000/downSample,(self.ysize*scale,self.xsize*scale),1)
        
        # scale video 
        vmin = np.mean(self.Cells[0].A/self.Cells[0].V) - 5 * np.std(self.Cells[0].A/self.Cells[0].V)
        vmax = np.mean(self.Cells[0].A/self.Cells[0].V) + 5 * np.std(self.Cells[0].A/self.Cells[0].V)
        
        for i in range(0,int(self.timepoints[-1]),downSample):
            
            frame = self.getFrame(i,'A')
            
            scaledIm = (frame-vmin)/(vmax-vmin)
            im8 = np.multiply(scaledIm,255).astype('uint8')
            
            frame = cv2.applyColorMap(im8,cmapy.cmap('inferno'))
            dataIndex = np.where(self.timepoints - i > 0)[0][0]-1
            frame[np.where(self.data[dataIndex]==0)] = [0,0,0]
            
            frame = cv2.resize(frame,[self.ysize*scale,self.xsize*scale],interpolation=cv2.INTER_NEAREST)
            # cv2.putText(frame,'t=%5.i' % i,(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,[255,255,255],4,cv2.LINE_AA)
            cv2.putText(frame,'generation %i' % (i // 1000 + 1),(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,[255,255,255],4,cv2.LINE_AA)
            
            vid.write(frame)
    
        vid.release()
        # cv2.destroyAllWindows() 
    
    def Bvideo(self,filePrefix,scale,downSample):
        vid = cv2.VideoWriter(filePrefix + '_Bvid.avi',cv2.VideoWriter_fourcc(*'MJPG'),1000/downSample,(self.ysize*scale,self.xsize*scale),1)
        
        # scale video 
        vmin = np.mean(self.Cells[0].B/self.Cells[0].V) - 5 * np.std(self.Cells[0].B/self.Cells[0].V)
        vmax = np.mean(self.Cells[0].B/self.Cells[0].V) + 5 * np.std(self.Cells[0].B/self.Cells[0].V)
        
        for i in range(0,int(self.timepoints[-1]),downSample):
            
            frame = self.getFrame(i,'B')
            
            scaledIm = (frame-vmin)/(vmax-vmin)
            im8 = np.multiply(scaledIm,255).astype('uint8')
            
            frame = cv2.applyColorMap(im8,cmapy.cmap('inferno'))
            dataIndex = np.where(self.timepoints - i > 0)[0][0]-1
            frame[np.where(self.data[dataIndex]==0)] = [0,0,0]
            
            frame = cv2.resize(frame,[self.ysize*scale,self.xsize*scale],interpolation=cv2.INTER_NEAREST)
            # cv2.putText(frame,'t=%5.i' % i,(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,[255,255,255],4,cv2.LINE_AA)
            # cv2.putText(frame,'generation %i' % (i // 1000 + 1),(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,[255,255,255],4,cv2.LINE_AA)
            
            vid.write(frame)
    
        vid.release()
        # cv2.destroyAllWindows()     
    
    def Cvideo(self,filePrefix,scale,downSample):
        vid = cv2.VideoWriter(filePrefix + '_Cvid.avi',cv2.VideoWriter_fourcc(*'MJPG'),1000/downSample,(self.ysize*scale,self.xsize*scale),1)
        
        # scale video 
        vmin = np.mean(self.Cells[0].C/self.Cells[0].V) - 5 * np.std(self.Cells[0].C/self.Cells[0].V)
        vmax = np.mean(self.Cells[0].C/self.Cells[0].V) + 5 * np.std(self.Cells[0].C/self.Cells[0].V)
        
        for i in range(0,int(self.timepoints[-1]),downSample):
            
            frame = self.getFrame(i,'C')
            
            scaledIm = (frame-vmin)/(vmax-vmin)
            im8 = np.multiply(scaledIm,255).astype('uint8')
            
            frame = cv2.applyColorMap(im8,cmapy.cmap('inferno'))
            dataIndex = np.where(self.timepoints - i > 0)[0][0]-1
            frame[np.where(self.data[dataIndex]==0)] = [0,0,0]
            
            frame = cv2.resize(frame,[self.ysize*scale,self.xsize*scale],interpolation=cv2.INTER_NEAREST)
            # cv2.putText(frame,'t=%5.i' % i,(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,[255,255,255],4,cv2.LINE_AA)
            # cv2.putText(frame,'generation %i' % (i // 1000 + 1),(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,[255,255,255],4,cv2.LINE_AA)
            
            vid.write(frame)
    
        vid.release()
        # cv2.destroyAllWindows() 

    def makeVideo(self,scale,downSample):
        timestamp = datetime.now()
        timestamptext = timestamp.strftime("%Y%m%d_%H%M%S")
        filetext = 'gridvid_' + timestamptext + '.avi'
        
        vid = cv2.VideoWriter(filetext,cv2.VideoWriter_fourcc(*'MJPG'),500/downSample,(self.ysize*scale,self.xsize*scale),1)
        
        for i in range(0,int(self.timepoints[-1]),downSample):
            dataIndex = np.where(self.timepoints - i > 0)[0][0]-1
            
            scaledIm = self.data[dataIndex] / len(self.Cells)
            im8 = np.multiply(scaledIm,255).astype('uint8')
            frame = cv2.applyColorMap(im8,cmapy.cmap('spring'))
            frame[np.where(self.data[dataIndex]==0)] = [0,0,0]
        
            frame = cv2.resize(frame,[self.ysize*scale,self.xsize*scale],interpolation=cv2.INTER_NEAREST)
            cv2.putText(frame,'t=%5.i' % i,(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,[255,255,255],4,cv2.LINE_AA)
            
            vid.write(frame)
    
        vid.release()
        cv2.destroyAllWindows()
    
    def moveCells(self,repDir,motherCell,timepoint):
        
        if repDir == 0:
            
            # start from cell and find next open spot 
            moveCellNum = 1 
            while self.data[timepoint,motherCell.yloc - moveCellNum,motherCell.xloc] != 0:
                moveCellNum += 1
            moveCellNum -= 1
            
            # start at furthest up cell and work down
            for i in range(motherCell.yloc-moveCellNum,motherCell.yloc):
                # get cell
                moveCell = self.Cells[int(self.data[timepoint,i,motherCell.xloc])-1]
                
                # update cell location
                moveCell.yloc = moveCell.yloc - 1 
                
                # update grid 
                self.data[timepoint,i-1,motherCell.xloc] = moveCell.ID
                self.data[timepoint,i,motherCell.xloc] = 0
                    
                # print('moved cell %i from %i, %i to %i, %i' % (moveCell.ID,i,motherCell.xloc,moveCell.yloc,moveCell.xloc))
        elif repDir == 1:
            # get number of cells to move 
            moveCellNum = 1
            while self.data[timepoint,motherCell.yloc,motherCell.xloc+moveCellNum] != 0:
                moveCellNum += 1
            moveCellNum -= 1
            
            # start at right edge and work left 
            for i in range(motherCell.xloc+moveCellNum,motherCell.xloc,-1):
                moveCell = self.Cells[int(self.data[timepoint,motherCell.yloc,i])-1]
                
                #update location 
                moveCell.xloc = moveCell.xloc + 1 
                
                # update grid 
                self.data[timepoint,motherCell.yloc,i+1] = moveCell.ID
                self.data[timepoint,motherCell.yloc,i] = 0
                
                # print('moved cell %i from %i, %i to %i, %i' % (moveCell.ID,i,motherCell.xloc,moveCell.yloc,moveCell.xloc))
                    
        elif repDir == 2:
            # get number of cells to move 
            moveCellNum = 1
            while self.data[timepoint,motherCell.yloc+moveCellNum,motherCell.xloc] != 0:
                moveCellNum += 1
            moveCellNum -= 1
            
            # start at bottom edge and move up 
            for i in range(motherCell.yloc+moveCellNum,motherCell.yloc,-1):
                # get cell
                moveCell = self.Cells[int(self.data[timepoint,i,motherCell.xloc])-1]
                
                # update cell location 
                moveCell.yloc = moveCell.yloc + 1 
                
                # update grid
                self.data[timepoint,i+1,motherCell.xloc] = moveCell.ID
                self.data[timepoint,i,motherCell.xloc] = 0 
                
                # print('moved cell %i from %i, %i to %i, %i' % (moveCell.ID,i,motherCell.xloc,moveCell.yloc,moveCell.xloc))
        else: 
            # get number of cells to move
            moveCellNum = 1
            while self.data[timepoint,motherCell.yloc,motherCell.xloc-moveCellNum] != 0:
                moveCellNum += 1
            moveCellNum -= 1
            
            # start from left and work right 
            for i in range(motherCell.xloc-moveCellNum,motherCell.xloc):
               
                # get cell 
                moveCell = self.Cells[int(self.data[timepoint,motherCell.yloc,i])-1]
                
                # update cell location 
                moveCell.xloc = moveCell.xloc - 1 
                
                # update grid 
                self.data[timepoint,motherCell.yloc,i-1] = moveCell.ID
                self.data[timepoint,motherCell.yloc,i] =0 
                
                # print('moved cell %i from %i, %i to %i, %i' % (moveCell.ID,i,motherCell.xloc,moveCell.yloc,moveCell.xloc))
    
    def getRepDir(self,motherCell,timepoint):
        # select replication direction (0=N, 1=E, 2=S, 3=W)
        dirCells = np.zeros(4)
        dirCells[0] = np.where(np.flip(self.data[timepoint,0:motherCell.yloc,motherCell.xloc])==0)[0][0]
        dirCells[1] = np.where(self.data[timepoint,motherCell.yloc,motherCell.xloc:self.xsize]==0)[0][0]
        dirCells[2] = np.where(self.data[timepoint,motherCell.yloc:self.ysize,motherCell.xloc]==0)[0][0]
        dirCells[3] = np.where(np.flip(self.data[timepoint,motherCell.yloc,0:motherCell.xloc])==0)[0][0]
        
        # get locations of minimum values
        minDirs = np.where(dirCells==np.min(dirCells))[0]
        
        # return random minimum value
        return minDirs[rng.integers(len(minDirs))]
                
    def getRepLoc(self,motherCell,repDir):
        
        # get replication location 
        if repDir == 0:
            repLoc = [motherCell.yloc-1,motherCell.xloc]
        elif repDir == 1:
            repLoc = [motherCell.yloc,motherCell.xloc+1]
        elif repDir == 2:
            repLoc = [motherCell.yloc+1,motherCell.xloc]
        else:
            repLoc = [motherCell.yloc,motherCell.xloc-1]
        
        return repLoc
    
    def getFrame(self,t,molecule):
        
        frame = np.zeros_like(self.data[0])
        
        # get all cells at timepoint 
        dataIndex = np.where(self.timepoints - t > 0)[0][0]-1
        cellNums = self.data[dataIndex][np.nonzero(self.data[dataIndex])]
        
        # for each cell, assign pixel value to concentration of M at nearest timepoint 
        for j in range(len(cellNums)):
            cell = self.Cells[int(cellNums[j]-1)]
            
            # find nearest timepoint 
            t_index = np.argmin(abs(t-cell.t))
            
            # get correct molecule
            if molecule == 'A':
                amt = cell.A[t_index]
            elif molecule == 'B':
                amt = cell.B[t_index]
            elif molecule == 'C':
                amt = cell.C[t_index]
            elif molecule == 'D':
                amt = cell.D[t_index]
            else:
                amt = cell.E[t_index]
            
            # assign pixel value to M concentration 
            frame[np.where(self.data[dataIndex]==cellNums[j])] = amt/cell.V[t_index]
    
        return frame
    
    def calcMoranI(self,neighborsize,frame,molecule,shape):
        dataIndex = np.where(self.timepoints - frame > 0)[0][0]-1
        
        # get number of cells at timepoint 
        N = np.count_nonzero(self.data[dataIndex])
        
        # get weight matrix 
        w = self.calcWeightMatrix(neighborsize,frame,shape)
        
        # get sum of weight matrix 
        W = np.sum(w)
        
        # get molecule frame 
        molFrame = self.getFrame(frame,molecule)
        
        # build 1D matrix of all molecule concentrations at frame
        concs = molFrame[np.nonzero(molFrame)]
        
        # calculate numerator value 
        numsum = 0
        for i in range(len(concs)):
            for j in range(len(concs)):
                numsum += w[i,j] * (concs[i]-np.mean(concs)) * (concs[j]-np.mean(concs))
        
        # calculate denominator 
        densum = 0
        for i in range(len(concs)):
            densum += (concs[i]-np.mean(concs))**2
        
        # print('number of cells: %i' % N)
        # print('weight matrix sum: %f' % W)
        # print('numerator sum: %f' % numsum)
        # print('denomenator sum: %f' % densum)
        
        # calculate Moran's I 
        return N / W * numsum / densum
        
    
    def calcWeightMatrix(self,neighborsize,frame,shape):
        dataIndex = np.where(self.timepoints - frame > 0)[0][0]-1
        
        # get locations of cells
        cellLocs = np.nonzero(self.data[dataIndex])

        # allocate weight matrix 
        w = np.zeros((np.size(cellLocs,1),np.size(cellLocs,1)))

        # set weight matrix
        for i in range(len(cellLocs[0])):
            for j in range(len(cellLocs[0])):
                y1,x1 = cellLocs[0][i],cellLocs[1][i]
                y2,x2 = cellLocs[0][j],cellLocs[1][j]
                
                if shape == 'discdist':
                    dist = np.sqrt((y1-y2)**2+(x1-x2)**2)
                    if i!= j and dist <= neighborsize:
                        w[i,j] = 1
                elif shape == 'discstep':
                    dist = np.sqrt((y1-y2)**2)+np.sqrt((x1-x2)**2)
                    if i!= j and dist <= neighborsize:
                        w[i,j] = 1
                elif shape == 'donut':
                    dist = np.sqrt((y1-y2)**2)+np.sqrt((x1-x2)**2)
                    if i!= j and dist == neighborsize:
                        w[i,j] = 1
                elif shape == 'gausdist':
                    dist = np.sqrt((y1-y2)**2+(x1-x2)**2)
                    if i!= j and dist == neighborsize:
                        w[i,j] = stats.norm.pdf(dist,0,neighborsize)
                else:
                    dist = np.sqrt((y1-y2)**2)+np.sqrt((x1-x2)**2)
                    if i!= j and dist == neighborsize:
                        w[i,j] = stats.norm.pdf(dist,0,neighborsize)
        
        return w
    
    
        def nextFrame(self,i):
            return plt.imshow(self.data[i],cmap='inferno',vmin=0,vmax=len(self.Cells))

        def play(self):
            fig,ax = plt.subplots(figsize=(12,10))
            plt.imshow(self.data[0],cmap='inferno',vmin=0,vmax=len(self.Cells))
            ax.spines['top'].set_linewidth(5)
            ax.spines['right'].set_linewidth(5)
            ax.spines['bottom'].set_linewidth(5)
            ax.spines['left'].set_linewidth(5)
            ax.set_xticks([])
            ax.set_yticks([])
            ani = animation.FuncAnimation(fig, self.nextFrame, frames=range(self.maxCells),repeat=0,interval=1000/20)
            return ani 
    
        
class Cell:
    def __init__(self,number,yloc,xloc,Tcc,varTcc,t):
        self.ID = number
        self.yloc = yloc
        self.xloc = xloc
        self.Tcc = Tcc
        self.varTcc = varTcc
        self.divTime = int(rng.normal(self.Tcc,self.varTcc))
        self.divTimes = np.array([self.divTime]) + t
        self.lineage = []
        self.prodA = 0
        self.prodB = 0
        self.prodC = 0
        self.k1 = 0
        self.k2 = 0 
        self.k3 = 0
        self.k4 = 0
        self.arrSize = int(1e9)
        self.t = np.array([0])
        self.V = np.array([1])
        self.A = np.array([0])
        self.B = np.array([0])
        self.C = np.array([0])
        self.D = np.array([0])
        self.E = np.array([0])
        self._init_buffers()

    def _init_buffers(self):
        self.t_array = np.empty(self.arrSize)
        self.V_array = np.empty(self.arrSize)
        self.A_array = np.empty(self.arrSize)
        self.B_array = np.empty(self.arrSize)
        self.C_array = np.empty(self.arrSize)
        self.D_array = np.empty(self.arrSize)
        self.E_array = np.empty(self.arrSize)

    def __getstate__(self):
        """Exclude pre-allocated buffer arrays from being pickled."""
        state = self.__dict__.copy()
        buffers = ['t_array', 'V_array', 'A_array', 'B_array', 'C_array', 'D_array', 'E_array']
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
        elif circuit == 'bind':
            self.prodA = params[0]
            self.prodB = params[1]
            self.k1 = params[2]
            
            self.A[0] = self.prodA * self.Tcc
            self.B[0] = self.prodB * self.Tcc
        elif circuit == 'prodsat':
            self.prodA = params[0]
            self.k1 = params[1]
            
            self.A[0] = self.prodA * self.Tcc
            self.B[0] = 3/2 * self.prodA * self.k1 * self.Tcc**2
            
        elif circuit == 'produnsat':
            self.prodA = params[0]
            self.prodB = params[1]
            self.k1 = params[2]
            self.k2 = params[3]
            
            Aest = int(self.prodA * self.Tcc - self.k1 * self.Tcc * self.prodA * self.Tcc * self.prodB * self.Tcc / (self.k2+self.prodA * self.Tcc * self.prodB * self.Tcc))
            if Aest > 0:
                self.A[0] = Aest
            else:
                self.A[0] = 0
            self.B[0] = int(self.prodB * self.Tcc)
            self.C[0] = int(4* self.k1 * self.Tcc * self.prodA * self.Tcc * self.prodB * self.Tcc / (self.k2+self.prodA * self.Tcc * self.prodB * self.Tcc))
            
        elif circuit =='cascade':
            self.prodA = params[0]
            self.k1 = params[1]
            self.k2 = params[2]
        
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
        elif circuit == 'diffTF':
            self.prodA = params[0]
            self.prodB = params[1]
            self.k1 = params[2]
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
            
        else:
            self.prodA = params[0]
            self.k1 = params[1]
            self.prodC = params[2]
            self.k2 = params[3]
            self.k3 = params[4]

        self.sampleCycle()
    
    def sampleCycle(self):
        
        growthRate = 1/self.divTime
        
        self.t_array[0] = self.t[-1]
        self.V_array[0] = self.V[-1]
        self.A_array[0] = self.A[-1]
        self.B_array[0] = self.B[-1]
        self.C_array[0] = self.C[-1]
        self.D_array[0] = self.D[-1]
        self.E_array[0] = self.E[-1]
        
        n = 1
        while self.V_array[n-1] < 2:
            
            # update arrays 
            self.V_array[n] = self.V_array[n-1]
            self.A_array[n] = self.A_array[n-1]
            self.B_array[n] = self.B_array[n-1]
            self.C_array[n] = self.C_array[n-1]
            self.D_array[n] = self.D_array[n-1]
            self.E_array[n] = self.E_array[n-1]
            
            # calculate reaction for time step 
            self.A_array,self.B_array,self.C_array,self.D_array,self.E_array,tau = self.reaction(n,self.A_array,self.B_array,self.C_array,self.D_array,self.E_array)
            
            # calculate cell growth 
            self.V_array[n] = self.V_array[n] + tau*growthRate
            
            # update time 
            self.t_array[n] = self.t_array[n-1] + tau
            
            # update counter
            n = n+1
            
        self.arrSize = int(n * 20)
        self._init_buffers()
    
    def cellCycle(self,partition='binomial'):
        self.runCycle()
        
        self.t = np.concatenate((self.t,np.array([self.divTimes[-1]])))
        self.updateDivTimes()
        
        if partition == 'binomial':
            self.V = np.concatenate((self.V,np.array([1])))
            self.A = np.concatenate((self.A,np.array([rng.binomial(self.A[-1],0.5)])))
            self.B = np.concatenate((self.B,np.array([rng.binomial(self.B[-1],0.5)])))
            self.C = np.concatenate((self.C,np.array([rng.binomial(self.C[-1],0.5)])))
            self.D = np.concatenate((self.D,np.array([rng.binomial(self.D[-1],0.5)])))
            self.E = np.concatenate((self.E,np.array([rng.binomial(self.E[-1],0.5)])))
        elif partition == 'perfect':
            self.V = np.concatenate((self.V,np.array([1])))
            self.A = np.concatenate((self.A,np.array([self.A[-1]//2])))
            self.B = np.concatenate((self.B,np.array([self.B[-1]//2])))
            self.C = np.concatenate((self.C,np.array([self.C[-1]//2])))
            self.D = np.concatenate((self.D,np.array([self.D[-1]//2])))
            self.E = np.concatenate((self.E,np.array([self.E[-1]//2])))
        elif partition == 'correlated':
            coef = rng.normal(0.5,0.1)
            self.V = np.concatenate((self.V,np.array([1])))
            self.A = np.concatenate((self.A,np.array([int(self.A[-1]*coef)])))
            self.B = np.concatenate((self.B,np.array([int(self.B[-1]*coef)])))
            self.C = np.concatenate((self.C,np.array([int(self.C[-1]*coef)])))
            self.D = np.concatenate((self.D,np.array([int(self.D[-1]*coef)])))
            self.E = np.concatenate((self.E,np.array([int(self.E[-1]*coef)])))
        else:
            print('invalid partition')
            return
    
    def runCycle(self):
        
        growthRate = 1/self.divTime
        
        self.t_array[0] = self.t[-1]
        self.V_array[0] = self.V[-1]
        self.A_array[0] = self.A[-1]
        self.B_array[0] = self.B[-1]
        self.C_array[0] = self.C[-1]
        self.D_array[0] = self.D[-1]
        self.E_array[0] = self.E[-1]
        
        n = 1
        while self.V_array[n-1] < 2:
            
            # update arrays 
            self.V_array[n] = self.V_array[n-1]
            self.A_array[n] = self.A_array[n-1]
            self.B_array[n] = self.B_array[n-1]
            self.C_array[n] = self.C_array[n-1]
            self.D_array[n] = self.D_array[n-1]
            self.E_array[n] = self.E_array[n-1]
            
            # calculate reaction for time step 
            self.A_array,self.B_array,self.C_array,self.D_array,self.E_array,tau = self.reaction(n,self.A_array,self.B_array,self.C_array,self.D_array,self.E_array)
            
            # calculate cell growth 
            self.V_array[n] = self.V_array[n] + tau*growthRate
            
            # update time 
            self.t_array[n] = self.t_array[n-1] + tau
            
            # update counter
            n = n+1
        
        # trim arrays
        self.t = np.concatenate((self.t,self.t_array[1:n]))
        self.V = np.concatenate((self.V,self.V_array[1:n]))
        self.A = np.concatenate((self.A,self.A_array[1:n]))
        self.B = np.concatenate((self.B,self.B_array[1:n]))
        self.C = np.concatenate((self.C,self.C_array[1:n]))
        self.D = np.concatenate((self.D,self.D_array[1:n]))
        self.E = np.concatenate((self.E,self.E_array[1:n]))
        
    def reaction(self,n,A_array,B_array,C_array,D_array,E_array):
        
        if self.circuit == 'single':
            # calculate probabilities
            prodA = self.prodA
            
            Rtot = prodA
            
            # generate random numbers
            r1 = rng.uniform()
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            A_array[n] = A_array[n] + 1
            
        
        elif self.circuit == 'bind':
            # get values
            A = A_array[n]
            B = B_array[n]
            
            # calculate probabilities
            prodA = self.prodA
            prodB = self.prodB
            prodC = self.k1 * A * B
            
            Rtot = prodA + prodB + prodC
            
            # generate random numbers
            r1 = rng.uniform()
            r2 = rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A_array[n] = A_array[n] + 1
            elif r2 < prodA + prodB:
                B_array[n] = B_array[n] + 1
            else:
                A_array[n] = A_array[n] - 1
                B_array[n] = B_array[n] - 1
                C_array[n] = C_array[n] + 1
        
        elif self.circuit == 'prodsat':
            
            # get values
            A = A_array[n]
            
            # calculate probabilities
            prodA = self.prodA
            prodB = self.k1 * A
            
            Rtot = self.prodA + prodB
            
            # generate random numbers
            r1 = rng.uniform()
            r2 = rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A_array[n] = A_array[n] + 1
            else:
                B_array[n] = B_array[n] + 1
        
        elif self.circuit == 'produnsat':
            
            A = A_array[n]
            B = B_array[n]
            
            prodA = self.prodA
            prodB = self.prodB
            prodC = self.k1/2 * (self.k2+A+B-np.sqrt((self.k2+A+B)**2-4*A*B))
            
            Rtot = prodA + prodB + prodC
            
            # generate random numbers
            r1 = rng.uniform()
            r2 = rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A_array[n] = A_array[n] + 1
            elif r2 < prodA + prodB:
                B_array[n] = B_array[n] + 1
            else:
                A_array[n] = A_array[n] - 1
                C_array[n] = C_array[n] + 1
        
        elif self.circuit == 'cascade':
            
            A = A_array[n]
            B = B_array[n]
            
            prodA = self.prodA
            prodB = self.k1 * A
            prodC = self.k2 * B
            
            Rtot = prodA + prodB + prodC
            
            # generate random numbers
            r1 = rng.uniform()
            r2 = rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A_array[n] = A_array[n] + 1
            elif r2 < prodA + prodB:
                B_array[n] = B_array[n] + 1
            else:
                C_array[n] = C_array[n] + 1
        
        elif self.circuit == 'proddeg':
            
            A = A_array[n]
            B = B_array[n]
            C = C_array[n]
            
            prodA = self.prodA
            prodB = self.prodB
            prodC = self.k1*A
            degC = self.k2/2*(self.k3+B+C-np.sqrt((self.k3+B+C)**2-4*B*C))
            
            Rtot = prodA + prodB + prodC + degC
            
            # generate random numbers
            r1 = rng.uniform()
            r2 = rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A_array[n] = A_array[n] + 1
            elif r2 < prodA + prodB:
                B_array[n] = B_array[n] + 1
            elif r2 < prodA + prodB + prodC:
                C_array[n] = C_array[n] + 1
            else:
                C_array[n] = C_array[n] - 1
                
        elif self.circuit == 'phos':
            
            A = A_array[n]
            B = B_array[n]
            C = C_array[n]
            
            prodA = self.prodA
            prodC = self.prodB
            prodB = self.k1*A
            prodD = self.k2*B*C
            
            Rtot = prodA + prodB + prodC + prodD
            
            # generate random numbers
            r1 = rng.uniform()
            r2 = rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A_array[n] = A_array[n] + 1
            elif r2 < prodA + prodB:
                A_array[n] = A_array[n] - 1
                B_array[n] = B_array[n] + 1
            elif r2 < prodA + prodB + prodC:
                C_array[n] = C_array[n] + 1
            else:
                A_array[n] = A_array[n] + 1
                B_array[n] = B_array[n] - 1
                C_array[n] = C_array[n] - 1
                D_array[n] = D_array[n] + 1
                
        elif self.circuit == 'diffTF':
            
            # get values
            A = A_array[n]
            B = B_array[n]
            C = C_array[n]
            
            # calculate probabilities
            prodA = self.prodA
            prodB = self.prodB
            prodC = self.k1 * A * B
            prodD = self.k2 * C
            
            Rtot = prodA + prodB + prodC + prodD
            
            # generate random numbers
            r1 = rng.uniform()
            r2 = rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A_array[n] = A_array[n] + 1
            elif r2 < prodA + prodB:
                B_array[n] = B_array[n] + 1
            elif r2 < prodA + prodB + prodC:
                A_array[n] = A_array[n] - 1
                B_array[n] = B_array[n] - 1
                C_array[n] = C_array[n] + 1
            else:
                D_array[n] = D_array[n] + 1
    
        elif self.circuit == 'cdg':
            
            A = A_array[n]
            B = B_array[n]
            C = C_array[n]
            
            prodA = self.prodA
            prodB = self.prodB
            prodC = self.k1*A
            degC = self.k2/2*(self.k3+B+C-np.sqrt((self.k3+B+C)**2-4*B*C))
            prodD = self.k4 * C
            
            Rtot = prodA + prodB + prodC + degC + prodD
            
            # generate random numbers
            r1 = rng.uniform()
            r2 = rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A_array[n] = A_array[n] + 1
            elif r2 < prodA + prodB:
                B_array[n] = B_array[n] + 1
            elif r2 < prodA + prodB + prodC:
                C_array[n] = C_array[n] + 1
            elif r2 < prodA + prodB + prodC + degC:
                C_array[n] = C_array[n] - 1
            else:
                D_array[n] = D_array[n] + 1
                
        else:
            
            A = A_array[n]
            B = B_array[n]
            C = C_array[n]
            D = D_array[n]
            
            prodA = self.prodA
            prodB = self.k1*A
            prodC = self.prodC
            prodD = self.k2*B*C
            prodE = self.k3*D
            
            Rtot = prodA + prodB + prodC + prodD + prodE
            
            # generate random numbers
            r1 = rng.uniform()
            r2 = rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA:
                A_array[n] = A_array[n] + 1
            elif r2 < prodA + prodB:
                A_array[n] = A_array[n] - 1
                B_array[n] = B_array[n] + 1
            elif r2 < prodA + prodB + prodC:
                C_array[n] = C_array[n] + 1
            elif r2 < prodA + prodB + prodC + prodD:
                A_array[n] = A_array[n] + 1
                B_array[n] = B_array[n] - 1
                C_array[n] = C_array[n] - 1
                D_array[n] = D_array[n] + 1
            else:
                E_array[n] = E_array[n] + 1
        
        
        return A_array,B_array,C_array,D_array,E_array,tau
    
    def updateDivTimes(self):
        self.divTime = int(rng.normal(self.Tcc,self.varTcc))
        self.divTimes = np.concatenate((self.divTimes,np.array([self.divTimes[-1] + self.divTime])))
        
    def updateLineage(self):
        self.lineage.append(self.ID)
    
    def inherit(self,motherCell):
        self.lineage = copy.copy(motherCell.lineage)
        self.divTimes = copy.copy(motherCell.divTimes)
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
        self.t = copy.copy(motherCell.t)
        self.V = copy.copy(motherCell.V)
        self.A = copy.copy(motherCell.A)
        self.B = copy.copy(motherCell.B)
        self.C = copy.copy(motherCell.C)
        self.D = copy.copy(motherCell.D)
        self.E = copy.copy(motherCell.E)
    
    def calcCousinNum(self,otherCell,timepoint):
        # print('comparing cell %i and cell %i' % (self.ID,otherCell.ID))
        
        # determine which generation to check based on timepoint 
        gen = 0
        while self.divTimes[gen] < timepoint:
            gen += 1
        
        if self.lineage[gen] == otherCell.lineage[gen]:
            return -1
        else:
            i = 0
            while self.lineage[i] == otherCell.lineage[i] and i < len(self.lineage)-1:
                # print('gen %i are the same' % i)
                i += 1
            
            return gen - i

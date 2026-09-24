# Grid Cells Function file 

import numpy as np
import matplotlib.animation as animation
from matplotlib import pyplot as plt 
import cv2, cmapy
from datetime import datetime
from scipy import stats 
import math
from las_model.utils.cell import Cell

class Grid:
    def __init__(self,ysize,xsize,maxCells,rng=None):
        self.xsize = xsize
        self.ysize = ysize
        self.maxCells = maxCells
        self.rng = rng if rng is not None else np.random.default_rng(seed=1000)
        self.timepoints = np.zeros(maxCells)
        self.data = np.zeros([maxCells,ysize,xsize])
        self.Cells = []
        
    def seed(self,circuit,params,Tcc,varTcc):
        starterCell = Cell(Tcc,varTcc,self.rng)
        starterCell.parameterize(circuit,params)
        starterCell.sampleCycle()
        starterCell.ID = 1
        starterCell.yloc, starterCell.xloc = self.ysize//2, self.xsize//2
        starterCell.lineage = [starterCell.ID]
        starterCell.lineageDivTimes = []
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
                
                # replicate cell: split the mother's end-of-cycle counts between mother and daughter 
                daughterState = motherCell.partition()
                motherCell.t = divTime
                daughterCell = Cell(motherCell.Tcc,motherCell.varTcc,self.rng,t0=divTime)
                daughterCell.inherit(motherCell,daughterState)
                daughterCell.ID = len(self.Cells)+1
                daughterCell.yloc, daughterCell.xloc = repLoc
                daughterCell.lineage = motherCell.lineage + [daughterCell.ID]
                daughterCell.lineageDivTimes = motherCell.lineageDivTimes + [divTime]
                motherCell.lineage.append(motherCell.ID)
                motherCell.lineageDivTimes.append(divTime)
                
                # run both cells' next cycles so their next division times are known 
                daughterCell.runCycle()
                motherCell.runCycle()
                
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
            neighborCousinNums[i] = self.cousinNum(self.Cells[cellNum-1],self.Cells[int(neighborCells[i]-1)],timepoint)
    
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

    def cousinNum(self,cell,otherCell,timepoint):
        # determine which generation to check based on timepoint 
        gen = 0
        while gen < len(cell.lineageDivTimes) and cell.lineageDivTimes[gen] < timepoint:
            gen += 1
        
        if cell.lineage[gen] == otherCell.lineage[gen]:
            return -1
        else:
            i = 0
            while cell.lineage[i] == otherCell.lineage[i] and i < len(cell.lineage)-1:
                i += 1
            return gen - i

    def cousinMap(self,cellNum,frame):
        cousinMap = np.zeros_like(self.data[-1])
        
        dataIndex = np.argmin(abs(self.timepoints-frame))
        
        for i in range(len(cousinMap)):
            for j in range(len(cousinMap[i])):
                if self.data[dataIndex,i,j] != 0:
                    cousinMap[i,j] = self.cousinNum(self.Cells[cellNum-1],self.Cells[int(self.data[dataIndex,i,j])-1],frame)
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
        cell0vals = self.Cells[0].molecules['ABCDEF'.index(molecule)]
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
        vmin = np.mean(self.Cells[0].molecules[0]) - 5 * np.std(self.Cells[0].molecules[0])
        vmax = np.mean(self.Cells[0].molecules[0]) + 5 * np.std(self.Cells[0].molecules[0])
        
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
        vmin = np.mean(self.Cells[0].molecules[1]) - 5 * np.std(self.Cells[0].molecules[1])
        vmax = np.mean(self.Cells[0].molecules[1]) + 5 * np.std(self.Cells[0].molecules[1])
        
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
        vmin = np.mean(self.Cells[0].molecules[2]) - 5 * np.std(self.Cells[0].molecules[2])
        vmax = np.mean(self.Cells[0].molecules[2]) + 5 * np.std(self.Cells[0].molecules[2])
        
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
        return minDirs[self.rng.integers(len(minDirs))]
                
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
        
        # for each cell, assign pixel value to concentration of M at nearest sampled timepoint 
        m = 'ABCDEF'.index(molecule)
        for j in range(len(cellNums)):
            cell = self.Cells[int(cellNums[j]-1)]
            t_index = np.argmin(abs(t-cell.sampleTimes))
            frame[np.where(self.data[dataIndex]==cellNums[j])] = cell.molecules[m,t_index]
    
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
        
        # center concentrations
        concs = concs - np.mean(concs)

        # calculate numerator value 
        numsum = 0
        for i in range(len(concs)):
            for j in range(len(concs)):
                numsum += w[i,j] * concs[i] * concs[j]
        
        # calculate denominator 
        densum = 0
        for i in range(len(concs)):
            densum += concs[i]**2
        
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

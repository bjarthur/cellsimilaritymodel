#%% figure code 

import pickle, os, numpy as np
from matplotlib import pyplot as plt 
from matplotlib.gridspec import GridSpec
from matplotlib import image as img
from matplotlib import patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit
from scipy import stats
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from las_model.utils.config import PROJECT_DIR 

letterLabelSize=42
axisFontSize=20
tickFontSize=16
tickLength=10
tickWidth=4
plotWidth = 2

cycleTime = 1000
breakerColor = [59/255,34/255,255/255]
randomColor = 'gray'
enzymeColor = [.992,.184,.259]
signalColor = [.271,.718,.8]
reactantColor = [181/255,124/255,177/255]
color_TFu = enzymeColor
color_L = [145/255,35/255,57/255]
color_TFb = signalColor
kcatColor = [255/255,149/255,21/255]
kcatColor2 = [255/255,162/255,111/255]
TccColor = [116/244,99/255,255/255]
TccSweepColor = [35/255,145/255,40/255]
TccSweepColor2 = [1/255,178/255,31/255]

def returnLogMinorTicks(start,stop):
    tickvals = []
    for i in range(start,stop):
        tickvals = np.concatenate((tickvals,np.linspace(10**i,10**(i+1),10)))
    return tickvals

def normdvar(kT):
    return 20/9*kT/(3+20/9*kT)

def logFit(x,xmax,Kx,n):
    return xmax*x**n/(Kx**n+x**n)

#%% Figure 1 Concept: pull data 

metricgraphic = img.imread(PROJECT_DIR / 'graphics/ngigraphic82.png')

#%% Figure 1 (Concept): plot 

f = plt.figure(figsize=(8,12))

# metric figure 
f.text(0.001,0.95,'A',fontsize=letterLabelSize,fontname='roboto')
f.text(0.001,0.52,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([0.01,0,.98,.98])
ax.imshow(metricgraphic)
ax.axis('off')


#%% Figure 2 (Toy Model): Pull Data 

def normdvar(kT):
    return 20/9*kT/(3+20/9*kT)

def logFit(x,xmax,Kx,n):
    return xmax*x**n/(Kx**n+x**n)

prodonly = img.imread(PROJECT_DIR / 'graphics/ngigraphic86.png')

LASpos = img.imread(PROJECT_DIR / 'graphics/ngigraphic79.png')
LASneg = img.imread(PROJECT_DIR / 'graphics/ngigraphic78.png')
differences_graphic = img.imread(PROJECT_DIR / 'graphics/ngigraphic80.png')
toymodel_enzymeA = img.imread(PROJECT_DIR / 'graphics/ngigraphic83.png')
toymodel_kcatA = img.imread(PROJECT_DIR / 'graphics/ngigraphic84.png')
toymodel_kcatB = img.imread(PROJECT_DIR / 'graphics/ngigraphic85.png')

amp_diagram = img.imread(PROJECT_DIR / 'graphics/ngigraphic87.png')

PprodAs = np.logspace(-3,2,6)
Tcc_prodAsweep = 1000
kcatA_prodAsweep = 0.1
drnds = np.zeros([len(PprodAs),1000,6])
dsiss = np.zeros_like(drnds)

for file in os.listdir(PROJECT_DIR / 'satprod/n1000'):
    index = int(file.split('_')[2].split('.')[0])

    filepath = os.path.join(PROJECT_DIR / 'satprod/n1000', file)
    with open(filepath,'rb') as f:
        PprodA,Tcc,kcatA,motherA,motherB,drnd,dsis = pickle.load(f)
    
    drnds[index] = drnd
    dsiss[index] = dsis

graphic = img.imread(PROJECT_DIR / 'graphics/ngigraphic40.png')

with open(PROJECT_DIR / 'asymmetric/asymmetric_bias_screen_2.pickle','rb') as f:
    biases,vardsis,vardrnd,normvar = pickle.load(f)

with open(PROJECT_DIR / 'satprod/satprod_prodAsweep.pickle','rb') as f:
    PprodAs,Tcc,kcatA,Aeqs,drndA,dsisA = pickle.load(f)
prodAs = np.logspace(-3,2,6)
As = prodAs * 1000 * 2

with open(PROJECT_DIR / 'burstSize/burstSize_prodA-2.pickle','rb') as f:
    burstSizes_0,prodAs_0,Aeqs_0,Beqs_0,normvarAs_0,normvarBs_0 = pickle.load(f)
with open(PROJECT_DIR / 'burstSize/burstSize_prodA-1.pickle','rb') as f:
    burstSizes_1,prodAs_1,Aeqs_1,Beqs_1,normvarAs_1,normvarBs_1 = pickle.load(f)
with open(PROJECT_DIR / 'burstSize/burstSize_prodA-0.pickle','rb') as f:
    burstSizes_2,prodAs_2,Aeqs_2,Beqs_2,normvarAs_2,normvarBs_2 = pickle.load(f)
colors = [enzymeColor,color_L,[68/255,10/255,21/255]]

with open(PROJECT_DIR / 'varTcc/single_var_cell_cycle_time/single_var_cell_cycle_time.pickle','rb') as f:
   varTccs, varTcc_results = pickle.load(f)

#%% Figure 2 (Toy Model): Plot 

As = PprodAs * Tcc_prodAsweep
Bs = 3/2 * PprodAs * kcatA_prodAsweep * Tcc_prodAsweep**2
dotsize= 100

def var_dA_sis(pprod,Tcc):
    return 2*pprod*Tcc

def var_dA_rnd(pprod,Tcc):
    return 2*pprod*Tcc

def var_dB_sis(kcat,Pprod,Tcc):
    return 3*kcat*Pprod*Tcc**2

def var_dB_rnd(kcat,Pprod,Tcc):
    return 3*kcat*Pprod*Tcc**2 + 20/9*kcat**2*Pprod*Tcc**3

f = plt.figure(figsize=(16,10))
gs = GridSpec(3,4,figure=f,wspace=0.5,hspace=0.5)

f.text(0.001,0.95,'A',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([0.01,0.6,.23,.5])
ax.imshow(prodonly)
ax.axis('off')

f.text(0.245,0.95,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([0.31,0.75,.15,.2])
ax.scatter(0,1,2,color='white',label='Numerical')
ax.scatter(As,np.var(drnds[:,:,0],axis=1),color=randomColor,label='$\Delta\sigma^2_{[A],rnd}$',s=dotsize,alpha=0.5)
ax.scatter(As,np.var(dsiss[:,:,0],axis=1),color=enzymeColor,label='$\Delta\sigma^2_{[A],sis}$',s=dotsize,alpha=0.5)
ax.hlines(0,1,2,color='white',label='Analytical')
ax.plot(As,var_dA_rnd(PprodAs,Tcc_prodAsweep),color=randomColor,linestyle='dashed',label='$\Delta\sigma^2_{[A],rnd}$')
ax.plot(As,var_dA_sis(PprodAs,Tcc_prodAsweep),color=enzymeColor,linestyle='dotted',label='$\Delta\sigma^2_{[A],sis}$')
ax.legend(frameon=0,fontsize=10,ncol=2,loc='upper left',bbox_to_anchor=[-.06,.7,.5,.5])
ax.set_xlabel('Enzyme Amt. ($[A]_{eq}$)',fontsize=axisFontSize,labelpad=0)
ylabel = ax.set_ylabel('Var. Diff. ($\sigma^2_{\Delta [A]}$)',fontsize=axisFontSize,labelpad=0)
ylabel.set_position((0,0.35))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([6*10**(-1),2*10**5])
ax.set_xticks(np.logspace(0,5,6))
ax.set_xticks(returnLogMinorTicks(0,5),[],minor=1)
ax.set_ylim([1,10**8])
ax.set_yticks(np.logspace(0,8,5))
ax.set_ylim([0.5,10**8])
ax.set_yticks(returnLogMinorTicks(0,8),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['left'].set_color(enzymeColor)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.tick_params(axis='y',colors=enzymeColor)
ax.tick_params(axis='y',which='minor',colors=enzymeColor)
ax = f.add_subplot([0.39,0.74,.08,0.12])
ax.imshow(LASneg)
ax.axis('off')

f.text(0.47,0.95,'C',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([0.54,0.75,.15,.2])
ax.scatter(0,1,2,color='white',label='Numerical')
ax.scatter(As,np.var(drnds[:,:,1],axis=1),color=randomColor,label='$\Delta\sigma^2_{[B],rnd}$',s=dotsize,alpha=0.5)
ax.scatter(As,np.var(dsiss[:,:,1],axis=1),color=signalColor,label='$\Delta\sigma^2_{[B],sis}$',s=dotsize,alpha=0.5)
ax.hlines(0,1,2,color='white',label='Analytical')
ax.plot(As,var_dB_rnd(kcatA_prodAsweep,PprodAs,Tcc_prodAsweep),color=randomColor,linestyle='dashed',label='$\Delta\sigma^2_{[B],rnd}$')
ax.plot(As,var_dB_sis(kcatA_prodAsweep,PprodAs,Tcc_prodAsweep),color=signalColor,linestyle='dotted',label='$\Delta\sigma^2_{[B],sis}$')
ax.legend(frameon=0,fontsize=10,ncol=2,loc='upper left',bbox_to_anchor=[-.05,.7,.5,.5])
ax.set_xlabel('Enzyme Amt. ($[A]_{eq}$)',fontsize=axisFontSize,labelpad=0)
ylabel = ax.set_ylabel('Var. Diff. ($\sigma^2_{\Delta [B]}$)',fontsize=axisFontSize,labelpad=0)
ylabel.set_position((0,0.35))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([6*10**(-1),2*10**5])
ax.set_xticks(np.logspace(0,5,6))
ax.set_xticks(returnLogMinorTicks(0,5),[],minor=1)
ax.set_yticks(np.logspace(2,12,6))
ax.set_yticks(returnLogMinorTicks(2,12),[],minor=1)
ax.set_ylim([10**2,10**12])
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['left'].set_color(signalColor)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='y',colors=signalColor)
ax.tick_params(axis='y',which='minor',colors=signalColor)
ax = f.add_subplot([0.61,0.74,.08,0.1])
ax.imshow(LASpos)
ax.axis('off')

f.text(0.7,0.95,'D',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([0.7,0.6,.3,.5])
ax.imshow(differences_graphic)
ax.axis('off')

f.text(0.001,0.62,'E',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([0.04,0.22,.96,.5])
ax.imshow(graphic)
ax.axis('off')

f.text(0.001,0.23,'F',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2,0])
ax.hlines(0,0,1,color='k',linestyle='dashed',linewidth=plotWidth)
ax.scatter(biases,normvar,color=randomColor)
ax.set_xlim([0.5,1.01])
ax.set_xticks(np.linspace(0.5,1,6))
ax.set_xticks(np.linspace(0.5,1,11),[],minor=1)
ax.set_ylim([-1,1])
ax.set_yticks(np.linspace(-1,1,5))
ax.set_yticks(np.linspace(-1,1,21),[],minor=1)
ax.set_xlabel('Partition Bias',fontsize=axisFontSize)
ylabel = ax.set_ylabel('LAS ($\Delta \hat{\sigma}^2_{\Delta [A]})$',fontsize=axisFontSize,labelpad=-6)
ylabel.set_position((0,0.5))
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.27,0.23,'G',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2,1])
ax.hlines(0,0,10**6,color='k',linestyle='dashed',linewidth=plotWidth)
ax.scatter(As,1-np.var(dsisA[:,:,0],axis=1)/np.var(drndA[:,:,0],axis=1),color=randomColor)
ax.set_xlabel('Concentration ($[A]_{eq}$)',fontsize=axisFontSize,labelpad=0)
# ax.set_ylabel('Norm. Similarity ($\sigma^2_{\Delta [A]}$)',fontsize=axisFontSize,labelpad=0)
ax.set_xscale('log')
ax.set_xlim([10**0,10**6])
ax.set_xticks(np.logspace(0,6,4))
ax.set_xticks(returnLogMinorTicks(0,6),[],minor=1)
ax.set_ylim([-1,1])
ax.set_yticks(np.linspace(-1,1,5))
ax.set_yticks(np.linspace(-1,1,21),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

colors=[[.2,.2,.2],[.4,.4,.4],[.6,.6,.6]]

f.text(0.5,0.23,'H',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2,2])
ax.hlines(0,-1,21,color='k',linestyle='dashed',linewidth=2)
ax.scatter(burstSizes_0,normvarAs_0,color=colors[0],label='$P_{prod,A}=10^{-2}$')
ax.scatter(burstSizes_1,normvarAs_1,color=colors[1],label='$P_{prod,A}=10^{-1}$')
ax.scatter(burstSizes_2,normvarAs_2,color=colors[2],label='$P_{prod,A}=10^{0}$')
ax.legend(frameon=1,fontsize=14,loc='upper left',bbox_to_anchor=[.25,.25,.5,.5])
ax.set_xlabel('$P_A$ Burst Size',fontsize=axisFontSize)
ax.set_xlim([0,20.4])
ax.set_xticks(np.linspace(0,20,5))
ax.set_xticks(np.linspace(0,20,21),[],minor=1)
ax.set_ylim([-1,1])
ax.set_yticks(np.linspace(-1,1,5))
ax.set_yticks(np.linspace(-1,1,21),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.74,0.23,'I',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2,3])
ax.hlines(0,0,10**4,color='k',linewidth=plotWidth,linestyle='dashed')
ax.scatter(varTccs,varTcc_results['normvar'][:,0],color=randomColor)
ax.set_xscale('log')
ax.set_xlabel('Cell Cycle Variation ($\sigma_{T_{cc}}$)',fontsize=axisFontSize)
ax.set_xticks(np.concatenate((np.linspace(1,10,10),np.linspace(10,100,10),np.linspace(100,1000,10))),[],minor=1)
ax.set_xlim([0.8,280])
ax.set_ylim([-1,1])
ax.set_yticks(np.linspace(-1,1,5))
ax.set_yticks(np.linspace(-1,1,21),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

plt.subplots_adjust(top = .85, bottom = 0.065, right = .97, left = 0.09)
plt.show()

#%% Figure 3 (Adjusting LAS): Pull Data 

with open(PROJECT_DIR / 'satprod/prodsat_sweep_kcat_low/prodsat_sweep_kcat_low/prodsat_sweep_kcat_low.pickle','rb') as f:
    kcatA_low_sweep_kcatAs, kcatA_low_sweep_results = pickle.load(f)

with open(PROJECT_DIR / 'satprod/prodsat_sweep_Tcc_low/prodsat_sweep_Tcc_low.pickle','rb') as f:
    Tcc_low_sweep_Tccs, Tcc_low_sweep_results = pickle.load(f)

with open(PROJECT_DIR / 'analyticalData/motifs_prodsat_sweepdata_reduced.pickle','rb') as f:
    Tccs,Tccvar_dsis,Tccvar_drnd,kcats_prodsat,kcatvar_dsis,kcatvar_drnd = pickle.load(f)
kcatMrange = np.logspace(-4,0,61)
Tccrange = np.logspace(2,4,11)
kcat = 0.01
Pprod = 0.01
Tcc = 1000
kTrange = np.logspace(-2,3,41)

prodAs = np.logspace(-2,0,3)
kcatAs = np.logspace(-4,0,9)
Tccs_sweep = np.logspace(2,4,5)

normvars_Tcc = 1-Tccvar_dsis[:,1]/Tccvar_drnd[:,1]
normvars_kcat = 1-kcatvar_dsis[:,1]/kcatvar_drnd[:,1]

params_Tcc,cov_Tcc = curve_fit(logFit,Tccs_sweep*kcat,normvars_Tcc,p0=[1,100,1])
params_kcat,cov_kcat = curve_fit(logFit,kcats_prodsat*Tcc,normvars_kcat,p0=[1,100,1])

kcatColorRange = [[255/255,184/255,98/255],[231/255,117/255,78/255]]
kcatColors = np.transpose(np.array((np.linspace(255/255,231/255,4),np.linspace(184/255,117/255,4),np.linspace(98/255,78/255,4))))

# kcatA2_kcats = np.zeros(13)
# kcatA2_normvarBs = np.zeros(13)
# for file in os.listdir(PROJECT_DIR / 'prodsat_sweep/prodsat_kcatsweep'):
#     kcatAindex = int(file.split('_')[3].split('.')[0])

#     filepath = os.path.join(PROJECT_DIR / 'prodsat_sweep/prodsat_kcatsweep', file)
#     with open(filepath,'rb') as f:
#         kcatA,normvarA,normvarB = pickle.load(f)
    
#     kcatA2_kcats[kcatAindex] = kcatA
#     kcatA2_normvarBs[kcatAindex] = normvarB

with open(PROJECT_DIR / 'satprod/prodsat_sweep_kcat_high/prodsat_sweep_kcat_high.pickle','rb') as f:
    kcatA_high_sweep_kcatAs, kcatA_high_sweep_results = pickle.load(f)

kcatA2_kcats = np.array(kcatA_high_sweep_kcatAs)
kcatA2_normvarBs = kcatA_high_sweep_results['normvar'][:,1]

# TODO: update Tcc2 sweep pull with new data
# NEW DATA LOCATION: 
# with open(PROJECT_DIR / 'satprod/prodsat_sweep_Tcc_high/prodsat_sweep_Tcc_high.pickle','rb') as f:
    # Tcc_high_sweep_Tccs, Tcc_high_sweep_results = pickle.load(f)

# OLD DATA LOCATION: 
Tccs2_Tcc = np.zeros(5)
Tccs2_normvarBs = np.zeros(5)
for file in os.listdir(PROJECT_DIR / 'prodsat_sweep/prodsat_Tccsweep'):
    Tccindex = int(file.split('_')[3].split('.')[0])
    
    filepath = os.path.join(PROJECT_DIR / 'prodsat_sweep/prodsat_Tccsweep', file)
    with open(filepath,'rb') as f:
        Tcc,normvarA,normvarB = pickle.load(f)
    
    Tccs2_Tcc[Tccindex] = Tcc
    Tccs2_normvarBs[Tccindex] = normvarB

Tcc = 1000

times = np.linspace(0,10,1001)
normvars = np.zeros([20,6,1001])

Rkcats = np.zeros(20)
for file in os.listdir(PROJECT_DIR / 'prodanddeg/Rkcatsweep2'):
    index = int(file.split('_')[1].split('.')[0])
    
    filepath = os.path.join(PROJECT_DIR / 'prodanddeg/Rkcatsweep2', file)
    with open(filepath,'rb') as f:
        kcatA_Rkcat_sweep,kcatB,normvar = pickle.load(f)
    
    normvars[index] = normvar
    Rkcats[index] = kcatB/kcatA_Rkcat_sweep

colors = np.ones([len(Rkcats),4])
colors[:,0] = np.linspace(112/255,52/255,len(Rkcats))
colors[:,1] = np.linspace(233/255,137/255,len(Rkcats))
colors[:,2] = np.linspace(255/255,153/255,len(Rkcats))
RkcatColors = ListedColormap(colors)
linestyles = ['solid','dotted','dashed','dashdot']

#%% Figure 3 (Adjusting LAS): Plot 

f = plt.figure(figsize=(8,6))
gs = GridSpec(2,2,figure=f,wspace=0.5,hspace=0.5)

f.text(-0.06,0.95,'A',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0,0])
ax.vlines(27/20,-.2,1,color='gray',linestyle=(0, (8, 8)),zorder=0)
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth,zorder=0)
ax.plot(kTrange,normdvar(kTrange),color='k',linewidth=plotWidth,zorder=1,label='Analytical')
ax.scatter(np.array(Tcc_low_sweep_Tccs)*kcatA,Tcc_low_sweep_results['normvar'][:,1],color=TccSweepColor,label='$T_{cc}$ sweep',marker='s')
ax.scatter(Tccs2_Tcc*10**-1,Tccs2_normvarBs,color=TccSweepColor2,label='$T_{cc}$ sweep 2',marker='+')
ax.scatter(kcatA_low_sweep_kcatAs*Tcc,kcatA_low_sweep_results['normvar'][:,1],color=kcatColor,label='$k_{cat}$ sweep',marker='s')
ax.scatter(kcatA2_kcats*Tcc,kcatA2_normvarBs,color=kcatColor2,label='$k_{cat}$ sweep 2',marker='+')
ax.legend(frameon=0,fontsize=10,loc='upper left',bbox_to_anchor=[0.45,0.37,.5,.5])
ax.set_xscale('log')
ax.set_xlabel('Amp. Factor ($k_{cat}T_{cc}$)',fontsize=axisFontSize,labelpad=0)
ylabel = ax.set_ylabel('Product\nLAS ($\Delta \hat{\sigma}^2_{\Delta [B]}$)',fontsize=axisFontSize,labelpad=-12)
ylabel.set_position((0,0.5))
ax.set_xlim([8*10**-2,1.2*10**3])
ax.set_ylim([-.2,1.02])
ax.set_xticks(np.logspace(-1,3,5))
ax.set_xticks(returnLogMinorTicks(-1,3),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.46,0.95,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([0.52,0.55,.48,0.45])
ax.imshow(amp_diagram)
ax.axis('off')

f.text(-0.06,0.43,'C',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1,0])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
ax.plot(times,normvars[0,0],label='A',color=enzymeColor,linewidth=2)
ax.plot(times,normvars[0,2],label='B',color=signalColor,linewidth=2)
ax.legend(frameon=0,fontsize=tickFontSize,loc='upper left',bbox_to_anchor=[0.6,.06,1,1])
ax.set_xlim([0,10])
ax.set_xlabel('Generations',fontsize=axisFontSize)
ax.set_ylabel('LAS ($\Delta \hat{\sigma}^2$)',fontsize=axisFontSize,labelpad=-10)
ax.set_ylim([-.2,1])
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)

f.text(0.46,0.43,'D',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1,1])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
ns = [0,5,9,18]
colors = np.zeros([len(ns),3])
colors[:,0] = np.linspace(112/255,52/255,len(ns))
colors[:,1] = np.linspace(233/255,137/255,len(ns))
colors[:,2] = np.linspace(255/255,153/255,len(ns))
for i in range(len(ns)):
    plt.plot(times,normvars[ns[i],2],color=colors[i],label='$R_{k_{cat}}=%.1f$' % Rkcats[ns[i]],linewidth=plotWidth,linestyle=linestyles[i])
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.set_xlabel('Generations',fontsize=axisFontSize)
ax.set_ylabel('Product\nLAS ($\Delta \hat{\sigma}^2_{\Delta [B]}$)',fontsize=axisFontSize,labelpad=-15)
ax.legend(frameon=0,fontsize=12,loc='upper left',bbox_to_anchor=[0.4,.06,1,1])
ax.set_xlim([0,10])
ax.set_ylim([-0.2,1])
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

plt.subplots_adjust(top = 0.99, bottom = 0.13, right = 0.98, left = 0.1)
plt.show()



#%% Figure 4 (Production Motif): Pull Data 

prod = img.imread(PROJECT_DIR / 'graphics/ngigraphic88.png')

with open(PROJECT_DIR / 'production/production_prodBsweep5.pickle','rb') as f:
    Ameans,Avars,Bmeans,Bvars,Cmeans,Cvars,normvarAs,normvarBs,normvarCs = pickle.load(f)
with open(PROJECT_DIR / 'production/production_prodAprodBsweep1.pickle','rb') as f:
    Ameans1,Avars1,Bmeans1,Bvars1,Cmeans1,Cvars1,normvarAs1,normvarBs1,normvarCs1 = pickle.load(f)

prodColorRange = [enzymeColor,[68/255,10/255,21/255]]
prodColors = np.transpose(np.array((np.linspace(prodColorRange[0][0],prodColorRange[1][0],16),
                                    np.linspace(prodColorRange[0][1],prodColorRange[1][1],16),
                                    np.linspace(prodColorRange[0][2],prodColorRange[1][2],16))))
prodColors2 = np.transpose(np.array((np.linspace(prodColorRange[0][0],prodColorRange[1][0],3),
                                    np.linspace(prodColorRange[0][1],prodColorRange[1][1],3),
                                    np.linspace(prodColorRange[0][2],prodColorRange[1][2],3))))
prodColorRange = [enzymeColor,[68/255,10/255,21/255]]
prodColors = np.transpose(np.array((np.linspace(prodColorRange[0][0],prodColorRange[1][0],len(Ameans1)),
                                    np.linspace(prodColorRange[0][1],prodColorRange[1][1],len(Ameans1)),
                                    np.linspace(prodColorRange[0][2],prodColorRange[1][2],len(Ameans1)),
                                    np.linspace(1,1,len(Ameans1)))))
prodColorMap = ListedColormap(prodColors)

#%% Figure 4 (Production Motif): Plot 

def calcProdRate(A,B,kcatA,Km):
    return kcatA/2*(A+B+Km-np.sqrt((A+B+Km)**2-4*A*B))

kcatA = 10**-1
Km = 10**3
PprodBs = np.logspace(-2,4,31)

f = plt.figure(figsize=(8,7))
gs = GridSpec(2,2,figure=f,wspace=0.5,hspace=0.4)

f.text(0.001,0.93,'A',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([-0.01,0.52,0.51,0.5])
ax.imshow(prod)
ax.axis('off')

f.text(0.49,0.93,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,1:2])
ax.hlines(0,0,2*10**7,color='k',linestyle='dashed',linewidth=plotWidth,zorder=0)
ax.scatter(Ameans,normvarBs,color=enzymeColor,label='A')
ax.scatter(Ameans,normvarAs,color=reactantColor,label='B\'')
ax.scatter(Ameans,normvarCs,color=signalColor,label='B')
ax.plot(Ameans[0:29],normvarBs[0:29],color=enzymeColor)
ax.plot(Ameans[0:29],normvarAs[0:29],color=reactantColor)
ax.plot(Ameans[0:29],normvarCs[0:29],color=signalColor)
ax.legend(frameon=0,fontsize=tickFontSize,loc='upper left',bbox_to_anchor=[-0.1,0.6,.5,.5])
ax.set_xscale('log')
xlabel = ax.set_xlabel('Reactant Amt. ($[B\']_{eq}$)',fontsize=axisFontSize,labelpad=-2)
xlabel.set_position((.45,0))
ax.set_xlim([10**-1,10**7])
ax.set_xticks(np.logspace(-1,7,5))
ax.set_xticks(returnLogMinorTicks(-1,7),[],minor=1)
ax.set_ylabel('LAS ($\Delta \hat{\sigma}^2$)',fontsize=axisFontSize,labelpad=-12)
ax.set_ylim([-.2,1.02])
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=1,labelsize=tickFontSize)

f.text(0.001,0.46,'C',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1:2,0:1])
ax.add_artist(mpatches.Rectangle((1,-0.2),100,1.2,facecolor=[76/255,222/255,82/255],zorder=0,alpha=0.2))
ax.vlines(1,-.2,1,color='gray',linestyle=(0, (8, 8)),zorder=0)
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth,zorder=0)
for i in range(len(Ameans1)):
    ax.scatter(Ameans1[i]/Bmeans1[i],normvarAs1[i],s=10,color=prodColorMap(i/len(Ameans1)),alpha=1,linewidth=0)
ax.set_xscale('log')
ax.set_xlabel('Reactant:Enzyme\n($[B\']_{eq}/[A]_{eq}$)',fontsize=axisFontSize,labelpad=0)
ax.set_xlim([10**-6,10**6])
ax.set_xticks(np.logspace(-6,6,5))
ax.set_xticks(returnLogMinorTicks(-6,6),[],minor=1)
ylabel = ax.set_ylabel('Reactant\nLAS ($\Delta \hat{\sigma}^2_{\Delta [B\']}$)',fontsize=axisFontSize,labelpad=-12)
ylabel.set_position((0,0.45))
ax.set_ylim([-.2,1.02])
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['left'].set_color(reactantColor)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=1,labelsize=tickFontSize)
ax.tick_params(axis='y',colors=reactantColor)
ax.tick_params(axis='y',which='minor',colors=reactantColor)
cax = f.add_subplot([0.26,0.29,.01,.2])
coef_cbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,3), cmap=prodColorMap),cax=cax,ticks=[0,1,2,3])
cax.yaxis.set_ticks_position('left')
coef_cbar.set_label('[Enzyme]',fontsize=tickFontSize,rotation=-90,labelpad=17)
coef_cbar.ax.tick_params(length=tickLength/2,width=tickWidth/2,labelsize=18)
coef_cbar.outline.set_linewidth(2)
coef_cbar.ax.set_yticklabels(['$10^{1}$','$10^2$','$10^3$','$10^4$'])

f.text(0.49,0.46,'D',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1:2,1:2])
ax.hlines(0,10**-4,10**4,color='k',linestyle='dashed',linewidth=plotWidth,zorder=0)
for i in range(len(Ameans1)):
    ax.scatter(Ameans1[i]/10**3,normvarCs1[i],color=prodColorMap(i/len(Ameans1)),s=10)
ax.set_xscale('log')
ax.set_xlabel('Saturation Ratio\n($[B\']_{eq}/K_M$)',fontsize=axisFontSize,labelpad=0)
ylabel = ax.set_ylabel('Product\nLAS ($\Delta \hat{\sigma}^2_{\Delta [B]}$)',fontsize=axisFontSize,labelpad=-15)
ylabel.set_position((0,0.45))
ax.add_artist(mpatches.Rectangle((1,-0.2),10,1.2,facecolor=[76/255,222/255,82/255],zorder=0,alpha=0.2))
ax.set_xlim([10**-4,10**4])
ax.set_xticks(np.logspace(-4,4,5))
ax.set_xticks(returnLogMinorTicks(-4,4),[],minor=1)
ax.set_ylim([-.2,1.02])
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['left'].set_color(signalColor)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=1,labelsize=tickFontSize)
ax.tick_params(axis='y',colors=signalColor)
ax.tick_params(axis='y',which='minor',colors=signalColor)
cax = f.add_subplot([0.93,0.25,.01,.2])
coef_cbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,3), cmap=prodColorMap),cax=cax,ticks=[0,1,2,3])
cax.yaxis.set_ticks_position('left')
coef_cbar.set_label('[Enzyme]',fontsize=tickFontSize,rotation=-90,labelpad=17)
coef_cbar.ax.tick_params(length=tickLength/2,width=tickWidth/2,labelsize=18)
coef_cbar.outline.set_linewidth(2)
coef_cbar.ax.set_yticklabels(['$10^{1}$','$10^2$','$10^3$','$10^4$'])

plt.subplots_adjust(top = 0.99, bottom = 0.16, right = 0.97, left = 0.17)
plt.show()



#%% Figure 5 (Full Pathways): Pull Data 

TCSpathway = img.imread(PROJECT_DIR / 'graphics/ngigraphic16.png')
diffTFpathway = img.imread(PROJECT_DIR / 'graphics/ngigraphic61.png')
cdGpathway = img.imread(PROJECT_DIR / 'graphics/ngigraphic17.png') # version with cdG label

with open(PROJECT_DIR / 'diffTF/diffTF_time/diffTF_time.pickle','rb') as f:
    results_diffTF = pickle.load(f)
normvar_diffTF = results_diffTF[0]['normvar'][0]

#TODO: after running figure_05/cdg_time.py, update the path to the new cdG data
# Old data 
with open(PROJECT_DIR / 'cdg/cdg_time.pickle','rb') as f:
    normvar_cdg = pickle.load(f)

# New data 
# with open(PROJECT_DIR / 'cdg/cdg_time/cdg_time.pickle','rb') as f:
#     results_cdg = pickle.load(f)
# normvar_cdg = results_cdg['normvar']

# TODO: after running figure_05/tcs_time.py, update the path to the new TCS data
# Old Data
with open(PROJECT_DIR / 'tcs/motifs_tcs3.pickle','rb') as f:
    results_tcs = pickle.load(f)

# New Data
# with open(PROJECT_DIR / 'tcs/tcs_time/tcs_time.pickle','rb') as f:
#     results_tcs = pickle.load(f)
# normvar_tcs = results_tcs['normvar'][0]


#%% Figure 5 (Full Pathways): Plot 

times = np.linspace(0,10,1001)

f = plt.figure(figsize=(16,9))
gs = GridSpec(3,3,figure=f,wspace=0.5,hspace=0.1)

f.text(0.001,0.95,'A',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_axes([0.05,.38,.45,.6],anchor='NW')
ax.imshow(diffTFpathway)
ax.axis('off')

f.text(0.001,0.33,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2:3,0:1])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
ax.plot(times,normvar_diffTF[0],color=enzymeColor,linewidth=plotWidth,label='$L$')
ax.plot(times,normvar_diffTF[1],color=signalColor,linewidth=plotWidth,label='$TF_u$')
ax.plot(times,normvar_diffTF[2],color=color_L,linewidth=plotWidth,label='$TF_b$')
ax.plot(times,normvar_diffTF[3],color='purple',linewidth=plotWidth,label='$P$')
ax.legend(frameon=0,fontsize=18)
ax.set_ylim([-0.2,1])
ax.set_xlim([0,10])
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.set_xlabel('Generations',fontsize=axisFontSize,labelpad=-2) 
ax.set_ylabel('LAS ($\Delta \hat{\sigma}^2$)',fontsize=axisFontSize,labelpad=-5)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.34,0.95,'C',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_axes([0.4,.38,.45,.6],anchor='NW')
ax.imshow(cdGpathway)
ax.axis('off')

f.text(0.34,0.33,'D',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2:3,1:2])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
ax.plot(times,normvar_cdg[0],color=enzymeColor,linewidth=plotWidth,label='$DGC$')
ax.plot(times,normvar_cdg[1],color=breakerColor,linewidth=plotWidth,label='$PDE$')
ax.plot(times,normvar_cdg[2],color=signalColor,linewidth=plotWidth,label='$cdG$')
ax.plot(times,normvar_cdg[3],color='purple',linewidth=plotWidth,label='$P$')
ax.legend(frameon=0,fontsize=18)
ax.set_ylim([-0.2,1])
ax.set_xlim([0,10])
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.set_xlabel('Generations',fontsize=axisFontSize,labelpad=-2) 
ax.set_ylabel('LAS ($\Delta \hat{\sigma}^2$)',fontsize=axisFontSize,labelpad=-5)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.69,0.95,'E',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_axes([0.74,.38,.45,.6],anchor='NW')
ax.imshow(TCSpathway)
ax.axis('off')

f.text(0.69,0.33,'F',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2:3,2:3])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
ax.plot(times,normvar_tcs[1],color=enzymeColor,linewidth=plotWidth,label='$SHK_P$')
ax.plot(times,normvar_tcs[4],color=signalColor,linewidth=plotWidth,label='$RR_P$')
ax.plot(times,normvar_tcs[5],color='purple',linewidth=plotWidth,label='$P$')
ax.set_xlabel('Generations',fontsize=axisFontSize,labelpad=-2) 
ax.set_ylabel('LAS ($\Delta \hat{\sigma}^2$)',fontsize=axisFontSize,labelpad=-5)
ax.legend(frameon=0,fontsize=tickFontSize,loc='upper left',bbox_to_anchor=[.5,.05,1,1])
ax.set_ylim([-0.2,1])
ax.set_xlim([0,10])
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

plt.subplots_adjust(top = 1, bottom = 0.07, right = .99, left = 0.08)
plt.show()


#%% Figure 6 (Spatial Simulations): Pull Data

lineage_simulation_graphic = img.imread(PROJECT_DIR / 'graphics/ngigraphic90.png')

pathway = img.imread(PROJECT_DIR / 'graphics/ngigraphic47.png')
A = img.imread(PROJECT_DIR / 'graphics/A_whiteborder.png')
B = img.imread(PROJECT_DIR / 'graphics/B_whiteborder.png')
C = img.imread(PROJECT_DIR / 'graphics/C_whiteborder.png')
satpathway = img.imread(PROJECT_DIR / 'graphics/ngigraphic62.png')

with open(PROJECT_DIR / 'cascade/cascade_spatial_analyze/cascade_spatial_analyze.pickle','rb') as f:
    cascade_spatial_results = pickle.load(f)
relatedness = cascade_spatial_results['relatedness']
imgs = cascade_spatial_results['cousinmaps']

with open(PROJECT_DIR / 'cascade/cascade_time/cascade_time.pickle','rb') as f:
    cascade_time_results = pickle.load(f)
normvar = cascade_time_results['normvar'][0]

ts = [2000,4000,6000,8000,10000]
molTimes = [4000,6000,8000,10000]
molAImgs,molBImgs,molCImgs = cascade_spatial_results['molConcMaps']
morIs_discdist = cascade_spatial_results['moranIs']



#%% Figure 6 (spatial): plot 

times = np.linspace(0,10,1001)

def expDecay(x,tau,x0):
    return x0*np.exp(-x/tau)

radii = np.linspace(1,9,9)

f = plt.figure(figsize=(16,11))
gs = GridSpec(3,3,figure=f,wspace=0.6,hspace=0.5)

f.text(0.01, 0.95, 'A', fontsize=letterLabelSize,fontname='roboto')
ax = f.add_axes([0.01,0.6,0.98,0.4],anchor='NW')
ax.imshow(lineage_simulation_graphic)
ax.axis('off')


f.text(0.01,0.62,'B',fontsize=letterLabelSize,fontname='roboto',color='white')
f.add_artist(mpatches.Rectangle((.01,.03),.68,.635,facecolor='k',zorder=0,alpha=1))
cbar_ax = f.add_axes([0.65,.05,.01,.18],zorder=1)
mean = np.mean(molAImgs[0][np.nonzero(molAImgs[0])])
var = 3*np.std(molAImgs[0][np.nonzero(molAImgs[0])])
coef_cbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(mean-var,mean+var), cmap='inferno'),cax=cbar_ax,ticks=[mean-var,mean,mean+var])
cbar_ax.yaxis.set_ticks_position('left')
coef_cbar.set_label('Concentration',fontsize=tickFontSize,rotation=-90,labelpad=20,color='white')
coef_cbar.ax.tick_params(length=tickLength/2,width=tickWidth/2,labelsize=18,colors='white')
coef_cbar.outline.set_linewidth(2)
coef_cbar.outline.set_color('white')
coef_cbar.ax.set_yticklabels(['$\mu -3\sigma$','$\mu$','$\mu +3\sigma$'])
for i in range(len(molAImgs)):
    ax = f.add_axes([.15*i+0.02,.43,.2,.2])
    mean = np.mean(molAImgs[i][np.nonzero(molAImgs[i])])
    var = 3*np.std(molAImgs[i][np.nonzero(molAImgs[i])])
    ax.imshow(molAImgs[i],cmap='inferno',vmin=mean-var,vmax=mean+var)
    ax.axis('off')
    ax.set_title('gen %i' % (molTimes[i]/1000),color='white',fontsize=tickFontSize,pad=-1000)
    
    ax = f.add_axes([.15*i+0.02,.23,.2,.2])
    mean = np.mean(molBImgs[i][np.nonzero(molBImgs[i])])
    var = 3*np.std(molBImgs[i][np.nonzero(molBImgs[i])])
    ax.imshow(molBImgs[i],cmap='inferno',vmin=mean-var,vmax=mean+var)
    ax.axis('off')
    
    ax = f.add_axes([.15*i+0.02,.03,.2,.2])
    mean = np.mean(molCImgs[i][np.nonzero(molCImgs[i])])
    var = 3*np.std(molCImgs[i][np.nonzero(molCImgs[i])])
    ax.imshow(molCImgs[i],cmap='inferno',vmin=mean-var,vmax=mean+var)
    ax.axis('off')
ax = f.add_axes([0.02,.51,.1,.05],anchor='NW')
ax.imshow(A)
ax.axis('off')
ax = f.add_axes([0.02,.31,.1,.05],anchor='NW')
ax.imshow(B)
ax.axis('off')
ax = f.add_axes([0.02,.11,.1,.05],anchor='NW')
ax.imshow(C)
ax.axis('off')

f.text(0.7,0.61,'C',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1,2])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
ax.plot(times,normvar[0,:],color='red',label='A',linewidth=plotWidth)
ax.plot(times,normvar[1,:],color='cyan',label='B',linewidth=plotWidth)
ax.plot(times,normvar[2,:],color='purple',label='C',linewidth=plotWidth)
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.set_xlabel('Generations',fontsize=axisFontSize,labelpad=-2)
ax.set_ylabel('LAS ($\Delta \hat{\sigma}^2$)',fontsize=axisFontSize,labelpad=-5)
ax.legend(frameon=0,fontsize=tickFontSize,loc='upper right')
ax.set_xlim([0,10])
ax.set_ylim([-0.2,1])
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.7,0.3,'D',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2,2])
plt.scatter(radii,morIs_discdist[:,0,-1],color='r',label='A')
params1,cov = curve_fit(expDecay,radii,morIs_discdist[:,0,-1],p0=[.5,1])
plt.plot(radii,expDecay(radii,params1[0],params1[1]),linestyle='dotted',color=enzymeColor)
plt.scatter(radii,morIs_discdist[:,1,-1],color='cyan',label='B')
params2,cov = curve_fit(expDecay,radii,morIs_discdist[:,1,-1],p0=[.5,1])
plt.plot(radii,expDecay(radii,params2[0],params2[1]),linestyle='dotted',color=signalColor)
plt.scatter(radii,morIs_discdist[:,2,-1],color='purple',label='C')
params3,cov = curve_fit(expDecay,radii,morIs_discdist[:,2,-1],p0=[.5,1])
plt.plot(radii,expDecay(radii,params3[0],params3[1]),linestyle='dotted',color='purple')
plt.legend(frameon=0,fontsize=tickFontSize,loc='upper left',bbox_to_anchor=[.6,.6,.5,.5])
ax.set_xlim([0.8,9.2])
ax.set_xticks(np.linspace(1,9,5))
ax.set_xticks(radii,[],minor=1)
ax.set_ylim([-.01,.3])
ax.set_yticks(np.linspace(0,.3,4))
ax.set_yticks(np.linspace(0,.3,7),[],minor=1)
ax.set_xlabel('Neighborhood Size\n(cell lengths)',fontsize=axisFontSize)
ylabel = ax.set_ylabel('Spatial Similarity\n(Moran\'s I)',fontsize=axisFontSize,labelpad=-2)
ylabel.set_position((0,0.35))
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

plt.subplots_adjust(top = .95, bottom = 0.11, right = .99, left = 0.08)
plt.show()

#%% Figure S1 (Framework): Pull Data 

framework = img.imread(PROJECT_DIR / 'graphics/ngigraphic32.png')

#%% Figure S1 (Framework): Plot

f = plt.figure(figsize=(16,14))

# metric figure 
f.text(0.001,0.96,'A',fontsize=letterLabelSize,fontname='roboto')
f.text(0.27,0.96,'B',fontsize=letterLabelSize,fontname='roboto')
f.text(0.645,0.96,'C',fontsize=letterLabelSize,fontname='roboto')
f.text(0.001,0.25,'D',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([0,0,.99,.99])
ax.imshow(framework)
ax.axis('off')

#%% Figure S2 (Distribution): Pull data

# TODO: plot after running satprod_PprodAsweep with full PprodA range
with open(PROJECT_DIR / 'satprod/satprod_PprodAsweep/satprod_PprodAsweep.pickle','rb') as f:
    PprodAs, results = pickle.load(f) 

#%% Figure S2 (Distributions): Plot 

def absM_dAsis_var(kcat,Pprod,Tcc):
    return 3*kcat*Pprod*Tcc**2

def absM_dArnd_var(kcat,Pprod,Tcc):
    return 3/2*kcat*Pprod*Tcc**2 + 10/9*kcat**2*Pprod*Tcc**3


f = plt.figure(figsize=(16,16))
gs = GridSpec(6,4,wspace=0.3,hspace=0.4)

motherAbounds = [[0,7],[8,32],[150,250],[1850,2150],[19500,20500],[198500,201500]]
dAbounds = [[-5,5],[-15,15],[-50,50],[-150,150],[-500,500],[-1500,1500]]
motherBbounds = [[0,750],[1500,5000],[25000,35000],[280000,320000],[2950000,3050000],[29500000,30500000]]
dBbounds = [[-200,200],[-750,750],[-2000,2000],[-7500,7500],[-20000,20000],[-75000,75000]]

for i in range(len(PprodAs)):
    ax = f.add_subplot(gs[i,0])
    ax.hist(results['mother_As'][i],color=enzymeColor,density=1,label='Numerical',bins=20)
    xmin = motherAbounds[i][0]
    xmax = motherAbounds[i][1]
    xs = np.linspace(xmin,xmax,xmax-xmin+1)
    ax.plot(xs,stats.poisson.pmf(xs,2*PprodAs[i]*Tcc),color='k',linestyle='dashed',linewidth=plotWidth,label='Analytical')
    ax.legend(frameon=0,ncol=2,fontsize=12,loc='upper left')
    if i==5:
        ax.set_xlabel('$A_{mother}$',fontsize=axisFontSize)
    ax.set_xlim([xmin,xmax])
    ax.set_ylabel('$P_{prod,A}=10^{%i}$' % (i-3),fontsize=tickFontSize)
    ymax = np.max(stats.poisson.pmf(xs,2*PprodAs[i]*Tcc))
    ax.set_ylim([0,ymax*1.5])
    ax.set_yticks([])
    ax.spines['left'].set_linewidth(0)
    ax.spines['bottom'].set_linewidth(tickWidth)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.tick_params(axis='x',length=tickLength,width=tickWidth,labelsize=tickFontSize)
    ax.tick_params(axis='x',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

    ax = f.add_subplot(gs[i,1])
    ax.hlines(0,1,2,color='white',label='Numerical')
    ax.hist(results['drnd'][i,0],density=1,color=randomColor,label='random',alpha=0.5,bins=20)
    ax.hist(results['dsis'][i,0],density=1,color=enzymeColor,label='sisters',alpha=0.5,bins=20)
    xmin = dAbounds[i][0]
    xmax = dAbounds[i][1]
    xs = np.linspace(xmin,xmax,xmax-xmin+1)
    ax.hlines(0,1,2,color='white',label='Analytical')
    ax.plot(xs,stats.poisson.pmf(xs+PprodAs[i]*Tcc,PprodAs[i]*Tcc),color=randomColor,linewidth=plotWidth,linestyle='dashed',label='random')
    ax.plot(xs,stats.poisson.pmf(xs+PprodAs[i]*Tcc,PprodAs[i]*Tcc),color=enzymeColor,linewidth=plotWidth,linestyle='dotted',label='sisters')
    ymax = np.max(stats.poisson.pmf(xs+PprodAs[i]*Tcc,PprodAs[i]*Tcc))
    ax.set_ylim([0,ymax*2])
    ax.set_xlim([xmin,xmax])
    if i==5:
        ax.set_xlabel('Pairwise Difference\n($\Delta [A]$)',fontsize=axisFontSize)
    ax.legend(frameon=0,fontsize=12,loc='upper left',ncol=2)
    ax.set_yticks([])
    ax.spines['left'].set_linewidth(0)
    ax.spines['bottom'].set_linewidth(tickWidth)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.tick_params(axis='x',length=tickLength,width=tickWidth,labelsize=tickFontSize)
    ax.tick_params(axis='x',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

    ax = f.add_subplot(gs[i,2])
    ax.hist(results['mother_Bs'][i],density=1,bins=20,color=signalColor,label='Numerical')
    xmin = motherBbounds[i][0]
    xmax = motherBbounds[i][1]
    xs = np.linspace(xmin,xmax,xmax-xmin+1)
    ax.set_xlim([xmin,xmax])
    meanB = 3*PprodAs[i]*kcatA*Tcc**2
    varB = absM_dArnd_var(kcatA,PprodAs[i],Tcc)
    ax.plot(xs,stats.norm.pdf(xs,meanB,np.sqrt(varB)),color='k',linestyle='dashed',linewidth=plotWidth,label='Analytical')
    ax.legend(frameon=0,fontsize=12,loc='upper left',ncol=2)
    ymax = np.max(stats.norm.pdf(xs,meanB,np.sqrt(varB)))
    ax.set_ylim([0,ymax*1.5])
    ax.set_yticks([])
    ax.set_xticks(np.linspace(xmin,xmax,3))
    if i==5:
        ax.set_xlabel('$B_{mother}$',fontsize=axisFontSize)
    ax.spines['left'].set_linewidth(0)
    ax.spines['bottom'].set_linewidth(tickWidth)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.tick_params(axis='x',length=tickLength,width=tickWidth,labelsize=tickFontSize)
    ax.tick_params(axis='x',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
    
    ax = f.add_subplot(gs[i,3])
    xmin = dBbounds[i][0]
    xmax = dBbounds[i][1]
    xs = np.linspace(xmin,xmax,xmax-xmin+1)
    xbins = np.linspace(xmin,xmax,20)
    ax.hlines(0,1,2,color='white',label='Numerical')
    ax.hist(results['drnd'][i,1],bins=xbins,density=1,color=randomColor,label='random',alpha=0.5)
    ax.hist(results['dsis'][i,1],bins=xbins,density=1,color=signalColor,label='sisters',alpha=0.5)
    
    meanB = 3/2*PprodAs[i]*kcatA*Tcc**2
    varBsis = round(absM_dAsis_var(kcatA,PprodAs[i],Tcc))
    varBrnd = round(absM_dArnd_var(kcatA,PprodAs[i],Tcc)/2)
    ax.hlines(0,1,2,color='white',label='Analytical')
    ax.plot(xs,stats.norm.pdf(xs,0,np.sqrt(varBrnd)),color=randomColor,linestyle='dashed',linewidth=plotWidth,label='random')
    ax.plot(xs,stats.norm.pdf(xs,0,np.sqrt(varBsis)),color=signalColor,linestyle='dotted',linewidth=plotWidth,label='sisters')
    ymax = np.max(stats.norm.pdf(xs,0,np.sqrt(varBsis)))
    ax.set_ylim([0,ymax*2])
    ax.set_xlim([xmin,xmax])
    if i==5:
        ax.set_xlabel('Pairwise Difference\n($\Delta [B]$)',fontsize=axisFontSize)
    ax.legend(frameon=0,fontsize=12,loc='upper left',ncol=2)
    ax.set_yticks([])
    ax.spines['left'].set_linewidth(0)
    ax.spines['bottom'].set_linewidth(tickWidth)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.tick_params(axis='x',length=tickLength,width=tickWidth,labelsize=tickFontSize)
    ax.tick_params(axis='x',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
    

plt.subplots_adjust(top = .98, bottom = 0.1, right = .97, left = 0.03)
plt.show()

#%% Figure S3 (Relative Errors): Pull data

#TODO: run after running satprod_PprodAsweep.py with full PprodA range 
with open(PROJECT_DIR / 'satprod/satprod_PprodAsweep/satprod_PprodAsweep.pickle','rb') as f:
    PprodAs, results = pickle.load(f) 

#%% Figure S3 (Relative Errors): Plot 

f= plt.figure(figsize=(8,6))

# compute cv^2 for amount of A across PprodAs 
var_mother_As = np.var(results['mother_As']/2,axis=1)
mean_mother_As = np.mean(results['mother_As']/2, axis=1)
cv_mother_As = var_mother_As/mean_mother_As**2 

# compute relative partition error across PprodAs 
partition_errors = np.abs(results['dsis'][:,0])
mean_partition_errors = np.mean(partition_errors,axis=1)
normalized_mean_partition_errors = mean_partition_errors/mean_mother_As

ax1 = f.add_subplot(1,1,1)
ax1.scatter(mean_mother_As, cv_mother_As, color=enzymeColor,s=100)
ax1.plot(mean_mother_As, cv_mother_As, color=enzymeColor,linestyle='dotted',linewidth=plotWidth)
ax1.hlines(0,0.9,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth,zorder=0)
plt.xscale('log')
ax1.set_xlabel('[A]',fontsize=axisFontSize)
ax1.set_ylabel('Molecular Noise\n($CV^2_{A_{mother}}$)',fontsize=axisFontSize)
ax1.set_ylim([-0.02,0.35])
ax1.spines['left'].set_linewidth(tickWidth)
ax1.spines['left'].set_color(enzymeColor)
ax1.spines['top'].set_linewidth(0)
ax1.spines['right'].set_linewidth(0)
ax1.spines['bottom'].set_linewidth(tickWidth)
ax1.tick_params(axis='y',colors=enzymeColor)
ax1.tick_params(axis='y',which='minor',colors=enzymeColor)
ax1.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax1.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

# dual y-axis 
ax2 = ax1.twinx()
ax2.scatter(mean_mother_As, normalized_mean_partition_errors, color='grey',s=100)
ax2.plot(mean_mother_As, normalized_mean_partition_errors, color='grey', linestyle='dotted',linewidth=plotWidth)
ax2.set_ylabel('Relative Partitioning Error\n($\overline{|\Delta A_{sister}|} / \mu_{A_{mother}}$)',fontsize=axisFontSize)
ax2.set_ylim([-0.07,1.2])
ax2.set_xticks(np.logspace(0,6,7))
ax2.set_xticks(returnLogMinorTicks(0,6),[],minor=1)
ax2.set_xlim([0.9,2*10**5])
ax2.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax2.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax2.tick_params(axis='y',colors='grey')
ax2.tick_params(axis='y',which='minor',colors='grey')
ax2.spines['left'].set_linewidth(0)
ax2.spines['right'].set_linewidth(tickWidth)
ax2.spines['right'].set_color('grey')
ax2.spines['top'].set_linewidth(0)
ax2.spines['bottom'].set_linewidth(tickWidth)

#%% Figure S4 (Saturated Production, kcat and Tcc sweep, full): pull data

# TODO: run after running Tcc_high_sweep

def normdvar(kT):
    return 20/9*kT/(3+20/9*kT)

os.chdir('//prfs.hhmi.org/sgrolab/mark/comp_proj/graphics')
satprod = img.imread(PROJECT_DIR / 'graphics/ngigraphic56.png')
prod_fixedB = img.imread(PROJECT_DIR / 'graphics/ngigraphic70.png')

# Old Data 
# with open(PROJECT_DIR / 'analyticalData/motifs_prodsat_sweepdata_reduced.pickle','rb') as f:
#     Tccs,Tccvar_dsis,Tccvar_drnd,kcats,kcatvar_dsis,kcatvar_drnd = pickle.load(f)
# kcatMrange = np.logspace(-4,0,61)
# Tccrange = np.logspace(2,4,11)
# kcat = 0.01
# Pprod = 0.01
# Tcc = 1000
# kTrange = np.logspace(-2,7,41)

# kcatA2_kcats = np.zeros(13)
# kcatA2_normvarBs = np.zeros(13)
# for file in os.listdir(PROJECT_DIR / 'prodsat_sweep/prodsat_kcatsweep'):
#     kcatAindex = int(file.split('_')[3].split('.')[0])
    
#     filepath = os.path.join(PROJECT_DIR / 'prodsat_sweep/prodsat_kcatsweep', file)
#     with open(filepath,'rb') as f:
#         kcatA,normvarA,normvarB = pickle.load(f)
    
#     kcatA2_kcats[kcatAindex] = kcatA
#     kcatA2_normvarBs[kcatAindex] = normvarB

# Tccs2_Tcc = np.zeros(5)
# Tccs2_normvarBs = np.zeros(5)
# for file in os.listdir(PROJECT_DIR / 'prodsat_sweep/prodsat_Tccsweep'):
#     Tccindex = int(file.split('_')[3].split('.')[0])
    
#     filepath = os.path.join(PROJECT_DIR / 'prodsat_sweep/prodsat_Tccsweep', file)
#     with open(filepath,'rb') as f:
#         Tcc_val,normvarA,normvarB = pickle.load(f)
    
#     Tccs2_Tcc[Tccindex] = Tcc_val
#     Tccs2_normvarBs[Tccindex] = normvarB



# New Data 
kcat = 10**-1
Tcc = 1000

with open(PROJECT_DIR / 'satprod/prodsat_sweep_kcat_low/prodsat_sweep_kcat_low/prodsat_sweep_kcat_low.pickle','rb') as f:
    kcatA_low_sweep_kcatAs, kcatA_low_sweep_results = pickle.load(f)

with open(PROJECT_DIR / 'satprod/prodsat_sweep_Tcc_low/prodsat_sweep_Tcc_low.pickle','rb') as f:
    Tcc_low_sweep_Tccs, Tcc_low_sweep_results = pickle.load(f)

with open(PROJECT_DIR / 'satprod/prodsat_sweep_kcat_high/prodsat_sweep_kcat_high.pickle','rb') as f:
    kcatA_high_sweep_kcatAs, kcatA_high_sweep_results = pickle.load(f)
kcatA_high_sweep_kcatAs = np.array(kcatA_high_sweep_kcatAs)

# with open(PROJECT_DIR / 'satprod/prodsat_sweep_Tcc_high/prodsat_sweep_Tcc_high.pickle','rb') as f:
#     Tcc_high_sweep_Tccs, Tcc_high_sweep_results = pickle.load(f)

#%% Figure S4 (Saturated Production, kcat and Tcc sweep, full): plot 

f = plt.figure(figsize=(8,3))
gs = GridSpec(1,2,figure=f,wspace=0.5,hspace=0.4)

f.text(0.001,0.85,'A',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([0,0.01,0.35,0.9])
ax.imshow(satprod)
ax.axis('off')

f.text(0.37,0.85,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0,1])
ax.vlines(27/20,-.2,1,color='gray',linestyle=(0, (8, 8)),zorder=0)
ax.hlines(0,0,2*10**6,color='k',linestyle='dashed',linewidth=plotWidth,zorder=0)
ax.plot(kTrange,normdvar(kTrange),color='k',linewidth=plotWidth,zorder=1,label='Analytical')
ax.scatter(np.array(Tcc_low_sweep_Tccs)*10**-2,Tcc_low_sweep_results['normvar'][:,1],color=TccSweepColor,label='$T_{cc}$ sweep',marker='s')
ax.scatter(Tcc_high_sweep_Tccs*10**-1,Tcc_high_sweep_results['normvar'][:,1],color=TccSweepColor2,label='$T_{cc}$ sweep 2',marker='+')
ax.scatter(kcatA_low_sweep_kcatAs*Tcc,kcatA_low_sweep_results['normvar'][:,1],color=kcatColor,label='$k_{cat}$ sweep',marker='s')
ax.scatter(kcatA_high_sweep_kcatAs*Tcc,kcatA_high_sweep_results['normvar'][:,1],color=kcatColor2,label='$k_{cat}$ sweep 2',marker='+')
ax.legend(frameon=0,fontsize=14,loc='upper left',bbox_to_anchor=[0.45,0.32,.5,.5])
ax.set_xscale('log')
ax.set_xlabel('Amp. Factor ($k_{cat}T_{cc}$)',fontsize=axisFontSize,labelpad=0)
ylabel = ax.set_ylabel('Product\nLAS ($\Delta \hat{\sigma}^2_{\Delta [B]}$)',fontsize=axisFontSize,labelpad=-12)
ylabel.set_position((0,0.5))
ax.set_xlim([8*10**-2,1.2*10**6])
ax.set_ylim([-.2,1.05])
ax.set_xticks(np.logspace(-1,6,8))
ax.set_xticks(returnLogMinorTicks(-1,6),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(signalColor)
ax.tick_params(axis='y',colors=signalColor)
ax.tick_params(axis='y',which='minor',colors=signalColor)

plt.subplots_adjust(top = 0.95, bottom = 0.25, right = 0.98, left = -0.1)
plt.show()

#%% Figure S5 (Saturated Production Amplification Factor 2D): pull data 

#TODO: run after running figure_S05/prodsat_sweep_kcatA_Tcc.py 

def normdvar(kT):
    return 20/9*kT/(3+20/9*kT)

with open(PROJECT_DIR / 'satprod/prodsat_sweep_kcatA_Tcc/prodsat_sweep_kcatA_Tcc.pickle','rb') as f:
    kcatAs,Tccs,results = pickle.load(f)

model = normdvar(np.outer(kcatAs, Tccs))


#normvarBs = np.zeros([len(prodAs),len(kcatAs),len(Tccs)])

# for i in range(len(model)):
#     for j in range(len(model[i])):
#             model[i,j] = normdvar(kcatAs[i]*Tccs[j])

# for file in os.listdir(PROJECT_DIR / 'prodsat_sweep/sweep1'):
    
#     prodAindex = int(file.split('_')[4])
#     kcatAindex = int(file.split('_')[6].split('.')[0])
    
#     filepath = os.path.join(PROJECT_DIR / 'prodsat_sweep/sweep1', file)
#     with open(filepath,'rb') as f:
#         prodA,kcatA,Tccs,divStates,dsis,drnd = pickle.load(f)

#     normvarBs[prodAindex,kcatAindex] = 1-np.var(dsis[:,:,1],axis=1)/np.var(drnd[:,:,1],axis=1)


#%% Figure S5 (Saturated Production Amplification Factor 2D): plot 

f = plt.figure(figsize=(8,4))
gs = GridSpec(1,2,figure=f,wspace=0.1,hspace=0.3)

f.text(0.001,0.89,'A',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,0:1])
im = ax.imshow(model,origin='lower',cmap='inferno',vmin=-.2,vmax=1)
ax.set_title('Analytical',fontsize=axisFontSize)
xlabel = ax.set_xlabel('Cell Cycle Time ($T_{cc}$)',fontsize=axisFontSize)
xlabel.set_position((1.5,6))
ax.set_ylabel('Rxn Rate ($k_{cat,A}$)',fontsize=axisFontSize)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.set_xticks([0,2,4],['$10^2$','$10^3$','$10^4$'])
ax.set_yticks([0,2,4,6,8],['$10^{-4}$','$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$'])
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.45,0.89,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,1:2])
im = ax.imshow(normvarBs[0],origin='lower',cmap='inferno',vmin=-.2,vmax=1)
ax.set_title('Simulation',fontsize=axisFontSize)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.set_xticks([0,2,4],['$10^2$','$10^3$','$10^4$'])
ax.set_yticks([0,2,4,6,8],['$10^{-4}$','$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$'])
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

cax = f.add_subplot([0.67,0.15,0.2,0.8])
cax.axis('off')
coef_cbar = ax.figure.colorbar(im,ax=cax,ticks=[-.2,0,.2,.4,.6,.8,1])
coef_cbar.set_label('Product LAS ($\Delta \hat{\sigma}^2_{\Delta [B]}$)',fontsize=tickFontSize,rotation=-90,labelpad=20,color='k')
coef_cbar.ax.tick_params(length=tickLength/2,width=tickWidth/2,labelsize=18,colors='k')
coef_cbar.outline.set_linewidth(2)

plt.subplots_adjust(top = 0.9, bottom = 0.2, right = 0.85, left = 0.05)
plt.show()

#%% Figure S6 (Production Motif Fixed Reactant - kcatA and PprodA sweep): Pull Data

prod_fixedB = img.imread(PROJECT_DIR / 'graphics/ngigraphic70.png')

with open(PROJECT_DIR / 'fixed_reactant/fixedReactant4.pickle','rb') as f:
    means_pA,variances_pA,vardSis_pA,vardRnd_pA,normvars_pA = pickle.load(f)

Tcc = 1000
kcatA = 10**-1
Km = 10**3

prodColorRange = [enzymeColor,[68/255,10/255,21/255]]
prodColors2 = np.transpose(np.array((np.linspace(prodColorRange[0][0],prodColorRange[1][0],3),
                                    np.linspace(prodColorRange[0][1],prodColorRange[1][1],3),
                                    np.linspace(prodColorRange[0][2],prodColorRange[1][2],3))))

with open(PROJECT_DIR / 'fixed_reactant/fixedReactant5.pickle','rb') as f:
    means_kA,variances_kA,vardSis_kA,vardRnd_kA,normvars_kA = pickle.load(f)

kcatAs = np.logspace(-2,2,5)
kcatColors = np.transpose(np.array((np.linspace(255/255,231/255,5),
                                    np.linspace(184/255,117/255,5),
                                    np.linspace(98/255,78/255,5))))


#%% Figure S6 (Production Motif Fixed Reactant - kcatA and PprodA sweep): Plot

def calcProdRate(A,B,Km,kcat):
    return kcat/2*(A+B+Km-np.sqrt((A+B+Km)**2-4*A*B))

def normdvar(kT):
    return 20/9*kT/(3+20/9*kT)

f = plt.figure(figsize=(16,4))
gs = GridSpec(1,3,figure=f,wspace=0.4,hspace=0.4)


f.text(0.001,0.89,'A',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([-0.01,0.1,.3,0.9])
ax.imshow(prod_fixedB)
ax.axis('off')

f.text(0.28,0.89,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,1:2])
for i in range(3):
    ax.scatter(Tcc*calcProdRate(means_pA[i,:,0],means_pA[i,:,1],Km,kcatA)/means_pA[i,:,0],normvars_pA[i,:,2],color=prodColors2[i],label='[A]=$10^{%i}$' % (i+1))
ax.vlines(27/20,-.2,1,color='gray',linestyle=(0, (8, 8)),zorder=0)
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth,zorder=0)
ax.plot(np.logspace(-2,3,101),normdvar(np.logspace(-2,3,101)),label='Analytical',color='k',linewidth=plotWidth,zorder=0)
ax.legend(frameon=0,fontsize=14,loc='upper left',bbox_to_anchor=[0.55,0.16,.5,.5])
ax.set_xscale('log')
xlabel = ax.set_xlabel(r'Amp. Factor ($\frac{r_B}{[A]_{eq}} T_{cc}$)',fontsize=axisFontSize,labelpad=-1)
xlabel.set_position((0.5,0))
ax.set_xlim([8*10**-3,2*10**2])
ax.set_xticks(np.logspace(-2,2,5))
ax.set_xticks(returnLogMinorTicks(-2,2),[],minor=1)
ax.set_ylabel('Product\nLAS ($\Delta \hat{\sigma}^2_{\Delta [B]}$)',fontsize=axisFontSize,labelpad=-12)
ax.set_ylim([-.2,1.02])
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['left'].set_color(signalColor)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.tick_params(axis='y',colors=signalColor)
ax.tick_params(axis='y',which='minor',colors=signalColor)

f.text(0.64,0.89,'C',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,2:3])
for i in range(len(means_kA)):
    ax.scatter(Tcc*calcProdRate(means_kA[i,:,0],means_kA[i,:,1],Km,kcatAs[i])/means_kA[i,:,0],normvars_kA[i,:,2],color=kcatColors[i],label='$k_{cat,A}=10^{%i}$' % (i-2))
ax.vlines(27/20,-.2,1,color='gray',linestyle=(0, (8, 8)),zorder=0)
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth,zorder=0)
ax.plot(np.logspace(-3,4,101),normdvar(np.logspace(-3,4,101)),label='Analytical',color='k',linewidth=plotWidth,zorder=0)
ax.legend(frameon=0,fontsize=14,loc='upper left',bbox_to_anchor=[0.55,0.45,.5,.5])
ax.set_xscale('log')
xlabel = ax.set_xlabel(r'Amp. Factor ($\frac{r_B}{[A]_{eq}} T_{cc}$)',fontsize=axisFontSize,labelpad=-1)
xlabel.set_position((0.5,0))
ax.set_xticks(np.logspace(-2,3,6))
ax.set_xticks(returnLogMinorTicks(-2,4),[],minor=1)
ax.set_xlim([7*10**-3,5*10**3])
ax.set_ylabel('Product\nLAS ($\Delta \hat{\sigma}^2_{\Delta [B]}$)',fontsize=axisFontSize,labelpad=-12)
ax.set_ylim([-.2,1.02])
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(signalColor)
ax.tick_params(axis='y',colors=signalColor)
ax.tick_params(axis='y',which='minor',colors=signalColor)

plt.subplots_adjust(top = 0.98, bottom = 0.23, right = 0.99, left = 0)
plt.show()


#%% Figure S7 (Irreversible and Reversible Binding): Pull Data

irreversible = img.imread(PROJECT_DIR / 'graphics/ngigraphic53.png')
reversible = img.imread(PROJECT_DIR / 'graphics/ngigraphic67.png')
a_slider = img.imread(PROJECT_DIR / 'graphics/a_slider.png')

# Old Data 
with open(PROJECT_DIR / 'prodRateSat/bind3.pickle','rb') as f:
    prodAs,prodBs,Aeqs,Beqs,Ceqs,varAs,varBs,varCs = pickle.load(f)

# New Data
# with open(PROJECT_DIR / 'binding/bind_sweep_PprodA_PprodB/bind_sweep_PprodA_PprodB.pickle','rb') as f:
#     PprodAs, PprodBs, results = pickle.load(f)

# Reversible Binding: Old Data 
with open(PROJECT_DIR / 'binding_rev/revbind4.pickle','rb') as f:
    prodAs_rev,prodBs_rev,Aeqs_rev,Beqs_rev,Ceqs_rev,varAs_rev,varBs_rev,varCs_rev,normvarAs_rev,normvarBs_rev,normvarCs_rev = pickle.load(f)

# Reversible Binding: New Data 
# with open(PROJECT_DIR / 'binding_rev/bind_sweep_PprodA_PprodB_rev/bind_sweep_PprodA_PprodB_rev.pickle','rb') as f:
    # PprodAs_rev, PprodBs_rev, results_rev = pickle.load(f)

prodColorRange = [enzymeColor,[68/255,10/255,21/255]]
prodColors = np.transpose(np.array((np.linspace(prodColorRange[0][0],prodColorRange[1][0],31),
                                    np.linspace(prodColorRange[0][1],prodColorRange[1][1],31),
                                    np.linspace(prodColorRange[0][2],prodColorRange[1][2],31))))

#%% Figure S7 (Irreversible and Reversible Binding): Plot

#TOOD: update with new data variables 

f = plt.figure(figsize=(16,7))
gs = GridSpec(2,4,figure=f,wspace=0.6,hspace=0.5)

f.text(0.001,0.93,'A',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([0.01,.55,.2,.45])
ax.imshow(irreversible)
ax.axis('off')


f.text(0.22,0.93,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,1:2])
ax.hlines(0,10**-1,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth,zorder=0)
for i in range(len(prodAs)):
    ax.scatter(Aeqs[:,i],varBs[:,i],color=prodColors[i])
ax.set_xscale('log')
ax.set_xlim([10**-1,2*10**5])
ax.set_xticks(np.logspace(-1,5,4))
ax.set_xticks(returnLogMinorTicks(-1,5),[],minor=1)
ax.set_xlabel('Unbound TF Amt. ($[B\']_{eq}$)',fontsize=axisFontSize)
ax.set_ylim([-0.2,1])
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.set_ylabel('Ligand\nLAS ($\Delta \hat{\sigma}^2_{\Delta [A]})$',fontsize=axisFontSize,labelpad=-10)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(enzymeColor)
ax.tick_params(axis='y',colors=enzymeColor)
ax.tick_params(axis='y',which='minor',colors=enzymeColor)
ax = f.add_subplot([0.28,0.81,0.1,0.16])
ax.imshow(a_slider)
ax.axis('off')

f.text(0.47,0.93,'C',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,2:3])
ax.hlines(0,10**-2,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth,zorder=0)
for i in range(len(prodAs)):
    ax.scatter(Aeqs[:,i],varAs[:,i],color=prodColors[i])
ax.set_xscale('log')
ax.set_xlim([10**-1,2*10**5])
ax.set_xticks(np.logspace(-1,5,4))
ax.set_xticks(returnLogMinorTicks(-1,5),[],minor=1)
ax.set_xlabel('Unbound TF Amt. ($[B\']_{eq}$)',fontsize=axisFontSize)
ax.set_ylim([-0.2,1])
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.set_ylabel('Unbound TF\nLAS ($\Delta \hat{\sigma}^2_{\Delta [B\']})$',fontsize=axisFontSize,labelpad=-10)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(reactantColor)
ax.tick_params(axis='y',colors=reactantColor)
ax.tick_params(axis='y',which='minor',colors=reactantColor)
ax = f.add_subplot([0.54,0.81,0.1,0.16])
ax.imshow(a_slider)
ax.axis('off')

f.text(0.73,0.93,'D',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,3:4])
ax.hlines(0,10**-2,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth,zorder=0)
for i in range(len(prodAs)):
    ax.scatter(Aeqs[:,i],varCs[:,i],color=prodColors[i])
ax.set_xscale('log')
ax.set_xlim([10**-1,2*10**5])
ax.set_xticks(np.logspace(-1,5,4))
ax.set_xticks(returnLogMinorTicks(-1,5),[],minor=1)
xlabel = ax.set_xlabel('Unbound TF Amt. ($[B\']_{eq}$)',fontsize=axisFontSize)
xlabel.set_position((.45,0))
ax.set_ylim([-0.2,1])
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.set_ylabel('Bound TF\nLAS ($\Delta \hat{\sigma}^2_{\Delta [B]})$',fontsize=axisFontSize,labelpad=-10)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(signalColor)
ax.tick_params(axis='y',colors=signalColor)
ax.tick_params(axis='y',which='minor',colors=signalColor)
ax = f.add_subplot([0.79,0.81,0.1,0.16])
ax.imshow(a_slider)
ax.axis('off')

f.text(0.001,0.42,'E',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([0.01,0.03,.2,.45])
ax.imshow(reversible)
ax.axis('off')

f.text(0.22,0.42,'F',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1:2,1:2])
ax.hlines(0,10**-2,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth,zorder=0)
for i in range(len(prodAs_rev)):
    ax.scatter(Aeqs_rev[:,i],normvarBs_rev[:,i],color=prodColors[i])
ax.set_xscale('log')
ax.set_xlim([10**-1,2*10**5])
ax.set_xticks(np.logspace(-1,5,4))
ax.set_xticks(returnLogMinorTicks(-1,5),[],minor=1)
ax.set_xlabel('Unbound TF Amt. ($[B\']_{eq}$)',fontsize=axisFontSize)
ax.set_ylim([-0.2,1])
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.set_ylabel('Ligand\nLAS ($\Delta \hat{\sigma}^2_{\Delta [A]})$',fontsize=axisFontSize,labelpad=-10)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(enzymeColor)
ax.tick_params(axis='y',colors=enzymeColor)
ax.tick_params(axis='y',which='minor',colors=enzymeColor)
ax = f.add_subplot([0.28,0.3,0.1,0.16])
ax.imshow(a_slider)
ax.axis('off')

f.text(0.47,0.42,'G',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1:2,2:3])
ax.hlines(0,10**-2,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth,zorder=0)
for i in range(len(prodAs_rev)):
    ax.scatter(Aeqs_rev[:,i],normvarAs_rev[:,i],color=prodColors[i])
ax.set_xscale('log')
ax.set_xlim([10**-1,2*10**5])
ax.set_xticks(np.logspace(-1,5,4))
ax.set_xticks(returnLogMinorTicks(-1,5),[],minor=1)
ax.set_xlabel('Unbound TF Amt. ($[B\']_{eq}$)',fontsize=axisFontSize)
ax.set_ylim([-0.2,1])
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.set_ylabel('Unbound TF\nLAS ($\Delta \hat{\sigma}^2_{\Delta [B\']})$',fontsize=axisFontSize,labelpad=-10)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(reactantColor)
ax.tick_params(axis='y',colors=reactantColor)
ax.tick_params(axis='y',which='minor',colors=reactantColor)
ax = f.add_subplot([0.54,0.3,0.1,0.16])
ax.imshow(a_slider)
ax.axis('off')

f.text(0.73,0.42,'H',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1:2,3:4])
ax.hlines(0,10**-2,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth,zorder=0)
for i in range(len(prodAs_rev)):
    ax.scatter(Aeqs_rev[:,i],normvarCs_rev[:,i],color=prodColors[i])
ax.set_xscale('log')
ax.set_xlim([10**-1,2*10**5])
ax.set_xticks(np.logspace(-1,5,4))
ax.set_xticks(returnLogMinorTicks(-1,5),[],minor=1)
xlabel = ax.set_xlabel('Unbound TF Amt. ($[B\']_{eq}$)',fontsize=axisFontSize)
xlabel.set_position((.45,0))
ax.set_ylim([-0.2,1])
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.set_ylabel('Bound TF\nLAS ($\Delta \hat{\sigma}^2_{\Delta [B]})$',fontsize=axisFontSize,labelpad=-10)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(signalColor)
ax.tick_params(axis='y',colors=signalColor)
ax.tick_params(axis='y',which='minor',colors=signalColor)
ax = f.add_subplot([0.79,0.3,0.1,0.16])
ax.imshow(a_slider)
ax.axis('off')

plt.subplots_adjust(top = 0.97, bottom = 0.12, right = 0.98, left = 0.05)
plt.show()


#%% Figure S08 (Duration sweep): Pull Data

circuit = img.imread(PROJECT_DIR / 'graphics/ngigraphic50.png')

times = np.linspace(0,10,1001)
Tccs = [500,1000,2000,5000,10000]

normvars_PprodAsweep = np.zeros([4,6,1001])
for file in os.listdir(PROJECT_DIR / 'satprod/time_PprodAsweep'):  
    index = int(file.split('_')[3].split('.')[0])

    filepath = os.path.join(PROJECT_DIR / 'satprod/time_PprodAsweep', file)
    with open(filepath,'rb') as f:
        normvar = pickle.load(f)
    normvars_PprodAsweep[index] = normvar

normvars_kcatAsweep = np.zeros([4,6,1001])
for file in os.listdir(PROJECT_DIR / 'satprod/time_kcatAsweep'):  
    index = int(file.split('_')[3].split('.')[0])

    filepath = os.path.join(PROJECT_DIR / 'satprod/time_kcatAsweep', file)
    with open(filepath,'rb') as f:
        normvar = pickle.load(f)
    normvars_kcatAsweep[index] = normvar

with open(PROJECT_DIR / 'satprod/time_Tccsweep/satprod_time_Tccsweep_0.pickle','rb') as f:
    normvar_Tccweep_0 = pickle.load(f)
with open(PROJECT_DIR / 'satprod/time_Tccsweep/satprod_time_Tccsweep_1.pickle','rb') as f:
    normvar_Tccweep_1 = pickle.load(f)
with open(PROJECT_DIR / 'satprod/time_Tccsweep/satprod_time_Tccsweep_2.pickle','rb') as f:
    normvar_Tccweep_2 = pickle.load(f)
with open(PROJECT_DIR / 'satprod/time_Tccsweep/satprod_time_Tccsweep_3.pickle','rb') as f:
    normvar_Tccweep_3 = pickle.load(f)

#%% Figure S08 (Duration sweep): Plot 

f = plt.figure(figsize=(16,9))
gs = GridSpec(3,4,figure=f,wspace=0.4,hspace=0.2)

f.text(0.001,0.95,'A',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([.35,.69,.3,.3])
ax.imshow(circuit)
ax.axis('off')

f.text(0.001,0.63,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1:2,0:1])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
for i in range(len(normvars_PprodAsweep)):
    plt.plot(times,normvars_PprodAsweep[i,0],linewidth=plotWidth,label='$P_{prod,A}=10^{%i}$' % (i-3),color=plt.cm.autumn(i/len(normvars_PprodAsweep)))
ax.legend(frameon=0,fontsize=tickFontSize,loc='upper left',bbox_to_anchor=[0.2,0.6,.5,.5])
ax.set_xlim([0,10])
ax.set_ylim([-.2,1])
ax.set_ylabel('Enzyme\nLAS ($\Delta \hat{\sigma}^2_{\Delta [A]}$)',fontsize=axisFontSize,labelpad=-10)
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(enzymeColor)
ax.tick_params(axis='y',colors=enzymeColor)
ax.tick_params(axis='y',which='minor',colors=enzymeColor)

f.text(0.001,0.31,'E',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2:3,0:1])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
for i in range(len(normvars_PprodAsweep)):
    plt.plot(times,normvars_PprodAsweep[i,1],linewidth=plotWidth,label='$P_{prod,A}=10^{%i}$' % (i-3),color=plt.cm.autumn(i/len(normvars_PprodAsweep)))
ax.legend(frameon=0,fontsize=tickFontSize,loc='upper left',bbox_to_anchor=[0.2,0.6,.5,.5])
ax.set_xlim([0,10])
ax.set_ylim([-.2,1])
ax.set_xlabel('Generations',fontsize=axisFontSize)
ax.set_ylabel('Product\nLAS ($\Delta \hat{\sigma}^2_{\Delta [B]}$)',fontsize=axisFontSize,labelpad=-10)
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(signalColor)
ax.tick_params(axis='y',colors=signalColor)
ax.tick_params(axis='y',which='minor',colors=signalColor)

f.text(0.27,0.63,'C',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1,1])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
for i in range(len(normvars_kcatAsweep)):
    plt.plot(times,normvars_kcatAsweep[i,0],linewidth=plotWidth,label='$k_{cat,A}=10^{%i}$' % (i-3),color=plt.cm.spring(i/len(normvars_kcatAsweep)))
ax.legend(frameon=0,fontsize=tickFontSize,loc='upper left',bbox_to_anchor=[0.2,0.6,.5,.5])
ax.set_xlim([0,10])
ax.set_ylim([-.2,1])
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(enzymeColor)
ax.tick_params(axis='y',colors=enzymeColor)
ax.tick_params(axis='y',which='minor',colors=enzymeColor)

f.text(0.27,0.31,'F',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2,1])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
for i in range(len(normvars_kcatAsweep)):
    plt.plot(times,normvars_kcatAsweep[i,1],linewidth=plotWidth,label='$k_{cat,A}=10^{%i}$' % (i-3),color=plt.cm.spring(i/len(normvars_kcatAsweep)))
ax.legend(frameon=0,fontsize=tickFontSize,loc='upper left',bbox_to_anchor=[0.2,0.6,.5,.5])
ax.set_xlim([0,10])
ax.set_ylim([-.2,1])
ax.set_xlabel('Generations',fontsize=axisFontSize)
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(signalColor)
ax.tick_params(axis='y',colors=signalColor)
ax.tick_params(axis='y',which='minor',colors=signalColor)

f.text(0.51,0.63,'D',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1,2])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
ax.plot(np.linspace(0,Tccs[0]*10,len(normvar_Tccweep_0[0])),normvar_Tccweep_0[0],linewidth=plotWidth,label='$T_{cc}$=' + str(Tccs[0]),color=plt.cm.winter(0/4))
ax.plot(np.linspace(0,Tccs[1]*10,len(normvar_Tccweep_1[0])),normvar_Tccweep_1[0],linewidth=plotWidth,label='$T_{cc}$=' + str(Tccs[1]),color=plt.cm.winter(1/4))
ax.plot(np.linspace(0,Tccs[2]*10,len(normvar_Tccweep_2[0])),normvar_Tccweep_2[0],linewidth=plotWidth,label='$T_{cc}$=' + str(Tccs[2]),color=plt.cm.winter(2/4))
ax.plot(np.linspace(0,Tccs[3]*10,len(normvar_Tccweep_3[0])),normvar_Tccweep_3[0],linewidth=plotWidth,label='$T_{cc}$=' + str(Tccs[2]),color=plt.cm.winter(3/4))
ax.legend(frameon=0,fontsize=tickFontSize,loc='upper left',bbox_to_anchor=[0.2,0.6,.5,.5])
ax.set_xlim([0,50000])
ax.set_ylim([-.2,1])
ax.set_xticks(np.linspace(0,50000,3))
ax.set_xticks(np.linspace(0,50000,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(enzymeColor)
ax.tick_params(axis='y',colors=enzymeColor)
ax.tick_params(axis='y',which='minor',colors=enzymeColor)

f.text(0.745,0.63,'D\'',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1,3])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
ax.plot(np.linspace(0,10,len(normvar_Tccweep_0[0])),normvar_Tccweep_0[0],linewidth=plotWidth,label='$T_{cc}$=' + str(Tccs[0]),color=plt.cm.winter(0/4))
ax.plot(np.linspace(0,10,len(normvar_Tccweep_1[0])),normvar_Tccweep_1[0],linewidth=plotWidth,label='$T_{cc}$=' + str(Tccs[1]),color=plt.cm.winter(1/4))
ax.plot(np.linspace(0,10,len(normvar_Tccweep_2[0])),normvar_Tccweep_2[0],linewidth=plotWidth,label='$T_{cc}$=' + str(Tccs[2]),color=plt.cm.winter(2/4))
ax.plot(np.linspace(0,10,len(normvar_Tccweep_3[0])),normvar_Tccweep_3[0],linewidth=plotWidth,label='$T_{cc}$=' + str(Tccs[3]),color=plt.cm.winter(3/4))
ax.legend(frameon=0,fontsize=tickFontSize,loc='upper left',bbox_to_anchor=[0.2,0.6,.5,.5])
ax.set_xlim([0,10])
ax.set_ylim([-.2,1])
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(enzymeColor)
ax.tick_params(axis='y',colors=enzymeColor)
ax.tick_params(axis='y',which='minor',colors=enzymeColor)

f.text(0.51,0.31,'G',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2:3,2:3])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
ax.plot(np.linspace(0,Tccs[0]*10,len(normvar_Tccweep_0[1])),normvar_Tccweep_0[1],linewidth=plotWidth,label='$T_{cc}$=' + str(Tccs[0]),color=plt.cm.winter(0/4))
ax.plot(np.linspace(0,Tccs[1]*10,len(normvar_Tccweep_1[1])),normvar_Tccweep_1[1],linewidth=plotWidth,label='$T_{cc}$=' + str(Tccs[1]),color=plt.cm.winter(1/4))
ax.plot(np.linspace(0,Tccs[2]*10,len(normvar_Tccweep_2[1])),normvar_Tccweep_2[1],linewidth=plotWidth,label='$T_{cc}$=' + str(Tccs[2]),color=plt.cm.winter(2/4))
ax.plot(np.linspace(0,Tccs[3]*10,len(normvar_Tccweep_3[1])),normvar_Tccweep_3[1],linewidth=plotWidth,label='$T_{cc}$=' + str(Tccs[3]),color=plt.cm.winter(3/4))
ax.legend(frameon=0,fontsize=tickFontSize,loc='upper left',bbox_to_anchor=[0.2,0.6,.5,.5])
ax.set_xlabel('Time',fontsize=axisFontSize)
ax.set_xlim([0,50000])
ax.set_ylim([-.2,1])
ax.set_xticks(np.linspace(0,50000,3))
ax.set_xticks(np.linspace(0,50000,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(signalColor)
ax.tick_params(axis='y',colors=signalColor)
ax.tick_params(axis='y',which='minor',colors=signalColor)

f.text(0.745,0.31,'G\'',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2:3,3:4])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
ax.plot(np.linspace(0,10,len(normvar_Tccweep_0[1])),normvar_Tccweep_0[1],linewidth=plotWidth,label='$T_{cc}$=' + str(Tccs[0]),color=plt.cm.winter(0/4))
ax.plot(np.linspace(0,10,len(normvar_Tccweep_1[1])),normvar_Tccweep_1[1],linewidth=plotWidth,label='$T_{cc}$=' + str(Tccs[1]),color=plt.cm.winter(1/4))
ax.plot(np.linspace(0,10,len(normvar_Tccweep_2[1])),normvar_Tccweep_2[1],linewidth=plotWidth,label='$T_{cc}$=' + str(Tccs[2]),color=plt.cm.winter(2/4))
ax.plot(np.linspace(0,10,len(normvar_Tccweep_3[1])),normvar_Tccweep_3[1],linewidth=plotWidth,label='$T_{cc}$=' + str(Tccs[3]),color=plt.cm.winter(3/4))
ax.legend(frameon=0,fontsize=tickFontSize,loc='upper left',bbox_to_anchor=[0.2,0.6,.5,.5])
ax.set_xlabel('Generations',fontsize=axisFontSize)
ax.set_xlim([0,10])
ax.set_ylim([-.2,1])
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(signalColor)
ax.tick_params(axis='y',colors=signalColor)
ax.tick_params(axis='y',which='minor',colors=signalColor)

plt.subplots_adjust(top = .99, bottom = 0.08, right = .99, left = 0.09)
plt.show()

#%% Figure S09 (Full Rkcat sweep): Analyze 

proddeg = img.imread(PROJECT_DIR / 'graphics/ngigraphic51.png')

normvars = np.zeros([20,6,1001])
Rkcats = np.zeros(20)
for file in os.listdir(PROJECT_DIR / 'prodanddeg/Rkcatsweep2'):
    index = int(file.split('_')[1].split('.')[0])
    
    filepath = os.path.join(PROJECT_DIR / 'prodanddeg/Rkcatsweep2', file)
    with open(filepath,'rb') as f:
        kcatA,kcatB,normvar = pickle.load(f)
    
    normvars[index] = normvar
    Rkcats[index] = kcatB/kcatA

colors = np.ones([len(Rkcats),4])
colors[:,0] = np.linspace(112/255,52/255,len(Rkcats))
colors[:,1] = np.linspace(233/255,137/255,len(Rkcats))
colors[:,2] = np.linspace(255/255,153/255,len(Rkcats))
RkcatColors = ListedColormap(colors)

#%% Figure S09 (Full Rkcat sweep): plot

times = np.linspace(0,10,1001)

f = plt.figure(figsize=(16,3.5))
gs = GridSpec(1,4,figure=f,wspace=0.6,hspace=0.2)

f.text(0.001,0.875,'A',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_axes([0.02,0.1,.8,.8],anchor='NW')
ax.imshow(proddeg)
ax.axis('off')

f.text(0.2,0.875,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,1:2])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
for i in range(len(Rkcats)):
    plt.plot(times,normvars[i,0],color=colors[i],label='$R_{k_{cat}}=%.2f$' % Rkcats[i],linewidth=plotWidth)
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.set_xlabel('Generations',fontsize=axisFontSize)
ax.set_ylabel('Producing Enzyme\nLAS ($\Delta \hat{\sigma}^2_{\Delta [A]}$)',fontsize=axisFontSize,labelpad=-10)
ax.set_xlim([0,10])
ax.set_ylim([-0.2,1])
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(enzymeColor)
ax.tick_params(axis='y',colors=enzymeColor)
ax.tick_params(axis='y',which='minor',colors=enzymeColor)
cax = f.add_subplot([0.45,0.6,.01,.3])
coef_cbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,2), cmap=RkcatColors),cax=cax,ticks=[0,1,2])
cax.yaxis.set_ticks_position('left')
coef_cbar.set_label('$R_{k_{cat}}$',fontsize=tickFontSize,rotation=-90,labelpad=20)
coef_cbar.ax.tick_params(length=tickLength/2,width=tickWidth/2,labelsize=18)
coef_cbar.outline.set_linewidth(2)

f.text(0.48,0.875,'C',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,2:3])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
for i in range(len(Rkcats)):
    plt.plot(times,normvars[i,2],color=colors[i],label='$R_{k_{cat}}=%.1f$' % Rkcats[i],linewidth=plotWidth)
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.set_xlabel('Generations',fontsize=axisFontSize)
ax.set_ylabel('Product\nLAS ($\Delta \hat{\sigma}^2_{\Delta [B]}$)',fontsize=axisFontSize,labelpad=-10)
ax.set_xlim([0,10])
ax.set_ylim([-0.2,1])
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(signalColor)
ax.tick_params(axis='y',colors=signalColor)
ax.tick_params(axis='y',which='minor',colors=signalColor)
cax = f.add_subplot([0.7,0.6,.01,.3])
coef_cbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,2), cmap=RkcatColors),cax=cax,ticks=[0,1,2])
cax.yaxis.set_ticks_position('left')
coef_cbar.set_label('$R_{k_{cat}}$',fontsize=tickFontSize,rotation=-90,labelpad=20)
coef_cbar.ax.tick_params(length=tickLength/2,width=tickWidth/2,labelsize=18)
coef_cbar.outline.set_linewidth(2)

f.text(0.72,0.875,'D',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,3:4])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
for i in range(len(Rkcats)):
    plt.plot(times,normvars[i,1],color=colors[i],label='$R_{k_{cat}}=%.1f$' % Rkcats[i],linewidth=plotWidth)
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.set_xlabel('Generations',fontsize=axisFontSize)
ax.set_ylabel('Degrading Enzyme\nLAS ($\Delta \hat{\sigma}^2_{\Delta [C]}$)',fontsize=axisFontSize,labelpad=-10)
ax.set_xlim([0,10])
ax.set_ylim([-0.2,1])
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
ax.spines['left'].set_color(breakerColor)
ax.tick_params(axis='y',colors=breakerColor)
ax.tick_params(axis='y',which='minor',colors=breakerColor)
cax = f.add_subplot([0.97,0.6,.01,.3])
coef_cbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,2), cmap=RkcatColors),cax=cax,ticks=[0,1,2])
cax.yaxis.set_ticks_position('left')
coef_cbar.set_label('$R_{k_{cat}}$',fontsize=tickFontSize,rotation=-90,labelpad=20)
coef_cbar.ax.tick_params(length=tickLength/2,width=tickWidth/2,labelsize=18)
coef_cbar.outline.set_linewidth(2)

plt.subplots_adjust(top = .93, bottom = 0.2, right = .99, left = 0.05)
plt.show()

#%% Figure S10 (Monofunctional Phosphorylation Motif): pull data 

phosphorylation = img.imread(PROJECT_DIR / 'graphics/ngigraphic58.png')

with open(PROJECT_DIR / 'prodRateSat/phos_int3.pickle','rb') as f:
    prodAs,prodBs,Aeqs,Beqs,Ceqs,Deqs,Eeqs,varAs,varBs,varCs,varDs,varEs = pickle.load(f)


#%% Figure S10 (Monofunctional Phosphorylation Motif): plot 

xmax = 30
ymax = 30

f = plt.figure(figsize=(16,8))
gs = GridSpec(3,5,figure=f,wspace=0.6,hspace=0.4)

f.text(0.001,0.94,'A',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([0,.72,.25,.25])
ax.imshow(phosphorylation)
ax.axis('off')

f.text(0.001,0.62,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1:2,0:1])
im = ax.imshow(np.log(Aeqs),origin='lower',vmin=-2,vmax=12,extent=[0,xmax,0,ymax])
ax.set_title('log($[A\']_{eq}$)',fontsize=axisFontSize)
ax.set_ylabel('$P_{prod,A\'}$',fontsize=axisFontSize)
ax.set_xticks(np.linspace(0,xmax,2),['$10^{-3}$','$10^{2}$'])
ax.set_xticks(np.linspace(0,xmax,6),[],minor=1)
ax.set_yticks(np.linspace(0,ymax,6),['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$'])
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(tickWidth)
ax.spines['top'].set_linewidth(tickWidth)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.19,0.62,'C',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1:2,1:2])
im = ax.imshow(np.log(Beqs),origin='lower',vmin=-2,vmax=12,extent=[0,xmax,0,ymax])
ax.set_title('log($[A]_{eq}$)',fontsize=axisFontSize)
ax.set_xticks(np.linspace(0,xmax,2),['$10^{-3}$','$10^{2}$'])
ax.set_xticks(np.linspace(0,xmax,6),[],minor=1)
ax.set_yticks(np.linspace(0,ymax,6),['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$'])
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(tickWidth)
ax.spines['top'].set_linewidth(tickWidth)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.38,0.62,'D',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1:2,2:3])
im = ax.imshow(np.log(Ceqs),origin='lower',vmin=-2,vmax=12,extent=[0,xmax,0,ymax])
ax.set_title('log($[B\']_{eq}$)',fontsize=axisFontSize)
ax.set_xticks(np.linspace(0,xmax,2),['$10^{-3}$','$10^{2}$'])
ax.set_xticks(np.linspace(0,xmax,6),[],minor=1)
ax.set_yticks(np.linspace(0,ymax,6),['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$'])
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(tickWidth)
ax.spines['top'].set_linewidth(tickWidth)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.57,0.62,'E',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1:2,3:4])
im = ax.imshow(np.log(Deqs),origin='lower',vmin=-2,vmax=12,extent=[0,xmax,0,ymax])
ax.set_title('log($[A B\']_{eq}$)',fontsize=axisFontSize)
ax.set_xticks(np.linspace(0,xmax,2),['$10^{-3}$','$10^{2}$'])
ax.set_xticks(np.linspace(0,xmax,6),[],minor=1)
ax.set_yticks(np.linspace(0,ymax,6),['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$'])
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(tickWidth)
ax.spines['top'].set_linewidth(tickWidth)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.75,0.62,'F',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1:2,4:5])
im = ax.imshow(np.log(Eeqs),origin='lower',vmin=-2,vmax=12,extent=[0,xmax,0,ymax])
ax.set_title('log($[B]_{eq}$)',fontsize=axisFontSize)
ax.set_xticks(np.linspace(0,xmax,2),['$10^{-3}$','$10^{2}$'])
ax.set_xticks(np.linspace(0,xmax,6),[],minor=1)
ax.set_yticks(np.linspace(0,ymax,6),['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$'])
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(tickWidth)
ax.spines['top'].set_linewidth(tickWidth)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

cax = f.add_axes([.86,.425,.1,.2])
cax.axis('off')
coef_cbar = ax.figure.colorbar(im,ax=cax,ticks=[-2,0,2,4,6,8,10,12])
coef_cbar.set_label('log([molecule])',fontsize=tickFontSize,rotation=-90,labelpad=12,color='k')
coef_cbar.ax.tick_params(length=tickLength/2,width=tickWidth/2,labelsize=18,colors='k')
coef_cbar.outline.set_linewidth(2)

f.text(0.001,0.3,'G',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2:3,0:1])
ax.set_title('$\Delta \hat{\sigma}^2_{\Delta [A\']}$',fontsize=axisFontSize)
im = ax.imshow(varAs,origin='lower',vmin=0,vmax=1,cmap='inferno')
ax.set_xlabel('$P_{prod,B\'}$',fontsize=axisFontSize)
ax.set_ylabel('$P_{prod,A\'}$',fontsize=axisFontSize)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(tickWidth)
ax.spines['top'].set_linewidth(tickWidth)
ax.set_xticks(np.linspace(0,xmax,2),['$10^{-3}$','$10^{2}$'])
ax.set_xticks(np.linspace(0,xmax,6),[],minor=1)
ax.set_yticks(np.linspace(0,ymax,6),['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$'])
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.19,0.3,'H',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2:3,1:2])
ax.set_title('$\Delta \hat{\sigma}^2_{\Delta [A]}$',fontsize=axisFontSize)
im = ax.imshow(varBs,origin='lower',vmin=0,vmax=1,cmap='inferno')
ax.set_xlabel('$P_{prod,B\'}$',fontsize=axisFontSize)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(tickWidth)
ax.spines['top'].set_linewidth(tickWidth)
ax.set_xticks(np.linspace(0,xmax,2),['$10^{-3}$','$10^{2}$'])
ax.set_xticks(np.linspace(0,xmax,6),[],minor=1)
ax.set_yticks(np.linspace(0,ymax,6),['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$'])
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.38,0.3,'I',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2:3,2:3])
ax.set_title('$\Delta \hat{\sigma}^2_{\Delta [B\']}$',fontsize=axisFontSize)
im = ax.imshow(varCs,origin='lower',vmin=0,vmax=1,cmap='inferno')
ax.set_xlabel('$P_{prod,B\'}$',fontsize=axisFontSize)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(tickWidth)
ax.spines['top'].set_linewidth(tickWidth)
ax.set_xticks(np.linspace(0,xmax,2),['$10^{-3}$','$10^{2}$'])
ax.set_xticks(np.linspace(0,xmax,6),[],minor=1)
ax.set_yticks(np.linspace(0,ymax,6),['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$'])
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.57,0.3,'J',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2:3,3:4])
ax.set_title('$\Delta \hat{\sigma}^2_{\Delta [A B\']}$',fontsize=axisFontSize)
im = ax.imshow(varDs,origin='lower',vmin=0,vmax=1,cmap='inferno')
ax.set_xlabel('$P_{prod,B\'}$',fontsize=axisFontSize)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(tickWidth)
ax.spines['top'].set_linewidth(tickWidth)
ax.set_xticks(np.linspace(0,xmax,2),['$10^{-3}$','$10^{2}$'])
ax.set_xticks(np.linspace(0,xmax,6),[],minor=1)
ax.set_yticks(np.linspace(0,ymax,6),['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$'])
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.75,0.3,'K',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2:3,4:5])
ax.set_title('$\Delta \hat{\sigma}^2_{\Delta [B]}$',fontsize=axisFontSize)
im = ax.imshow(varEs,origin='lower',vmin=0,vmax=1,cmap='inferno')
ax.set_xlabel('$P_{prod,B\'}$',fontsize=axisFontSize)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(tickWidth)
ax.spines['top'].set_linewidth(tickWidth)
ax.set_xticks(np.linspace(0,xmax,2),['$10^{-3}$','$10^{2}$'])
ax.set_xticks(np.linspace(0,xmax,6),[],minor=1)
ax.set_yticks(np.linspace(0,ymax,6),['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^{0}$','$10^{1}$','$10^{2}$'])
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

cax = f.add_axes([.86,.11,.1,.2])
cax.axis('off')
coef_cbar = ax.figure.colorbar(im,ax=cax,ticks=[-0.2,0,.2,.4,.6,.8,1])
coef_cbar.set_label('LAS ($\Delta\hat{\sigma}^2$)',fontsize=tickFontSize,rotation=-90,labelpad=15,color='k')
coef_cbar.ax.tick_params(length=tickLength/2,width=tickWidth/2,labelsize=18,colors='k')
coef_cbar.outline.set_linewidth(2)

plt.subplots_adjust(top = .96, bottom = 0.11, right = .93, left = 0.07)
plt.show()

#%% Figure S11 (Bifunctional TCS): Pull Data 

pathway = img.imread(PROJECT_DIR / 'graphics/ngigraphic60.png')

with open(PROJECT_DIR / 'prodRateSat/phos2_0.pickle','rb') as f:
    prodAs,prodBs,Aeqs,Beqs,Ceqs,Eeqs,Deqs,Feqs,varAs,varBs,varCs,varDs,varEs,varFs = pickle.load(f)

#%% Figure S11 (Bifunctional TCS): Plot

f = plt.figure(figsize=(16,4))
gs = GridSpec(1,3,figure=f,wspace=0.8,hspace=0.1)

f.text(0.001,0.89,'A',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([-0.02,.05,.4,.9])
ax.imshow(pathway)
ax.axis('off')

f.text(0.33,0.89,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,1:2])
im = ax.imshow(varDs[0:21,:],origin='lower',vmin=0,vmax=1,cmap='inferno',aspect=1.5)
ax.set_xticks(np.linspace(0,30,6),['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^0$','$10^1$','$10^2$'])
ax.set_yticks(np.linspace(0,20,4),['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^0$'])
ax.set_xlabel('$P_{prod,B\'}$',fontsize=axisFontSize)
ax.set_ylabel('$P_{prod,A\'}$',fontsize=axisFontSize)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
cax = f.add_axes([.52,.23,.1,.74])
cax.axis('off')
coef_cbar = ax.figure.colorbar(im,ax=cax,ticks=[-0.2,0,.2,.4,.6,.8,1])
coef_cbar.set_label('Product LAS ($\Delta\hat{\sigma}^2_{\Delta [B]}$)',fontsize=tickFontSize,rotation=-90,labelpad=25,color='k')
coef_cbar.ax.tick_params(length=tickLength/2,width=tickWidth/2,labelsize=18,colors='k')
coef_cbar.outline.set_linewidth(2)

f.text(0.67,0.89,'C',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,2:3])
im = ax.imshow(Feqs[0:21,:]/(prodAs[0:21,:]*1000),origin='lower',vmin=0,vmax=1,cmap='hot',aspect=1.5)
ax.set_xticks(np.linspace(0,30,6),['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^0$','$10^1$','$10^2$'])
ax.set_yticks(np.linspace(0,20,4),['$10^{-3}$','$10^{-2}$','$10^{-1}$','$10^0$'])
ax.set_xlabel('$P_{prod,B\'}$',fontsize=axisFontSize)
ax.set_ylabel('$P_{prod,A\'}$',fontsize=axisFontSize)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
cax = f.add_axes([.855,.23,.1,.74])
cax.axis('off')
coef_cbar = ax.figure.colorbar(im,ax=cax,ticks=[0,.2,.4,.6,.8,1])
coef_cbar.set_label('$[A\'B]/[A_T]$',fontsize=tickFontSize,rotation=-90,labelpad=20,color='k')
coef_cbar.ax.tick_params(length=tickLength/2,width=tickWidth/2,labelsize=18,colors='k')
coef_cbar.outline.set_linewidth(2)

plt.subplots_adjust(top = .99, bottom = 0.2, right = .93, left = 0.08)
plt.show()


#%% Figure S12 (5 Step Cascade): pull

cascade = img.imread(PROJECT_DIR / 'graphics/ngigraphic75.png')

with open(PROJECT_DIR / 'cascade/cascade5_time/cascade5_time.pickle','rb') as f:
    results = pickle.load(f)
times = np.linspace(0,20,2001)

#%% Figure S12 (5 Step Cascade): plot 

f = plt.figure(figsize=(8,4))
gs = plt.GridSpec(1,3,wspace=0.3,hspace=0.1)

f.text(0.001,0.88,'A',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([0.05,0.08,.3,0.9])
ax.imshow(cascade)
ax.axis('off')

f.text(0.36,0.88,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0,1:3])
ax.hlines(0,0,20,linestyle='dashed',linewidth=plotWidth,color='k')
ax.plot(times,results['normvar'][0],color=enzymeColor,label='A')
ax.plot(times,results['normvar'][1],color=signalColor,label='B')
ax.plot(times,results['normvar'][2],color=breakerColor,label='C')
ax.plot(times,results['normvar'][3],color=color_L,label='D')
ax.plot(times,results['normvar'][4],color=kcatColor,label='E')
ax.legend(frameon=0,fontsize=tickFontSize,ncol=2)
ax.set_xlabel('Generations',fontsize=axisFontSize)
ax.set_xlim([0,20])
ax.set_xticks(np.linspace(0,20,5))
ax.set_xticks(np.linspace(0,20,21),[],minor=1)
ax.set_ylabel('LAS ($\Delta \hat{\sigma}^2$)',fontsize=axisFontSize,labelpad=-15)
ax.set_ylim([-0.2,1])
ax.set_yticks(np.linspace(-0.2,1,7))
ax.set_yticks(np.linspace(-0.2,1,13),[],minor=1)

ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

plt.subplots_adjust(top = .95, bottom = 0.2, right = .98, left = 0.2)
plt.show()

#%% Figure S13 (Spatial Simulation graphic): Pull 

spacealgo = img.imread(PROJECT_DIR / 'graphics/ngigraphic26.png')

#%% Figure S13 (Spatial Simulation graphic): plot 

f = plt.figure(figsize=(16,4))
ax = f.add_subplot([0,0,1,1])
ax.imshow(spacealgo)
ax.axis('off')

#%% Figure S14 (cousin maps different reference cells): pull

with open(PROJECT_DIR / 'cascade/cascade_spatial_analyze/cascade_spatial_analyze.pickle','rb') as f:
    imgs = pickle.load(f)['cousinmaps_all']
cousinNums = [100,200,300,400,500]
ts = [0,2000,4000,6000,8000,10000]

#%% Figure S14 (cousin maps different reference cells): plot

f = plt.figure(figsize=(16,14))

f.add_artist(mpatches.Rectangle((0,0),1,1,facecolor='k',zorder=0,alpha=1))
for j in range(len(imgs)):
    for i in range(len(imgs[j])):
        ax = f.add_axes([.15*i,.8-.2*j,.18,.18])
        ax.imshow(imgs[j][i],vmin=-2,vmax=8)
        ax.axis('off')
        if i==0:
            f.text(0.02,0.86-.2*j,'cell %i' % cousinNums[j],fontsize=tickFontSize,color='white',rotation=90,fontname='roboto')
        if j==0:
            ax.set_title('t=%4i' % ts[i],color='white',fontsize=tickFontSize)

cbar_ax = f.add_axes([0.95,.8,.01,.18])
coef_cbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-1,8), cmap='viridis'),cax=cbar_ax,ticks=[-1,2,5,8])
cbar_ax.yaxis.set_ticks_position('left')
coef_cbar.ax.set_yticklabels([8,5,2,-1])
coef_cbar.set_label('Cousin Number',fontsize=tickFontSize,rotation=-90,labelpad=20,color='white')
coef_cbar.ax.tick_params(length=tickLength/2,width=tickWidth/2,labelsize=18,colors='white')
coef_cbar.outline.set_linewidth(2)
coef_cbar.outline.set_color('white')

#%% Figure S15 (relatedness curves different grids): pull

with open(PROJECT_DIR / 'gridcells/randomseeds/grid_1000_relatedness.pickle','rb') as f:
    relatedness_1000 = pickle.load(f)
with open(PROJECT_DIR / 'gridcells/randomseeds/grid_1001_relatedness.pickle','rb') as f:
    relatedness_1001 = pickle.load(f)
with open(PROJECT_DIR / 'gridcells/randomseeds/grid_1002_relatedness.pickle','rb') as f:
    relatedness_1002 = pickle.load(f)

#%% Figure S15 (relatedness curves different grids): plot

f,ax = plt.subplots(figsize=(8,5))

ax.scatter(np.linspace(1,8,8),np.mean(relatedness_1000,axis=0),color=randomColor,label='seed=1000')
ax.errorbar(np.linspace(1,8,8),np.mean(relatedness_1000,axis=0),yerr=np.std(relatedness_1000,axis=0),fmt='None',ecolor=randomColor)
ax.scatter(np.linspace(0.9,7.9,8),np.mean(relatedness_1001,axis=0),color=TccColor,label='seed=1001')
ax.errorbar(np.linspace(0.9,7.9,8),np.mean(relatedness_1001,axis=0),yerr=np.std(relatedness_1001,axis=0),fmt='None',ecolor=TccColor)
ax.scatter(np.linspace(1.1,8.1,8),np.mean(relatedness_1002,axis=0),color=kcatColor,label='seed=1002')
ax.errorbar(np.linspace(1.1,8.1,8),np.mean(relatedness_1002,axis=0),yerr=np.std(relatedness_1002,axis=0),fmt='None',ecolor=kcatColor)
ax.legend(frameon=0,fontsize=tickFontSize,loc='lower right')
ax.set_ylim([0,7])
ax.set_xlabel('Neighborhood Size (Cell Lengths)',fontsize=axisFontSize)
ax.set_ylabel('Relatedness (Ave. Cousin #)',fontsize=axisFontSize)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

plt.subplots_adjust(top = .98, bottom = 0.19, right = .99, left = 0.1)
plt.show()

#%% Figure S16 (Moran's I weight matrices): pull

weightdefs = img.imread(PROJECT_DIR / 'graphics/ngigraphic27.png')

def load_moranIs(shape,radii=range(1,10)):
    morIs = []
    for r in radii:
        name = f'cascade_spatial_moranI_{shape}_r{r}'
        with open(PROJECT_DIR / 'cascade' / name / f'{name}.pickle','rb') as f:
            morIs.append(pickle.load(f))
    return np.stack(morIs,axis=0)

morIs_discdist = load_moranIs('discdist')
morIs_donut = load_moranIs('donut')
morIs_gausdisc = load_moranIs('gausdist')

#%% Figure S16 (Moran's I weight matrices): plot

def expDecay(x,tau,x0):
    return x0*np.exp(-x/tau)

radii = np.linspace(1,9,9)
plotradii = np.linspace(0,9,100)
times = np.linspace(0,10,101)

f = plt.figure(figsize=(16,8))
gs = GridSpec(3,5,figure=f,wspace=0.5,hspace=0.3)

f.text(0.001,0.94,'A',fontsize=letterLabelSize,fontname='roboto')
f.text(0.001,0.61,'D',fontsize=letterLabelSize,fontname='roboto')
f.text(0.001,0.33,'G',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([-.03,0.09,.55,.9])
ax.imshow(weightdefs)
ax.axis('off')

f.text(0.47,0.94,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,3:4])
for i in range(len(radii)):
    ax.plot(times,morIs_discdist[i,2,:],color=plt.cm.Greys((i+len(radii))/(2*len(radii))))
ax.set_xlim([0,10])
ax.set_ylim([-1,1])
ax.set_ylabel('Spatial Sim.\n(Moran\'s I)', fontsize=axisFontSize, labelpad=-2)
ax.set_xticks(np.linspace(0,10,3))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-1,1,5))
ax.set_yticks(np.linspace(-1,1,21),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.755,0.94,'C',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,4:5])
ax.scatter(radii,morIs_discdist[:,0,-1],color=enzymeColor,label='A')
params,cov = curve_fit(expDecay,radii,morIs_discdist[:,0,-1],p0=[.5,1])
plt.plot(plotradii,expDecay(plotradii,params[0],params[1]),linestyle='dashed',color=enzymeColor)
ax.scatter(radii,morIs_discdist[:,1,-1],color=signalColor,label='B')
params,cov = curve_fit(expDecay,radii,morIs_discdist[:,1,-1],p0=[.5,1])
plt.plot(plotradii,expDecay(plotradii,params[0],params[1]),linestyle='dashed',color=signalColor)
ax.scatter(radii,morIs_discdist[:,2,-1],color='purple',label='C')
params,cov = curve_fit(expDecay,radii,morIs_discdist[:,2,-1],p0=[.5,1])
plt.plot(plotradii,expDecay(plotradii,params[0],params[1]),linestyle='dashed',color='purple')
ax.legend(frameon=0,fontsize=tickFontSize,loc='upper left',bbox_to_anchor=[.6,.6,.5,.5])
ax.set_xlim([0.8,9.2])
ax.set_xticks(np.linspace(1,9,5))
ax.set_xticks(np.linspace(1,9,9),[],minor=1)
ax.set_yticks(np.linspace(0,.4,5))
ax.set_yticks(np.linspace(0,.4,9),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.47,0.61,'E',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1:2,3:4])
for i in range(len(radii)):
    ax.plot(times,morIs_donut[i,2,:],color=plt.cm.Greys((i+len(radii))/(2*len(radii))))
ax.set_xlim([0,10])
ax.set_ylim([-1,1])
ax.set_ylabel('Spatial Sim.\n(Moran\'s I)', fontsize=axisFontSize, labelpad=-2)
ax.set_xticks(np.linspace(0,10,3))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-1,1,5))
ax.set_yticks(np.linspace(-1,1,21),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.76,0.61,'F',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1:2,4:5])
ax.scatter(radii,morIs_donut[:,0,-1],color=enzymeColor,label='A')
params,cov = curve_fit(expDecay,radii,morIs_donut[:,0,-1],p0=[.5,1])
plt.plot(plotradii,expDecay(plotradii,params[0],params[1]),linestyle='dashed',color=enzymeColor)
ax.scatter(radii,morIs_donut[:,1,-1],color=signalColor,label='B')
params,cov = curve_fit(expDecay,radii,morIs_donut[:,1,-1],p0=[.5,1])
plt.plot(plotradii,expDecay(plotradii,params[0],params[1]),linestyle='dashed',color=signalColor)
ax.scatter(radii,morIs_donut[:,2,-1],color='purple',label='C')
params,cov = curve_fit(expDecay,radii,morIs_donut[:,2,-1],p0=[.5,1])
plt.plot(plotradii,expDecay(plotradii,params[0],params[1]),linestyle='dashed',color='purple')
ax.legend(frameon=0,fontsize=tickFontSize,loc='upper left',bbox_to_anchor=[.6,.6,.5,.5])
ax.set_xlim([0.8,9.2])
ax.set_xticks(np.linspace(1,9,5))
ax.set_xticks(np.linspace(1,9,9),[],minor=1)
ax.set_yticks(np.linspace(0,.4,5))
ax.set_yticks(np.linspace(0,.4,9),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.47,0.33,'H',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2:3,3:4])
for i in range(len(radii)):
    ax.plot(times,morIs_gausdisc[i,2,:],color=plt.cm.Greys((i+len(radii))/(2*len(radii))))
ax.set_xlim([0,10])
ax.set_ylim([-1,1])
ax.set_xlabel('Cell Generations',fontsize=axisFontSize,labelpad=-2)
ax.set_ylabel('Spatial Sim.\n(Moran\'s I)', fontsize=axisFontSize, labelpad=-2)
ax.set_xticks(np.linspace(0,10,3))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-1,1,5))
ax.set_yticks(np.linspace(-1,1,21),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.76,0.33,'I',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[2:3,4:5])
ax.scatter(radii,morIs_gausdisc[:,0,-1],color=enzymeColor,label='A')
params,cov = curve_fit(expDecay,radii,morIs_gausdisc[:,0,-1],p0=[.5,1])
plt.plot(plotradii,expDecay(plotradii,params[0],params[1]),linestyle='dashed',color=enzymeColor)
ax.scatter(radii,morIs_gausdisc[:,1,-1],color=signalColor,label='B')
params,cov = curve_fit(expDecay,radii,morIs_gausdisc[:,1,-1],p0=[.5,1])
plt.plot(plotradii,expDecay(plotradii,params[0],params[1]),linestyle='dashed',color=signalColor)
ax.scatter(radii,morIs_gausdisc[:,2,-1],color='purple',label='C')
params,cov = curve_fit(expDecay,radii,morIs_gausdisc[:,2,-1],p0=[.5,1])
plt.plot(plotradii,expDecay(plotradii,params[0],params[1]),linestyle='dashed',color='purple')
ax.legend(frameon=0,fontsize=tickFontSize,loc='upper left',bbox_to_anchor=[.6,.6,.5,.5])
ax.set_xlabel('Neighborhood Size\n(cell lengths)',fontsize=axisFontSize,labelpad=-2)
ax.set_xlim([0.8,9.2])
ax.set_ylim([-0.1,0.4])
ax.set_xticks(np.linspace(1,9,5))
ax.set_xticks(np.linspace(1,9,9),[],minor=1)
ax.set_yticks(np.linspace(0,.4,5))
ax.set_yticks(np.linspace(0,.4,9),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

plt.subplots_adjust(top = .95, bottom = 0.14, right = .99, left = -.15)
plt.show()

#%% Figure S17 (Seed Cell Schematic): Pull Data 

seedcells = img.imread(PROJECT_DIR / 'graphics/ngigraphic46.png')

#%% Figure S17 (Seed Cell Schematic): Plot 

f = plt.figure(figsize=(16,5.5))
ax = f.add_subplot([0,0,1,1])
ax.imshow(seedcells)
ax.axis('off')

#%% Figure S18 (Correlation Coefficient): Pull Data

prodonly = img.imread(PROJECT_DIR / 'graphics/ngigraphic50.png')

with open(PROJECT_DIR/'correlationcoef/samplerun.pickle','rb') as f:
   dM_rnd,dM_sis,dA_rnd,dA_sis,rM_sis,rM_rnd,rA_sis,rA_rnd = pickle.load(f)

times = np.linspace(0,10,10000)

#%% Figure S18 (Correlation Coefficient): Plot

f = plt.figure(figsize=(16,6))
gs = GridSpec(2,4,figure=f,wspace=0.6,hspace=0.3)

f.text(0.001,0.92,'A',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([-.01,.2,.15,.7])
ax.imshow(prodonly)
ax.axis('off')

f.text(0.15,0.92,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,1:2])
ax.plot(times,np.var(dM_rnd,axis=0),color='gray',linewidth=2,label='random')
ax.plot(times,np.var(dM_sis,axis=0),color='r',linewidth=2,label='related')
ax.legend(frameon=0,fontsize=18,loc='upper left',bbox_to_anchor=[.3,.1,1,1])
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_xlim([0,10])
ax.set_ylim([5,30])
ax.set_yticks(np.linspace(10,30,3))
ax.set_yticks(np.linspace(5,30,26),[],minor=1)
ylabel = ax.set_ylabel('Pairwise Diff.\nVariance ($\sigma^2_{\Delta [A]}$)',fontsize=axisFontSize,labelpad=-5)
ylabel.set_position((.7,.45))
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.425,0.92,'D',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,2:3])
ax.plot(times,np.var(dA_rnd,axis=0),color=randomColor,linewidth=2,label='random')
ax.plot(times,np.var(dA_sis,axis=0),color=signalColor,linewidth=2,label='related')
ax.legend(frameon=0,fontsize=18)
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_xlim([0,10])
ax.set_ylim([0,2000])
ax.set_yticks(np.linspace(0,2000,5))
ax.set_yticks(np.linspace(0,2000,21),[],minor=1)
ylabel = ax.set_ylabel('Pairwise Diff.\nVariance ($\sigma^2_{\Delta [B]}$)',fontsize=axisFontSize,labelpad=-3)
ylabel.set_position((0,.4))
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.72,0.92,'F',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,3:4])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
ax.plot(times,1-np.var(dM_sis,axis=0)/np.var(dM_rnd,axis=0),color=enzymeColor,linewidth=2,label='A')
ax.plot(times,1-np.var(dA_sis,axis=0)/np.var(dA_rnd,axis=0),color=signalColor,linewidth=2,label='B')
ax.legend(frameon=0,fontsize=18)
ax.set_xlim([0,10])
ax.set_ylim([-.2,1])
ax.set_ylabel('LAS ($\Delta \hat{\sigma}^2$)',fontsize=axisFontSize,labelpad=-5)
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.15,0.45,'C',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1:2,1:2])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
ax.plot(times,rM_rnd,color=randomColor,linewidth=2,label='random')
ax.plot(times,rM_sis,color=enzymeColor,linewidth=2,label='related')
ax.legend(frameon=0,fontsize=18)
ax.set_xlim([0,10])
ax.set_ylim([-.2,1])
ax.set_xlabel('Generations',fontsize=axisFontSize)
ax.set_ylabel('Correlation ($r_{\Delta [A]}$)',fontsize=axisFontSize,labelpad=-5)
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.425,0.45,'E',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1:2,2:3])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
ax.plot(times,rA_rnd,color=randomColor,linewidth=2,label='random')
ax.plot(times,rA_sis,color=signalColor,linewidth=2,label='related')
ax.legend(frameon=0,fontsize=18)
ax.set_xlim([0,10])
ax.set_ylim([-.2,1])
ax.set_xlabel('Generations',fontsize=axisFontSize)
ax.set_ylabel('Correlation ($r_{\Delta [B]}$)',fontsize=axisFontSize,labelpad=-5)
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.72,0.45,'G',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[1:2,3:4])
ax.hlines(0,0,2*10**5,color='k',linestyle='dashed',linewidth=plotWidth)
ax.plot(times,rM_sis,color=enzymeColor,linewidth=2,label='A (related)')
ax.plot(times,rA_sis,color=signalColor,linewidth=2,label='B (related)')
ax.legend(frameon=0,fontsize=18)
ax.set_xlim([0,10])
ax.set_ylim([-.2,1])
ax.set_xlabel('Generations',fontsize=axisFontSize)
ax.set_ylabel('Correlation ($r$)',fontsize=axisFontSize,labelpad=-5)
ax.set_xticks(np.linspace(0,10,6))
ax.set_xticks(np.linspace(0,10,11),[],minor=1)
ax.set_yticks(np.linspace(-.2,1,7))
ax.set_yticks(np.linspace(-.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

plt.subplots_adjust(top = .98, bottom = 0.15, right = .99, left = -0.05)
plt.show()

#%% Figure S19 (Toy Model Brekaout): pull data

pathway = img.imread(PROJECT_DIR / 'graphics/ngigraphic89.png')

#%% Figure S19 (Toy Model Brekaout): Plot

f = plt.figure(figsize=(8,4.5))
ax = f.add_subplot([0,0,1,1])
ax.imshow(pathway)
ax.axis('off')



#%% Figure S20 (Phosphorylation Monocycles): pull data 

os.chdir('//prfs.hhmi.org/sgrolab/mark/comp_proj/graphics')
pathway = img.imread(PROJECT_DIR / 'graphics/ngigraphic69.png')
a_slider = img.imread(PROJECT_DIR / 'graphics/a_slider.png')

with open(PROJECT_DIR / 'phos_cycle/phoscycle2.pickle','rb') as f:
    prodAs_MA,prodBs_MA,Aeqs_MA,Beqs_MA,Ceqs_MA,Deqs_MA,varAs_MA,varBs_MA,varCs_MA,varDs_MA,normvarAs_MA,normvarBs_MA,normvarCs_MA,normvarDs_MA = pickle.load(f)

prodColorRange = [enzymeColor,[68/255,10/255,21/255]]
prodColors = np.transpose(np.array((np.linspace(prodColorRange[0][0],prodColorRange[1][0],len(prodAs_MA)),
                                    np.linspace(prodColorRange[0][1],prodColorRange[1][1],len(prodAs_MA)),
                                    np.linspace(prodColorRange[0][2],prodColorRange[1][2],len(prodAs_MA)),
                                    np.linspace(1,1,len(prodAs_MA)))))
prodColorMap = ListedColormap(prodColors)

with open(PROJECT_DIR/'satphos/satphos3.pickle','rb') as f:
    prodBs_MM,kms_MM,Aeqs_MM,Beqs_MM,Ceqs_MM,Deqs_MM,varAs_MM,varBs_MM,varCs_MM,varDs_MM = pickle.load(f)

#%% Figure S20 (Phosphorylation Monocycles): plot 

def logFit(x,xmax,Kx,n):
    return xmax*x**n/(Kx**n+x**n)

f = plt.figure(figsize=(16,5))
gs = GridSpec(1,3,figure=f,wspace=0.5,hspace=0.6)

f.text(0.001,0.91,'A',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([-0.01,.05,.3,.9])
ax.imshow(pathway)
ax.axis('off')

f.text(0.27,0.91,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,1:2])
ax.set_title('Mass-Action Kinetics',fontsize=axisFontSize)
ax.hlines(0,10**-5,10**6,color='k',linestyle='dashed',linewidth=plotWidth)
for i in range(len(prodAs_MA)):
    ax.scatter(Beqs_MA[i]+Deqs_MA[i],normvarDs_MA[i],color=prodColorMap(i/len(prodAs_MA)))
    params,cov = curve_fit(logFit,Beqs_MA[i]+Deqs_MA[i],normvarDs_MA[i],p0=[1,100,1])
    ax.plot(np.logspace(-1,6,100),logFit(np.logspace(-1,6,100),params[0],params[1],params[2]),linestyle='dotted',color=prodColors[i])
ax.set_xscale('log')
ax.set_xlim([0.8*10**0,1.2*10**5])
ax.set_ylim([-.2,1.05])
ax.set_xlabel('Substrate Amount ($[B\']+[B]$)',fontsize=axisFontSize)
ax.set_ylabel('Product\nLAS ($\Delta \hat{\sigma}^2_{\Delta [B]}$)',fontsize=axisFontSize,labelpad=-10)
ax.set_yticks(np.linspace(-0.2,1,7))
ax.set_yticks(np.linspace(-0.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)
cax = f.add_subplot([0.6,0.35,.01,.4])
coef_cbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,3), cmap=prodColorMap),cax=cax,ticks=[0,1,2,3])
cax.yaxis.set_ticks_position('left')
coef_cbar.set_label('[Enzyme]',fontsize=tickFontSize,rotation=-90,labelpad=17)
coef_cbar.ax.tick_params(length=tickLength/2,width=tickWidth/2,labelsize=18)
coef_cbar.outline.set_linewidth(2)
coef_cbar.ax.set_yticklabels(['$10^{1}$','$10^2$','$10^3$','$10^4$'])

f.text(0.65,0.91,'C',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,2:3])
ax.set_title('Enzyme Kinetics',fontsize=axisFontSize)
ax.hlines(0,10**-5,10**6,color='k',linestyle='dashed',linewidth=plotWidth)
for i in range(len(kms_MM)):
    ax.scatter(Beqs_MM[i]+Deqs_MM[i],varDs_MM[i],color=plt.cm.bwr((np.log10(Aeqs_MM[i,0]/kms_MM[i,0])+2)/4))
    params,cov = curve_fit(logFit,prodBs_MM[i,0:-3]*1000,varDs_MM[i,0:-3],p0=[1,1,1])
    ax.plot(np.logspace(-1,6,100),logFit(np.logspace(-1,6,100),params[0],params[1],params[2]),linestyle='dotted',color=plt.cm.bwr((np.log10(Aeqs_MM[i,0]/kms_MM[i,0])+2)/4))
ax.set_xscale('log')
ax.set_xlim([8*10**0,4*10**5])
ax.set_ylim([-.2,1.05])
ax.set_xlabel('Substrate Amount ($[B\'] + [B]$)',fontsize=axisFontSize)
ax.set_ylabel('Product\nLAS ($\Delta \hat{\sigma}^2_{\Delta [B]}$)',fontsize=axisFontSize,labelpad=-10)
ax.set_yticks(np.linspace(-0.2,1,7))
ax.set_yticks(np.linspace(-0.2,1,13),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',length=tickLength,width=tickWidth,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

cbar_ax = f.add_axes([0.96,.35,.01,.4])
coef_cbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-2,2), cmap='bwr'),cax=cbar_ax,ticks=[-2,-1,0,1,2])
cbar_ax.yaxis.set_ticks_position('left')
coef_cbar.ax.set_yticklabels([-2,-1,0,1,2])
coef_cbar.set_label('$log_{10}([A]_{eq}/K_M)$',fontsize=tickFontSize,rotation=-90,labelpad=20)
coef_cbar.ax.tick_params(length=tickLength/2,width=tickWidth/2,labelsize=18)
coef_cbar.outline.set_linewidth(2)

plt.subplots_adjust(top = 0.93, bottom = 0.18, right = 0.99, left = -0.01)
plt.show()



#%% Figure S21 (Reactant Similarity): pull data 

def calcProdRate(A,B,kcatA,Km):
    return kcatA/2*(A+B+Km-np.sqrt((A+B+Km)**2-4*A*B))

prod = img.imread(PROJECT_DIR / 'graphics/ngigraphic68.png')

with open(PROJECT_DIR / 'orderAnalysis/calcOrder4.pickle','rb') as f:
   substrateMean,substrateVar,substrateLow,substrateHigh,substrateDrnd,substrateDsis,substrateSim,enzymeMean,enzymeVar,enzymeDrnd,enzymeDsis,enzymeSim,productMean,productVar,productDrnd,productDsis,productSim,orderMean,orderStd = pickle.load(f)

# nCells = 1000
# Tcc = 1000
# PprodA = 10**-1
kcatA = 10**-1
PprodBs = np.logspace(-2,4,31)
Km = 10**3

#%% Figure S21 (Reactant Similarity): Plot 

f = plt.figure(figsize=(16,3.5))
gs = GridSpec(1,4,figure=f,wspace=0.6,hspace=0.4)

f.text(0.001,0.87,'A',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot([0.01,.07,.2,.9])
ax.imshow(prod)
ax.axis('off')

f.text(0.22,0.87,'B',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0:1,1:2])
ax.hlines(1,10**-4,10**4,color='k',linestyle='dashed',zorder=0)
ax.vlines(10**0,-10,10**10,color='k',linestyle='dashed',zorder=0)
ax.scatter(substrateMean/Km,substrateVar/substrateMean,color=signalColor)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Saturation Ratio\n($[B\']_{eq}/K_M$)',fontsize=axisFontSize)
ax.set_xticks(np.logspace(-3,3,4))
ax.set_xticks(returnLogMinorTicks(-4,4),[],minor=1)
ax.set_xlim([0.4*10**-3,2*10**3])
ylabel = ax.set_ylabel('Reactant Fano\nFactor ($\sigma^2_{B\'}/\mu_{B\'}$)',fontsize=axisFontSize,labelpad=-5)
ylabel.set_position((0,.35))
ax.set_ylim([6*10**-1,10**2])
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

f.text(0.5,0.87,'C',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0,2])
ax.vlines(10**0,-10,10**10,color='k',linestyle='dashed',zorder=0)
ax.errorbar(substrateMean/Km,orderMean,orderStd,fmt='.',color=signalColor,zorder=2)
ax.set_xscale('log')
ax.set_xlabel('Saturation Ratio\n($[B\']_{eq}/K_M$)',fontsize=axisFontSize)
ax.set_xticks(np.logspace(-3,3,4))
ax.set_xticks(returnLogMinorTicks(-4,4),[],minor=1)
ax.set_xlim([0.4*10**-3,10*10**3])
ylabel = ax.set_ylabel('Reaction Order',fontsize=axisFontSize,labelpad=0)
ylabel.set_position((0,.35))
ax.set_ylim([-0.05,1.02])
ax.set_yticks(np.linspace(0,1,3))
ax.set_yticks(np.linspace(0,1,11),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)

ax = f.add_subplot([0.69,0.6,0.05,0.3])
ax.hlines(1,10**-4,10**4,color='k',linestyle='dashed',zorder=0)
ax.vlines(10**0,-10,10**10,color='k',linestyle='dashed',zorder=0)
ax.scatter(substrateMean/Km,orderStd,color=signalColor,s=10)
ax.set_xscale('log')
# ax.set_xlabel('Saturation Ratio\n($[B\']_{eq}/K_M$)',fontsize=axisFontSize)
ax.set_xticks(np.logspace(-3,3,2))
ax.set_xticks(np.logspace(-3,3,7),[],minor=1)
ax.set_xlim([0.4*10**-3,2*10**3])
ylabel = ax.set_ylabel('$\sigma_{order}$',fontsize=axisFontSize,labelpad=-20)
ylabel.set_position((0,.5))
ax.set_ylim([-0.002,0.075])
ax.set_yticks(np.linspace(0,0.075,2))
ax.set_yticks(np.linspace(0,0.075,5),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=12)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=12)


f.text(0.73,0.87,'D',fontsize=letterLabelSize,fontname='roboto')
ax = f.add_subplot(gs[0,3])
ax.vlines(10**0,-10,10**10,color='k',linestyle='dashed',zorder=0)
ax.scatter(substrateMean/Km,calcProdRate(substrateMean,enzymeMean,kcatA,Km)/PprodBs[0:27],color=signalColor)
ax.set_xscale('log')
ax.set_xlabel('Saturation Ratio\n($[B\']_{eq}/K_M$)',fontsize=axisFontSize)
ax.set_xticks(np.logspace(-3,3,4))
ax.set_xticks(returnLogMinorTicks(-4,4),[],minor=1)
ax.set_xlim([0.4*10**-3,2*10**3])
ylabel = ax.set_ylabel('Reactant Rate Ratio\n(Consumption:Production)',fontsize=18)
ylabel.set_position((0.3,.35))
ax.set_ylim([-0.02,0.06])
ax.set_yticks(np.linspace(0,.6,3))
ax.set_yticks(np.linspace(0,.6,7),[],minor=1)
ax.spines['left'].set_linewidth(tickWidth)
ax.spines['bottom'].set_linewidth(tickWidth)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
ax.tick_params(axis='both',width=tickWidth,length=tickLength,labelsize=tickFontSize)
ax.tick_params(axis='both',which='minor',length=tickLength/2,width=tickWidth/2,labelsize=tickFontSize)


plt.subplots_adjust(top = 0.95, bottom = 0.34, right = 0.99, left = 0.04)
plt.show()

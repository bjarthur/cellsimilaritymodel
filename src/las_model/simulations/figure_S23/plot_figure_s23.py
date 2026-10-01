import pickle 
from las_model.utils.config import PROJECT_DIR
import matplotlib.pyplot as plt 
import numpy as np 

BURST_COLOR = 'deeppink'

def plot_dvar_grid(vardsis_no_burst,vardrnd_no_burst,normvar_no_burst,vardsis_burst,vardrnd_burst,normvar_burst):

    plt.figure(figsize=(16,8))
    plt.subplot(2,4,1)
    plt.plot(vardsis_no_burst[0],color='r')
    plt.plot(vardrnd_no_burst[0],color='grey')

    plt.subplot(2,4,2)
    plt.plot(vardsis_no_burst[1],color='b')
    plt.plot(vardrnd_no_burst[1],color='grey')

    plt.subplot(2,4,3)
    plt.plot(normvar_no_burst[0],color='r')
    plt.plot(normvar_no_burst[1],color='b')

    plt.subplot(2,4,5)
    plt.plot(vardsis_burst[0],color='r')
    plt.plot(vardrnd_burst[0],color='grey')

    plt.subplot(2,4,6)
    plt.plot(vardsis_burst[1],color='b')
    plt.plot(vardrnd_burst[1],color='grey')

    plt.subplot(2,4,7)
    plt.plot(normvar_burst[0],color='r')
    plt.plot(normvar_burst[1],color='b')

    plt.subplot(2,4,8)
    plt.plot(normvar_no_burst[1], color='k', label='no bursting')
    plt.plot(normvar_burst[1],color='b',label='bursting')
    plt.legend(frameon=0)

    plt.show()

def plot_normvar_comparison(normvar_no_burst,normvar_burst):

    plt.figure(figsize=(8,3))

    plt.subplot(1,2,1)
    plt.hlines(0,0,1000,color='k',linestyle='dashed',zorder=0)
    plt.plot(normvar_no_burst[0],label='No Bursting',color='gray')
    plt.plot(normvar_burst[0],label='Bursting',color=BURST_COLOR)
    plt.legend(frameon=0)
    plt.xlim([0,1000])
    plt.ylim([-0.2,1])
    plt.xticks(np.linspace(0,1000,11),np.linspace(0,10,11).astype(int))
    plt.xlabel('Generations')
    plt.ylabel('LAS A ($\Delta \hat{\sigma}^2_{\Delta [A]}$)')

    plt.subplot(1,2,2)
    plt.hlines(0,0,1000,color='k',linestyle='dashed',zorder=0)
    plt.plot(normvar_no_burst[1],label='No Bursting',color='gray')
    plt.plot(normvar_burst[1],label='Bursting',color=BURST_COLOR)
    plt.legend(frameon=0)
    plt.xlim([0,1000])
    plt.ylim([-0.2,1])
    plt.xticks(np.linspace(0,1000,11),np.linspace(0,10,11).astype(int))
    plt.xlabel('Generations')
    plt.ylabel('LAS B ($\Delta \hat{\sigma}^2_{\Delta [B]}$)')

    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    with open(PROJECT_DIR / 'satprod/satprod_burst_time/satprod_burst_time.pickle','rb') as f:
        results = pickle.load(f)
    vardsis_no_burst, vardrnd_no_burst, normvar_no_burst = (results['no_burst'][k] for k in ('vardsis', 'vardrnd', 'normvar'))
    vardsis_burst, vardrnd_burst, normvar_burst = (results['burst'][k] for k in ('vardsis', 'vardrnd', 'normvar'))
    
    plot_normvar_comparison(normvar_no_burst,normvar_burst)
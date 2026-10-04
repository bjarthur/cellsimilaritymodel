import pickle 
from las_model.utils.config import PROJECT_DIR
import matplotlib.pyplot as plt 
import numpy as np 

def plot_vard_grid(vardsis_inherited,vardrnd_inherited,vardsis_scrambled,vardrnd_scrambled,normvar_inherited,normvar_scrambled):

    plt.figure(figsize=(16,8))
    plt.subplot(2,4,1)
    plt.plot(vardsis_inherited[0],color='r')
    plt.plot(vardrnd_inherited[0],color='grey')

    plt.subplot(2,4,2)
    plt.plot(vardsis_inherited[1],color='b')
    plt.plot(vardrnd_inherited[1],color='grey')

    plt.subplot(2,4,3)
    plt.plot(vardsis_inherited[2],color='purple')
    plt.plot(vardrnd_inherited[2],color='grey')

    plt.subplot(2,4,4)
    plt.plot(normvar_inherited[0],color='r')
    plt.plot(normvar_inherited[1],color='b')
    plt.plot(normvar_inherited[2],color='purple')

    plt.subplot(2,4,5)
    plt.plot(vardsis_scrambled[0],color='r')
    plt.plot(vardrnd_scrambled[0],color='grey')

    plt.subplot(2,4,6)
    plt.plot(vardsis_scrambled[1],color='b')
    plt.plot(vardrnd_scrambled[1],color='grey')

    plt.subplot(2,4,7)
    plt.plot(vardsis_scrambled[2],color='purple')
    plt.plot(vardrnd_scrambled[2],color='grey')

    plt.subplot(2,4,8)
    plt.plot(normvar_scrambled[0],color='r')
    plt.plot(normvar_scrambled[1],color='b')
    plt.plot(normvar_scrambled[2],color='purple')

    plt.show()

def plot_compare_normvar(inerhited_B,scrambled_B,inherited_A_B,scrambled_A_B):
    plt.figure(figsize=(12,3))

    plt.subplot(1,3,1)
    plt.hlines(0,0,1000,color='k',linestyle='dashed',zorder=0)
    plt.plot(inerhited_B[0],label='inherited',color='gray')
    plt.plot(scrambled_B[0],label='scrambled B only',color='r')
    plt.plot(scrambled_A_B[0],label='scrambled A and B',color='r',linestyle='dashed')
    plt.legend(frameon=0)
    plt.xlim([0,1000])
    plt.ylim([-0.2,1])
    plt.xticks(np.linspace(0,1000,11),np.linspace(0,10,11).astype(int))
    plt.xlabel('Generations')
    plt.ylabel('LAS A ($\\Delta \\hat{\\sigma}^2_{\\Delta [A]}$)')

    plt.subplot(1,3,2)
    plt.hlines(0,0,1000,color='k',linestyle='dashed',zorder=0)
    plt.plot(inerhited_B[1],label='inherited',color='gray')
    plt.plot(scrambled_B[1],label='scrambled B only',color='cyan')
    plt.plot(scrambled_A_B[1],label='scrambled A and B',color='cyan',linestyle='dashed')
    plt.legend(frameon=0)
    plt.xlim([0,1000])
    plt.ylim([-0.2,1])
    plt.xticks(np.linspace(0,1000,11),np.linspace(0,10,11).astype(int))
    plt.xlabel('Generations')
    plt.ylabel('LAS B ($\\Delta \\hat{\\sigma}^2_{\\Delta [B]}$)')

    plt.subplot(1,3,3)
    plt.hlines(0,0,1000,color='k',linestyle='dashed',zorder=0)
    plt.plot(inerhited_B[2],label='inherited',color='gray')
    plt.plot(scrambled_B[2],label='scrambled B only',color='purple')
    plt.plot(scrambled_A_B[2],label='scrambled A and B',color='purple',linestyle='dashed')
    plt.legend(frameon=0)
    plt.xlim([0,1000])
    plt.ylim([-0.2,1])
    plt.xticks(np.linspace(0,1000,11),np.linspace(0,10,11).astype(int))
    plt.xlabel('Generations')
    plt.ylabel('LAS C ($\\Delta \\hat{\\sigma}^2_{\\Delta [C]}$)')

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    with open(PROJECT_DIR / 'cascade_scramble/cascade_scramble_B_only/cascade_scramble_B_only.pickle','rb') as f:
        scramble_B = pickle.load(f)

    with open(PROJECT_DIR / 'cascade_scramble/cascade_scramble_A_and_B/cascade_scramble_A_and_B.pickle','rb') as f:
        scramble_A_B = pickle.load(f)

    plot_compare_normvar(scramble_B['inherited']['normvar'],scramble_B['scrambled']['normvar'],
                         scramble_A_B['inherited']['normvar'],scramble_A_B['scrambled']['normvar'])
    # plot_vard_grid(scramble_B['inherited']['vardsis'],scramble_B['inherited']['vardrnd'],
    #                scramble_B['scrambled']['vardsis'],scramble_B['scrambled']['vardrnd'],
    #                scramble_B['inherited']['normvar'],scramble_B['scrambled']['normvar'])

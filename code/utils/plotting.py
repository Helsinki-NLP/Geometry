import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.logger import logger

class Plt:
    
    def makeplot(metrics, outdir):
    '''
    metrics[modname] = {'selfsim':selfsim,
                   'selfsim_isotropic': selfsim - b1,
                   'intrasentsim': insentsim,
                   'intrasentsim_isotropic': insentsim - b2,
                   'mev': mev,
                   'mev_isotropic_V1': mev - b3,
                   'mev_isotropic_V2': mev - b3bis,
                    }
    '''
    
    for modname in metrics:
        logger.info(f'plotting metrics for model {modname}')

        n_layers = metrics[modname]['selfsim'].shape[1]
        
        #selfsim
        per_layer_average = np.mean(metrics[modname]['selfsim'], 0).reshape((n_layers,1))
        fig = plt.plot(1:n_layers, per_layer_average, '-.')
        plt.title(f'{modname} Word Self-Sim [Uncorrected]')
        plt.ylim([0, 1])
        plt.xticks(1:n_layers)
        plt.grid(True)
        plt.savefig(f'{outdir}/{modname}.selfsim.png')    

        #selfsim_isotropic
        per_layer_average = np.mean(metrics[modname]['selfsim_isotropic'], 0).reshape((n_layers,1))
        fig = plt.plot(1:n_layers, per_layer_average, '-.')
        plt.title(f'{modname} Word Self-Sim [Isotropic]')
        plt.ylim([0, 1])
        plt.xticks(1:n_layers)
        plt.grid(True)
        plt.savefig(f'{outdir}/{modname}.selfsim_isotropic.png')    

        #intrasentsim
        per_layer_average = np.mean(metrics[modname]['intrasentsim'], 0).reshape((n_layers,1))
        fig = plt.plot(1:n_layers, per_layer_average, '-.')
        plt.title(f'{modname} Intra-Sent Sim [Uncorrected]')
        plt.ylim([0, 1])
        plt.xticks(1:n_layers)
        plt.grid(True)
        plt.savefig(f'{outdir}/{modname}.intrasentsim.png')

        #intrasentsim_isotropic
        per_layer_average = np.mean(metrics[modname]['intrasentsim_isotropic'], 0).reshape((n_layers,1))
        fig = plt.plot(1:n_layers, per_layer_average, '-.')
        plt.title(f'{modname} Intra-Sent Sim [Isotopic]')
        plt.ylim([0, 1])
        plt.xticks(1:n_layers)
        plt.grid(True)
        plt.savefig(f'{outdir}/{modname}.intrasentsim_isotropic.png')        

        #mev
        per_layer_average = np.mean(metrics[modname]['mev'], 0).reshape((n_layers,1))
        fig = plt.plot(1:n_layers, per_layer_average, '-.')
        plt.title(f'{modname} Max Explainable Var [Uncorrected]')
        plt.ylim([0, 1])
        plt.xticks(1:n_layers)
        plt.grid(True)
        plt.savefig(f'{outdir}/{modname}.mev.png')

        #mev_isotropic_V1
        per_layer_average = np.mean(metrics[modname]['mev_isotropic_V1'], 0).reshape((n_layers,1))
        fig = plt.plot(1:n_layers, per_layer_average, '-.')
        plt.title(f'{modname} Max Explainable Var [Isotropic V1]')
        plt.ylim([0, 1])
        plt.xticks(1:n_layers)
        plt.grid(True)
        plt.savefig(f'{outdir}/{modname}.mev_v1.png')

        #mev_isotropic_V2
        per_layer_average = np.mean(metrics[modname]['mev_isotropic_V2'], 0).reshape((n_layers,1))
        fig = plt.plot(1:n_layers, per_layer_average, '-.')
        plt.title(f'{modname} Max Explainable Var [Isotropic V2]')
        plt.ylim([0, 1])
        plt.xticks(1:n_layers)
        plt.grid(True)
        plt.savefig(f'{outdir}/{modname}.mev_v2.png')

   

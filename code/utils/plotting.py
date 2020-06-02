import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.logger import logger
import argparse


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
        
        fig = plt.figure()

    	#selfsim
        plt.subplot(6, 1, 1)
    	per_layer_average = np.mean(metrics[modname]['selfsim'], 0).reshape((n_layers,1))
    	plt.plot(1:n_layers, per_layer_average, '-.')
    	plt.title(f'{modname} Word Self-Sim [Uncorrected]')
    	plt.ylim([0, 1])
    	plt.xticks(1:n_layers)
    	plt.grid(True)

    	#selfsim_isotropic
        plt.subplot(6, 1, 2)
    	per_layer_average = np.mean(metrics[modname]['selfsim_isotropic'], 0).reshape((n_layers,1))
    	plt.plot(1:n_layers, per_layer_average, '-.')
    	plt.title(f'{modname} Word Self-Sim [Isotropic]')
    	plt.ylim([0, 1])
    	plt.xticks(1:n_layers)
    	plt.grid(True)

    	#intrasentsim
        plt.subplot(6, 1, 3)
    	per_layer_average = np.mean(metrics[modname]['intrasentsim'], 0).reshape((n_layers,1))
    	plt.plot(1:n_layers, per_layer_average, '-.')
    	plt.title(f'{modname} Intra-Sent Sim [Uncorrected]')
    	plt.ylim([0, 1])
    	plt.xticks(1:n_layers)
    	plt.grid(True)

    	#intrasentsim_isotropic
        plt.subplot(6, 1, 4)
    	per_layer_average = np.mean(metrics[modname]['intrasentsim_isotropic'], 0).reshape((n_layers,1))
    	plt.plot(1:n_layers, per_layer_average, '-.')
    	plt.title(f'{modname} Intra-Sent Sim [Isotopic]')
    	plt.ylim([0, 1])
    	plt.xticks(1:n_layers)
    	plt.grid(True)

    	#mev
        plt.subplot(6, 1, 5)
    	per_layer_average = np.mean(metrics[modname]['mev'], 0).reshape((n_layers,1))
    	plt.plot(1:n_layers, per_layer_average, '-.')
    	plt.title(f'{modname} Max Explainable Var [Uncorrected]')
    	plt.ylim([0, 1])
    	plt.xticks(1:n_layers)
    	plt.grid(True)

    	#mev_isotropic_V1
    	#per_layer_average = np.mean(metrics[modname]['mev_isotropic_V1'], 0).reshape((n_layers,1))
    	#fig = plt.plot(1:n_layers, per_layer_average, '-.')
    	#plt.title(f'{modname} Max Explainable Var [Isotropic V1]')
    	#plt.ylim([0, 1])
    	#plt.xticks(1:n_layers)
    	#plt.grid(True)
    	#plt.savefig(f'{outdir}/{modname}.mev_v1.png')

    	#mev_isotropic_V2
        plt.subplot(6, 1, 6)
    	per_layer_average = np.mean(metrics[modname]['mev_isotropic_V2'], 0).reshape((n_layers,1))
    	plt.plot(1:n_layers, per_layer_average, '-.')
    	plt.title(f'{modname} Max Explainable Var [Isotropic]')
    	plt.ylim([0, 1])
    	plt.xticks(1:n_layers)
    	plt.grid(True)
    	
        plt.savefig(f'{outdir}/{modname}.metrics.png')

 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,  nargs='+',
                        help="Path to dictionary with metrics to make plot. This file is outputed from compare_contextEmbeddings.py")

    parser.add_argument("--output", type=str, required=False,  nargs='+', default='../outputs/'
                        help="Path to saving plots directory")
    
    opt = parser.parse_args()
    metrics = {}
    for in_path in opt.input:
        if os.path.isdir(in_path):
            for f in os.listdir(in_path):
                if f.find('similarity.pkl') > -1:
                    thismetrics=pickle.load(open(in_path+'/'+f,'rb'))
                    for key, value in thismetrics.items():
                        metrics[key] = value
        else:
            thismetrics=pickle.load(open(in_path,'rb'))
            for key, value in thismetrics.items():
                metrics[key] = value

    makeplot(metrics,opt.output)
  

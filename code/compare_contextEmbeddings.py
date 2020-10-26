#!/usr/bin/env python3 -u

"""
Compute similarity metrics for contextualized embeddings.
"""
from collections import Counter
from datetime import datetime 
from tqdm import tqdm
from pathlib import Path
import numpy as np

import os 
import ipdb
import re
import functools
import operator
import random 
import pickle
import torch

from utils import opts
from utils.logger import logger
from utils import embeddings_manager as Emb
from utils import similarity_measures as Sim
from utils import plotting as Plt

def  main(opt):
    
    samples = load_samples(opt)
    toks_and_embeddings_paths = Emb.compute_embeddings(opt, samples)
    
    if opt.only_save_embs:
        print('Saved embeddings ... finishing job')
        raise SystemExit 

    logger.info('Loading embeddings & tokenized sentences...') 
    metrics={}
    for modelname, path in toks_and_embeddings_paths.items():
        tokdsents, embs = Emb.load_embeddings(path)
        logger.info(f'[*] MODEL: {modelname} ')
        w2s=make_indexer(opt=opt,sents=tokdsents,modname=modelname)
        logger.info('Computing metrics')

        metrics[modelname] = compute_similarity_metrics(w2s, modelname, embs)
        logger.info('Saving metrics')
        dump_similarity_metrics(opt, metrics)

    if opt.plot_results:
        Plt.makeplot(metrics,opt.outdir)
    else:
        logger.info('finishing ... to make the plots call `utils/ploting.py path/to/savedMetrics.pkl` ')
        logger.info(' or use option --plot_results')
    
def load_samples(opt):
    if len(opt.data_path) == 1:
        logger.info('Loading data samples from '+str(opt.data_path))
        with open(opt.data_path, 'r') as f:
            samples = f.readlines()
        
        samples = [sent.strip() for sent in samples]
        if opt.dev_params:
            samples=samples[:20]
    else:
        src_path, tgt_path = opt.data_path
        logger.info(f'Loading data samples from {src_path} and {tgt_path}')
        with open(src_path, 'r') as f:
            src_samples = f.readlines()
        
        src_samples = [sent.strip() for sent in src_samples]
        if opt.dev_params:
            src_samples=src_samples[:20]

        with open(tgt_path, 'r') as f:
            tgt_samples = f.readlines()
        
        tgt_samples = [sent.strip() for sent in tgt_samples]
        if opt.dev_params:
            tgt_samples = tgt_samples[:20]

        samples = (src_samples, tgt_samples)

    return samples


def make_indexer(opt,sents,modname):
    '''
    index word occurences in the given corpus
    INPUT:
        - opt
        - sents[dict]: list of tokenized sentences. KEYS: 'source' 'target'
    OUTPUT:
        - w2s[class]: class to manage the dictionary that points words to their occurences in the corpus as tuples. 
    '''
    logger.info('   Selecting words that occured at least 5 times')
    allwords=Counter(functools.reduce(operator.iconcat, sents['source'], []))
    bow5x= [key for key,count in allwords.items() if count > 4]
    #clean symbols
    for sym in [',', '.', ':', '?', '!', ';', '_', '-', "'", '"','[CLS]', '[SEP]', '</s>']:
        if sym in bow5x:
            _ = bow5x.pop(bow5x.index(sym))
    
    w2s = Emb.w2s(
            sents=sents['source'], 
            wsample=bow5x
        )
    w2s_2 = None
    if sents.get('target',None):
        logger.info('   Selecting words that occured at least 5 times for the target side')
        allwords=Counter(functools.reduce(operator.iconcat, sents['target'], []))
        bow5x= [key for key,count in allwords.items() if count > 4]
        #clean symbols
        for sym in [',', '.', ':', '?', '!', ';', '_', '-', "'", '"','[CLS]', '[SEP]', '</s>']:
            if sym in bow5x:
                _ = bow5x.pop(bow5x.index(sym))
            
        w2s_2 = Emb.w2s(
                    sents=sents['target'], 
                    wsample=bow5x, 
                    shifted_by=len(sents['source'])
                )        
    
    W2S = {'source': w2s, 'target':w2s_2}
    fname=f'w2s_{modname}' 
    logger.info(f'   Dumping word2sentence indexer at location: {opt.outdir}/{fname}.pkl')
    pickle.dump(W2S,open(f'{opt.outdir}/{fname}.pkl','wb'), protocol=pickle.HIGHEST_PROTOCOL)

    return W2S



def compute_similarity_metrics(w2s, modname, embeddings):
    N_LAYERS, _ , HDIM = embeddings[0].shape
    nsents= len(embeddings) // 2 if w2s['target'] else len(embeddings)
    # BASELINES [FOR ANISOTROPY CORRECTION]
    logger.info('   computing baselines for anisotropy correction ')
    b1, b3 = Sim.get_baselines(embeddings, w2s, N_LAYERS)

    # SELF-SIMILARITY & MAXIMUM EXPLAINABLE VARIANCE
    logger.info('   computing self-similarity and max explainable variance ')
    selfsim = torch.zeros((len(w2s['source'].w2sdict), N_LAYERS))
    mev = torch.zeros((len(w2s['source'].w2sdict), N_LAYERS ))
    for wid, occurrences in tqdm(enumerate(w2s['source'].w2sdict.values())):
        for layer in range(N_LAYERS):
            embs4thisword = torch.zeros((len(occurrences), HDIM))
            for i, idx_tuple in enumerate(occurrences):
                sentence_id, word_id = idx_tuple
                embs4thisword[i,:] = embeddings[sentence_id][layer, word_id,:]

            selfsim[wid, layer] = Sim.self_similarity(embs4thisword).item() #- b1[layer]
            mev[wid, layer] = Sim.max_expl_var(embs4thisword) #- b3[layer]
    
    if w2s['target']:
        N_LAYERS_dec, _ , HDIM = embeddings[-1].shape
        selfsim_dec = torch.zeros((len(w2s['target'].w2sdict), N_LAYERS_dec))
        mev_dec = torch.zeros((len(w2s['target'].w2sdict), N_LAYERS_dec ))
        for wid, occurrences in tqdm(enumerate(w2s['target'].w2sdict.values())):
            for layer in range(N_LAYERS_dec):
                embs4thisword = torch.zeros((len(occurrences), HDIM))
                for i, idx_tuple in enumerate(occurrences):
                    sentence_id, word_id = idx_tuple
                    embs4thisword[i,:] = embeddings[sentence_id][layer, word_id,:]

                selfsim_dec[wid, layer] = Sim.self_similarity(embs4thisword).item() #- b1[layer]
                mev_dec[wid, layer] = Sim.max_expl_var(embs4thisword) #- b3[layer]

    # INTRA-SENTENCE SIMILARITY
    logger.info('   computing intra-sentence similarity ')
    insentsim = torch.zeros((nsents, N_LAYERS)) 
    for layer in range(N_LAYERS):
        logger.info(f'        layer: {layer}')
        for sid in range(nsents): 
            insentsim[sid,layer] = Sim.intra_similarity(embeddings[sid][layer,:,:]) #- b2[layer]

    if w2s['target']:
        insentsim_dec = torch.zeros((nsents, N_LAYERS_dec)) 
        for layer in range(N_LAYERS_dec):
            logger.info(f'        layer: {layer+N_LAYERS}')
            for sid in range(nsents): 
                insentsim_dec[sid,layer] = Sim.intra_similarity(embeddings[nsents+sid][layer,:,:]) #- b2[layer]

    logger.info('    intra sentence similarity finished')

    #b2 = insentsim.mean(dim=0)
    #b3bis = mev.mean(dim=0)# INTERPRETATION 2: baseline_metric3 <- this is the WRONG one
    metricsdict = {'selfsim':selfsim,
                   'selfsim_isotropic': selfsim - b1[:N_LAYERS],
                   'intrasentsim': insentsim,
                   'intrasentsim_isotropic': insentsim - b1[:N_LAYERS],
                   'mev': mev,
                   'mev_isotropic': mev - b3[:N_LAYERS],
                   'baseline1': b1,
                   'baseline3': b3  }
    
    if w2s['target']:
        metricsdict.update(
            selfsim_decoder=selfsim_dec,
            selfsim_decoder_isotropic=selfsim_dec - b1[-N_LAYERS_dec:],
            mev_decoder=mev_dec,
            mev_decoder_isotropic=mev_dec - b3[-N_LAYERS_dec:],
            insentsim_decoder=insentsim_dec,
            insentsim_decoder_isotropic=insentsim_dec - b1[-N_LAYERS_dec:]
        )
    return metricsdict

def dump_similarity_metrics(opt, metrics):
    # DUMP PICKLES
    #if opt.save_results:        

    suffix='_simMetrics.pkl' if not opt.dev_params else '_simMetrics_dev.pkl'
    outfilename = str(opt.outdir)+'/'+'_'.join( metrics.keys() ) + suffix
    logger.info(f'   dumping similarity measures to {outfilename}')
    pickle.dump(metrics, open(outfilename, 'bw'))

    #else:
    #    logger.info('   use --save_results options to save computed metrics before averaging')

def update_opts_to_devmode(opts):
    logger.info('dev mode activated... overriding parameters:')
    print('                                --debug_mode = True,')
    print('                                --use_samples = True,')
    print('                                --intrasentsim_samplesize = 10,')
    print('                                --isotropycorrection_samplesize = 25')
    print('                                --selfsim_samplesize = 1')
    opts.debug_mode = True
    #opts.use_samples=False
    opts.intrasentsim_samplesize = 10
    opts.isotropycorrection_samplesize = 25
    opts.selfsim_samplesize = 3
    #opts.save_results = True
    #opts.huggingface_models = 'bert-base-uncased'


if __name__ == '__main__':
    parser = opts.get_parser()
    opt = parser.parse_args()
    if opt.dev_params:
        update_opts_to_devmode(opt)
    if opt.debug_mode:
        with ipdb.launch_ipdb_on_exception():
            main(opt)
    else:
        main(opt)

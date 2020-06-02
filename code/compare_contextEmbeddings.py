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

def  main(opts):
    
    logger.info('Loading data samples from '+str(opt.data_path))
    with open(opt.data_path, 'r') as f:
        samples = f.readlines()
    
    samples = [sent.strip() for sent in samples]
    if opt.dev_params:
        bad_sent = samples[35619]
        samples=samples[:20]
        samples.append(bad_sent)

    
    all_toks, all_embeddings = Emb.compute_or_load_embeddings(opt, samples)

    
    ssample = sample_sents(opt,samples)
    
    metrics={}
    logger.info('Computing metrics')
    for tokdsents, (modname, embs) in zip(* (all_toks.values(), all_embeddings.items()) ):
        #w2s=make_or_load_w2s(opt,tokdsents)
        logger.info('[*] MODEL: {0} '.format(modname))
        w2s=make_indexer(opt,tokdsents,modname)
        metrics[modname] = compute_similarity_metrics(w2s, ssample, modname, embs)

    logger.info('Saving metrics')
    dump_similarity_metrics(opt, metrics)
    
    if opt.plot_results:
        Plt.makeplot(metrics)
    else:
        logger.info('finishing ... to make a plot like ours call `utils/ploting.py path/to/savedMetrics.pkl` ')
        logger.info(' you can also use option --plot_results')
    

def make_or_load_w2s(opt, sents):
    ''' 
    load indexer if --load_w2s_path given or if it has been created before. 
    If none of the above, then create w2s and save it. 
    '''
    fbasename=os.path.basename(opt.data_path).strip('.txt')
    if opt.load_w2s:
        logger.info(f'loading w2s indexer from {opt.load_w2s}') 
        w2s =  pickle.load(open(opt.load_w2s,'rb')) 
    elif Path(f'../embeddings/{fbasename}_w2s.pkl').is_file():
        logger.info(f'loading w2s indexer from ../embeddings/{fbasename}_w2s.pkl') 
        w2s =  pickle.load(open(f'../embeddings/{fbasename}_w2s.pkl','rb')) 
    else:
        w2s = make_indexer(opt,sents) 
    
    return w2s

def make_indexer(opt,sents,modname):
    '''
    index word occurences in the given corpus
    INPUT:
        - opt
        - sents[list]: list of tokenized sentences
    OUTPUT:
        - w2s[class]: class to manage the dictionary that points words to their occurences in the corpus as tuples. 
    '''
    logger.info('   Selecting words that occured at least 5 times')
    allwords=Counter(functools.reduce(operator.iconcat, sents, []))
    bow5x= [key for key,count in allwords.items() if count > 4]
    #clean symbols
    for sym in [',', '.', ':', '?', '!', ';', '_', '-', "'", '"','[CLS]', '[SEP]']:
        if sym in bow5x:
            _ = bow5x.pop(bow5x.index(sym))

    logger.info('   Selecting words that occured less than 1001 times') # Kawin did this, i think
    bow1k= set([key for key,count in allwords.items() if count <= 1000])
    bow5x = set(bow5x).intersection(bow1k)
    '''
    # sample words for self-similarity
    logger.info('Generating w2s dictionary')
    if opt.use_samples:
        logger.info("Sampling {0} words to compute self-similarity ... ".format(opt.selfsim_samplesize))
        wsample = random.sample(bow5x,opt.selfsim_samplesize)
        w2s = Emb.w2s(sents,wsample)


        #UPDATE w2s sentence index, to match the smaller subset of sentences
        logger.info('Updating  dictionary indexes')
        w2s.update_indexes(sents)
    else:
        w2s = Emb.w2s(sents,bow5x)
    '''
    w2s = Emb.w2s(sents,bow5x)
    fbasename=os.path.basename(opt.data_path).strip('.txt')
    fname=f'w2s_{fbasename}_{modname}' 
    logger.info(f'   Dumping word2sentence indexer at location: ../outputs/{fname}.pkl')
    pickle.dump(w2s,open(f'../outputs/{fname}.pkl','wb'))

    return w2s

def sample_sents(opt,sents):
    '''
    sample sentences if required in opt
    INPUT:
        - opt 
        - sents
    OUTPUT:
        - ssample[tuple]:  (sampled indexes, sampled sentences)
    '''
    if opt.use_samples:
        logger.info("Sampling {0} sentences to compute intra-sentence similarity ... ".format(opt.intrasentsim_samplesize))
        # sample sentences for intra sent similarity
        ssample_idx = [random.randint(0,len(sents)-1) for i in range(opt.intrasentsim_samplesize)]
        ssample = [sents[idx] for idx in ssample_idx]
    else:
        ssample_idx = [i for i in range(len(sents))]
        ssample = sents
    return (ssample_idx, ssample)


def compute_similarity_metrics(w2s,ssample,modname, embeddings):
    N_LAYERS, _ , HDIM = embeddings[0].shape

    # BASELINES [FOR ANISOTROPY CORRECTION]
    logger.info('   computing baselines for anisotropy correction ')
    b1, b3 = Sim.get_baselines(embeddings, w2s, N_LAYERS)

    # SELF-SIMILARITY & MAXIMUM EXPLAINABLE VARIANCE
    logger.info('   self-similarity and max explainable variance ')
    selfsim = torch.zeros((len(w2s.w2sdict), N_LAYERS ))
    mev = torch.zeros((len(w2s.w2sdict), N_LAYERS ))
    for wid, occurrences in tqdm(enumerate(w2s.w2sdict.values())):
        for layer in range(N_LAYERS):
            embs4thisword = torch.zeros((len(occurrences), HDIM))
            for i, idx_tuple in enumerate(occurrences):
                sentence_id = idx_tuple[0]
                word_id = idx_tuple[1]
                embs4thisword[i,:] = embeddings[sentence_id][layer, word_id,:]

            selfsim[wid, layer] = Sim.self_similarity(embs4thisword).item() #- b1[layer]
            mev[wid, layer] = Sim.max_expl_var(embs4thisword).item() #- b3[layer]
    

    # INTRA-SENTENCE SIMILARITY
    logger.info('   intra-sentence similarity ')
    insentsim = torch.zeros((len(ssample[0]), N_LAYERS))
    for layer in range(N_LAYERS):
        for sid, sentence in zip(*ssample):
            insentsim[sid,layer] = Sim.intra_similarity(embeddings[sid][layer]) #- b2[layer]
    
    b2 = insentsim.mean(dim=0)
    b3bis = mev.mean(dim=0)# INTERPRETATION 2: baseline_metric3


    metricsdict = {'selfsim':selfsim,
                   'selfsim_isotropic': selfsim - b1,
                   'intrasentsim': insentsim,
                   'intrasentsim_isotropic': insentsim - b2,
                   'mev': mev,
                   'mev_isotropic_V1': mev - b3,
                   'mev_isotropic_V2': mev - b3bis,
                    }
    return metricsdict

def dump_similarity_metrics(opt, metrics):
    # DUMP PICKLES
    if opt.save_results:        

        suffix='similarity.pkl' if not opt.dev_params else 'similarity_dev.pkl'
        outfilename=prefix = str(opt.outdir)+'/'+'_'.join( metrics.keys() ) + suffix
        logger.info(f'   dumping similarity measures to {outfilename}')
        pickle.dump(metrics, open(outfilename, 'bw'))

    else:
        logger.info('   use --save_results options to save computed metrics before averaging')

    
 

def update_opts_to_devmode(opts):
    logger.info('dev mode activated... overriding parameters:')
    print('                                --debug_mode = True,')
    print('                                --use_samples = True,')
    print('                                --intrasentsim_samplesize = 10,')
    print('                                --isotropycorrection_samplesize = 25')
    print('                                --selfsim_samplesize = 1')
    #print('                          huggingface_models = bert-base-uncased')
    opts.debug_mode = True
    opts.use_samples=False
    opts.intrasentsim_samplesize = 10
    opts.isotropycorrection_samplesize = 25
    opts.selfsim_samplesize = 3
    opts.save_results = True
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

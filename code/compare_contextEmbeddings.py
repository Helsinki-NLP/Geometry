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

def  main(opts):
    
    logger.info('Loading data samples from '+str(opt.data_path))
    STS_path = opt.data_path
    with open(STS_path, 'r') as f:
        samples = f.readlines()

    sents = []
    for sent in samples:
        sent = sent.strip()
        sent = re.findall(r'[\w]+|\.|,|\?|\!|;|:|\'|\(|\)|/',sent) # tokenization is a model_loader class attribute. Need to undo this there.
        sents.append(sent)

    w2s=make_or_load_w2s(opt,sents)
        
    all_embeddings = Emb.compute_or_load_embeddings(opt,w2s)
    ssample = sample_sents(opt,sents)
    metrics={}
    for modname, embs in all_embeddings.items():
        metrics[modname] = compute_similarity_metrics(w2s,ssample,modname, embs)




    '''

    #load & compute embeddings from MT model
    logger.info('Running sentences containing sampled words through the mt-encoder')
    all_mt_embedds = get_encodings_from_onmt_model(w2s.new_sents,opt) # get MT-embedds from set of sentences with the words needed
    bpedsents = all_mt_embedds.pop('sentences')

    # merge embeddings from splitted words (bpe)
    if opt.bpe_model:
        all_mt_embedds, w2s, _ = merge_bped_embedds(bpedsents, all_mt_embedds, w2s)

    # get embeddings of the word positions
    print('Extracting word embeddings')
    mt_embedds = extract_embeddings(w2s,all_mt_embedds)

    # SELF-SIMILARITY
    self_similarities = {}
    for word, embedds in mt_embedds.items():
        if embedds['layer0']['normalized'].shape[0] > 1:
            for layer,embs_type in embedds.items():
                #print('entered using:', word,layer)
                self_similarities.setdefault(word,{}).setdefault(layer,{})
                self_similarities[word][layer] = self_similarity(word, embs_type)

    print('Computing self-similarities...')
    w2s_4_selfsim = {w:pos for w,pos in w2s.items() if len(pos)>1}
    self_similarities = {}
    numlayers = len(mt_embedds[list(w2s_4_selfsim.keys())[0]])
    tensor_selfsimilarities = torch.zeros((len(w2s_4_selfsim),numlayers,2)) # [word_index, num_layers, |{normalized,unnormalized}| ]

    for wordid, word in enumerate(w2s_4_selfsim):
        print('computing self similarity for ', word)
        for layerid, (layer,embs_type) in enumerate(mt_embedds[word].items()):
            tensor_selfsimilarities[wordid,layerid,:] = self_similarity(word, embs_type)


    '''

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

def make_indexer(opt,sents):
    '''
    index word occurences in the given corpus
    INPUT:
        - opt
        - sents[list]: list of tokenized sentences
    OUTPUT:
        - w2s[class]: class to manage the dictionary that points words to their occurences in the corpus as tuples. 
    '''
    logger.info('Selecting words that occured at least 5 times')
    allwords=Counter(functools.reduce(operator.iconcat, sents, []))
    bow5x= [key for key,count in allwords.items() if count > 4]
    #clean symbols
    for sym in [',', '.', ':','?','!',';','_']:
        _ = bow5x.pop(bow5x.index(sym))

    logger.info('Selecting words that occured less than 1001 times') # Kawin did this, i think
    bow1k= set([key for key,count in allwords.items() if count <= 1000])
    bow5x = set(bow5x).intersection(bow1k)

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
        w2s.update(sents)
    fbasename=os.path.basename(opt.data_path).strip('.txt')
    fname=f'{fbasename}_w2s' 
    logger.info(f'Dumping word2sentence indexer at location: ../embeddings/{fname}.pkl')
    pickle.dump(w2s,open(f'../embeddings/{fname}.pkl','wb'))

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
    logger.info('   self-similarity and max explainable variance ')
    N_LAYERS, _ , HDIM = embeddings[0].shape
    # SELF-SIMILARITY & MAXIMUM EXPLAINABLE VARIANCE
    selfsim = np.zeros((len(w2s.w2sdict), N_LAYERS ))
    mev = np.zeros((len(w2s.w2sdict), N_LAYERS ))
    for wid, occurrences in enumerate(w2s.w2sdict.values()):
        for layer in range(N_LAYERS):
            embs4thisword = torch.zeros((len(occurrences), HDIM))
            for i, idx_tuple in enumerate(occurrences):
                sentence_id = idx_tuple[0]
                word_id = idx_tuple[1]
                embs4thisword[i,:] = embeddings[sentence_id][layer, word_id,:]

            selfsim[wid, layer] = Sim.self_similarity(embs4thisword).item()
            mev[wid, layer] = Sim.max_expl_var(embs4thisword)


    # INTRA-SENTENCE SIMILARITY
    logger.info('   intra-sentence similarity ')
    intrasentsim = np.zeros((len(ssample[0]), N_LAYERS))
    for layer in range(N_LAYERS):
        for sid, sentence in zip(*ssample):
            intrasentsim[sid,layer] = Sim.intra_similarity(embeddings[sid][layer])

    
    return selfsim, intrasentsim, mev

def dump_similarity_metrics(opt, metrics):
    # DUMP PICKLES
    if opt.save_results:
        logger.info('   dumping similarity measures to '+str(opt.outdir) )
        
        outfilename=str(opt.outdir)+'similarity.pkl' if not opt.dev_params else str(opt.outdir)+'similarity_dev.pkl'
        pickle.dump(metrics, open(outfilename, 'bw'))

    else:
        logger.info('   use --save_results options to save results')

    
 

def update_opts_to_devmode(opts):
    logger.info('dev mode activated... overriding parameters:')
    print('                                --debug_mode = True,')
    print('                                --use_samples = True,')
    print(f'                                --data_path = {opt.data_path.strip(".txt")}_dev.txt,')
    print('                                --intrasentsim_samplesize = 10,')
    print('                                --isotropycorrection_samplesize = 25')
    print('                                --selfsim_samplesize = 1')
    #print('                          huggingface_models = bert-base-uncased')
    opts.debug_mode = True
    opts.use_samples=True
    opt.data_path=f'..{opt.data_path.strip(".txt")}_dev.txt'
    opts.intrasentsim_samplesize = 10
    opts.isotropycorrection_samplesize = 25
    opts.selfsim_samplesize = 3
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

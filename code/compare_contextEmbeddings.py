#!/usr/bin/env python3 -u

"""
Compute similarity metrics for contextualized embeddings.
"""
from collections import Counter
from datetime import datetime 
from tqdm import tqdm

import numpy as np

import ipdb
import re
import functools
import operator
import random 
import pickle
import torch

from utils import opts
from utils import embeddings_manager as Emb
from utils import model_loaders as Loader
from utils import similarity_measures as Sim

def  main(opts):
    if opt.dev_params:
        update_opts_to_devmode(opts)

    logger('Loading data samples from '+str(opt.data_path))
    STS_path = opt.data_path
    with open(STS_path, 'r') as f:
        samples = f.readlines()

    sents = []
    for sent in samples:
        sent = sent.strip()
        sent = re.findall(r'[\w]+|\.|,|\?|\!|;|:|\'|\(|\)|/',sent) # tokenization is a model_loader class attribute. Need to undo this there.
        sents.append(sent)

    
    w2s, ssample = create_samples(opt,sents)

    #load & compute embeddings using huggingface 
    logger('HUGGINGFACE models.')
    if isinstance(opt.huggingface_models,list):
        for hfmodel in opt.huggingface_models:
            logger('   {0} embeddings: computing similarity metrics'.format(hfmodel))
            hf_selfsim, hf_instrasentsim, hf_mev = huggingface_compute_metrics(w2s,ssample,opt,hf_model)
    else:
        logger('   {0} embeddings: computing similarity metrics'.format(opt.huggingface_models))
        hf_selfsim, hf_instrasentsim, hf_mev = huggingface_compute_metrics(w2s,ssample,opt, opt.huggingface_models)

    #load & compute embeddings using BERT
    logger('BERT embeddings: computing similarity metrics')
    bert_selfsim, bert_instrasentsim, bert_mev = BERT_compute_metrics(w2s,ssample,opt)
    
    #load & compute embeddings using huggingface
    logger('HUGGINGFACE models.')
    for hfmodel in opt.huggingface_model:
        logger('   {0} embeddings: computing similarity metrics'.format(hfmodel))
        hf_selfsim, hf_instrasentsim, hf_mev = huggingface_compute_metrics(w2s,ssample,opt,hf_model)

    #load & compute embeddings from MT model
    logger('Running sentences containing sampled words through the mt-encoder')
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



def update_opts_to_devmode(opts):
    logger('dev mode activated... overriding parameters:')
    print('                          intrasentsim_samplesize = 10,')
    print('                          isotropycorrection_samplesize = 25')
    print('                          selfsim_samplesize = 1')
    #print('                          huggingface_models = bert-base-uncased')
    opts.debug_mode = True
    opts.intrasentsim_samplesize = 10
    opts.isotropycorrection_samplesize = 25
    opts.selfsim_samplesize = 1
    #opts.huggingface_models = 'bert-base-uncased'

def create_samples(opt,sents):
    logger('Selecting words that occured at least 5 times')
    allwords=Counter(functools.reduce(operator.iconcat, sents, []))
    bow5x= [key for key,count in allwords.items() if count > 4]
    #clean symbols
    for sym in [',', '.', ':','?','!',';','_']:
        _ = bow5x.pop(bow5x.index(sym))

    logger('Selecting words that occured less than 1001 times') # Kawin did this, i think
    bow1k= set([key for key,count in allwords.items() if count <= 1000])
    bow5x = set(bow5x).intersection(bow1k)

    logger("Sampling {0} words to compute self-similarity ... ".format(opt.selfsim_samplesize))
     # sample words for self-similarity
    wsample = random.sample(bow5x,opt.selfsim_samplesize)
    
    logger('Generating w2s dictionary')
    w2s = Emb.w2s(sents,wsample)
    #UPDATE w2s sentence index, to match the smaller subset of sentences
    logger('Updating  dictionary indexes')
    w2s.update_indexes(sents)

    logger("Sampling {0} sentences to compute self-similarity ... ".format(opt.intrasentsim_samplesize))
    # sample sentences for intra sent similarity
    ssample_idx = [random.randint(0,len(sents)-1) for i in range(opt.intrasentsim_samplesize)]
    ssample = [sents[idx] for idx in ssample_idx]
    
    return w2s, ssample

def BERT_compute_metrics(w2s,ssample,opt):

    # load model + tokenizer
    model = Loader.bertModel(opt.bert_model, opt.cuda)

    # SELF-SIMILARITY OF WORDS
    logger('   self-similarity and max explainable variance ')
    #compute BERT embeddings
    bert_tokens, bert_tokenization =  model.tokenize(w2s.new_sents)
    bert_encodings = model.encode(bert_tokens)
    bert_encodings = model.correct_bert_tokenization(bert_encodings, bert_tokenization)

    # get selfsim
    self_similarities = {}
    mev={}
    to_pickle = np.zeros((model.N_BERT_LAYERS, opt.selfsim_samplesize))
    wid = 0
    for word,occurrences in w2s.w2sdict.items():
        for layer in range(model.N_BERT_LAYERS):
            embs4thisword = torch.zeros((len(occurrences), model.ENC_DIM))
            #encodings = []
            for i, idx_tuple in enumerate(occurrences):
                sentence_id = idx_tuple[0]
                word_id = idx_tuple[1]
                embs4thisword[i,:] = bert_encodings[sentence_id][layer][0,word_id,:]
                #encodings.append(bert_encodings[sentence_id][layer][0,word_id,:])

            self_similarities.setdefault(word,{}).setdefault(layer,{})
            self_similarities[word][layer] = Sim.self_similarity(embs4thisword).item()
            mev.setdefault(word,{}).setdefault(layer,{})
            mev[word][layer] = Sim.max_expl_var(embs4thisword)
            to_pickle[layer,wid] = self_similarities[word][layer]
            #print(word, layer, self_similarities[word][layer])
        wid += 1

    # INTRA-SENTENCE SIMILARITY
    logger('   intra-sentence similarity ')
    #compute BERT embeddings
    ssample_bert_tokens, ssample_bert_tokenization =  model.tokenize(ssample)
    ssample_bert_encodings = model.encode(ssample_bert_tokens)
    ssample_bert_encodings = model.correct_bert_tokenization(ssample_bert_encodings, ssample_bert_tokenization)
    
    # get intrasentsim
    intrasentsim = torch.zeros((opt.intrasentsim_samplesize, model.N_BERT_LAYERS))
    for lid, layer in enumerate(range(model.N_BERT_LAYERS)):
        for sid, sentence in enumerate(ssample):
            intrasentsim[sid,lid] = Sim.intra_similarity(ssample_bert_encodings[sid][lid][0])


    # DUMP PICKLES
    if opt.save_results:
        logger('   dumping similarity measures to '+str(opt.outdir) )
        to_pickle = to_pickle[:, 0:wid]
        selfsimfname=str(opt.outdir)+'BERTselfsim_samplesize'+str(opt.selfsim_samplesize)+'.pkl'
        pickle.dump(to_pickle, open(selfsimfname, 'bw'))
        
        intrasimfname=str(opt.outdir)+'BERTintrasentsim_samplesize'+str(opt.intrasentsim_samplesize)+'.pkl'
        pickle.dump(tensor_intrasentsim.numpy(), open(intrasimfname, 'bw'))

        wordsfname=str(opt.outdir)+'sampledwords4selfsim_samplesize'+str(opt.selfsim_samplesize)+'.pkl'
        pickle.dump(wsample,open(wordsfname,'wb'))
    else:
        logger('   use --save_results options to save results')

    return self_similarities, intrasentsim, mev
 
def huggingface_compute_metrics(w2s,ssample,opt,hf_model):
    model = Loader.huggingfaceModel(hf_model, opt.cuda)

    logger('   self-similarity and max explainable variance ')
    #compute BERT embeddings
    hf_tokens, hf_tokenization =  model.tokenize(w2s.new_sents)
    hf_encodings = model.encode(hf_tokens)
    hf_encodings = model.correct_bert_tokenization(hf_encodings, hf_tokenization)

    # get selfsim
    self_similarities = {}
    mev={}
    to_pickle = np.zeros((model.N_BERT_LAYERS, opt.selfsim_samplesize))
    wid = 0
    for word,occurrences in w2s.w2sdict.items():
        for layer in range(model.N_BERT_LAYERS):
            embs4thisword = torch.zeros((len(occurrences), model.ENC_DIM))
            #encodings = []
            for i, idx_tuple in enumerate(occurrences):
                sentence_id = idx_tuple[0]
                word_id = idx_tuple[1]
                embs4thisword[i,:] = bert_encodings[sentence_id][layer][0,word_id,:]
                #encodings.append(bert_encodings[sentence_id][layer][0,word_id,:])

            self_similarities.setdefault(word,{}).setdefault(layer,{})
            self_similarities[word][layer] = Sim.self_similarity(embs4thisword).item()
            mev.setdefault(word,{}).setdefault(layer,{})
            mev[word][layer] = Sim.max_expl_var(embs4thisword)
            to_pickle[layer,wid] = self_similarities[word][layer]
            #print(word, layer, self_similarities[word][layer])
        wid += 1

    # INTRA-SENTENCE SIMILARITY
    logger('   intra-sentence similarity ')
    #compute BERT embeddings
    ssample_bert_tokens, ssample_bert_tokenization =  model.tokenize(ssample)
    ssample_bert_encodings = model.encode(ssample_bert_tokens)
    ssample_bert_encodings = model.correct_bert_tokenization(ssample_bert_encodings, ssample_bert_tokenization)
    
    # get intrasentsim
    intrasentsim = torch.zeros((opt.intrasentsim_samplesize, model.N_BERT_LAYERS))
    for lid, layer in enumerate(range(model.N_BERT_LAYERS)):
        for sid, sentence in enumerate(ssample):
            intrasentsim[sid,lid] = Sim.intra_similarity(ssample_bert_encodings[sid][lid][0])


    # DUMP PICKLES
    if opt.save_results:
        logger('   dumping similarity measures to '+str(opt.outdir) )
        to_pickle = to_pickle[:, 0:wid]
        selfsimfname=str(opt.outdir)+'BERTselfsim_samplesize'+str(opt.selfsim_samplesize)+'.pkl'
        pickle.dump(to_pickle, open(selfsimfname, 'bw'))
        
        intrasimfname=str(opt.outdir)+'BERTintrasentsim_samplesize'+str(opt.intrasentsim_samplesize)+'.pkl'
        pickle.dump(tensor_intrasentsim.numpy(), open(intrasimfname, 'bw'))

        wordsfname=str(opt.outdir)+'sampledwords4selfsim_samplesize'+str(opt.selfsim_samplesize)+'.pkl'
        pickle.dump(wsample,open(wordsfname,'wb'))
    else:
        logger('   use --save_results options to save results')

    return self_similarities, intrasentsim, mev

def logger(string):
    print(' | ',datetime.now().replace(microsecond=0), '|   '+string)



if __name__ == '__main__':
    parser = opts.get_parser()
    opt = parser.parse_args()
    if opt.debug_mode:
        with ipdb.launch_ipdb_on_exception():
            main(opt)
    else:
        main(opt)

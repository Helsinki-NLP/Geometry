''' 
Contains the functions to compute similarity measures as described in Ethayarajh(2019)
"How contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings"
'''

import torch
import random
import numpy as np
from tqdm import tqdm
from utils.logger import logger
from sklearn.decomposition import TruncatedSVD, PCA


def self_similarity(word_embs):
    '''
    compute self-similarity as described in Ethayarajh(2019)
    INPUT: 
        - word_embs[Tendsor]: embedding for current word. Tensor of size [word_occurrences, hdim_embs]
    
    OUTPUT:
        - current_selfsim[tensor]: computed self-similarity for inputed word_embeddings
    '''
    n = len(word_embs)

    current_selfsim = 0
    for i in range(n):
        for j in range(i+1,n):
            current_selfsim += torch.cosine_similarity(word_embs[i],word_embs[j],dim=0)

    coeff = 2/(n**2 - n)
    current_selfsim *= coeff
    curr_var = None
    
    return current_selfsim

def intra_similarity(sent_embs):
    '''
    compute intra-sentence similarity as described in Ethayarajh(2019)
    INPUT: 
        - sent_embs[tensor]: sentence embedding. Tensor of size [sent_len, hdim_embs]. 
    
    OUTPUT:
        - current_selfsim[tensor]: computed self-similarity for inputed word_embeddings
    '''
    sent_len = len(sent_embs)
    mean_emb = torch.mean(sent_embs, dim=0)
    current_intrasim = 0
    for i in range(sent_len):
        current_intrasim += torch.cosine_similarity(mean_emb, sent_embs[i, :], dim=0)
    current_intrasim *= 1 / sent_len
    return current_intrasim

def max_expl_var(word_embs):
    '''
    compute maximum explainable varianve as described in Ethayarajh(2019)
    INPUT: 
        - word_embs[Tendsor]: embedding for current word. Tensor of size [word_occurrences, hdim_embs]
    
    OUTPUT:
        - current_selfsim[tensor]: computed self-similarity for inputed word_embeddings
    '''
    pca = PCA(n_components=1)
    pca.fit(word_embs)

    #pca_svd = TruncatedSVD(n_components=100)
    #pca_svd.fit(word_embs)
    
    variance_explained = min(1.0, round(pca.explained_variance_ratio_[0], 4))
 
    return variance_explained


def sample_gen(n, forbid):
    state = dict()
    track = dict()
    for (i, o) in enumerate(forbid):
        x = track.get(o, o)
        t = state.get(n-i-1, n-i-1)
        state[x] = t
        track[t] = x
        state.pop(n-i-1, None)
        track.pop(o, None)
    del track
    
    for remaining in range(n-len(forbid), 0, -1):
        i = random.randrange(remaining)
        yield state.get(i, i)
        state[i] = state.get(remaining - 1, remaining - 1)
        state.pop(remaining - 1, None)

def get_baselines(embeddings, w2s, nlayers):
    
    baselines_metric1 = torch.zeros((nlayers,))
    #baselines_metric2 = torch.zeros((nlayers,)) #<- computed later for efficiency purposes
    baselines_metric3 = torch.zeros((nlayers,))

    '''
    all_embs = []
    for i in range(len(embeddings)):
        all_embs.append(embeddings[i][:,w2s.s2idxdict[i],:])
    all_embs = torch.cat(all_embs, dim=1)
    '''
    all_embs = torch.cat([embeddings[i][:,w2s.s2idxdict[i],:] for i in range(len(embeddings))], dim=1)
    
    # hyperparams for estimator computation
    nembs=all_embs.shape[1]
    bsz=max(2,round(nembs*0.001)) # bsz = 0.1% of the data
    n=(bsz**2-bsz)/2       # num of elements from which we average for computing the cosine-similarity on each batch

    trsh = 0.001  # stopping treshhold 
    patience = 10 

    for layer in tqdm(range(nlayers)):
        logger.info(f'     LAYER {layer}')
        # init params for sampling on this layer
        means = []  
        gen = sample_gen(nembs,[])
        patience_count = 0
        for i in np.arange(nembs/bsz-1): 
            batch = [next(gen) for j in range(bsz)]
            means.append( self_similarity(all_embs[layer,batch,:]).item() )
            # the variance of the means is the sample variance (see "The Method of Batch Means")
            sample_var = np.nan if len(means)==1 else np.var(means, ddof=1)

            logger.info(f'      mean estim.: {round(means[-1],5)}, var: {sample_var}, patience counter: {patience - patience_count}')
            # break if is has converged
            if sample_var<trsh:
                patience_count += 1 # allows to accum at least 10 means
                if patience_count > patience:
                    break
            else:
                patience_count = 0
        # the mean of the means, once it stops varying too much (see "The Method of Batch Means")
        baselines_metric1[layer] = np.mean(means) # not open to interpretation, given in the paper
        #baselines_metric1[layer] = self_similarity(all_embs[layer,:,:]).item() # not open to interpretation, given in the paper
        baselines_metric3[layer] = max_expl_var(all_embs[layer,:,:]) # INTERPRETATION 1: baseline_metric3
        '''
        temp = 0
        for i in range(len(embeddings)):        
            sent_embs = embeddings[i]#[:,w2s.s2idxdict[i],:]
            temp += intra_similarity( sent_embs[layer,:,:] ) 
        baselines_metric2[layer] = temp/len(embeddings) # INTERPRETATION 1: baseline_metric2
        '''

    return baselines_metric1, baselines_metric3
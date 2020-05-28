''' 
Contains the functions to compute similarity measures as described in Ethayarajh(2019)
"How contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings"
'''

import torch
#import numpy as np
from tqdm import tqdm
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

def get_baselines(embeddings, w2s, nlayers):
    
    baselines_metric1 = torch.zeros((nlayers,))
    #baselines_metric2 = torch.zeros((nlayers,)) #<- computed later for efficiency purposes
    baselines_metric3 = torch.zeros((nlayers,))
    '''
    for layer in range(nlayers):
        all_embs_list = []
        for wid, occurrences in enumerate(w2s.w2sdict.values()):
            for occ in occurrences:
                sentence_id, word_id = occ
                all_embs_list.append(embeddings[sentence_id][layer,word_id,:])

        all_embs = torch.stack(all_embs_list, dim=0)
        baselines_metric1[layer] = Sim.self_similarity(all_embs).item()
        baselines_metric3[layer] = Sim.max_expl_var(all_embs).item() #FIXME: Is this correct formula?
    '''

    
    all_embs = torch.cat([embeddings[i][:,w2s.s2idxdict[i],:] for i in range(len(embeddings))], dim=1)
    for layer in range(nlayers):
        baselines_metric1[layer] = self_similarity(all_embs[layer,:,:]).item() # not open to interpretation, given in the paper
        baselines_metric3[layer] = max_expl_var(all_embs[layer,:,:]).item() # INTERPRETATION 1: baseline_metric3
        '''
        temp = 0
        for i in range(len(embeddings)):        
            sent_embs = embeddings[i]#[:,w2s.s2idxdict[i],:]
            temp += intra_similarity( sent_embs[layer,:,:] ) 
        baselines_metric2[layer] = temp/len(embeddings) # INTERPRETATION 1: baseline_metric2
        '''

    return baselines_metric1, baselines_metric3
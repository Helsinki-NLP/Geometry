''' 
Contains the functions to compute similarity measures as described in Ethayarajh(2019)
"How contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings"
'''

import torch
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
                           Relevant for transformer architectures.
    
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

    pca_svd = TruncatedSVD(n_components=100)
    pca_svd.fit(word_embs)

    variance_explained = min(1.0, round(pca.explained_variance_ratio_[0], 4))

    return variance_explained


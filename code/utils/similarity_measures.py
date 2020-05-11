''' 
Contains the functions to compute similarity measures as described in Ethayarajh(2019)
"How contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings"
'''

import torch

def self_similarity(word,embs_type):
    '''
    compute self-similarity as described in Ethayarajh(2019)
    INPUT: 
        - word[str]: token from which the embeddings were produced
        - embs_type[dict]: Dictionary containing 'normalized' and/or 'unnormalized' embeddings as Tensors. 
                           Relevant for transformer architectures.
    
    OUTPUT:
        - current_selfsim[dict]: computed self-similarity for inputed word and the given 'normalized' or 'unnormalized' embeddings
    '''
    n, hdim = embs_type['normalized'].shape
    current_selfsim = torch.Tensor([0,0])
    #import ipdb; ipdb.set_trace()
    # fill matrix of cosine_similarities
    for i in range(n):
        for j in range(i+1,n):
            current_selfsim[0] += torch.cosine_similarity(embs_type['normalized'][i],  embs_type['normalized'][j],  dim=0) # ['normalized']
            current_selfsim[1] += torch.cosine_similarity(embs_type['unnormalized'][i],embs_type['unnormalized'][j],dim=0) # ['unnormalized']
    #for key, embs in embs_type.items():
    #    for i in range(n):
    #        for j in range(i+1,n):
    #           current_selfsim[key] += torch.cosine_similarity(embs[i],embs[j],dim=0)
            #print(word, i,j,current_selfsim)
    coeff = 2/(n**2 - n)
    current_selfsim[0] *= coeff # ['normalized']
    current_selfsim[1] *= coeff # ['unnormalized']
    return current_selfsim

def intra_similarity(sent_embs):
    sent_len = len(sent_embs)
    mean_emb = torch.mean(sent_embs, dim=0)
    current_intrasim = 0
    for i in range(sent_len):
        current_intrasim += torch.cosine_similarity(mean_emb, sent_embs[i, :], dim=0)
    current_intrasim *= 1 / sent_len
    return current_intrasim

def max_expl_var():
    print('Not implemented yet ... passing')
    pass


#!/usr/bin/env python3 -u

"""
Compute similarity metrics for contextualized embeddings.
"""
from collections import Counter
import ipdb
import re
import functools
import operator
import random 
from datetime import datetime 

from utils import opts
from utils import embeddings_manager as Emb

def  main(opts):
    print(datetime.now().time(), '|   Loading data samples from ', opt.data_path)
    STS_path = opt.data_path
    with open(STS_path, 'r') as f:
        samples = f.readlines()

    sents = []
    for sent in samples:
        sent = sent.strip()
        sent = re.findall(r'[\w]+|\.|,|\?|\!|;|:|\'|\(|\)|/',sent)
        sents.append(sent)

    print(datetime.now().time(), '|   Selecting words that occured at least 5 times')
    allwords=Counter(functools.reduce(operator.iconcat, sents, []))
    bow5x= [key for key,count in allwords.items() if count > 4]
    #clean symbols
    for sym in [',', '.', ':','?','!',';','_']:
        _ = bow5x.pop(bow5x.index(sym))

    print(datetime.now().time(), '|   Selecting words that occured less than 1001 times') # Kawin did this, i think
    bow1k= set([key for key,count in allwords.items() if count <= 1000])
    bow5x = set(bow5x).intersection(bow1k)

    print(datetime.now().time(), "|   Sampling {0} words to compute self-similarity ... ".format(opt.selfsim_samplesize))
     # sample words for self-similarity
    wsample = random.sample(bow5x,opt.selfsim_samplesize)
    
    print(datetime.now().time(), '|   Generating w2s dictionary')
    w2s = Emb.w2s(sents,wsample)
    #UPDATE w2s sentence index, to match the smaller subset of sentences
    print(datetime.now().time(), '|   Updating  dictionary indexes')
    w2s.update_indexes(sents)

    
    #load/compute embeddings from MT model
    print(datetime.now().time(), '|   Running sentences containing sampled words through the mt-encoder')
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

    import pickle






    print(datetime.now().time(), "|   Sampling {0} sentences to compute self-similarity ... ".format(opt.intrasentsim_samplesize))
    # sample sentences for intra sent similarity
    ssample_idx = [random.randint(0,len(sents)-1) for i in range(opt.intrasentsim_samplesize)]
    ssample = [sents[idx] for idx in ssample_idx]




 

if __name__ == '__main__':
    parser = opts.get_parser()
    opt = parser.parse_args()
    if opt.debug_mode:
        with ipdb.launch_ipdb_on_exception():
            main(opt)
    else:
        main(opt)

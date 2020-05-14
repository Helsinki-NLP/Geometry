
# Replicate Kawin E.'s work + be able to compute results with any given embeddings

'''
PUHTI bash:
    module purge
    source /projappl/project_2000945/.bashrc
    export PYTHONUSERBASE=/projappl/project_2000945
    export PATH=/projappl/project_2000945/bin:$PATH
    conda activate senteval
    cd /projappl/project_2001970/Geometry_jrvc/scripts
    python comparing_context_embedds.py 
    #ipython
'''
#
#
#
import random
import argparse
import sys 
import torch
import re
from copy import deepcopy
#import bertify


#####     MT MODEL EMBEDDINGS:     #####

def apply_bpe(sents_to_bpe, bpe_model):
    '''
    INPUT:
        sents_to_bpe: list of list of sentences. Each sentence is a list of words.
        bpe_model (str): path to the location of the BPE model
    '''
    PATH_TO_BPE="/projappl/project_2000945/subword-nmt/subword_nmt" 
    PATH_TO_BPE="/projappl/project_2001970/subword-nmt/subword_nmt" 
    sys.path.insert(0, PATH_TO_BPE)                                         
    from apply_bpe import BPE 
    import codecs
    codes = codecs.open(bpe_model, encoding='utf-8')
    bpe = BPE(codes)
    bpedsents=[]
    for sent in sents_to_bpe:
        str1 = ' '.join(sent)
        bpeversion=bpe.process_line(str1)
        bpedsents.append(bpeversion)
    return bpedsents

def invert_permutation(p):
    p = p - min(p) if not(min(p)==0) else p
    p = p.tolist()
    return torch.tensor([p.index(l) for l in range(len(p))])

def pad_batch(layers,longest_sent, pading_value=0):
    bsize, sentlen, hdim = layers['layer0']['normalized'].shape
    if not(sentlen == longest_sent):
        padder = torch.zeros(bsize,longest_sent-sentlen,hdim)
        for layer in layers:
            for key in layers[layer]:
                layers[layer][key] = torch.cat((layers[layer][key],padder), dim=1)
    
    return layers

def get_encodings_from_onmt_model(sents, opt):

    # STEP: get rid of empty lines
    samples = [sent if sent != [] else ['.'] for sent in sents]
    
    #  STEP: apply BPE to the sentences in the db
    bpe_model = opt.bpe_model
    sents = apply_bpe(samples, opt.bpe_model) if opt.apply_bpe else samples

    # STEP: load ONMT model
    if opt.cuda:
        torch.cuda.set_device(0)
        cur_device = "cuda"
        gpu_id=0
    else:
        cur_device = "cpu"
        gpu_id=None

    PATH_TO_ONMT='/scratch/project_2001970/AleModel/OpenNMT-py'
    sys.path.insert(0, PATH_TO_ONMT)
    from onmt import inputters
    from onmt import model_builder

    checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage)
    model_opt = checkpoint['opt']
    fields = checkpoint['vocab']
    model = model_builder.build_base_model(model_opt,fields,cur_device=='gpu',checkpoint,gpu_id)
    model.to(cur_device)
    model.eval()

    reader=inputters.str2reader['text'].from_opt(opt)    
    data = inputters.Dataset(fields=fields,
                             data=[("src",sents)],
                             readers=[reader],
                             dirs=[None],
                             sort_key=inputters.str2sortkey['text'])
    bsize=max(1024,round(len(sents)/100))
    data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=bsize, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

    #from onmt.translate.translator import Translator
    longest_sent = max([len(ex.src[0]) for ex in data.examples]) #needed for padding 
    dump_dict={'sentences':sents}
    for BATCH in data_iter:
        permutation = BATCH.indices
        src, src_lengths = BATCH.src if isinstance(BATCH.src, tuple) \
                       else (BATCH.src, None)
        
        enc_states, memory_bank, src_lengths, layers = model.encoder(src, src_lengths)
    
        # reorder and convert to numpy
        for layer in layers:
            for key in layers[layer]:
                layers[layer][key] = layers[layer][key][invert_permutation(permutation),:,:].detach()#.numpy() #shape=[batch_size,sentlen,rnn_size]        
                # dump to a dictionary
                layers = pad_batch(layers,longest_sent)
                dump_dict.setdefault(layer,{}).setdefault(key,torch.Tensor())
                #print(dump_dict[layer][key].shape, layers[layer][key].shape)
                dump_dict[layer][key] = torch.cat((dump_dict[layer][key], layers[layer][key]),dim=0)
                #print(dump_dict[layer][key].shape)
    
    return dump_dict

########################################

def index_w2s(sents,wsample):
    #makes a dictionary. Given a word/list of words point to all sentences (&position) where it appears
    w2s={word:[] for word in wsample} # word:[(sent1,pos1),(sent2,pos2)]
    new_sentences = []
    for i,sentence in enumerate(sents):
        for j,word in enumerate(sentence):
            if word in wsample:
                w2s[word].append((i,j))
                new_sentences.append(i)
    return w2s, set(new_sentences)

def extract_embeddings(w2s, all_mt_embedds):
    ''' given a dict and the embeddings, creates a dict of the form 
            {'word': {'layeri': {'normalized': [tensor], 'unnormalized': [tensor]} } } '''
    embedds = {}
    for word,occur in w2s.items():
        sentidx, wordidx = zip(*occur)
        for layer in all_mt_embedds:
            for key in all_mt_embedds[layer]:
                embedds.setdefault(word,{}).setdefault(layer,{}).setdefault(key,torch.Tensor())
                embedds[word][layer][key] = all_mt_embedds[layer][key][sentidx,wordidx,:]
                
    return embedds

def merge_bped_embedds(bpedsents, all_mt_embedds, w2s=None):
    '''find position in the bped version + average those embeddings'''
    tokd_bpedsents=[]
    for sent in bpedsents:
        sent = sent.strip()
        sent = sent.split(' ')
        tokd_bpedsents.append(sent)
    
    from itertools import groupby
    from operator import itemgetter
    
    repeated_indices = []
    for sentidx, sent in enumerate(tokd_bpedsents):
        # find the bpe splits
        # TODO: add special chars in the regex!!!
        subwordunits=re.findall(r"\S+@@",bpedsents[sentidx])
        temp=[-1]
        for subword in subwordunits:
            temp.append(sent.index(subword,temp[-1]+1))
        temp.remove(-1)
        repeated_indices.append(deepcopy(temp))

        subwordidx = []
        for k, g in groupby(enumerate(temp), lambda x: x[0]-x[1]):
            subwordidx.append(list(map(itemgetter(1), g)))
        for k in range(len(subwordidx)): 
            subwordidx[k].append(subwordidx[k][-1]+1)

        # average the corresponding embeddings:
        for layer in all_mt_embedds:
            for embs_type in all_mt_embedds[layer]:
                for aidis in subwordidx:
                    mean_vector = torch.mean(all_mt_embedds[layer][embs_type][sentidx,aidis,:], dim=0) # compute mean emb
                    all_mt_embedds[layer][embs_type][sentidx,aidis,:] = mean_vector  # replace all subword embs, for the mean_emb 
                    # cannot delete a column like this!
                    #all_mt_embedds[layer][embs_type][sentidx,:,:]  = torch.cat((all_mt_embedds[layer][embs_type][sentidx,:aidis[0],:],all_mt_embedds[layer][embs_type][sentidx,aidis[-1]:,:] ) )
        
        # correct index shifting
        if w2s:
            # unBPE the splitted words in the current sentence 
            unsplitted_words = []
            for subword in subwordidx:
                this_subword=[]
                for j in subword:
                    this_subword.append(''.join(sent[j].strip('@@')))
                unsplitted_words.append(''.join(this_subword))

            # replace one of the bpe occurences in the sentence    
            for chunk,w in enumerate(subwordidx):
                sent[w[0]] = unsplitted_words[chunk] 

            #replace index in w2s
            for word,position in w2s.items():
                ss,ww = zip(*position)
                indices = [i for i, x in enumerate(ss) if x == sentidx] 
                for i in indices:
                    ##import ipdb; ipdb.set_trace()
                    if word in sent:
                        w2s[word][i] = (w2s[word][i][0], sent.index(word))
                    else:
                        print('Failed to find word < {0} > in sentence < {1} > \n Original sentence < {2} > '.format(word, sent, bpedsents[sentidx]) ) 
                        import ipdb; ipdb.set_trace()
                    #prev_splits_count=0
                    #w2s[word][i] = tokd_bpedsents[sent_wordid[0]][sent_wordid[1]+prev_splits_count]

    return all_mt_embedds, w2s, repeated_indices

def self_similarity(word,embs_type):
    n, hdim = embs_type['normalized'].shape 
    current_selfsim = {'normalized':0, 'unnormalized':0}
    for key, embs in embs_type.items():
        # fill matrix of cosine_similarities
        for i in range(n):
            for j in range(i+1,n):
                current_selfsim[key] += torch.cosine_similarity(embs[i],embs[j],dim=0)
                #print(word, i,j,current_selfsim)
    #coeff = 1/(n**2 - n)
    coeff = 2 / (n**2 - n)
    current_selfsim['normalized']   *= coeff
    current_selfsim['unnormalized'] *= coeff
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
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=False,
                        help='path to read onmt_model')
    parser.add_argument("--cuda", action='store_true',
                        help="Whether to use a cuda device.") 
    parser.add_argument("--apply_bpe", action='store_true',
                        help="Does your model needs BPEd data as input?")
    parser.add_argument("--bpe_model", required=False,      
                        help='path to read onmt_model')
    opt = parser.parse_args() 
    #################### TAKE AWAY!!!!!!!:
    opt.model = "/scratch/project_2001970/AleModel/tr.6l.8ah.en-de_step_200000.pt" # TAKE AWAY overwrites
    opt.apply_bpe = True # TAKE AWAY overwrites
    opt.bpe_model = "/scratch/project_2001970/AleModel/bpe-model.de-en-35k.wmt19-news-para.norm.tok.tc" # TAKE AWAY overwrites
    #--------------
    # TODO: make it more general so the data can be given as an argument from command line, instead of hardcoding STS
    STS_path = '/scratch/project_2001970/SentEval/data/allSTS.txt'
    with open(STS_path, 'r') as f:
        samples = f.readlines()
    
    sents = []
    bow = set()
    for sent in samples:
        sent = sent.strip()
        #sent = sent.split(' ')
        sent = re.findall(r'[\w]+|\.|,|\?|\!|;|:|\'|\(|\)|/',sent)
        bow.update(set(sent)) 
        sents.append(sent)
    
    wsample = random.sample(bow,2500) # sample words: around 10% of vocab size
    # sample sentences for intra sent similarity
    ssample_idx = [random.randint(0,len(sents)-1) for i in range(500)] 
    ssample = [sents[idx] for idx in ssample_idx] 

    w2s, new_sentences_idx = index_w2s(sents,wsample[0:15])
    new_sentences = [sents[idx] for idx in new_sentences_idx]
    #UPDATE w2s sentence index, to match the smaller subset of sentences
    for key, val in w2s.items(): 
        for i, tpl in enumerate(val): 
            new_idx = list(new_sentences_idx).index(tpl[0]) 
            #print(key, tpl, new_idx, new_sentences[new_idx], )
            w2s[key][i] = (new_idx, tpl[1])

    #load/compute embeddings from MT model
    all_mt_embedds = get_encodings_from_onmt_model(new_sentences,opt) # get MT-embedds from set of sentences with the words needed
    bpedsents = all_mt_embedds.pop('sentences')
    
    # merge embeddings from splitted words (bpe)
    if opt.bpe_model:
        all_mt_embedds, w2s, _ = merge_bped_embedds(bpedsents, all_mt_embedds, w2s)

    # get embeddings of the word positions
    mt_embedds = extract_embeddings(w2s,all_mt_embedds)
    
    #import ipdb; ipdb.set_trace()
    
    # SELF-SIMILARITY
    '''
    self_similarities = {}
    for word, embedds in mt_embedds.items():
        if embedds['layer0']['normalized'].shape[0] > 1:
            for layer,embs_type in embedds.items():
                #print('entered using:', word,layer)
                self_similarities.setdefault(word,{}).setdefault(layer,{})
                self_similarities[word][layer] = self_similarity(word, embs_type)
    

    for word, emb_type in self_similarities.items(): 
        print('\t ### '+word+' ### \t')
        print('LAYER \t', 'NORMALIZED \t \t', 'UNNORMALIZED \t')
        for layername, layer in emb_type.items():
            print(layername+'\t', layer['normalized'].tolist(),'\t', layer['unnormalized'].tolist())
    '''

    #import ipdb; ipdb.set_trace()

    # INTRA-SENTENCE SIMILARITY    
    ssample_mt_embedds = get_encodings_from_onmt_model(ssample,opt) # get MT-embedds from set of sentences with the words needed
    bpedssample = ssample_mt_embedds.pop('sentences')
    
    # merge embeddings from splitted words (bpe)
    if opt.bpe_model:
        ssample_mt_embedds, _, repeated_indices = merge_bped_embedds(bpedsents, ssample_mt_embedds)
        
        #for sentence, indices in zip(bpedsents, repeated_indices):
        #    print(indices, sentence)


    #import ipdb; ipdb.set_trace()

    intra_similarities = {}
    tensor_intrasimilarities = torch.zeros((len(bpedsents), 7, 2))
    for lid, layer in enumerate(['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6']):
        for nid, norm in enumerate(['normalized', 'unnormalized']):
            for sid, sentence in enumerate(bpedsents):
                tok_sentence = sentence.strip().split(' ')
                bpe_length = len(tok_sentence)
                sent_emb = ssample_mt_embedds[layer][norm][sid,0:bpe_length,:]
                #import ipdb; ipdb.set_trace()
                unique_indices = list(set(range(bpe_length)).difference(set(repeated_indices[sid])))
                sent_emb = sent_emb[unique_indices,:]
                tensor_intrasimilarities[sid, lid, nid] = intra_similarity(sent_emb)
   
    #print(tensor_intrasimilarities[0,:,:])

    #load/compute embeddings from Bert
    # BERT: 
    #bert_sample = random.sample(sents,1)[0]
    #print(bert_sample)
    #bertify.tokenize(bert_sample)
    #bertify.encode(bert_sample)




######################################




# Replicate Kawin E.'s work + be able to compute results with any given embeddings

'''
PUHTI bash:
    module purge
    source /projappl/project_2000945/.bashrc
    export PYTHONUSERBASE=/projappl/project_2000945
    export PATH=/projappl/project_2000945/bin:$PATH
    conda activate senteval
    ipython
'''
#
#
#
import random
import argparse
import sys 






def index_w2s(sents,wsample):
    #makes a dictionary. Given a word/list of words point to all sentences (&position) where it appears
    w2s={word:[] for word in wsample} # word:[(sent1,pos1),(sent2,pos2)]
    for i,sentence in enumerate(sents):
        for j,word in enumerate(sentence):
            if word in wsample:
                w2s[word].append((i,j))
    return 


def self_similarity(word,embeddings):
    n = len(occurences)

    pass

def intra_similarity():
    pass

def max_expl_var():
    pass



#####     MT MODEL EMBEDDINGS:     #####

def apply_bpe(sents_to_bpe, bpe_model):
    '''
    INPUT:
        sents_to_bpe: list of list of sentences. Each sentence is a list of words.
        bpe_model (str): path to the location of the BPE model
    '''
    PATH_TO_BPE="/projappl/project_2000945/subword-nmt/subword_nmt" 
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
    #################### TAKE AWAY!!!!!!!:
    opt.model = "/scratch/project_2001970/AleModel/tr.6l.8ah.en-de_step_200000.pt" # TAKE AWAY overwrites
    opt.apply_bpe = True # TAKE AWAY overwrites
    opt.bpe_model = "/scratch/project_2001970/AleModel/bpe-model.de-en-35k.wmt19-news-para.norm.tok.tc" # TAKE AWAY overwrites
    #--------------

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
    import torch
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
    bsize=round(len(sents)/100)
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
                print(dump_dict[layer][key].shape)
    
return dump_dict

########################################

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
    # TODO: make it more general so the data can be given as an argument from command line, instead of hardcoding STS
    STS_path = '/scratch/project_2001970/SentEval/data/allSTS.txt'
    with open(STS_path, 'r') as f:
        samples = f.readlines()
    
    sents = []
    bow = set()
    for sent in samples:
        sent = sent.strip()
        sent = sent.split(' ')
        bow.update(set(sent)) 
        sents.append(sent)
    
    wsample = random.sample(bow,2500) # around 10% of vocab size

    w2s = index_w2s(sents,wsample)
    
    #load/compute embeddings    
    get_encodings_from_onmt_model(sents,opt)




'''
Contains functions to manage embedding representations
'''
import h5py
import pickle
import os

from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict
import torch
import random

from utils.logger import logger
from utils import model_loaders as Loader

class w2s:
    '''
    manages the word2sentence indexing dictionary
    '''
    def __init__(self, sents, wsample, shifted_by=0):
        '''
        A dictionary that points to sentence and word indexes.  
        Given a word/list of words point to all sentences (&position) where it appears
        INPUT:
            - sents[list]: list of tok'd sentences
            - wsample[list]: list of sampled words/tokens
    
        attributes:
            - w2sdict[dict]: dictionary to index word/token in wsample and sentences, position of appearence in that sentence.  
                         {'token_1': (sent_i1, pos_j1), (sent_i2,pos_j2), ... ,
                          'token_2': ... }
            - newidxs[set]: subset of sentences, where sampled words appear
        '''
        self.w2sdict={word:[] for word in wsample} # word:[(sent1,pos1),(sent2,pos2)]
        self.s2idxdict={idx:[] for idx in range(shifted_by,len(sents)+shifted_by)} # sentnum:[w_1, w_2,... w_n] ; for w_i \in  wsample

        for i,sentence in tqdm(enumerate(sents)):
            for j,word in enumerate(sentence):
                if word in wsample:
                    self.w2sdict[word].append((i+shifted_by,j))
                    self.s2idxdict[i+shifted_by].append(j)
         
        # for very frequent words (e.g., stopwords), there are too many pairwise comparisons
        # so for faster estimation, take a random sample of no more than 1000 pairs     
        logger.info('   Selecting 1000 occurences from words that occured more times.') # Kawin did this, i think
        for word, occ in self.w2sdict.items():
            if len(occ) > 1000:
                index_pairs = random.sample(occ, 1000)
                self.w2sdict[word] = index_pairs
    '''
    def update_indexes(self,sents):
        #override old indexes to match the new subset of sentences
        
        self.new_sents = [sents[idx] for idx in self.idx2newsents]
        for key, val in self.w2sdict.items():
            for i, tpl in enumerate(val):
                new_idx = list(self.idx2newsents).index(tpl[0])
                self.w2sdict[key][i] = (new_idx, tpl[1])

    def update(self,sents):
        self.new_sents = [sents[idx] for idx in self.idx2newsents]

    def save_w2s(self,fname):
        outfile=f'../outputs/{fname}.pkl'
        with open(outfile, 'w') as fout:
            for idx,embs in tqdm(enumerate(embeddings)):
                fout.create_dataset(str(idx), embs.shape, dtype='float32', data=embs)
    ''' 


def index_w2s(sents,wsample):
    '''
    Make a dictionary to point to sentence and word indexes. 
    Given a word/list of words point to all sentences (&position) where it appears
    INPUT:
        - sents[list]: list of tok'd sentences
        - wsample[list]: list of sampled words/tokens
    
    OUTPUT:
        - w2s[dict]: dictionary to index word/token in wsample and sentences, position of appearence in that sentence.  
                     {'token_1': (sent_i1, pos_j1), (sent_i2,pos_j2), ... ,
                      'token_2': ... }
        - new_sentences[set]: subset of sentences, where sampled words appear
    '''
    w2s={word:[] for word in wsample} # word:[(sent1,pos1),(sent2,pos2)]
    new_sentences = []
    for i,sentence in enumerate(sents):
        for j,word in enumerate(sentence):
            if word in wsample:
                w2s[word].append((i,j))
                new_sentences.append(i)
    return w2s, set(new_sentences)

def extract_embeddings(w2s, all_mt_embedds):
    '''
    Given a dict and the embeddings, creates a dict of the form 
            {'word': {'layeri': {'normalized': [tensor], 'unnormalized': [tensor]} } } 
    '''
    embedds = {}
    for word,occur in w2s.items():
        sentidx, wordidx = zip(*occur)
        for layer in all_mt_embedds:
            for key in all_mt_embedds[layer]:
                embedds.setdefault(word,{}).setdefault(layer,{}).setdefault(key,torch.Tensor())
                embedds[word][layer][key] = all_mt_embedds[layer][key][sentidx,wordidx,:]

    return embedds

def average_selfsim_per_layer(w2s,embeddings):
    avg_similarities = {}
    to_pickle = numpy.zeros((N_BERT_LAYERS, 1))
    for layer in range(N_BERT_LAYERS):
        encodings = []
        for word in w2s.keys():
            for occurrence in w2s[word]:
                sentence_id = occurrence[0]
                word_id = occurrence[1]
                encodings.append(bert_encodings[sentence_id][layer][0,word_id,:])

        avg_similarities[layer] = calculate_similarity(encodings).item()
        to_pickle[layer,0] = avg_similarities[layer]
        print('Avg similarity layer ', layer+1, avg_similarities[layer])
    
    pickle.dump(to_pickle, open('../results/avg_similarities_of_layers_' + str(opt.n_words) + '_' + opt.case + '.pkl', 'bw'))


    return avg_similarities

def loadh5file(load_path, numsents=-1, has_tgt=False):
    '''load embeddings and convert to list of tensors'''

    logger.info('  - Loading embeddings from {0}'.format(load_path))
    h5f = h5py.File(load_path, 'r')
    if numsents == -1:
        setlen = len(h5f) 
        loaded_embs = [torch.FloatTensor(h5f.get(str(i))[()]) for i in range(setlen)]    
    else:
        if has_tgt:
            setlen = len(h5f) // 2
            loaded_embs1 = [torch.FloatTensor(h5f.get(str(i))[()]) for i in range(numsents)] 
            loaded_embs2 = [torch.FloatTensor(h5f.get(str(i))[()]) for i in range(setlen,numsents+setlen)] 
            loaded_embs = loaded_embs1 + loaded_embs2 

        else:
            setlen = numsents 
            loaded_embs = [torch.FloatTensor(h5f.get(str(i))[()]) for i in range(setlen)]    
        
    h5f.close()
    # correct for sents of 1 token
    loaded_embs = [ emb.unsqueeze(1).detach() if len(emb.shape) == 2 else emb.detach() for emb in loaded_embs ] 
    return loaded_embs

def saveh5file(outdir, fname, embeddings):
    '''save embeddings in h5 format'''

    outfile=f'{outdir}/embeddings/{fname}.h5'
    logger.info('     saving embeddings to {0}'.format(outfile))
    os.system(f'mkdir -p {os.path.dirname(outfile)}')
    with h5py.File(outfile, 'w') as fout:
        for idx,embs in enumerate(embeddings):
            fout.create_dataset(str(idx), embs.shape, dtype='float32', data=embs)
    return outfile   


def BERT_compute_embeddings(samples, opt, bmodel, model_state_dict=None):

    # load model + tokenizer
    model = Loader.bertModel(bmodel, opt.cuda)
    
    if model_state_dict:
        logger.info('     HUOM! Loading pre-aligned BERT\'s state_dict')
        fsuffix=model_state_dict.get('modelname', '')
        model_state_dict = model_state_dict.get('state_dict', None) 
        model_state_dict = OrderedDict({k.replace('bert.',''):v for k,v in model_state_dict.items()})
        # TODO: Decide what to do with the elements of the last layer (used to project BERT into the same dimension as MT models)
        _, _ = model_state_dict.pop('linear.weight'), model_state_dict.pop('linear.bias')
        model.model.load_state_dict(model_state_dict)

    #compute BERT embeddings
    bert_tokens, bert_tokenization =  model.tokenize(samples)
    bert_encodings = model.encode(bert_tokens)
    corrected_sentences, bert_encodings = model.correct_bert_tokenization(bert_encodings, bert_tokenization)
    
    # save embeddings 
    fname0 = str(os.path.basename(opt.data_path[0]).split('.')[0])+'_'+str(bmodel)
    fname = fname0 if not model_state_dict else f'{fname0}_{fsuffix}'
    outfile= fname if not opt.dev_params else f'{fname}_dev'
    bert_encodings_path = saveh5file(outdir=opt.outdir, fname=outfile, embeddings=bert_encodings)
    
    #save sentences
    logger.info('     saving tokenized sentences to {0}'.format(f'{opt.outdir}/embeddings/{outfile}_tokd.pkl'))
    corr_sents_path = f'{opt.outdir}/embeddings/{outfile}_tokd.pkl'
    pickle.dump(corrected_sentences, open(corr_sents_path,'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    return  {'embeddings':bert_encodings_path, 'tokd_sents':corr_sents_path}
    #return [corr_sents_path, bert_encodings_path]


def huggingface_compute_embeddings(samples, opt, hfmodel, model_state_dict=None):
    # load model + tokenizer
    model = Loader.huggingfaceModel(hfmodel, opt.cuda)
    if model_state_dict:
        logger.info(f'   HUOM! Loading pre-aligned {hfmodel}\'s state_dict')
        fsuffix=model_state_dict.get('modelname', '')
        model_state_dict = model_state_dict.get('state_dict', None) 
        model_state_dict = OrderedDict({k.replace('mt_model.',''):v for k,v in model_state_dict.items()})
        model.model.load_state_dict(model_state_dict)

        
    #compute embeddings    
    hf_tokd, hf_encodings =  model(samples)
    corrected_sentences, hf_encodings = model.correct_tokenization(hf_tokd, hf_encodings)
    
    # save embeddings
    fname0 = str(os.path.basename(opt.data_path[0]).split('.')[0])+'_'+str(os.path.basename(hfmodel))
    fname = fname0 if not model_state_dict else f'{fname0}_{fsuffix}'
    outfile= fname if not opt.dev_params else f'{fname}_dev'
    hf_encodings_path = saveh5file(outdir=opt.outdir, fname=outfile, embeddings=hf_encodings)

    # save sentences
    logger.info('   saving tokenized sentences to {0}'.format(f'{opt.outdir}/embeddings/{outfile}_tokd.pkl'))
    corr_sents_path = f'{opt.outdir}/embeddings/{outfile}_tokd.pkl'
    pickle.dump(corrected_sentences, open(corr_sents_path,'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    return  {'embeddings':hf_encodings_path, 'tokd_sents':corr_sents_path}
    #return [corr_sents_path, hf_encodings_path]




def load_embeddings(paths, nsents=-1):
    '''
    loads embeddings 
    INPUT:
        - path. Path to h5 file -the tokenized embeddings have the same name but are saved in a pickle

    OUTPUT:
        - tokenized sentences, embeddings 

    '''
    # load embeddings
    path1 = paths.get('tokd_sents')
    logger.info(f'  - Loading tokenized sentences from {path1}') 
    toks = pickle.load(open(path1,'br'))
    if not nsents == -1:
        toks = {k:v[:nsents] for k,v in toks.items()}
    
    path2 = paths.get('embeddings')
    embeddings = loadh5file(path2, nsents, toks.get('target',False))

    return [toks, embeddings]


def compute_embeddings(opt, samples):   
    '''
    computes embeddings and saves them later on to use loads_embeddings 
    if --load_embeddings_path is given, then only calls load_embeddings
    otherwise computes and saves embeddings given by --bert_model and --huggingface_models

    INPUT:
        - opt
        - samples:list - 1 sentece per list entry
    OUTPUT:
        - toks_and_embeddings_paths:dict[modelname:str : embeddings:torch.Tensor] - dictionary with paths where embeddings where saved
                                                        

    '''
    toks_and_embeddings_paths={}
    
    # compute embeddings using prealigned models
    if not (opt.prealigned_models[0]=='None'):
        logger.info('Pre-aligned models:')
        for mpath, mtype in zip(opt.prealigned_models, opt.prealigned_model_types): 
            #prealigned_compute_embeddings(samples, )
            device='cuda' if opt.cuda else 'cpu'
            model_state_dict = torch.load(mpath, map_location=torch.device(device))
            modelname=os.path.basename(os.path.dirname(mpath))
            model_state_dict.update(modelname=modelname.replace('_',''))
            if 'bert' in mtype:
                bsamples = deepcopy(samples[0]) if isinstance(samples,tuple) else deepcopy(samples)
                model_type = 'bert-base-cased' # change this!
                used_paths = BERT_compute_embeddings(bsamples, opt, bmodel=model_type, model_state_dict=model_state_dict)
            else:
                srclang, tgtlang = mtype.split('-')
                used_paths= huggingface_compute_embeddings(deepcopy(samples),opt, hfmodel=f'Helsinki-NLP/opus-mt-{srclang}-{tgtlang}', model_state_dict=model_state_dict)
            
            fname = os.path.basename(used_paths['embeddings']).split('.')[0]
            toks_and_embeddings_paths[fname] = used_paths
         
        return toks_and_embeddings_paths
            


    # compute embeddings using BERT
    if not (opt.bert_models[0]=='None'):
        logger.info('BERT models:')
        for i, bmodel in enumerate(opt.bert_models):
            logger.info(' [{0}] {1} embeddings: computing & saving embeddings'.format(i, bmodel))
            bsamples = deepcopy(samples[0]) if isinstance(samples,tuple) else deepcopy(samples)
            used_paths = BERT_compute_embeddings(bsamples, opt, bmodel)
            
            fname = os.path.basename(used_paths['embeddings']).split('.')[0]
            toks_and_embeddings_paths[fname] = used_paths
    
    # compute embeddings using HUGGINGFACE models
    if not (opt.huggingface_models[0]=='None'):
        logger.info('HUGGINGFACE models:')

        for i, hfmodel in enumerate(opt.huggingface_models):
            srclang, tgtlang = hfmodel.split('-')
            logger.info(f' [{i}] Helsinki-NLP/opus-mt-{srclang}-{tgtlang} embeddings: computing & saving embeddings')
            used_paths= huggingface_compute_embeddings(deepcopy(samples),opt,f'Helsinki-NLP/opus-mt-{srclang}-{tgtlang}')
            fname = os.path.basename(used_paths['embeddings']).split('.')[0]
            toks_and_embeddings_paths[fname] = used_paths

            if isinstance(samples,tuple):
                logger.info(f' [{i}] Helsinki-NLP/opus-mt-{tgtlang}-{srclang} embeddings: computing & saving embeddings')
                samples2=(samples[1],samples[0]) 
                used_paths = huggingface_compute_embeddings(deepcopy(samples2),opt,f'Helsinki-NLP/opus-mt-{tgtlang}-{srclang}')
                fname = os.path.basename(used_paths['embeddings']).split('.')[0]
                toks_and_embeddings_paths[fname] = used_paths

    return toks_and_embeddings_paths

def emul_compute_embeddings(opt):
    '''
    emulate the dictionary formed in compute_embeddings 
    OUT:
     toks_and_embeddings_paths:dict[modelname:str : embeddings:torch.Tensor] - dictionary with paths where embeddings where saved

    '''
    toks_and_embeddings_paths={}
    

   
    for path in opt.load_embeddings_path:
        if os.path.isdir(path):
            for file in os.listdir(path):
                fname, extension = os.path.basename(file).split('.')
                fname = fname.strip('_tokd')
                tmpdict = toks_and_embeddings_paths.get(fname,{})
                if extension == 'h5':
                    embeddings_path = f'{path}/{file}'
                    tmpdict.update(embeddings = embeddings_path)

                if extension == 'pkl':
                    #logger.info('   loading from {0}'.format(f'{path}/{file}'))
                    toks_path = f'{path}/{file}'
                    tmpdict.update(tokd_sents = toks_path)
            
                toks_and_embeddings_paths[fname] = tmpdict
                
        else:
            fname = os.path.basename(path).split('.')[0]
            fname = fname.strip('_tokd')
            embeddings_path = path
            toks_path = f'{os.path.splitext(path)[0]}_tokd.pkl'

            toks_and_embeddings_paths[fname] = {'embeddings':embeddings_path, 'tokd_sents':toks_path}

    return toks_and_embeddings_paths
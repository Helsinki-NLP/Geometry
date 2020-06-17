'''
Contains functions to manage embedding representations
'''
import h5py
import pickle
import os
from tqdm import tqdm
from copy import deepcopy
import torch


from utils.logger import logger
from utils import model_loaders as Loader

class w2s:
    '''
    manages the word2sentence indexing dictionary
    '''
    def __init__(self, sents, wsample):
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
        self.s2idxdict={idx:[] for idx in range(len(sents))} # sentnum:[w_1, w_2,... w_n] ; for w_i \in  wsample
        new_sentences = []
        for i,sentence in tqdm(enumerate(sents)):
            for j,word in enumerate(sentence):
                if word in wsample:
                    self.w2sdict[word].append((i,j))
                    self.s2idxdict[i].append(j)
                    #if i not in new_sentences:
                    #    new_sentences.append(i)
        #self.idx2newsents = set(new_sentences)

    def update_indexes(self,sents):
        '''
        override old indexes to match the new subset of sentences
        '''
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

def loadh5file(load_path):
    '''load embeddings and convert to list of tensors'''
    logger.info('   loading from {0}'.format(load_path))
    h5f = h5py.File(load_path, 'r')
    setlen = len(h5f)
    loaded_embs = [torch.FloatTensor(h5f.get(str(i))[()]) for i in range(setlen)]    
    h5f.close()
    # correct for sents of 1 token
    loaded_embs = [ emb.unsqueeze(1).detach() if len(emb.shape) == 2 else emb.detach() for emb in loaded_embs ] 
    return loaded_embs

def saveh5file(outdir,fname,embeddings):
    '''save embeddings in h5 format'''
    outfile=f'{outdir}/embeddings/{fname}.h5'
    logger.info('     saving embeddings to {0}'.format(outfile))
    os.system(f'mkdir -p {os.path.dirname(outfile)}')
    with h5py.File(outfile, 'w') as fout:
        for idx,embs in enumerate(embeddings):
            fout.create_dataset(str(idx), embs.shape, dtype='float32', data=embs)
            


def BERT_compute_embeddings(samples,opt,bmodel):

    # load model + tokenizer
    model = Loader.bertModel(bmodel, opt.cuda)

    #compute BERT embeddings
    bert_tokens, bert_tokenization =  model.tokenize(samples)
    #if (opt.dev_params and len(bert_tokens)>17):
    #    bert_tokens = bert_tokens[0:17] 
    bert_encodings = model.encode(bert_tokens)
    corrected_sentences, bert_encodings = model.correct_bert_tokenization(bert_encodings, bert_tokenization)
    
    outfile=f'{bmodel}' if not opt.dev_params else f'{bmodel}_dev'
    saveh5file(opt.outdir,outfile,bert_encodings)
    logger.info('     saving tokenized sentences to {0}'.format(f'{opt.outdir}/embeddings/{outfile}_tokd.pkl'))
    pickle.dump(corrected_sentences, open(f'{opt.outdir}/embeddings/{outfile}_tokd.pkl','wb'))
    return corrected_sentences, bert_encodings


def huggingface_compute_embeddings(samples,opt,hfmodel):
    # load model + tokenizer
    model = Loader.huggingfaceModel(hfmodel, opt.cuda)


    #compute embeddings

    #tokd_sentences = model.tokenize(w2s.new_sents.copy())
    
    hf_tokd, hf_encodings =  model.encode(samples)
    corrected_sentences, hf_encodings = model.correct_tokenization(hf_tokd, hf_encodings)#, hf_tokenization)

    outfile=os.path.basename(hfmodel) if not opt.dev_params else f'{os.path.basename(hfmodel)}_dev'
    saveh5file(opt.outdir,outfile,hf_encodings)
    logger.info('   saving tokenized sentences to {0}'.format(f'{opt.outdir}/{outfile}_tokd.pkl'))
    pickle.dump(corrected_sentences, open(f'{opt.outdir}/embeddings/{outfile}_tokd.pkl','wb'))
    return corrected_sentences, hf_encodings






def compute_or_load_embeddings(opt, samples):
    '''
    loads embeddings if --load_embeddings_path is given
    otherwise computes and saves embeddings given by --bert_model and --huggingface_models

    INPUT:
        - opt
        - samples[list]: 1 sentece per list entry
    OUTPUT:
        - all_embeddings[dict[modelname:embeddings]]: dictionary with all embeddings.
                                                      either loads them with --load_embeddings_path

    '''
    all_embeddings={}
    all_toks = {}
    if not opt.load_embeddings_path:

        # compute embeddings using BERT
        if not (opt.bert_models[0]=='None'):
            logger.info('BERT models:')
            for i, bmodel in enumerate(opt.bert_models):
                logger.info(' [{0}] {1} embeddings: computing & saving embeddings'.format(i, bmodel))
                all_toks[bmodel], all_embeddings[bmodel] = BERT_compute_embeddings(deepcopy(samples), opt, bmodel)
        
        
        # compute embeddings using HUGGINGFACE 
        if not (opt.huggingface_models[0]=='None'):
            logger.info('HUGGINGFACE models:')
            for i, hfmodel in enumerate(opt.huggingface_models):
                logger.info(' [{0}] {1} embeddings: computing & saving embeddings'.format(i,hfmodel))
                all_toks[hfmodel], all_embeddings[hfmodel] = huggingface_compute_embeddings(deepcopy(samples),opt,hfmodel)
        
    else:
        # load embeddings
        logger.info('Loading embeddings & tokenized sentences...') 
        ''' 
        if isinstance(opt.load_embeddings_path,list):
            print(opt.load_embeddings_path)
            logger.info('Is a list')
            for file in opt.load_embeddings_path:
                fname=os.path.basename(file).split('.')[0]
                all_embeddings[fname] = loadh5file(file)
        else:
            path = opt.load_embeddings_path
            logger.info(f'{path}...')
            if os.path.isdir(path):
                logger.info('is a directory')
                for file in os.listdir(path):
                    fname, extension =os.path.basename(file).split('.')
                    if extension == 'h5':
                        all_embeddings[fname] = loadh5file(path+file)
        
            else:
                logger.info('is a file')
                fname = os.path.basename(path).split('.')[0]
                all_embeddings[fname]= loadh5file(path) 
        '''
        for path in opt.load_embeddings_path:
            if os.path.isdir(path):
                for file in os.listdir(path):
                    if os.path.isfile(path+'/'+file):
                        fname, extension = os.path.basename(file).split('.')
                        if extension == 'h5':
                            all_embeddings[fname] = loadh5file(path+'/'+file)
                        if extension == 'pkl':
                            logger.info('   loading from {0}'.format(f'{path}/{file}'))
                            all_toks[fname] = pickle.load(open(path+'/'+file,'br'))
            else:
                fname = os.path.basename(path).split('.')[0]
                all_embeddings[fname]= loadh5file(path)
                logger.info(f'Loading tokenized sentences from {os.path.splitext(path)[0]}_tokd.pkl') 

                all_toks[fname] = pickle.load(open(os.path.splitext(path)[0]+'_tokd.pkl','br'))
                

    return all_toks, all_embeddings

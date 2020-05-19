'''
Contains functions to manage embedding representations
'''
import h5py
import pickle
import os
from tqdm import tqdm


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
        new_sentences = []
        for i,sentence in tqdm(enumerate(sents)):
            for j,word in enumerate(sentence):
                if word in wsample:
                    self.w2sdict[word].append((i,j))
                    new_sentences.append(i)
        self.idx2newsents = set(new_sentences)
    
    def update_indexes(self,sents):
        '''
        override old indexes to match the new subset of sentences
        '''
        self.new_sents = [sents[idx] for idx in self.idx2newsents]
        for key, val in self.w2sdict.items():
            for i, tpl in enumerate(val):
                new_idx = list(self.idx2newsents).index(tpl[0])
                self.w2sdict[key][i] = (new_idx, tpl[1])

    def save_w2s(self,fname):
        outfile=f'../embeddings/{fname}.plk'
        print(coso)
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
    to_pickle = numpy.zeros((12, 1))
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


def saveh5file(outdir,fname,embeddings):
    outfile=f'{outdir}/{fname}.h5'
    logger.info('   saving to {0}'.format(outfile))
    with h5py.File(outfile, 'w') as fout:
        for idx,embs in tqdm(enumerate(embeddings)):
            fout.create_dataset(str(idx), embs.shape, dtype='float32', data=embs)
            


def BERT_compute_embeddings(w2s,opt):

    # load model + tokenizer
    model = Loader.bertModel(opt.bert_model, opt.cuda)

    # SELF-SIMILARITY OF WORDS
    logger.info('   self-similarity and max explainable variance ')
    #compute BERT embeddings
    bert_tokens, bert_tokenization =  model.tokenize(w2s.new_sents.copy())
    if (opt.dev_params and len(bert_tokens)>17):
        bert_tokens = bert_tokens[0:17] 
    bert_encodings = model.encode(bert_tokens)
    bert_encodings = model.correct_bert_tokenization(bert_encodings, bert_tokenization)
    
    outfile=f'{opt.bert_model}' if not opt.dev_params else f'{opt.bert_model}_dev'
    saveh5file(opt.outdir,outfile,bert_encodings)
    return bert_encodings


def huggingface_compute_embeddings(w2s,opt,hfmodel):
    # load model + tokenizer
    model = Loader.huggingfaceModel(hfmodel, opt.cuda)


    #compute embeddings

    #tokd_sentences = model.tokenize(w2s.new_sents.copy())
    sentences= w2s.new_sents.copy()[0:19] if opt.dev_params else w2s.new_sents.copy()
    hf_tokd, hf_encodings =  model.encode(sentences)
    hf_encodings = model.correct_tokenization(hf_tokd, hf_encodings)#, hf_tokenization)

    outfile=os.path.basename(hfmodel) if not opt.dev_params else f'{os.path.basename(hfmodel)}_dev'
    saveh5file(opt.outdir,outfile,hf_encodings)
    return hf_encodings






def compute_or_load_embeddings(opt,w2s):
    '''
    loads embeddings if --load_embeddings_path is given
    otherwise computes and saves embeddings given by --bert_model and --huggingface_models

    INPUT:
        - opt
        - w2s[class]: see above
    OUTPUT:
        - all_embeddings[dict[modelname:embeddings]]: dictionary with all embeddings.
                                                      either loads them with --load_embeddings_path

    '''
    all_embeddings={}
    if not opt.load_embeddings_path:
        # compute embeddings using BERT
        logger.info('BERT embeddings: computing & saving embeddings')
        all_embeddings[opt.bert_model] = BERT_compute_embeddings(w2s,opt)
    
        # compute embeddings using huggingface 
        logger.info('HUGGINGFACE models:')
        if isinstance(opt.huggingface_models,list):
            for i,hfmodel in enumerate(opt.huggingface_models):
                logger.info('   [{0}] {1} embeddings: computing & saving embeddings'.format(i,hfmodel))
                all_embeddings[hfmodel] = huggingface_compute_embeddings(w2s,opt,hfmodel)
        else:
            hfmodel=opt.huggingface_models
            logger.info('   [0] {0} embeddings: computing & saving embeddings'.format(hfmodel))
            all_embeddings[hfmodel] = huggingface_compute_embeddings(w2s,opt,hfmodel)


    else:
        # load embeddings
        logger.info('Loading embeddings...')
        if isinstance(opt.load_embeddings_path,list):
            for file in opt.load_embeddings_path:
                fname=os.path.basename(file)
                all_embeddings[fname] = 'prueba'   
        else:
            fname=os.load.basename(opt.load_embeddings_path)
            all_embeddings[fname]= h5py.File(opt.load_embeddings_path, 'r')
    
    return all_embeddings
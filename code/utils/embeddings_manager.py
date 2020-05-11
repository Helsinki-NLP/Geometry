'''
Contains functions to manage embedding representations
'''

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
        for i,sentence in enumerate(sents):
            for j,word in enumerate(sentence):
                if word in wsample:
                    self.w2sdict[word].append((i,j))
                    new_sentences.append(i)
        self.idx2newsents = set(new_sentences)
    
    def update_indexes(self,sents):
        self.new_sents = [sents[idx] for idx in self.idx2newsents]
        for key, val in self.w2sdict.items():
            for i, tpl in enumerate(val):
                new_idx = list(self.idx2newsents).index(tpl[0])
                self.w2sdict[key][i] = (new_idx, tpl[1])
        


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


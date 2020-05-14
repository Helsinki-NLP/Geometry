
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
import numpy
import pickle
from copy import deepcopy
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import logging

N_BERT_LAYERS = 12
ENC_DIM = 768

###### BERT-RELATED ######

def tokenize(sentences, tokenizer):
    # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
    # logging.basicConfig(level=logging.INFO)

    tokenized_sentences = []
    tokens_tensors = []
    for i in range(len(sentences)):
        if isinstance(sentences[i], str):
            sentences[i] = sentences[i].split(' ')
        sentences[i] = ' '.join(['[CLS]'] + sentences[i] + ['[SEP]'])
        #print(sentences[i])
        tokenized_sentences.append(tokenizer.tokenize(sentences[i]))
        #print('tokenized: ', tokenized_sentences[i])

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentences[i])

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensors.append(tokens_tensor)

    return tokens_tensors, tokenized_sentences


def encode(tokens_tensors, model):
    encoded_sentences = []

    for i in range(len(tokens_tensors)):
        # If you have a GPU, put everything on cuda
        tokens_tensors[i] = tokens_tensors[i].to('cuda')

        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensors[i])
            #print(encoded_layers[0].shape)
        
        encoded_sentences.append(encoded_layers)
    
    return encoded_sentences


def correct_bert_tokenization(bert_encodings, bert_sentences):
   
    corrected_sentences = []
    for bert_encoding, bert_sentence in zip(bert_encodings, bert_sentences):
        #print(bert_sentence)

        all_layers = []
        for layer in range(N_BERT_LAYERS):
            current_layer = []

            prev_token = bert_encoding[layer][0,0,:] # This is [CLS]!
            sequence_len = bert_encoding[layer].shape[1]

            accum = 1
            for token_id in range(1,sequence_len):
                if  bert_sentence[token_id][:2] != '##':
                    current_layer.append(prev_token.view(1,1,ENC_DIM) / accum) # Average pooling
                    accum = 1
                    prev_token = bert_encoding[layer][0,token_id,:]
                else:
                    prev_token += bert_encoding[layer][0,token_id,:] # Average pooling
                    accum += 1
            # Add the final token too:
            current_layer.append(prev_token.view(1,1,ENC_DIM) / accum) # Average pooling

            current_layer_tensor = torch.cat(current_layer, dim=1)
            # Get rid of the [CLS] and [SEP]
            current_layer_tensor = current_layer_tensor[:,1:-1,:]

            all_layers.append(current_layer_tensor)

        corrected_sentences.append(all_layers)
       
    #print('cbt: ', bert_sentences[0])
    #print('cbt: ', corrected_sentences[0][0].shape)

    return corrected_sentences





##### AUXILARY #####

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




##### KAWIN-METRICS #####

def calculate_similarity(encodings):
    n = len(encodings)

    current_selfsim = 0
    for i in range(n):
        for j in range(i+1,n):
            current_selfsim += torch.cosine_similarity(encodings[i],encodings[j],dim=0)

    coeff = 2/(n**2 - n)
    current_selfsim *= coeff

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




##### MAIN #####

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action='store_true',
                        help="Whether to use a cuda device.") 
    parser.add_argument('--case', type=str, default='uncased')
    parser.add_argument('--n_words', type=int, default=1000)
    opt = parser.parse_args() 
     
   
    # Load pre-trained model tokenizer (vocabulary)
    if opt.case == 'uncased':
        lower = True
    else:
        lower = False
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-' + opt.case, do_lower_case=lower)

    # Load pre-trained model (weights)
    bert_model = BertModel.from_pretrained('bert-base-' + opt.case)
    bert_model.eval()
    bert_model.to('cuda')


    STS_path = '/scratch/project_2001970/SentEval/data/allSTS.txt'
    with open(STS_path, 'r') as f:
        samples = f.readlines()

    sents = []
    bow = set()
    for sent in samples:
        sent = sent.strip()
        sent = re.findall(r'[\w]+|\.|,|\?|\!|;|:|\'|\(|\)|/',sent)
        bow.update(set(sent)) 
        sents.append(sent)
    
    # sample words for average word similarity: around 10% of vocab size
    wsample = random.sample(bow, 2500) # was 2500
    
    # sample sentences for intra sent similarity
    ssample_idx = [random.randint(0,len(sents)-1) for i in range(50)] # was 500
    ssample = [sents[idx] for idx in ssample_idx] 


    w2s, new_sentences_idx = index_w2s(sents,wsample[0:opt.n_words])
    new_sentences = [sents[idx] for idx in new_sentences_idx]
    for key, val in w2s.items(): 
        for i, tpl in enumerate(val): 
            new_idx = list(new_sentences_idx).index(tpl[0]) 
            w2s[key][i] = (new_idx, tpl[1])


    #compute BERT embeddings
    bert_tokens, bert_tokenization = tokenize(new_sentences, bert_tokenizer)
    bert_encodings = encode(bert_tokens, bert_model)
    bert_encodings = correct_bert_tokenization(bert_encodings, bert_tokenization)

    #mt_embedds = extract_embeddings(w2s,all_mt_embedds)
   
    
    # AVERAGE-SIMILARITY OF EVERY LAYER
    avg_similarities = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
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

    sys.exit(1)

    # SELF-SIMILARITY OF WORDS
    self_similarities = {}
    to_pickle = numpy.zeros((12,2500))
    wid = 0
    for word in w2s.keys():
        if len(w2s[word]) > 1: #eg. 'invaded': [(154, 32), (162, 7), (140, 34)]
            for layer in range(N_BERT_LAYERS):
                encodings = []
                for occurrence in w2s[word]:
                    sentence_id = occurrence[0]
                    word_id = occurrence[1]
                    encodings.append(bert_encodings[sentence_id][layer][0,word_id,:])

                self_similarities.setdefault(word,{}).setdefault(layer,{})
                self_similarities[word][layer] = calculate_similarity(encodings).item()
                to_pickle[layer,wid] = self_similarities[word][layer]
                print(word, layer, self_similarities[word][layer])
            wid += 1

    to_pickle = to_pickle[:, 0:wid]
    pickle.dump(to_pickle, open('../results/self_similarities_1000_cased.pkl', 'bw'))
        

    '''
    for word, emb_type in self_similarities.items(): 
        print('\t ### '+word+' ### \t')
        print('LAYER \t', 'NORMALIZED \t \t', 'UNNORMALIZED \t')
        for layername, layer in emb_type.items():
            print(layername+'\t', layer['normalized'].tolist(),'\t', layer['unnormalized'].tolist())
    '''

    # INTRA-SENTENCE SIMILARITY    
    '''
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
    '''




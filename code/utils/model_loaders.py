'''
Utilities for loading models - needed for embedding extraction
'''

#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM

from copy import deepcopy
from typing import List
import time
import transformers 

import torch
import torch.nn as nn

from utils.logger import logger


class bertModel(nn.Module):

    def __init__(self, bert_type='bert-base-uncased', cuda=False):
        super().__init__()
        self.N_LAYERS = 13
        self.ENC_DIM = 768

        self.tokenizer = BertTokenizer.from_pretrained(bert_type)

        config = BertConfig.from_pretrained(bert_type, output_hidden_states=True)
        self.model = BertModel.from_pretrained(bert_type, config=config)
        
        device='cuda' if cuda else 'cpu'
        self.device = device      
        self.model.eval()
        self.model.to(device)

    def forward(self, sentences):
        tokd_tensors, tokd_sents = self.tokenize(sentences)
        return tokd_sents, self.encode(tokd_tensors)

    def tokenize(self, sentences):
        '''OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
                    # logging.basicConfig(level=logging.INFO) '''
        logger.info('     tokenizing...')
        tokenized_sentences = []
        tokens_tensors = []
        for i in range(len(sentences)):
            # add BERT tags
            sentences[i] = '[CLS] ' + sentences[i] + ' [SEP]'
            tokenized_sentences.append(self.tokenizer.tokenize(sentences[i]))
            
            # Convert token to vocabulary indices
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sentences[i])
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            tokens_tensors.append(tokens_tensor)
        

        return tokens_tensors, tokenized_sentences


    def encode(self, tokens_tensors):
        logger.info('   encoding...')
        if self.device =='cpu':
            logger.info('   WARNING: using CPU... this might take a while.')
            print('                               If you have a GPU capable device, use --cuda option.')
        encoded_sentences = []
        
        nsents = len(tokens_tensors)
        check = int(nsents * 0.25 ) # LOG EVERY 25%:
        initime= time.time()

        for i in range(nsents):
            # If you have a GPU, put everything on cuda
            tokens_tensors[i] = tokens_tensors[i].to(self.device)
           
            with torch.no_grad():
                _, _, encoded_layers = self.model(tokens_tensors[i])

            encoded_layers = [i.detach().to('cpu') for i in encoded_layers]
            encoded_sentences.append(encoded_layers)

            # LOG:
            if len(encoded_sentences) % check  == 0 :
                logger.info(f'      ... step {len(encoded_sentences)} / {nsents} time: {time.time()-initime:.1f} secs')

        return encoded_sentences

    def correct_bert_tokenization(self, bert_encodings, bert_sentences):
        logger.info('     correcting for BERT subword tokenization...')
        corrected_encodings = []
        for bert_encoding, bert_sentence in zip(bert_encodings, bert_sentences):
            #print(bert_sentence)

            all_layers = []
            for layer in range(self.N_LAYERS):
                current_layer = []

                prev_token = bert_encoding[layer][0,0,:] # This is [CLS]!
                sequence_len = bert_encoding[layer].shape[1]

                accum = 1
                for token_id in range(1,sequence_len):
                    if  bert_sentence[token_id][:2] != '##':
                        current_layer.append(prev_token.view(1,1,self.ENC_DIM) / accum) # Average pooling
                        accum = 1
                        prev_token = bert_encoding[layer][0,token_id,:]
                    else:
                        prev_token += bert_encoding[layer][0,token_id,:] # Average pooling
                        accum += 1
                # Add the final token too:
                current_layer.append(prev_token.view(1,1,self.ENC_DIM) / accum) # Average pooling

                current_layer_tensor = torch.cat(current_layer, dim=1)
                # Get rid of the [CLS] and [SEP]
                current_layer_tensor = current_layer_tensor[:,1:-1,:]

                all_layers.append(current_layer_tensor.squeeze().detach())
 
            corrected_encodings.append(torch.stack(all_layers)) # [n_sents, N_LAYERS, sent_length, h_hidden]

        corrected_sentences = []        
        for tokdsent in bert_sentences:
            corrected_sentences.append(' '.join(tokdsent[1:-1]).replace(' ##','').split(' ') )
        
        corrected_sentencesdict = {'source':corrected_sentences}

        #print('cbt: ', bert_sentences[0])
        #print('cbt: ', corrected_sentences[0][0].shape)
        return corrected_sentencesdict, corrected_encodings


def load_onmt_model():
    pass

class huggingfaceModel(nn.Module):
    def __init__(self, modelname='Helsinki-NLP/opus-mt-en-fi', cuda=False):
        super().__init__()        

        logger.info('     loading model {}'.format(modelname))
        
        config_overrider={'output_attentions':True, 'output_hidden_states':True}
        self.model = transformers.MarianMTModel.from_pretrained(modelname, **config_overrider)
        
        logger.info('     loading tokenizer')
        self.tokenizer = transformers.MarianTokenizer.from_pretrained(modelname)    

        device='cuda' if cuda else 'cpu'
        self.device = device     
        self.model.eval()
        self.model.to(device)
        
        self.N_ENC_LAYERS = self.model.config.encoder_layers
        self.N_DEC_LAYERS = self.model.config.decoder_layers
        self.N_LAYERS = 1+self.N_ENC_LAYERS +self.N_DEC_LAYERS 
        self.DIM_HIDDEN = self.model.config.d_model # dimendion of the model
        self.bsz = 256 if cuda else 16
    
    def forward(self, sentences):
        return self.encode(sentences)

    def tokenize(self, sentences): 
        return [self.tokenizer.tokenize(sent) for sent in sentences] 

    def encode(self, sentences):
        '''
        encodes sentences
        OUTPUT:
            - encoded_sentences[list]: list of size |batches| . Each element of the list contains a list with the encodings 
                                       of embeddings layer, encoder_layers, decoder_layers
        '''
        logger.info('     tokenizing and computing embeddings ...')
        
        encoded_sentences = []
        tgt_tokd_sentences = []
        src_tokd_sentences = []

        if isinstance(sentences,tuple):
            src_sentences, tgt_sentences = sentences
            given_tgt_sents = True
        else:
            src_sentences = sentences
            tgt_sentences = [None for _ in range(len(sentences))]
            given_tgt_sents = False
            
        #src_tokd_sentences = self.tokenize(src_sentences)
        nsents = len(src_sentences)
        check = int(nsents * 0.25 ) # LOG EVERY 25%:
        initime= time.time()
        for src_sent, tgt_sent in zip(src_sentences,tgt_sentences):
            src_sent=[src_sent]
            if tgt_sent:
                tgt_sent = [tgt_sent] 
            tokdsent = self.tokenizer.prepare_seq2seq_batch(src_texts=src_sent, tgt_texts=tgt_sent) 
            tokdsent = {k:v.to(self.device) for k,v in tokdsent.items()}
            m_outputs = self.model.forward(return_dict=True, output_hidden_states=True, **tokdsent) 
            
            encoded_sentences.append( [x.detach().to('cpu').squeeze() for x in m_outputs.encoder_hidden_states+m_outputs.decoder_hidden_states]  )
            src_tokd = self.tokenizer.convert_ids_to_tokens(tokdsent['input_ids'][0])[:-1] # strip the '</s>' token
            tgt_tokd = self.tokenizer.convert_ids_to_tokens(tokdsent.get('labels', torch.zeros_like(tokdsent['input_ids']))[0])[:-1] # strip the '</s>' token
            src_tokd_sentences.append(src_tokd)
            tgt_tokd_sentences.append(tgt_tokd)
            # LOG:
            if len(encoded_sentences) % check  == 0 :
                logger.info(f'      ... step {len(encoded_sentences)} / {nsents} time: {time.time()-initime:.1f} secs')


        ''' CHECK IF TOKENIZATION WAS DONE RIGHT:
        allidx = set([i for i in range(len(src_tokd_sentences))])
        badidx = []
        for i,sent in enumerate(src_tokd_sentences):
            if not (len(sent) + 1 == encoded_sentences[i][0].size(0)):
                badidx.append(i)
        if tgt_tokd_sentences:
            idx_tokeep2 = []
            badidx2 = []
            for i,sent in enumerate(tgt_tokd_sentences):
                if not (len(sent) + 1 == encoded_sentences[i][12].size(0)):
                    badidx2.append(i)

        realbad = set(badidx).union(set(badidx2))
        idx_tokeep = allidx.difference(realbad)
        with open('../outputs/wrongly_tokd_sentences.txt','w') as f:
            for i in badidx:
                f.writelines(' '.join(src_tokd_sentences[i])+' \n')
            if tokd_tgt_sentences:
                for i in badidx2:
                    f.writelines(' '.join(tgt_tokd_sentences[i])+' \n')
        
        src_tokd_sentences = [src_tokd_sentences[i] for i in idx_tokeep]
        encoded_sentences = [encoded_sentences[i] for i in idx_tokeep]
        if given_tgt_sents:
            tgt_tokd_sentences = [tgt_tokd_sentences[i] for i in idx_tokeep]
        else:
            tgt_tokd_sentences = None
        '''

        # THIS IS HERE TEMPORARILY, TO SEE IF WE CAN GET RID OF THE COMMENTED PART ABOVE
        for i,sent in enumerate(src_tokd_sentences):
            if not (len(sent)+1  == encoded_sentences[i][0].size(0)):
                print(' ERROR: tokenization is not working for some sentences in src side')
        for i,sent in enumerate(tgt_tokd_sentences):
            if not (len(sent)+1  == encoded_sentences[i][12].size(0)):
                print(' ERROR: tokenization is not working for some sentences in tgt side')

        tgt_tokd_sentences = tgt_tokd_sentences if given_tgt_sents  else None 
        tokd_sentences = {'source':src_tokd_sentences, 'target':tgt_tokd_sentences}

        return tokd_sentences, encoded_sentences


    
    def correct_tokenization(self, tokd_sentences, encodings):
        logger.info('     correcting for tokenization...')
        corrected_encodings = []
        
        nlayers = self.N_ENC_LAYERS+1 if tokd_sentences['target'] else self.N_LAYERS
        for encoding, sentence in zip(encodings, tokd_sentences['source']):
            all_layers = []
            for layer in range(nlayers):
                current_layer = []
                prev_token = encoding[layer][0,:] 
                sequence_len = encoding[layer].shape[0]-1 # sentence does not include <eos>
                accum = 1
                for token_id in range(1,sequence_len):
                    if  sentence[token_id][0] == '▁':
                        current_layer.append(prev_token / accum) # Average pooling
                        accum = 1
                        prev_token = encoding[layer][token_id,:]
                    else:
                        prev_token += encoding[layer][token_id,:] # Average pooling
                        accum += 1
                
                # Add the last token too:
                current_layer.append(prev_token / accum) # Average pooling

                current_layer_tensor = torch.stack(current_layer, dim=0)

                all_layers.append(current_layer_tensor.detach().squeeze())
        
            corrected_encodings.append(torch.stack(all_layers)) # [n_sents, N_LAYERS, sent_length, h_hidden]
        
        if tokd_sentences['target']:
            for encoding, sentence in zip(encodings, tokd_sentences['target']):
                all_layers = []
                for layer in range(nlayers,self.N_LAYERS):
                    current_layer = []

                    prev_token = encoding[layer][0,:] 

                    sequence_len = encoding[layer].shape[0]-1 # sentence does not include <eos>
                    
                    accum = 1
                    for token_id in range(1,sequence_len):
                        if  sentence[token_id][0] == '▁':
                            current_layer.append(prev_token / accum) # Average pooling
                            accum = 1
                            prev_token = encoding[layer][token_id,:]
                        else:
                            prev_token += encoding[layer][token_id,:] # Average pooling
                            accum += 1
                    
                    # Add the last token too:
                    current_layer.append(prev_token / accum) # Average pooling

                    current_layer_tensor = torch.stack(current_layer, dim=0)

                    all_layers.append(current_layer_tensor.detach().squeeze())
            
                corrected_encodings.append(torch.stack(all_layers)) # [n_sents, N_LAYERS, sent_length, h_hidden]

        src_corr_sentences = []
        for sent in tokd_sentences['source']:
            src_corr_sentences.append(''.join(sent).split('▁')[1:] )
        tgt_corr_sentences  =[]
        for sent in tokd_sentences['target']:
            tgt_corr_sentences.append(''.join(sent).split('▁')[1:] )

        corrected_sentences = {'source':src_corr_sentences, 'target':tgt_corr_sentences}

        return corrected_sentences, corrected_encodings


        

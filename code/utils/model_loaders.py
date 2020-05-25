'''
Utilities for loading models - needed for embedding extraction
'''

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from copy import deepcopy
from tqdm import tqdm
from typing import List

import transformers 

import torch
from utils.logger import logger


class bertModel():

    def __init__(self, bert_type='bert-base-uncased', cuda=False):
        self.N_LAYERS = 12
        self.ENC_DIM = 768

        self.tokenizer = BertTokenizer.from_pretrained(bert_type)
        self.model = BertModel.from_pretrained(bert_type)
        
        device='cuda' if cuda else 'cpu'
        self.device = device      
        self.model.eval()
        self.model.to(device)
    
    def tokenize(self, sentences):
        '''OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
                    # logging.basicConfig(level=logging.INFO) '''
        logger.info('   tokenizing...')
        tokenized_sentences = []
        tokens_tensors = []
        for i in tqdm(range(len(sentences))):
            # add BERT tags
            sentences[i] = ' '.join(['[CLS]'] + sentences[i] + ['[SEP]'])
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

        for i in tqdm(range(len(tokens_tensors))):
            # If you have a GPU, put everything on cuda
            tokens_tensors[i] = tokens_tensors[i].to(self.device)
            
            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensors[i])
            
            encoded_layers = [i.to('cpu') for i in encoded_layers]
            encoded_sentences.append(encoded_layers)
            
        return encoded_sentences

    def correct_bert_tokenization(self, bert_encodings, bert_sentences):
        logger.info('   correcting for BERT subword tokenization...')
        corrected_sentences = []
        for bert_encoding, bert_sentence in tqdm(zip(bert_encodings, bert_sentences)):
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

                all_layers.append(current_layer_tensor.squeeze())
 
            corrected_sentences.append(torch.stack(all_layers)) # [n_sents, N_LAYERS, sent_length, h_hidden]
           
        #print('cbt: ', bert_sentences[0])
        #print('cbt: ', corrected_sentences[0][0].shape)
        return corrected_sentences


def load_onmt_model():
    pass

class huggingfaceModel():
    def __init__(self, modelname='bert-base-uncased', cuda=False):

        
        
        #configfile=transformers.AutoConfig.from_pretrained(modelname)
        #configfile=transformers.PretrainedConfig.get_config_dict(modelname)[0]
        #for key,val in config_overrider.items(): configfile[key]=val 

        logger.info('     loading model')
        
        #self.model = transformers.AutoModelWithLMHead.from_pretrained(modelname, **config_overrider)
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
    
    def tokenize(self, sentences): 
        return [self.tokenizer.tokenize(' '.join(sent)) for sent in sentences] 


    def unbatch(self, encoded_sentences_batched, tokd_batches, numsents):
        '''
        # correct dimensions from batching
        OUTPUT
            - encoded_sentences [list[list[tensor]]]. list of size numsents. 
                                                      each element is a list of size nlayers(1+N_ENC_LAYERS+N_DEC_LAYERS)
                                                     each entry a tensor of size [sentlength, self.DIM_HIDDEN]
        '''
        
        encoded_sentences = [[None for i in range(self.N_LAYERS)] for j in range(numsents)]
        
        for i, (batch, encodings) in enumerate(zip(tokd_batches, encoded_sentences_batched)):
            for nlay, layer_values in enumerate(encodings): 
                sentlevel = [ encsent[batch['attention_mask'][nsent].bool()] for nsent,encsent in enumerate(layer_values) ]
                for nsent, encsent in enumerate(sentlevel):
                    real_nsent = i*self.bsz+nsent
                    encoded_sentences[real_nsent][nlay] = encsent#.to('cpu')
        
        #for idx,sent in enumerate(encoded_sentences): 
        #    encoded_sentences[idx] = torch.stack(sent)

        encoded_sentences = [torch.stack(sent) for sent in encoded_sentences]
        return encoded_sentences

    def encode(self, sentences):
        '''
        encodes sentences
        OUTPUT:
            - encoded_sentences[list]: list of size |batches| . Each element of the list contains a list with the encodings 
                                       of embeddings layer, encoder_layers, decoder_layers
        '''
        logger.info('   tokenizing and computing embeddings ...')

        # compute by batches, because it is faster
        encoded_sentences = []
        '''
        tokd_batches = []
        for batch_id in tqdm(range(0,len(sentences),self.bsz)):
            last_id = min(batch_id+self.bsz, len(sentences)) # the last batch could be smaller than bsz
            thisbatch = [' '.join(sentences[i]) for i in range(batch_id,last_id)]
            tokdbatch = self.tokenizer.prepare_translation_batch(src_texts= thisbatch ) 
            tokdbatch = {k:v.to(self.device) for k,v in tokdbatch.items()}
            model_outputs = self.model.forward(**tokdbatch) 
            #tokdbatch = {k:v.to('cpu') for k,v in tokdbatch.items()}
            #tokd_batches.append(tokdbatch)
            #encoded_sentences.append( [x.to('cpu') for x in model_outputs[4]+model_outputs[1]]  )
        
        
        encoded_sentences = self.unbatch(encoded_sentences, tokd_batches, len(sentences))
        '''
        
        encoded_sentences = []
        for sent in tqdm(sentences):
            tokdsent = self.tokenizer.prepare_translation_batch(src_texts=[' '.join(sent)]) 
            tokdsent = {k:v.to(self.device) for k,v in tokdsent.items()}
            model_outputs = self.model.forward(**tokdsent) 
            #dec_out=model_outputs[:3] # x, all_hidden_states(output_hidden_states=True), all_self_attns(output_attentions=True)
            #enc_out=model_outputs[3:] # x, encoder_states(output_hidden_states=True), all_attentions(output_attentions=True)
            encoded_sentences.append( [x.detach().to('cpu') for x in model_outputs[4]+model_outputs[1]]  )

        tokd_sentences = self.tokenize(sentences)
        return tokd_sentences, encoded_sentences


    
    def correct_tokenization(self, tokd_sentences, encodings):
        logger.info('   correcting for tokenization...')
        corrected_sentences = []

        for encoding, sentence in tqdm(zip(encodings, tokd_sentences)):
            #sentence.append('▁<eos>')
            all_layers = []
            for layer in range(self.N_LAYERS):
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

                all_layers.append(current_layer_tensor.detach())
        
            corrected_sentences.append(torch.stack(all_layers)) # [n_sents, N_LAYERS, sent_length, h_hidden]
           
        return corrected_sentences


        

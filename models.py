
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm, trange
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline

from transformers import BertTokenizer, BertModel
from transformers import BartConfig, BartTokenizer, BartForConditionalGeneration
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

import faiss

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import glob


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'device is {device}')



class BARTTrain(torch.nn.Module):
    def __init__(self):
        super(BARTTrain, self).__init__()
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    
    def forward(self, encodings, mode):

        if mode == 'train':
            model_outputs = self.model(
                input_ids=encodings['input_ids_source_merged'].to(device), 
                attention_mask=encodings['attention_mask_source_merged'].to(device),
                decoder_input_ids=encodings['input_ids_target'].to(device),
                decoder_attention_mask=encodings['attention_mask_target'].to(device),
                labels=encodings['input_ids_target'].to(device))
        else:
            print('Running model in eval mode')
            model_outputs = self.model.generate(encodings)     

        return model_outputs

def get_latest_checkpoint(checkpoint, model, MODEL_STORE, q_string=''):

    checkpoints = sorted(glob.glob(f'{MODEL_STORE}/{checkpoint}*-[0-9]*'))
    if len(checkpoints):
        global_step = int(checkpoints[0].split('-')[-1])
        ckpt_name = '{}-{}'.format(checkpoint, global_step)
        print("Loading model from checkpoint %s" % ckpt_name)
        
        PATH = f'{MODEL_STORE}/{ckpt_name}/{checkpoint}{q_string}-{str(global_step)}.pt'
        # PATH = f'{MODEL_STORE}/{ckpt_name}/q_checkpoint-{str(global_step-1)}.pt'
        model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        print("model successfully loaded %s" % ckpt_name)
    
    else:
        print("No checkpoints available right now")
    return model

class Question_Model(torch.nn.Module):
    def __init__(self):
        super(Question_Model, self).__init__()
        self.question_model = BertModel.from_pretrained('bert-base-uncased')
    
    def forward(self, input_ids, attention_mask):
        question_outputs = self.question_model(input_ids, attention_mask)
        return question_outputs


class Context_Model(torch.nn.Module):
    def __init__(self):
        super(Context_Model, self).__init__()
        self.context_model = BertModel.from_pretrained('bert-base-uncased')
    
    def forward(self, input_ids, attention_mask):
        context_outputs = self.context_model(input_ids, attention_mask)
        return context_outputs


class EnsembleTokens(torch.nn.Module):
    def __init__(self):
        super(EnsembleTokens, self).__init__()
        self.question_model = Question_Model().to(device)
        self.context_model = Context_Model().to(device)

        # Freezing context model's params
        for param in list(self.context_model.children()):
            param.requires_grad = False

    
    def forward(self, encoding, mode='train'):

        if mode == 'train':
            question_encoded = self.question_model(encoding['input_ids_questions'].to(device), encoding['attention_mask_questions'].to(device))
            context_encoded = self.context_model(encoding['input_ids_context'].to(device), encoding['attention_mask_context'].to(device))
            return question_encoded, context_encoded

        if mode == 'inference':
            question_encoded = self.question_model(encoding['input_ids_questions'].to(device), encoding['attention_mask_questions'].to(device))
            return question_encoded




# def inference(question, bart_tokenizer, bart_model):

#     # Get Pretrained BERT encodings

#     ge = GetEncodings(type='questions')
#     encoded_question = ge.encode(question, max_length=30)

#     # Find top matching documents
#     ss = SearchSimilar(iterator = df_context['context'].values.tolist(), filename='index.bin', embeddings=model_op, shape=768, device=device)
#     similar_contexts = ss.get_n_similar_vectors(encoded_question, 3)
#     similar_contexts.insert(0, question)

#     combined_tokens = '</s></s>'.join(similar_contexts)

#     print(f'Top similar document outputs is {combined_tokens}')

#     # Prepare data for BART Inferencing

#     source_encoding = tokenizer(
#             combined_tokens,
#             max_length=1024,
#             padding='max_length',
#             add_special_tokens=True,
#             truncation=True,
#             return_tensors="pt")
   

#     # Inference BART Model
#     output = bart_model(source_encoding['input_ids'].to(device), mode = 'eval')
#     output = tokenizer.decode(output[0])
#     print(output)
#     return output

class GetEncodings:
    def __init__(self, MODEL_STORE, type='context'):
        self.MODEL_STORE = MODEL_STORE
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = Question_Model().to(device)

        # Load model's pretrained weights
        if type == 'questions':
            self.model = get_latest_checkpoint(checkpoint='qna_checkpoint', model=self.model, MODEL_STORE=self.MODEL_STORE, q_string='-q')
    

    def encode(self, text, max_length=200):
        '''Tokenize data and get encoded embeddings from model'''
        tk = self.tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                return_attention_mask=True,
                add_special_tokens=True,
                truncation=True,
                return_tensors="pt"
            )
        model_op = self.model(input_ids=tk['input_ids'].to(device), attention_mask=tk['attention_mask'].to(device))['pooler_output']
        return model_op



class SearchSimilar:
    def __init__(self, iterator = None, filename=None, embeddings=None, shape=None, device="cpu"):

        self.iterator = iterator
        
        if os.path.exists(filename) == True:

            print(f'Index file {filename}')
            self.index = faiss.read_index(filename)  # index2 is identical to index
        
        else:

            self.index = faiss.index_factory(shape, "Flat", faiss.METRIC_INNER_PRODUCT)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            faiss.write_index(self.index, filename)
            print(f'Index written at {filename}')
 
        
        if device == "cuda":
            print('Now running on CUDA')
            self.index = faiss.index_cpu_to_all_gpus(self.index)

  
        print(f'Index trained - {self.index.is_trained}')


    def get_n_similar_vectors(self, text, n=2):
        '''Find the top n similar vector from text using cosine similarity FAISS'''
        text = text.cpu().detach().numpy()
        faiss.normalize_L2(text)
        distance, index = self.index.search(text, n)

        context = []

        for i in index.tolist()[0]:
            context.append(self.iterator[i])
        return context
    
    def get_n_dissimilar_vectors(self, text, n=2):
        '''Find the top n dissimilar vector from text using cosine similarity FAISS'''
        pass
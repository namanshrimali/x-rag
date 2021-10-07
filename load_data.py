import pandas as pd
import os
import torch
from transformers import BertTokenizer, BertModel
import time
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'device is {device}')


import torch

class TextTokenizerBERT(torch.utils.data.Dataset):
    def __init__(
        self,
        data:pd.DataFrame, 
        tokenizer:BertTokenizer,
        source_max_length: int=30,
        ):
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_length = source_max_length



    def __len__(self):
        return len(self.data)


    def __getitem__(self, index:int):
        data_row = self.data.iloc[index]
        question_encoding = tokenizer(
            data_row['question'],
            max_length=self.source_max_length,
            padding='max_length',
            return_attention_mask=True,
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt")
        
        context_encoding = tokenizer(
            data_row['context'],
            max_length=300,
            padding='max_length',
            return_attention_mask=True,
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt")


        return dict(
            input_ids_questions = question_encoding['input_ids'].flatten(),
            attention_mask_questions = question_encoding['attention_mask'].flatten(),
            input_ids_context = context_encoding['input_ids'].flatten(),
            attention_mask_context = context_encoding['attention_mask'].flatten()
        )


# install huggingface BART MOdel
class BARTTokenizer(torch.utils.data.Dataset):
    def __init__(
        self,
        data:pd.DataFrame, 
        tokenizer:BartTokenizer,
        context_list:list,
        source_max_length: int=1024,
        target_max_length: int=30,
        device='cpu'
        ):

        self.tokenizer = tokenizer
        self.data = data
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length

        self.context_list = context_list

        self.ss = SearchSimilar(iterator = context_list, filename='index.bin', embeddings=model_op, shape=768, device=device)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index:int):
        data_row = self.data.iloc[index]

        context_index = data_row['group_id'].item()
        context = ge.encode(self.context_list[context_index])


        similar_contexts = self.ss.get_n_similar_vectors(context, 3)
        similar_contexts.insert(0, data_row['question'])

        combined_tokens = '</s></s>'.join(similar_contexts)

        source_encoding = tokenizer(
            combined_tokens,
            max_length=1024,
            padding='max_length',
            return_attention_mask=True,
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt")

        
        target_encoding = tokenizer(
            data_row['answer'],
            max_length=self.target_max_length,
            padding='max_length',
            return_attention_mask=True,
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt")
        

        # labels = target_encoding['input_ids']
        # decoder_input_ids = shift_tokens_right(labels, model.config.pad_token_id).flatten()
        # labels[labels[:, :] == self.model.config.pad_token_id] = -100
        # print(label)
    


        return dict(
            input_ids_source_merged = source_encoding['input_ids'].flatten(),
            attention_mask_source_merged = source_encoding['attention_mask'].flatten(),
            input_ids_target = target_encoding['input_ids'].flatten(),
            attention_mask_target = target_encoding['attention_mask'].flatten(),
        )

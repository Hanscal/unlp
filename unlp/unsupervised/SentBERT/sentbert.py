# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
refer:
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(paraphrase-multilingual-MiniLM-L12-v2)
"""
import os
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"


def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling - Take attention mask into account for correct averaging
    """
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class SentBERT(object):
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)

    def encode(self, sentences):
        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_is_string = True

        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        all_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        all_embeddings = np.asarray([emb.cpu().numpy() for emb in all_embeddings])

        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings


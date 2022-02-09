# -*- coding: utf-8 -*-
import time
import os
import torch
import numpy as np
file_root = os.path.dirname(__file__)
import sys
sys.path.append(file_root)
from BM25.bm25 import BM25Okapi
from mutils.similarity import cos_sim
from mutils.tokenizer import Tokenizer
from Word2Vec.word2vec import Word2Vec
from SentBERT.sentbert import SentBERT

class SemanticSearch(object):
    def __init__(self, model_type, corpus, model_path='', **kwargs):
        self.suport_type = ['bm25', 'sentbert', 'w2v']
        self.corpus = corpus
        self.tokenizer = Tokenizer()
        if isinstance(self.corpus, str):
            self.corpus = [self.corpus]
        if model_type == 'bm25':
            self.corpus_embeddings = {k: self.tokenizer.tokenize(k) for k in self.corpus}
            self.bm25_instance = BM25Okapi(corpus=list(self.corpus_embeddings.values()))
        elif model_type =='sentbert':
            self.model = SentBERT(model_path) if os.path.exists(model_path) else SentBERT()
            self.corpus_embeddings = self.model.encode(self.corpus)
        elif model_type == 'w2v':
            w2v_kwargs = {'binary': True}
            w2v_kwargs.update(kwargs)
            self.model = Word2Vec(model_name_or_path=model_path, w2v_kwargs=w2v_kwargs) if os.path.exists(model_path) else Word2Vec(w2v_kwargs=kwargs)
            corpus_word_list = [self.tokenizer.tokenize(k) for k in self.corpus]
            self.corpus_embeddings = self.model.encode(corpus_word_list)
        else:
            print("suport model_type: {}".format("##".join(self.suport_type)))
            os._exit(-1)
        self.model_type = model_type

    def run_search(self, query:list, top_k=5):
        out = []
        if self.model_type == 'bm25':
            for query in queries:
                scores = self.get_bm25_scores(query)
                rank_n = np.argsort(scores)[::-1]
                out.append([(self.corpus[i], scores[i]) for i in rank_n[:top_k]])
        elif self.model_type == 'w2v':
            query_list = [self.tokenizer.tokenize(q) for q in query]
            query_embedding = self.model.encode(query_list)
            hits = self.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k)
            for hit in hits:  # torch四维
                tmp = []
                for h in hit:
                    tmp.append((self.corpus[h['corpus_id']], h['score']))
                out.append(tmp)
        else:
            query_embedding = self.model.encode(query)
            hits = self.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k)
            for hit in hits: # torch四维
                tmp = []
                for h in hit:
                    tmp.append((self.corpus[h['corpus_id']], h['score']))
                out.append(tmp)
        return out


    def get_bm25_scores(self, query):
        """
        Get scores between query and docs
        :param query: input str
        :return: numpy array, scores for query between docs
        """
        tokens = self.tokenizer.tokenize(query)
        return self.bm25_instance.get_scores(query=tokens)

    def semantic_search(self,
                        query_embeddings,
                        corpus_embeddings,
                        query_chunk_size: int = 100,
                        corpus_chunk_size: int = 500000,
                        top_k: int = 10,
                        score_function=cos_sim):
        """
        This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
        It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

        :param query_embeddings: A 2 dimensional tensor with the query embeddings.
        :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
        :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
        :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
        :param top_k: Retrieve top k matching entries.
        :param score_function: Funtion for computing scores. By default, cosine similarity.
        :return: Returns a sorted list with decreasing cosine similarity scores. Entries are dictionaries with the keys 'corpus_id' and 'score'
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if isinstance(query_embeddings, (np.ndarray, np.generic)):
            query_embeddings = torch.from_numpy(query_embeddings)
        elif isinstance(query_embeddings, list):
            query_embeddings_tensor = [torch.from_numpy(i) for i in query_embeddings if isinstance(i, (np.ndarray, np.generic))]
            query_embeddings = torch.stack(query_embeddings_tensor)

        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.unsqueeze(0)

        if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
            corpus_embeddings = torch.from_numpy(corpus_embeddings)
        elif isinstance(corpus_embeddings, list):
            corpus_embeddings = torch.stack(corpus_embeddings)

        # Check that corpus and queries are on the same device
        query_embeddings = query_embeddings.to(device)
        corpus_embeddings = corpus_embeddings.to(device)

        queries_result_list = [[] for _ in range(len(query_embeddings))]

        for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
            # Iterate over chunks of the corpus
            for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
                # Compute cosine similarites
                cos_scores = score_function(query_embeddings[query_start_idx:query_start_idx + query_chunk_size],
                                            corpus_embeddings[corpus_start_idx:corpus_start_idx + corpus_chunk_size])

                # Get top-k scores
                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k, len(cos_scores[0])),
                                                                           dim=1, largest=True, sorted=False)
                cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
                cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(cos_scores)):
                    for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr],
                                                    cos_scores_top_k_values[query_itr]):
                        corpus_id = corpus_start_idx + sub_corpus_id
                        query_id = query_start_idx + query_itr
                        queries_result_list[query_id].append({'corpus_id': corpus_id, 'score': score})

        # Sort and strip to top_k results
        for idx in range(len(queries_result_list)):
            queries_result_list[idx] = sorted(queries_result_list[idx], key=lambda x: x['score'], reverse=True)
            queries_result_list[idx] = queries_result_list[idx][0:top_k]

        return queries_result_list

if __name__ == '__main__':
    # Corpus with example sentences
    corpus = [
        '花呗',
        '开通',
        '现在',
        '我们',
        'A man is eating food.',
        'A man is eating a piece of bread.',
        'The girl is carrying a baby.',
        'A man is riding a horse.',
        'A woman is playing violin.',
        'Two men pushed carts through the woods.',
        'A man is riding a white horse on an enclosed ground.',
        'A monkey is playing drums.',
        'A cheetah is running behind its prey.'
    ]
    # Query sentences:
    queries = [
        '我们',
        '如何更换花呗绑定银行卡',
        'A man is eating pasta.',
        'Someone in a gorilla costume is playing a set of drums.',
        'A cheetah chases prey on across a field.']

    ########  use semantic_search to perform cosine similarty + topk
    # TODO 还有w2v有点问题
    ssearch = SemanticSearch(model_type='w2v',corpus=corpus, model_path="/Volumes/work/project/unlp/unlp/transformers/word2vec/light_Tencent_AILab_ChineseEmbedding.bin")
    # model_type in ['bm25', 'sentbert', 'w2v']
    print("using semantic search!")
    b0 = time.time()
    res =ssearch.run_search(queries)
    print("cost {:.2f}s".format(time.time() - b0))
    print(res)




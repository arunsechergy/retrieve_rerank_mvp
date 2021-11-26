import json
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import gzip
import os
import torch


def search(query, n_ranked_hits, top_k, corpus_embeddings, passages, gpu):
    """
    :param query:
    :param n_ranked_hits:
    :param top_k:
    :param corpus_embeddings:
    :param passagees:
    :param gpu:
    :return:
    """
    #semantic encoding of query
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    if gpu:
        query_embedding.cuda()
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]

    #Rerank
    cross_encoder_input = [(query, passages[hit['corpus_id']]) for hit in hits]
    cross_scores = cross_encoder.predict(cross_encoder_input)

    #Sort results by cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross_score'] = cross_scores[idx]

    #Output top n reranker hits
    hits = sorted(hits, key = lambda x: x['cross_score'], reverse=True)

    return hits[0:n_ranked_hits]


def format_hits(passages, hits):
    """
    :param passages:
    :param hits:
    :return:
    """
    for hit in hits:
        print(f"{hit['cross_score']} {passages[hit['corpus_id']]}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    gpu = True
    if not torch.cuda.is_available():
        print("GPU not found")
        gpu = False

    #we use biencoder to encode all passages, so that we use it with semantic search
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    bi_encoder.max_sequence_length = 256
    top_k = 32 # number of documents to be retrieved from biencoder

    #Biencoder will retrieve n documents, cross encoder will rerank
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    #dataset here is wikipedia
    wikipedia_filepath = 'data/simplewiki-2020-11-01.jsonl.gz'

    if not os.path.exists(wikipedia_filepath):
        util.http_get('http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz', wikipedia_filepath)

    passages = []

    with gzip.open(wikipedia_filepath, 'rt', encoding='utf-8') as fIn:
        for line in fIn:
            data = json.loads(line.strip())

            passages.append(data['paragraphs'][0])

    print(f"{len(passages)}")

    #encoder all passages into vector space
    corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)

    query = 'What is the capital of the United States' #read from path location
    n_ranked_hits = 3
    reranked_hits = search(query, n_ranked_hits, top_k, corpus_embeddings, passages, gpu)

    # format hits in readable format
    format_hits(passages, reranked_hits)

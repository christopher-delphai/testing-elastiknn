""" 
Create a sample index and try some basic Elastic operations with vectors.
"""

from elasticsearch import Elasticsearch
from elastiknn.client import ElastiknnClient
from vectors import generate_vectors, get_vector, get_similars, get_random_vecs
import json
import time

# angular is exactly the same as cosine similarity
search_strategy = 'angular'
# search_strategy = 'l2'


def create_index(es_client):
    mapping = {
        "mappings": {
            "properties": {
                "keyword": {
                    "type": "text"
                },
                "keyword_vector": {
                    "type": "elastiknn_dense_float_vector",
                    "elastiknn": {
                        "dims": 200,
                        "model": "lsh",
                        "similarity": search_strategy,
                        "L": 99,
                        "k": 5
                    }
                }
            }
        }
    }
    if search_strategy == 'l2':
        mapping['mappings']['properties']['keyword_vector']['elastiknn']['w'] = 3

    # es_client.indices.delete(index='keyword_index')
    es_client.indices.delete(index='keyword_index', ignore=404)
    es_client.indices.create(index='keyword_index', body=mapping)


def do_index(es_client):
    vecs_dict = generate_vectors()
    rdm_vecs = get_random_vecs(905000)
    # breakpoint()
    start_t = time.time()
    index_dict(es_client, vecs_dict)
    index_dict(es_client, rdm_vecs)
    end_t = time.time()
    print(f'indexing took {end_t-start_t:.3f}s')


def index_dict(es_client, vecs_dict):
    for i, (keyw, vec) in enumerate(vecs_dict.items()):
        doc = {
            "keyword": keyw,
            "keyword_vector": {
                "values": vec
            }
        }
        es_client.index(index='keyword_index', body=doc)
        if i > 0 and i % 10000 == 0:
            print(f'\t{i} docs indexed...')


def do_query(es_client, term, print_results=False):
    term_vec = get_vector(term)
    search_query = {
        "size": 10,
        "query": {
            "elastiknn_nearest_neighbors": {
                "field": "keyword_vector",
                "vec": {
                    "values": term_vec
                },
                "model": "lsh",
                "similarity": search_strategy,
                "candidates": 50
            }
        }
    }
    if search_strategy == 'l2':
        search_query['query']['elastiknn_nearest_neighbors']['probes'] = 3

    start_t = time.time()
    res = es_client.search(index='keyword_index', body=json.dumps(search_query), _source=['keyword'])
    end_t = time.time()
    res_cnt = res.get("hits", {}).get("total", {}).get("value", 0)
    single_results = res.get("hits", {}).get('hits', [])
    res_keywords = [(r['_source']['keyword'], r['_score'] - 1) for r in single_results]
    print(f'\tquery {term:<20} took {end_t-start_t:.3f}s\t ({res_cnt} results)')
    if len(res_keywords) >= 1:
        print(f'\t\tmost similar: {res_keywords[0][0]:<20} with cosine sim {res_keywords[0][1]:.2f}')
    if print_results:
        print(f'{res_cnt} results for {term}')
        print(json.dumps(res_keywords, indent=4))
        print('compared with w2v output:')
        print(json.dumps(get_similars(term), indent=4))
        print()
        print()



def main():
    es_client = Elasticsearch('localhost:9200')
    # knn_client = ElastiknnClient(es_client)
    create_index(es_client)
    do_index(es_client)
    # do_index(client)
    do_query(es_client, 'mobility')
    do_query(es_client, 'sunlight')
    do_query(es_client, 'device')
    do_query(es_client, 'electric_cars')
    do_query(es_client, 'computers')


if __name__ == "__main__":
    main()
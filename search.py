import os
import sys; sys.path.insert(0, 'ColBERT/')
import pandas as pd
from colbert import Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Collection
from prepare_data import prepare_queries, prepare_collection


def search(queries_path, collection_path, index_name):
    """
    Uses the previously created index to perform efficient search of the queries
    
    Args:
        collection: list of strings, each string is a document
        queries: list of strings, each string is a query
        index_name: str, name of the index to use
        
    Returns:
    
    """
    if not os.path.exists(os.path.join("experiments/song_lyrics/indexes", index_name)):
        raise ValueError(f"Index {index_name} does not exist")
       
    with Run().context(RunConfig(nranks=2, experiment="song_lyrics")):
        config = ColBERTConfig(
            root="/mnt/netstore1_home/jakub.suran/COS470/Assignment5/experiments/",
        )

        collection = Collection(collection_path)
        searcher = Searcher(index=index_name, collection=collection, config=config)
        
    queries = pd.read_csv(queries_path, sep='\t', header=None, names=['ID', 'Lyrics'])
    results = []
    for _, (q_id, query) in queries.iterrows():
        results.append((q_id, searcher.search(query, k=10)))

    return results

# query_id Q0 doc_id rank score run_tag
def save_as_trec(results):
    columns = ["query_id", "Q0", "doc_id", "rank", "score", "run_tag"]
    data_list = []
    for q_id, result in results:
        for doc_id, rank, score in zip(*result):
            data_list.append((q_id, 'Q0', doc_id, rank, score, 'jakub.suran'))
    
    df = pd.DataFrame(data_list, columns=columns)
    df.to_csv('results.tsv', sep='\t', index=False)
    
    
def main():
    index_name = 'genre_lyrics.train.2bits'

    queries_path = prepare_queries()
    collection_path = prepare_collection()
    
    print("Searching...")
    results = search(queries_path, collection_path, index_name)
    save_as_trec(results)
    print("Saved results to results.tsv")

if __name__ == '__main__':
    main()
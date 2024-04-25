import sys; sys.path.insert(0, 'ColBERT/')

from colbert import Indexer
from colbert.data import Collection
from colbert.infra import Run, RunConfig, ColBERTConfig
from prepare_data import prepare_collection
import pandas as pd
import os


def index(collection_path, index_name):
    """
    For an efficient search, ColBERT pre-computes representation of each passage and indexes them.
    
    Args:
        collection: list of strings, each string is a document
    
    Returns:
        str: path to the index
    """
    if os.path.exists(os.path.join("experiments/song_lyrics/indexes", index_name)):
        print("Index already exists")
        return index_name
    
    nbits = 2
    checkpoint = 'colbert-ir/colbertv2.0'    

    with Run().context(RunConfig(nranks=2, experiment='song_lyrics')):  # nranks specifies the number of GPUs to use
        config = ColBERTConfig(nbits=nbits, kmeans_niters=4) # kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
                                                                                    # Consider larger numbers for small datasets.
        collection = Collection(collection_path)
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)

    index_path = indexer.get_index() 
    print(f"Index created at: {index_path}")
    
    return index_path



def main():
    index_name = 'genre_lyrics.train.2bits'
    collection_path = prepare_collection()
    
    print("\nIndexing collection...")
    index(collection_path, index_name)
    
   
if __name__ == '__main__':
    main()    

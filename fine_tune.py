import sys; sys.path.insert(0, 'ColBERT/') 
import pandas as pd
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer
from prepare_data import prepare_train_triples, prepare_queries, prepare_collection


def fine_tune(collection_path, queries_path, triples_path):
    with Run().context(RunConfig(nranks=2, experiment="song_lyrics")):

        config = ColBERTConfig(
            bsize=32,
        )
        trainer = Trainer(
            triples=triples_path,
            queries=queries_path,
            collection=collection_path,
            config=config,
        )

        trainer.train()

        print(f"Saved checkpoint...")


if __name__ == '__main__':
    collection_path = prepare_collection()
    queries_path = prepare_queries()
    triples_path = prepare_train_triples()
    
    fine_tune(collection_path, queries_path, triples_path)
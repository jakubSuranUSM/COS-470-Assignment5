import os
import pandas as pd
import json

def prepare_collection():
    """
    Prepares collection data for indexing and returns a list of passages
    """
    filename = "data/collection.tsv"
    dataset_path = prepare_train_dataset()
    
    if os.path.exists(filename):
        print("Passages file already exists")
        return filename
    
    df = pd.read_csv(dataset_path, sep='\t')
    df = df[["ID", "Lyrics"]]
    
    df.to_csv(filename, sep='\t', header=False, index=False)
    return filename


def prepare_train_dataset():
    filename = "data/song_dataset.tsv"
    if os.path.exists(filename):
        return filename
    
    # cleaning data
    df = pd.read_csv("data/genre_lyrics.tsv", sep='\t')
    df = df[df['Lyrics'] != 'Lyrics not found']
    df.drop_duplicates(subset='Lyrics', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['ID'] = df.index
    df = df[["ID", "Genre", "Lyrics"]]
    df.to_csv(filename, sep='\t', index=False)
    return filename
    
    
def prepare_test_dataset():
    filename = 'data/test_dataset.tsv'
    if os.path.exists(filename):
        return filename
       
    genre_recode = {
        'Blues': 'Blues',
        'Country': 'Country',
        'Metal': 'HeavyMetal',
        'Pop': 'Pop',
        'Rock': 'RockandRoll',
        'Rap': 'HipHop',
    }
    genres = os.listdir('data/Test Songs')
    song_dic = {'Genre': [], 'Lyrics': []}

    for genre in genres:
        songs = os.listdir('data/Test Songs/' + genre)
        for song in songs:
            with open('data/Test Songs/' + genre + '/' + song, 'r') as file:
                lyrics = file.read()
            lyrics = lyrics.replace('\n', ' ')
            song_dic['Genre'].append(genre)
            song_dic['Lyrics'].append(lyrics)

    df = pd.DataFrame(song_dic)
    df['ID'] = df.index
    df['Genre'] = df['Genre'].map(genre_recode)
    df = df[["ID", "Genre", "Lyrics"]]
    df.to_csv(filename, sep='\t', index=False)
    return filename
    

def prepare_queries():
    """
    Prepares queries data for searching
    """
    filename = "data/queries.tsv"
    test_dataset_path = prepare_test_dataset()
    
    if os.path.exists(filename):
        print("Queries file already exists")
        return filename
    
    df = pd.read_csv(test_dataset_path, sep='\t')
    df = df[["ID", "Lyrics"]]
    df.to_csv(filename, sep='\t', header=False, index=False)
    
    return filename


def prepare_train_triples():
    filename = 'data/triples.json'
    if os.path.exists(filename):
        print("Triples file already exists")
        return filename
    
    collection_path = prepare_train_dataset()
    queries_path = prepare_test_dataset()
    
    collection = pd.read_csv(collection_path, sep='\t')
    queries = pd.read_csv(queries_path, sep='\t')
    
    triples_dic = {
        'Query_ID': [],
        'Positive_ID': [],
        'Negative_ID': []
    }
    
    for _, (q_id, q_genre, _) in queries.iterrows():
        positive_samples = collection[collection['Genre'] == q_genre].sample(10)
        negativesamples = collection[collection['Genre'] != q_genre]
        negativesamples = negativesamples.groupby('Genre').apply(lambda x: x.sample(n=2)).reset_index(drop=True)
        assert len(positive_samples) == len(negativesamples)
        for pos_id, neg_id in zip(positive_samples['ID'], negativesamples['ID']):
            triples_dic['Query_ID'].append(q_id)
            triples_dic['Positive_ID'].append(pos_id)
            triples_dic['Negative_ID'].append(neg_id)
    
    print(len(triples_dic['Query_ID']))
    with open(filename, 'w') as file:
        for q_id, pos_id, neg_id in zip(triples_dic['Query_ID'], triples_dic['Positive_ID'], triples_dic['Negative_ID']):
            json.dump([q_id, pos_id, neg_id], file)
            file.write('\n')
    return filename
    
if __name__ == '__main__':
    prepare_collection()
    prepare_queries()
    prepare_train_triples()

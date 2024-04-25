# COS 470 - Assignment 5 - ColBERT Index, Search, and Fine-Tune

This repository contains scripts for Indexing, Searching, and Fine-Tuning ColBERT information retrieval model.

### Installation

Run the following commads to pull the ColBERT model and create appropriate environment. (if you are using CPU only change the `conda_env.yml` to `conda_env_cpu.yml`)

```
git -C ColBERT/ pull || git clone https://github.com/stanford-futuredata/ColBERT.git
conda env create -f ColBERT/conda_env.yml
conda activate colbert
```

**Note:** If you ever run into an error `fatal error: crypt.h: No such file or directory
   44 | #include <crypt.h>` you have to copy the `crypt.h` library into your environment. Use the following command to do so.

```
cp /usr/include/crypt.h [your_path_to_conda_environments]/colbert/include/python3.8/
```

## Indexing

Run the `index.py` file to index all songs from the Assignment 3 and 4 (can be found in `data/genre_lyrics.tsv`). This script will create a new folder `experiments` where the index will be saved.

## Searching

Run the `search_file.py` to search queries in the collection using the created index (queries are extracted form `data/Test Songs` folder that was provided with the assignment). The script will create a `results.tsv` file in the root directory which will contain the retrieval results in TREC format.

## Fine-Tuning

Run the `fine_tune.py` file to fine-tune the ColBERT model with a small subset of songs from the train dataset. The model is trained only on 300 songs to prevent exhaustive training which is not the scope of this assignment.

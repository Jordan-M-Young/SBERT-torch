# SBERT-torch
Pytorch Implementation of Sentence-BERT architecture from the 2019 Paper [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084).

# Data

The paper uses the [SNLI](https://nlp.stanford.edu/projects/snli/) and [MNLI](https://cims.nyu.edu/~sbowman/multinli/) datasets to train SBERT models, we make use of these datasets as well.

To download these datasets, run:

```bash
poetry run pull_data
```

You'll see a `data` directory populated with two zip files. Extract these files and you're set to go.


# Training

To train an SBERT model simply modify the config variables in `main.py` and run:

```bash
poetry run train
```

The main variables you'll want to pay attention to are the `BERT_MODEL` and `OBJECTVIE` variables. These control which variant o the BERT family of
models you'll use in your SBERT model and which of the training objectives (detailed in the paper, section 3. page 3) you want to use. 



# Improvements

Per the paper, several concantentation strategies were tested for use in the classification objective. I've only implemented the strategy that had the best results
`(u,v,|u-v|)`. Implementing the other strategies could be a nice improvement.

I've also only implemented two of the training objectives, classification and regression. A third training objective, triplet objective, was used by the authors. I haven't implemented that either. 

Finally, the `BERTModels` enum in `constants.py` is fairly limited in which models are available for use. Adding more models like DistilBERT would be interesting.


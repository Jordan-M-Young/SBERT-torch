"""Main Training Loop."""

from app.sbert import SentenceBERT


def main():
    """Main Training Loop."""
    BERT_MODEL = "google-bert/bert-base-uncased"
    CONCAT_STRATEGY = "u,v,u-v"
    _model = SentenceBERT(bert_model=BERT_MODEL, concat_strat=CONCAT_STRATEGY)
    pass


if __name__ == "__main__":
    main()

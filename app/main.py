"""Main Training Loop."""

from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer

from app.constants import BERTModels, ConcatStrategies, TrainingObjective
from app.data import SentencePairDataset, load_data
from app.sbert import SentenceBERT
from app.training import evaluate, train
from app.utils import log_epoch


def main():
    """Main Training Loop."""
    # type of bert model you want to train.
    BERT_MODEL = BERTModels.TINY_BERT

    # model name and layer size of bert model
    MODEL_NAME, _LAYER_SIZE = BERT_MODEL.value

    # training objective. classification or regression.
    OBJECTIVE = TrainingObjective.CLASSIFICATION

    # concatentation strategy. if training objective is regression dont worry about this
    CONCAT_STRATEGY = ConcatStrategies.UVUsubV

    # jsonl file containing training data.
    TRAINING_FILE = "./data/snli_1.0/snli_1.0_dev.jsonl"

    # training hyperparams
    TEST_FRACTION = 0.2
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.0001

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    pairs, labels = load_data(TRAINING_FILE)

    # use a small portion of data..
    pairs = pairs[:1000]
    labels = labels[:1000]

    sentence_pair_dataset = SentencePairDataset(
        sentence_pairs=pairs, labels=labels, tokenizer=tokenizer, head=OBJECTIVE
    )

    # split dataset
    size = len(sentence_pair_dataset)
    train_size = int((1 - TEST_FRACTION) * size)
    test_size = size - train_size
    train_dataset, test_dataset = random_split(
        sentence_pair_dataset, [train_size, test_size]
    )

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # initialize model
    model = SentenceBERT(
        bert_model=BERT_MODEL,
        concat_strat=CONCAT_STRATEGY,
        head=OBJECTIVE,
    )

    # initialize loss function
    if OBJECTIVE == TrainingObjective.CLASSIFICATION:
        # if training objective is classification use crossentropy
        loss_func = CrossEntropyLoss()
    else:
        # if training objective is regresssion use mean square error
        loss_func = MSELoss()
    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop
    for epoch in range(EPOCHS):
        train_loss = train(train_dataloader, model, loss_func, optimizer)
        test_loss = evaluate(test_dataloader, model, loss_func)
        log_epoch(epoch, train_loss, test_loss)


if __name__ == "__main__":
    main()

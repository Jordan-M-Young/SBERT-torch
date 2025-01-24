"""Main Training Loop."""
from app.data import load_data, SentencePairDataset
from torch.utils.data import DataLoader, random_split
from torch import optim, tensor
from app.sbert import SentenceBERT
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.nn import CrossEntropyLoss
from app.training import train, evaluate
from app.utils import log_epoch
def main():
    """Main Training Loop."""
    BERT_MODEL = "prajjwal1/bert-tiny"
    CONCAT_STRATEGY = "u,v,u-v"
    TRAINING_FILE = './data/snli_1.0/snli_1.0_dev.jsonl'
    TEST_FRACTION =  0.2
    BATCH_SIZE = 32
    EPOCHS = 30
    #load tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    #load dataset
    pairs, labels = load_data(TRAINING_FILE)
    
    #use a small portion of data..
    pairs = pairs[:1000]
    labels = labels[:1000]

    sentence_pair_dataset = SentencePairDataset(sentence_pairs=pairs, labels=labels, tokenizer=tokenizer)
    
    # split dataset
    size = len(sentence_pair_dataset)
    train_size = int((1-TEST_FRACTION) * size)
    test_size = size - train_size
    train_dataset, test_dataset = random_split(sentence_pair_dataset,[train_size, test_size])

    # create dataloaders
    train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    #initialize model
    model = SentenceBERT(bert_model=BERT_MODEL, concat_strat=CONCAT_STRATEGY)

    #initialize loss function
    loss_func = CrossEntropyLoss()

    #initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    #training loop
    for epoch in range(EPOCHS):
        train_loss = train(train_dataloader,model,loss_func,optimizer)
        test_loss = evaluate(test_dataloader, model, loss_func)
        log_epoch(epoch, train_loss, test_loss)

if __name__ == "__main__":
    main()

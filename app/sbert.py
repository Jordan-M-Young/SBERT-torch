"""SBERT Model Class."""

import torch
from transformers import BertModel

from app.constants import BERTModels, ConcatStrategies, TrainingObjective


class SentenceBERT(torch.nn.Module):
    """SBERT Class extending pytorch Module."""

    def __init__(
        self,
        bert_model: BERTModels = BERTModels.TINY_BERT,
        concat_strat: ConcatStrategies = ConcatStrategies.UV,
        head: TrainingObjective = TrainingObjective.CLASSIFICATION,
    ):
        """Initialize SBERT Class."""
        super(SentenceBERT, self).__init__()

        self.model_name = bert_model.value[0]
        self.layer_size = bert_model.value[1]
        self.strat = concat_strat
        self.bert = BertModel.from_pretrained(self.model_name)
        self.size_multiple = self.get_size_multiplier()
        self.fc_classifier = torch.nn.Linear(self.layer_size * self.size_multiple, 3)
        self.softmax = torch.nn.Softmax(dim=1)
        self.cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.head = head

    def forward(self, left, right):
        """Model forward pass."""
        l_input_ids = left["input_ids"]
        l_input_ids = l_input_ids.squeeze(1)
        l_attention_mask = left["attention_mask"]

        r_input_ids = right["input_ids"]
        r_input_ids = r_input_ids.squeeze(1)
        r_attention_mask = right["attention_mask"]

        left = self.bert(l_input_ids, attention_mask=l_attention_mask)
        right = self.bert(r_input_ids, attention_mask=r_attention_mask)

        l_attention_mask = l_attention_mask.squeeze(1)
        r_attention_mask = r_attention_mask.squeeze(1)

        left = self.concise_mean_pooling(left, l_attention_mask)
        right = self.concise_mean_pooling(right, r_attention_mask)

        if self.head == TrainingObjective.CLASSIFICATION:
            output = self.run_concat(left, right)
            output = self.fc_classifier(output)
            output = self.softmax(output)

        else:
            output = self.cossim(left, right)
        return output

    def run_concat(
        self, left_input: torch.Tensor, right_input: torch.Tensor
    ) -> torch.Tensor:
        """Run concatenation operation based on concatenation strategy."""
        concat = torch.cat((left_input, right_input), dim=1)
        if self.strat == ConcatStrategies.UV:
            return concat
        if self.strat == ConcatStrategies.UVUsubV:
            sub = left_input - right_input
            concat = torch.cat((concat, sub))
            return concat

    def concise_mean_pooling(self, model_output, attention_mask):
        """More concise mean pooling function."""
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_size_multiplier(self) -> int:
        """Returns an integer which describes how many times the size of
        our input will be multipled based on a concat strategy.
        e.g. the bert model has a 128 size output, and were concatenating
         the left, right, and left-right vectors. Thus our multiple is 3.
        """
        if self.strat == ConcatStrategies.UV:
            return 2

        if self.strat == ConcatStrategies.UVUsubV:
            return 3

"""SBERT Model Class."""

from torch import Tensor, cat
from torch.nn import Module
from transformers import BertModel


class SentenceBERT(Module):
    """SBERT Class extending pytorch Module."""

    def __init__(self, bert_model: str, concat_strat="u,v,u-v"):
        """Initialize SBERT Class."""
        super(SentenceBERT, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model)
        self.strat = concat_strat

    def forward(self, left_input, right_input):
        """Model forward pass."""
        left_ouput = self.bert(left_input)
        right_ouput = self.bert(right_input)

        # Todo, add 'pooling' layer.

        output = self.run_concat(left_ouput, right_ouput)

        return output

    def run_concat(self, left_input: Tensor, right_input: Tensor) -> Tensor:
        """Run concatenation operation based on concatenation strategy."""
        concat = cat((left_input, right_input), dim=2)
        if self.strat == "u,v,u-v":
            return concat

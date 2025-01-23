"""SBERT Model Class."""

import torch
from transformers import BertModel


class SentenceBERT(torch.nn.Module):
    """SBERT Class extending pytorch Module."""

    def __init__(self, bert_model: str, concat_strat="u,v,u-v"):
        """Initialize SBERT Class."""
        super(SentenceBERT, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model)
        self.strat = concat_strat

    def forward(self, left, right):
        """Model forward pass."""
        left = self.bert(left)
        right = self.bert(right)

        left = self.mean_pooling(left)
        right = self.mean_pooling(right)

        output = self.run_concat(left, right)

        return output

    def run_concat(self, left_input: torch.Tensor, right_input: torch.Tensor) -> torch.Tensor:
        """Run concatenation operation based on concatenation strategy."""
        concat = torch.cat((left_input, right_input), dim=2)
        if self.strat == "u,v,u-v":
            return concat

    def concise_mean_pooling(self, model_output, attention_mask):
        """More concise mean pooling function."""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def mean_pooling(self, features):
        """Custom Mean pooling logic according to the paper."""
        output_vectors = []
        token_embeddings = features["token_embeddings"]
        attention_mask = (
            features["attention_mask"]
            if "attention_mask" in features
            else torch.ones(token_embeddings.shape[:-1], device=token_embeddings.device, dtype=torch.int64)
        )

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        sum_embeddings = sum(token_embeddings * input_mask_expanded, 1)

        # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
        if "token_weights_sum" in features:
            sum_mask = features["token_weights_sum"].unsqueeze(-1).expand(sum_embeddings.size())
        else:
            sum_mask = input_mask_expanded.sum(1)

        sum_mask = torch.clamp(sum_mask, min=1e-9)

        if self.pooling_mode_mean_tokens:
            output_vectors.append(sum_embeddings / sum_mask)

        output_vector = torch.cat(output_vectors, 1)
        features["sentence_embedding"] = output_vector
        return features

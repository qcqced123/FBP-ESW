import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoModel

import configuration
from model.model_utils import freeze, reinit_topk


class NERModel(nn.Module):
    """
    Model class For NER Task Pipeline, in this class no pooling layer
    This pipeline apply B.I.O Style, so the number of classes is 15 which is 7 unique classes original
    Each of 7 unique classes has sub 2 classes (B, I) => 14 classes
    And 1 class for O => 1 class
    14 + 1 = 15 classes
    Args:
        cfg: configuration.CFG
    """
    def __init__(self, cfg: configuration.CFG) -> None:
        super().__init__()
        self.cfg = cfg
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model,
            output_hidden_states=True
        )
        self.model = AutoModel.from_pretrained(
            cfg.model,
            config=self.auto_cfg
        )
        self.fc = nn.Linear(self.auto_cfg.hidden_size, 15)  # BIO Style NER Task

        if self.cfg.reinit:
            self._init_weights(self.fc)
            reinit_topk(self.model, cfg.num_reinit)

        if self.cfg.freeze:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:cfg.num_freeze])

        if self.cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def _init_weights(self, module) -> None:
        """ over-ride initializes weights of the given module function (+initializes LayerNorm) """
        if isinstance(module, nn.Linear):
            if self.cfg.init_weight == 'normal':
                module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            elif self.cfg.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif self.cfg.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif self.cfg.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif self.cfg.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif self.cfg.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """ reference from torch.nn.Layernorm with elementwise_affine=True """
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def feature(self, inputs: dict):
        outputs = self.model(**inputs)
        return outputs

    def forward(self, inputs: dict[Tensor, Tensor, Tensor]) -> Tensor:
        """
        No Pooling Layer for word-level task
        Args:
            inputs: Dict type from AutoTokenizer
            => {input_ids: Tensor[], attention_mask: Tensor[], token_type_ids: Tensor[]}
        """
        outputs = self.feature(inputs)
        logit = self.fc(outputs.last_hidden_state)
        return logit

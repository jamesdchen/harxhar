import torch
import torch.nn as nn
from transformers import PatchTSMixerConfig, PatchTSMixerModel, PreTrainedModel


class PatchTSMixerForecaster(PreTrainedModel):
    config_class = PatchTSMixerConfig

    def __init__(self, config):
        super().__init__(config)
        self.backbone = PatchTSMixerModel(config)

        dummy_input = torch.zeros(1, config.context_length, config.num_input_channels)
        with torch.no_grad():
            dummy_out = self.backbone(past_values=dummy_input).last_hidden_state

        self.num_patches = dummy_out.shape[2]
        self.flat_dim = self.num_patches * config.d_model

        self.head = nn.Linear(self.flat_dim, config.prediction_length)
        self.head.weight.data.normal_(0, 0.001)
        self.head.bias.data.fill_(0.0)

        self.post_init()

    def forward(self, past_values, future_values=None):
        outputs = self.backbone(past_values=past_values)
        last_hidden_state = outputs.last_hidden_state
        batch_size, num_channels, _, _ = last_hidden_state.shape
        flattened = last_hidden_state.view(batch_size, num_channels, -1)
        return self.head(flattened)


def get_model(cfg):
    config = PatchTSMixerConfig(
        context_length=cfg['context_len'],
        prediction_length=1,
        num_input_channels=cfg['num_input_channels'],
        d_model=cfg['hidden_dim'],
        num_layers=cfg.get('num_layers', 4),
        dropout=cfg['dropout'],
        patch_length=cfg['patch_len'],
        patch_stride=cfg['stride'],
        gated_attn=False,
        norm_type='layernorm',
        scaling=None
    )
    return PatchTSMixerForecaster(config)

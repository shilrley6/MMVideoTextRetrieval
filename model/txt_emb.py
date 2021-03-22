from base import BaseModel
from model.utils import GatedEmbeddingUnit, ReduceDim

import torch as th
import torch.nn as nn
import torch.nn.functional as F

class TxtProjection(BaseModel):
    """project the extracted text feature into the common space and use gru/bert/other method process the features"""

    def __init__(self, modalities, txt_pro, text_dim, same_dim, normalize_experts, txt_wgh, txt_bert_params):
        """modalities: all modalities used to form video features
           txt_pro: models used to project the text into the common space(gbn:gated with bn, gem: gated without bn, lin: linear)
           text_dim: the feature dimension of extracted text
           same_dim: the feature dimension of common space
           normalize_experts:
           txt_wgh: the method to calculate the text weights"""
        super().__init__()
        self.modalities = modalities
        self.txt_pro = txt_pro
        self.normalize_experts = normalize_experts
        self.txt_wgh = txt_wgh



        self.text_GU = nn.ModuleDict()
        for mod in self.modalities:
          if self.txt_pro == 'gbn':
            self.text_GU[mod] = GatedEmbeddingUnit(text_dim,
                                                   same_dim,
                                                   use_bn=True,
                                                   normalize=self.normalize_experts)
          elif self.txt_pro == 'gem':
            self.text_GU[mod] = GatedEmbeddingUnit(text_dim,
                                                   same_dim,
                                                   use_bn=False,
                                                   normalize=self.normalize_experts)
          elif self.txt_pro == 'lin':
            self.text_GU[mod] = ReduceDim(text_dim, same_dim)

        # Weightening of each modality similarity
        if self.txt_wgh == 'emb':
          self.moe_fc_txt = nn.ModuleDict()
          dout_prob = txt_bert_params['hidden_dropout_prob']
          self.moe_txt_dropout = nn.Dropout(dout_prob)
          for mod in self.modalities:
            self.moe_fc_txt[mod] = nn.Linear(text_dim, 1)


    def forward(self, text, token_ids):
        b, captions_per_video, max_text_words, feat_dim = token_ids.size()
        # From the text representation, compute as many embeddings as there are
        # modalities
        text_embd = {}
        for mod in self.modalities:
          layer = self.text_GU[mod]
          text_ = layer(text)
          text_ = text_.view(b, captions_per_video, -1)
          text_embd[mod] = text_
        text = text.view(b, captions_per_video, -1)

        return text, text_embd

    def compute_txt_weights(self, text, b, captions_per_video, m, device):
        if self.txt_wgh == 'emb':
          text_weights = self.compute_weights_from_emb(text)
        elif self.txt_wgh == 'none':
          text_weights = th.ones(b, captions_per_video, m).to(device)
        else:
          msg = 'txt weighting mode {} not supported'
          raise NotImplementedError(msg.format(self.txt_wgh))

        # Normalize text weights so that they sum to one
        text_weights = nn.functional.normalize(text_weights, p=1, dim=-1)

        return text_weights


    def compute_weights_from_emb(self, embd):
       # Compute the modality weights given an embedding

       # text emb
       if len(embd.size()) == 3:
         embd = self.moe_txt_dropout(embd)
         b, k, d = embd.size()
         m = len(self.modalities)
         embd = embd.view(b * k, d)
         moe_weights = th.cat(
             [self.moe_fc_txt[mod](embd) for mod in self.modalities], dim=-1)
         moe_weights = F.softmax(moe_weights, dim=1)
         moe_weights = moe_weights.view(b, k, m)

       return moe_weights

    def compute_weights_from_norm(self, embds):
      # Compute the modality weights according to their norm

      device = embds[self.modalities[0]].device
      # text emb
      if len(embds[self.modalities[0]].size()) == 3:
        b, k, d = embds[self.modalities[0]].size()
        for idx, mod in self.modalities:
          embds[mod] = embds[mod].view(b * k, d)
        b = b * k

      m = len(self.modalities)
      norm_embd = th.zeros(b, m).to(device)
      for idx, mod in enumerate(self.modalities):
        norm_embd[:, idx] = th.norm(embds[mod], p=2, dim=1)

      sum_norm = th.sum(norm_embd, dim=1)  # b
      sum_norm = sum_norm.unsqueeze(1)  # b x 1

      weights = th.div(norm_embd, sum_norm)

      return weights

# Copyright 2020 Valentin Gabeur
# Copyright 2020 Samuel Albanie, Yang Liu and Arsha Nagrani
# Copyright 2018 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cross-modal Architecture models.

Code based on the implementation of "Collaborative Experts":
https://github.com/albanie/collaborative-experts

Code based on the implementation of "Mixture of Embedding Experts":
https://github.com/antoine77340/Mixture-of-Embedding-Experts
"""

import collections
import itertools
import logging
import re
import types

from base import BaseModel
from model.bert import BertModel
from model.lstm import LSTMModel
from model.net_vlad import NetVLAD
from model.txt_embeddings import TxtEmbeddings
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert import BertModel as TxtBertModel
from utils.util import get_len_sequences

from model.vid_emb import VidProjection
from model.txt_extract import TxtExtract
from model.txt_emb import TxtProjection

logger = logging.getLogger(__name__)


class CENet(BaseModel):
  """Whole cross-modal architecture."""

  def __init__(self,
               l2renorm,
               expert_dims,
               tokenizer,
               keep_missing_modalities,
               test_caption_mode,
               freeze_weights=False,
               mimic_ce_dims=False,
               concat_experts=False,
               concat_mix_experts=False,
               use_experts='origfeat',
               txt_inp=None,
               txt_agg=None,
               txt_pro=None,
               txt_wgh=None,
               vid_inp=None,
               vid_cont=None,
               vid_wgh=None,
               pos_enc=None,
               out_tok=None,
               use_mask='nomask',
               same_dim=512,
               vid_bert_params=None,
               txt_bert_params=None,
               agg_dims=None,
               normalize_experts=True):
    super().__init__()

    self.sanity_checks = False
    modalities = list(expert_dims.keys())
    self.expert_dims = expert_dims
    self.modalities = modalities
    logger.debug(self.modalities)
    self.mimic_ce_dims = mimic_ce_dims
    self.concat_experts = concat_experts
    self.concat_mix_experts = concat_mix_experts
    self.test_caption_mode = test_caption_mode
    self.freeze_weights = freeze_weights
    self.use_experts = use_experts
    self.use_mask = use_mask
    self.keep_missing_modalities = keep_missing_modalities
    self.l2renorm = l2renorm
    self.same_dim = same_dim
    self.txt_inp = txt_inp
    self.txt_agg = txt_agg
    self.txt_pro = txt_pro
    self.txt_wgh = txt_wgh
    self.vid_inp = vid_inp
    self.vid_cont = vid_cont
    self.vid_wgh = vid_wgh
    self.pos_enc = pos_enc
    self.out_tok = out_tok
    self.vid_bert_params = vid_bert_params
    self.normalize_experts = normalize_experts


    self.vid_emb_model = VidProjection(modalities, expert_dims, same_dim, vid_inp, vid_cont, vid_wgh, vid_bert_params, pos_enc, out_tok, keep_missing_modalities)

    self.txt_feat_model = TxtExtract(modalities, txt_agg, txt_inp, txt_bert_params, vid_bert_params)
    self.txt_emb_model = TxtProjection(modalities, txt_pro, self.txt_feat_model.text_dim, same_dim, normalize_experts, txt_wgh, txt_bert_params)

    self.debug_dataloader = False
    if self.debug_dataloader:
      self.tokenizer = tokenizer

  def forward(self,
              token_ids,
              features,
              features_t,
              features_ind,
              features_avgpool,
              features_maxpool,
              query_masks,
              out='conf',
              device=None,
              debug=None):

    self.device = device
    experts_feats = features
    experts_feats_t = features_t
    experts_feats_ind = features_ind
    ind = {}
    for mod in self.modalities:
      ind[mod] = th.max(experts_feats_ind[mod], 1)[0]
    pooled_experts = {}

    for _, mod in enumerate(self.modalities):
      pooled_experts[f'{mod}_avgpool'] = features_avgpool[mod]
      pooled_experts[f'{mod}_maxpool'] = features_maxpool[mod]

    # Notation: B = batch size, M = number of modalities

    text = self.txt_feat_model(token_ids, device)
    text, text_embd = self.txt_emb_model(text, token_ids)
    text_weights = self.txt_emb_model.compute_txt_weights(text, token_ids.size()[0], token_ids.size()[1], len(self.modalities), device)

    vid_embd, experts = self.vid_emb_model(pooled_experts, experts_feats, experts_feats_t, ind, token_ids.size()[0], experts_feats_ind, device)
    vid_weights = self.vid_emb_model.compute_vid_wights(experts, vid_embd, ind, token_ids.size()[0], len(self.modalities), device)

    # L2 Normalization of each expert
    if self.normalize_experts:
      for _, modality in enumerate(self.modalities):
        experts[modality] = nn.functional.normalize(experts[modality], dim=-1)
        text_embd[modality] = nn.functional.normalize(text_embd[modality],
                                                      dim=-1)

    if self.training:
      merge_caption_similiarities = 'avg'
    else:
      merge_caption_similiarities = self.test_caption_mode
    self.merge_caption_similarities = merge_caption_similiarities

    if out == 'conf':  # Output confusion matrix
      cross_view_conf_matrix = sharded_cross_view_inner_product(
          vid_embds=experts,
          text_embds=text_embd,
          vid_weights=vid_weights,
          text_weights=text_weights,
          subspaces=self.modalities,
          merge_caption_similiarities=merge_caption_similiarities,
      )
      return {
          'modalities': self.modalities,
          'cross_view_conf_matrix': cross_view_conf_matrix,
      }
    else:  # Output the embeddings
      # Transform the dictionaries into tensors
      vid_embds_list = []
      text_embds_list = []
      for idx, mod in enumerate(self.modalities):
        vid_embds_list.append(experts[mod].unsqueeze(1))
        text_embds_list.append(text_embd[mod].unsqueeze(1))
      vid_embds = th.cat(vid_embds_list, 1)
      text_embds = th.cat(text_embds_list, 1)

      return {
          'vid_embds': vid_embds,
          'text_embds': text_embds,
          'vid_weights': vid_weights,
          'text_weights': text_weights,
      }

  def display_minibatch(self, token_ids, input_ids, attention_mask,
                        token_type_ids, position_ids, features):
    for i in range(1):
      logger.debug()
      # logger.debug(f'Sample {i}:')
      logger.debug('Text:')
      ids = token_ids[i, 0, :, 0].cpu().numpy()
      logger.debug(ids)

      tokens = self.tokenizer.convert_ids_to_tokens(ids)
      logger.debug(tokens)

      logger.debug('Video:')
      # logger.debug(f'input_ids: {input_ids[i]}')
      # logger.debug(f'attention_mask: {attention_mask[i]}')
      # logger.debug(f'token_type_ids: {token_type_ids[i]}')
      # logger.debug(f'position_ids: {position_ids[i]}')
      logger.debug(features[i].shape)


def sharded_cross_view_inner_product(vid_embds,
                                     text_embds,
                                     vid_weights,
                                     text_weights,
                                     subspaces,
                                     merge_caption_similiarities='avg'):
  """Compute similarities between all captions and videos."""

  b = vid_embds[subspaces[0]].size(0)
  device = vid_embds[subspaces[0]].device
  num_caps = text_embds[subspaces[0]].size(1)
  m = len(subspaces)

  # unroll separate captions onto first dimension and treat them separately
  sims = th.zeros(b * num_caps, b, device=device)

  text_weights = text_weights.view(b * num_caps, -1)
  vid_weights = vid_weights.view(b, -1)

  moe_weights = vid_weights[None, :, :] * text_weights[:, None, :]

  import numpy as np
  import random
  moe_file = "/youtu_pedestrian_detection/tayllorliu/mmt/mmt_base/save_dir/weights/" + str(random.random()) + ".npy"
  np.save(moe_file, moe_weights.cpu().detach().numpy())
  print(moe_file)

  norm_weights = th.sum(moe_weights, dim=2)
  norm_weights = norm_weights.unsqueeze(2)
  # If only one modality is used and is missing in some videos, moe_weights will
  # be zero.
  # To avoid division by zero, replace zeros by epsilon
  # (or anything else, in that case moe_weights are zero anyway)
  norm_weights[norm_weights == 0] = 1E-5
  moe_weights = th.div(moe_weights, norm_weights)

  assert list(moe_weights.size()) == [b * num_caps, b, m]

  for idx, mod in enumerate(subspaces):
    text_embds[mod] = text_embds[mod].view(b * num_caps, -1)
    sims += moe_weights[:, :, idx] * th.matmul(text_embds[mod],
                                               vid_embds[mod].t())

  if num_caps > 1:
    # aggregate similarities from different captions
    if merge_caption_similiarities == 'avg':
      sims = sims.view(b, num_caps, b)
      sims = th.mean(sims, dim=1)
      sims = sims.view(b, b)
    elif merge_caption_similiarities == 'indep':
      pass
    else:
      msg = 'unrecognised merge mode: {}'
      raise ValueError(msg.format(merge_caption_similiarities))
  return sims

import types
import itertools
import collections

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from model.bert import BertModel
from model.utils import GatedEmbeddingUnit, ReduceDim, GatedEmbeddingUnitReasoning


class VidProjection(BaseModel):
    """project the extracted video feature into the common space and use gru/bert/other method process the features"""


    def __init__(self, modalities, expert_dims, same_dim, vid_inp, vid_cont, vid_wgh, vid_bert_params, pos_enc, out_tok, keep_missing_modalities):
        """modalities: all modalities used to form video features
           expert_dims: dict, the feature dimension for each modality
           same_dim: the dimension of the common space
           vid_inp: video
           vid_cont: the model used to embed the features (coll: collaborative gating; bert)
           vid_wgh: the method to compute the weight
           pos_enc: used in vid_cont=bert"""
        super().__init__()

        self.modalities = modalities
        self.expert_dims = expert_dims
        self.same_dim = same_dim
        self.vid_inp = vid_inp
        self.vid_cont = vid_cont
        self.vid_wgh = vid_wgh
        self.vid_bert_params = vid_bert_params
        self.pos_enc = pos_enc
        self.out_tok = out_tok
        self.keep_missing_modalities = keep_missing_modalities

        self.video_dim_reduce = nn.ModuleDict()
        for mod in self.modalities:
          in_dim = expert_dims[mod]['dim']
          if self.vid_inp in ['agg', 'both', 'all', 'temp']:
            self.video_dim_reduce[mod] = ReduceDim(in_dim, same_dim)

        if self.vid_cont == 'coll':
          self.g_reason_1 = nn.Linear(same_dim * 2, same_dim)
          dout_prob = vid_bert_params['hidden_dropout_prob']
          self.coll_g_dropout = nn.Dropout(dout_prob)
          self.g_reason_2 = nn.Linear(same_dim, same_dim)

          self.f_reason_1 = nn.Linear(same_dim, same_dim)
          self.coll_f_dropout = nn.Dropout(dout_prob)
          self.f_reason_2 = nn.Linear(same_dim, same_dim)
          self.f_reason_3 = nn.Linear(same_dim, same_dim)
          self.batch_norm_g1 = nn.BatchNorm1d(same_dim)
          self.batch_norm_g2 = nn.BatchNorm1d(same_dim)

          self.batch_norm_f1 = nn.BatchNorm1d(same_dim)
          self.batch_norm_f2 = nn.BatchNorm1d(same_dim)

          self.video_GU = nn.ModuleDict()
          for mod in self.modalities:
            self.video_GU[mod] = GatedEmbeddingUnitReasoning(same_dim)

        # If Bert architecture is employed for video
        elif self.vid_cont == 'bert':
          vid_bert_config = types.SimpleNamespace(**vid_bert_params)
          self.vid_bert = BertModel(vid_bert_config)

        elif self.vid_cont == 'none':
          pass

        if self.vid_wgh == 'emb':
          self.moe_fc_vid = nn.ModuleDict()
          dout_prob = vid_bert_params['hidden_dropout_prob']
          self.moe_vid_dropout = nn.Dropout(dout_prob)
          for mod in self.modalities:
            self.moe_fc_vid[mod] = nn.Linear(self.same_dim, 1)

    def forward(self, pooled_experts, experts_feats, experts_feats_t, ind, b, experts_feats_ind, device):
        """pooled_experts: dict, pooled features for each modality
           ind: experts_feature_ind > 1 in the dimension = 1
           b: batch_size"""

        # Output experts
        experts = collections.OrderedDict()
        vid_embd = None

        if self.vid_inp in ['agg', 'both', 'all']:
          agg_experts = collections.OrderedDict()
          mnp_experts = collections.OrderedDict()
          maxp_experts = collections.OrderedDict()

          # Embed all features to a common dimension
          for mod in self.modalities:
            layer = self.video_dim_reduce[mod]
            mnp_experts[mod] = layer(pooled_experts[f'{mod}_avgpool'])
            maxp_experts[mod] = layer(pooled_experts[f'{mod}_maxpool'])

          for mod in self.modalities:
            agg_experts[mod] = maxp_experts[mod]

        if self.vid_inp in ['both', 'temp', 'all']:
          for mod in self.modalities:
            layer = self.video_dim_reduce[mod]
            experts_feats[mod] = layer(experts_feats[mod])

        if self.vid_cont in ['none', 'coll']:
          for _, modality in enumerate(self.modalities):
            experts[modality] = agg_experts[modality]

        # If we use collaborative gating to compute a mask (called T in the
        # Collaborative experts paper)
        if self.vid_cont == 'coll':
          masks = {}
          all_combinations = list(itertools.permutations(agg_experts, 2))
          assert len(self.modalities) > 1, 'use_ce requires multiple modalities'

          for _, modality in enumerate(self.modalities):

            mask_num = 0
            curr_mask = 0
            temp_dict = {}
            avai_dict = {}

            for modality_pair in all_combinations:
              mod0, mod1 = modality_pair
              if mod0 == modality:
                new_key = '_'.join(modality_pair)
                fused = th.cat((agg_experts[mod0], agg_experts[mod1]),
                               1)  # -> B x 2D
                temp = self.g_reason_1(fused)  # B x 2D -> B x D
                temp = self.coll_g_dropout(temp)
                temp = self.g_reason_2(F.relu(temp))  # B x D -> B x D
                temp_dict[new_key] = temp
                avail = (ind[mod0].float() * ind[mod1].float()).to(device)
                avai_dict[new_key] = avail

            # Combine the paired features into a mask through elementwise summation
            for mm in temp_dict:
              curr_mask += temp_dict[mm] * avai_dict[mm].unsqueeze(1)
              mask_num += avai_dict[mm]

            curr_mask = th.div(curr_mask, (mask_num + 0.00000000001).unsqueeze(1))
            curr_mask = self.f_reason_1(curr_mask)
            curr_mask = self.coll_f_dropout(curr_mask)
            curr_mask = self.f_reason_2(F.relu(curr_mask))
            masks[modality] = curr_mask

            mod_gu = self.video_GU[modality]
            experts[modality] = mod_gu(experts[modality], masks[modality])

        # If Bert architecture is employed
        if self.vid_cont == 'bert':
          # 0=[CLS] 1=[SEP] 2=[AGG] 3=[MAXP] 4=[MNP] 5=[VLAD] 6=[FEA]
          input_ids_list = []
          token_type_ids_list = []  # Modality id
          # Position (0 = no position, 1 = unknown, >1 = valid position)
          position_ids_list = []
          features_list = []  # Semantics
          attention_mask_list = []  # Valid token or not

          modality_to_tok_map = collections.OrderedDict()

          # [CLS] token
          tok_id = 0
          ids_size = (b,)
          input_ids_list.append(th.full(ids_size, 0, dtype=th.long))
          token_type_ids_list.append(th.full(ids_size, 0, dtype=th.long))
          position_ids_list.append(th.full(ids_size, 0, dtype=th.long).to(device))
          features_list.append(
              th.full((b, self.same_dim), 0, dtype=th.float).to(device))
          attention_mask_list.append(th.full(ids_size, 1, dtype=th.long).to(device))

          # Number of temporal tokens per modality
          if self.vid_inp in ['temp', 'both', 'all']:
            max_expert_tokens = collections.OrderedDict()
            for _, modality in enumerate(self.modalities):
              max_expert_tokens[modality] = experts_feats[modality].size()[1]

          # Make the features_t and raw_captions_t start at the minimum value
          if self.pos_enc == 'tint':

            # Clamp the position encoding to [0, max_position_embedding - 1]
            max_pos = self.vid_bert_params['max_position_embeddings'] - 1
            for _, modality in enumerate(self.modalities):
              experts_feats_t[modality].clamp_(min=0, max=max_pos)
              experts_feats_t[modality] = experts_feats_t[modality].long().to(
                  device)

          for _, modality in enumerate(self.modalities):
            token_type = self.expert_dims[modality]['idx']

            # Add an aggregated feature token
            if self.vid_inp in ['agg', 'both', 'all']:
              tok_id += 1
              modality_to_tok_map[modality] = tok_id
              input_ids_list.append(th.full(ids_size, 2, dtype=th.long))
              token_type_ids_list.append(
                  th.full(ids_size, token_type, dtype=th.long))
              position_ids_list.append(
                  th.full(ids_size, 0, dtype=th.long).to(device))
              if self.out_tok == 'sep':
                features_list.append(
                    th.full((b, self.same_dim), 0, dtype=th.float).to(device))
              elif self.out_tok == 'mxp':
                features_list.append(maxp_experts[modality])
              elif self.out_tok == 'mnp':
                features_list.append(mnp_experts[modality])
              attention_mask_list.append(ind[modality].to(dtype=th.long).to(device))
            if self.vid_inp in ['temp', 'both', 'all']:
              for frame_id in range(max_expert_tokens[modality]):
                if self.pos_enc == 'ordr':
                  position_ids_list.append(
                      th.full(ids_size, frame_id + 1, dtype=th.long).to(device))
                elif self.pos_enc == 'tint':
                  position_ids_list.append(experts_feats_t[modality][:, frame_id])
                elif self.pos_enc == 'type':
                  position_ids_list.append(
                      th.full(ids_size, 1, dtype=th.long).to(device))
                tok_id += 1
                input_ids_list.append(th.full(ids_size, 6, dtype=th.long))
                token_type_ids_list.append(
                    th.full(ids_size, token_type, dtype=th.long))
                features_list.append(experts_feats[modality][:, frame_id, :])
                attention_mask_list.append(
                    experts_feats_ind[modality][:, frame_id].to(dtype=th.long))

          features = th.stack(features_list, dim=1).to(device)
          input_ids = th.stack(input_ids_list, dim=1).to(device)
          token_type_ids = th.stack(token_type_ids_list, dim=1).to(device)
          if self.pos_enc != 'none':
            position_ids = th.stack(position_ids_list, dim=1).to(device)
          else:
            position_ids = None
          attention_mask = th.stack(attention_mask_list, dim=1).to(device)

          vid_bert_output = self.vid_bert(input_ids,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids,
                                          position_ids=position_ids,
                                          features=features)

          last_layer = vid_bert_output[0]
          vid_embd = last_layer[:, 0]

          for _, modality in enumerate(self.modalities):
            experts[modality] = last_layer[:, modality_to_tok_map[modality]]
        return vid_embd, experts

    def compute_vid_wights(self, experts, vid_embd, ind, b, m, device, ):
        if self.vid_wgh == 'nrm':
          vid_weights = self.compute_weights_from_norm(experts)
        elif self.vid_wgh == 'emb':
          vid_weights = self.compute_weights_from_emb(vid_embd)
        elif self.vid_wgh == 'none':
          vid_weights = th.ones(b, m).to(device)
        else:
          msg = 'video weighting mode {} not supported'
          raise NotImplementedError(msg.format(self.vid_wgh))

        if not self.keep_missing_modalities:
          # Zero padding of the missing modalities
          available = th.zeros(b, m).to(device)
          for idx, mod in enumerate(self.modalities):
            available[:, idx] = ind[mod].float()  # B x M
          vid_weights = vid_weights * available

        # Normalize video weights so that they sum to one
        vid_weights = nn.functional.normalize(vid_weights, p=1, dim=-1)
        return vid_weights

    def compute_weights_from_emb(self, embd):
      # Compute the modality weights given an embedding

      # vid emb
      if len(embd.size()) == 2:
        embd = self.moe_vid_dropout(embd)
        moe_weights = th.cat(
            [self.moe_fc_vid[mod](embd) for mod in self.modalities], dim=-1)
        moe_weights = F.softmax(moe_weights, dim=1)

      return moe_weights

    def compute_weights_from_norm(self, embds):
      # Compute the modality weights according to their norm

      device = embds[self.modalities[0]].device
      # vid emb
      if len(embds[self.modalities[0]].size()) == 2:
        b, d = embds[self.modalities[0]].size()

      m = len(self.modalities)
      norm_embd = th.zeros(b, m).to(device)
      for idx, mod in enumerate(self.modalities):
        norm_embd[:, idx] = th.norm(embds[mod], p=2, dim=1)

      sum_norm = th.sum(norm_embd, dim=1)  # b
      sum_norm = sum_norm.unsqueeze(1)  # b x 1

      weights = th.div(norm_embd, sum_norm)

      return weights

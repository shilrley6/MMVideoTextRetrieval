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
"""Training losses.

Code based on the implementation of "Collaborative Experts":
https://github.com/albanie/collaborative-experts

Code based on the implementation of "Mixture of Embedding Experts":
https://github.com/antoine77340/Mixture-of-Embedding-Experts
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init


class MaxMarginRankingLoss(nn.Module):
    """Implementation of the Max-margin ranking loss."""

    def __init__(self, margin=1, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = th.nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x):
        n = x.size()[0]

        x1 = th.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = th.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = th.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = th.ones(x.shape) - th.eye(x.shape[0])
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = th.nonzero(th.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                self.device = x1.device
                keep_idx = keep_idx.to(self.device)
            x1_ = th.index_select(x1, dim=0, index=keep_idx)
            x2_ = th.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))
        return max_margin.mean()


class InfoNceLoss(nn.Module):
    """Implementation of the noise-constrastive estimation loss."""

    def __init__(self):
        super().__init__()
        self.loss = th.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        n = x.size()[0]
        target = th.arange(n)
        if x.is_cuda:
            self.device = x.device
            target = target.to(x.device)

        return self.loss(x, target) + self.loss(th.transpose(x, 0, 1), target)


class Contrastive_Loss(nn.Module):
    """docstring for contrastive_loss"""

    def __init__(self):
        super(Contrastive_Loss, self).__init__()
        self.lambda_softmax = 20
        self.epsilon = 1e-08
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, sim):
        self.batch_size = sim.shape[0]
        self.device = sim.device
        self.labels = th.arange(self.batch_size).to(self.device)
        i2t_pred = F.softmax(sim * self.lambda_softmax, dim=1)

        contrastive_loss = self.criterion(i2t_pred, self.labels)

        return contrastive_loss


class MaxMarginContrastive_Loss(nn.Module):
    def __init__(self, margin=0.05, cross_attn="t2i", fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = th.nn.MarginRankingLoss(margin)
        self.margin = margin
        self.lambda_softmax = 20
        self.epsilon = 1e-08
        self.criterion = nn.CrossEntropyLoss()
        self.contrastive = ContrastiveLoss_RAM(cross_attn=cross_attn)

    def forward(self, x, video_features, text_features):
        sim = x.clone()
        self.batch_size = sim.shape[0]
        self.device = sim.device
        self.labels = th.arange(self.batch_size).to(self.device)
        i2t_pred = F.softmax(sim * self.lambda_softmax, dim=1)

        contrastive_loss = self.criterion(i2t_pred, self.labels)

        x1 = th.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(self.batch_size, self.batch_size)
        x1 = x1.contiguous().view(-1, 1)
        x1 = th.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = th.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = th.ones(x.shape) - th.eye(x.shape[0])
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = th.nonzero(th.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                self.device = x1.device
                keep_idx = keep_idx.to(self.device)
            x1_ = th.index_select(x1, dim=0, index=keep_idx)
            x2_ = th.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        words_l = torch.ones(32, dtype=int) * 30
        contrastive_align = self.contrastive(video_features, text_features, words_l)

        return max_margin.mean() + 1e-4 * contrastive_align
        # return max_margin.mean()


class MILNCELoss(th.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[0], -1)
        nominator = x * th.eye(x.shape[0])[:, :, None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = th.logsumexp(nominator, dim=1)
        denominator = th.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        denominator = th.logsumexp(denominator, dim=1)
        return th.mean(denominator - nominator)


class ContrastiveLoss(nn.Module):
    """
  Compute contrastive loss
  """

    def __init__(self, margin=0.2, cross_attn="i2t", max_violation=True, lambda_lse=6., lambda_softmax=9., eps=1e-8):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.lambda_lse = lambda_lse
        self.lambda_softmax = lambda_softmax
        self.eps = eps
        self.raw_feature_norm = "clipped_l2norm"
        self.agg_func = "LogSumExp"
        self.cross_attn = cross_attn

    def l1norm(self, X, dim):
        """
    L1-normalize columns of X
    """
        norm = torch.abs(X).sum(dim=dim, keepdim=True) + self.eps
        X = torch.div(X, norm)
        return X

    def l2norm(self, X, dim):
        """
    L2-normalize columns of X
    """
        norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + self.eps
        X = torch.div(X, norm)
        return X

    def func_attention(self, query, context, smooth):
        """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
        batch_size_q, queryL = query.size(0), query.size(1)
        batch_size, sourceL = context.size(0), context.size(1)

        # Get attention
        # --> (batch, d, queryL)
        queryT = torch.transpose(query, 1, 2)

        # (batch, sourceL, d)(batch, d, queryL)
        # --> (batch, sourceL, queryL)
        attn = torch.bmm(context, queryT)
        if self.raw_feature_norm == "softmax":
            # --> (batch*sourceL, queryL)
            attn = attn.view(batch_size * sourceL, queryL)
            attn = nn.Softmax()(attn)
            # --> (batch, sourceL, queryL)
            attn = attn.view(batch_size, sourceL, queryL)
        elif self.raw_feature_norm == "l2norm":
            attn = self.l2norm(attn, 2)
        elif self.raw_feature_norm == "clipped_l2norm":
            attn = nn.LeakyReLU(0.1)(attn)
            attn = self.l2norm(attn, 2)
        elif self.raw_feature_norm == "l1norm":
            attn = self.l1norm(attn, 2)
        elif self.raw_feature_norm == "clipped_l1norm":
            attn = nn.LeakyReLU(0.1)(attn)
            attn = self.l1norm(attn, 2)
        elif self.raw_feature_norm == "clipped":
            attn = nn.LeakyReLU(0.1)(attn)
        elif self.raw_feature_norm == "no_norm":
            pass
        else:
            raise ValueError("unknown first norm type:", self.raw_feature_norm)
        # --> (batch, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        # --> (batch*queryL, sourceL)
        attn = attn.view(batch_size * queryL, sourceL)
        attn = nn.Softmax(dim=1)(attn * smooth)
        # --> (batch, queryL, sourceL)
        attn = attn.view(batch_size, queryL, sourceL)
        # --> (batch, sourceL, queryL)
        attnT = torch.transpose(attn, 1, 2).contiguous()

        # --> (batch, d, sourceL)
        contextT = torch.transpose(context, 1, 2)
        # (batch x d x sourceL)(batch x sourceL x queryL)
        # --> (batch, d, queryL)
        weightedContext = torch.bmm(contextT, attnT)
        # --> (batch, queryL, d)
        weightedContext = torch.transpose(weightedContext, 1, 2)

        return weightedContext, attnT

    def cosine_similarity(self, x1, x2, dim=1):
        """
    Returns cosine similarity between x1 and x2, computed along dim.
    """
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=self.eps)).squeeze()

    def xattn_score_t2i(self, images, captions, cap_lens):
        """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
        similarities = []
        n_image = images.size(0)
        n_caption = captions.size(0)
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i].item()
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            """
      word(query): (n_image, n_word, d)
      image(context): (n_image, n_regions, d)
      weiContext: (n_image, n_word, d)
      attn: (n_image, n_region, n_word)
      """
            weiContext, attn = self.func_attention(cap_i_expand, images, smooth=self.lambda_softmax)
            # (n_image, n_word)
            row_sim = self.cosine_similarity(cap_i_expand, weiContext, dim=2)
            if self.agg_func == 'LogSumExp':
                row_sim.mul_(self.lambda_lse).exp_()
                row_sim = row_sim.sum(dim=1, keepdim=True)
                row_sim = torch.log(row_sim) / self.lambda_lse
            elif self.agg_func == 'Max':
                row_sim = row_sim.max(dim=1, keepdim=True)[0]
            elif self.agg_func == 'Sum':
                row_sim = row_sim.sum(dim=1, keepdim=True)
            elif self.agg_func == 'Mean':
                row_sim = row_sim.mean(dim=1, keepdim=True)
            else:
                raise ValueError("unknown aggfunc: {}".format(self.agg_func))
            similarities.append(row_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)
        return similarities

    def xattn_score_i2t(self, images, captions, cap_lens):
        """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
        similarities = []
        n_image = images.size(0)
        n_caption = captions.size(0)
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i].item()
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            # (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            """
      word(query): (n_image, n_word, d)
      image(context): (n_image, n_region, d)
      weiContext: (n_image, n_region, d)
      attn: (n_image, n_word, n_region)
      """
            weiContext, attn = self.func_attention(images, cap_i_expand, smooth=self.lambda_softmax)
            # (n_image, n_region)
            row_sim = self.cosine_similarity(images, weiContext, dim=2)
            if self.agg_func == 'LogSumExp':
                row_sim.mul_(self.lambda_lse).exp_()
                row_sim = row_sim.sum(dim=1, keepdim=True)
                row_sim = torch.log(row_sim) / self.lambda_lse
            elif self.agg_func == 'Max':
                row_sim = row_sim.max(dim=1, keepdim=True)[0]
            elif self.agg_func == 'Sum':
                row_sim = row_sim.sum(dim=1, keepdim=True)
            elif self.agg_func == 'Mean':
                row_sim = row_sim.mean(dim=1, keepdim=True)
            else:
                raise ValueError("unknown aggfunc: {}".format(self.agg_func))
            similarities.append(row_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)
        return similarities

    def forward(self, im, s, s_l):
        # compute image-sentence score matrix
        if self.cross_attn == 'i2t':
            scores = self.xattn_score_i2t(im, s, s_l)
        elif self.cross_attn == 't2i':
            scores = self.xattn_score_t2i(im, s, s_l)
        elif self.cross_attn == 'both':
            scores = self.xattn_score_i2t(im, s, s_l) + self.xattn_score_t2i(im, s, s_l)
        else:
            raise ValueError("unknown first norm type:", self.cross_attn)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        I = (torch.eye(scores.size(0)) > .5).cuda()
        cost_s = cost_s.cuda().masked_fill_(I, 0)
        cost_im = cost_im.cuda().masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class ContrastiveLoss_RAM(nn.Module):
    """
  Stacked Cross Attention Network (SCAN) model
  """

    def __init__(self, embed_size=1024, eps=1e-8, iteration_step=3, lambda_softmax=9, cross_attn="t2i",
                 raw_feature_norm="clipped_l2norm", no_IMRAM_norm=True):
        super(ContrastiveLoss_RAM, self).__init__()
        # Build Models
        self.embed_size = embed_size
        self.eps = eps
        self.raw_feature_norm = raw_feature_norm
        self.iteration_step = iteration_step
        self.no_IMRAM_norm = no_IMRAM_norm
        self.lambda_softmax = lambda_softmax
        self.cross_attn = cross_attn
        self.max_violation = True
        self.margin = 0.2

        self.linear_t2i = nn.Linear(self.embed_size * 2, self.embed_size).cuda()
        self.gate_t2i = nn.Linear(self.embed_size * 2, self.embed_size).cuda()
        self.linear_i2t = nn.Linear(self.embed_size * 2, self.embed_size).cuda()
        self.gate_i2t = nn.Linear(self.embed_size * 2, self.embed_size).cuda()

    def l1norm(self, X, dim):
        """
    L1-normalize columns of X
    """
        norm = torch.abs(X).sum(dim=dim, keepdim=True) + self.eps
        X = torch.div(X, norm)
        return X

    def l2norm(self, X, dim):
        """
    L2-normalize columns of X
    """
        norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + self.eps
        X = torch.div(X, norm)
        return X

    def func_attention(self, query, context, smooth, weight=None):
        """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
        batch_size_q, queryL = query.size(0), query.size(1)
        batch_size, sourceL = context.size(0), context.size(1)

        # Get attention
        # --> (batch, d, queryL)
        queryT = torch.transpose(query, 1, 2)

        # (batch, sourceL, d)(batch, d, queryL)
        # --> (batch, sourceL, queryL)
        attn = torch.bmm(context, queryT)
        if self.raw_feature_norm == "softmax":
            # --> (batch*sourceL, queryL)
            attn = attn.view(batch_size * sourceL, queryL)
            attn = F.softmax(attn, dim=1)
            # --> (batch, sourceL, queryL)
            attn = attn.view(batch_size, sourceL, queryL)
        elif self.raw_feature_norm == "l2norm":
            attn = self.l2norm(attn, 2)
        elif self.raw_feature_norm == "clipped_l2norm":
            attn = nn.LeakyReLU(0.1)(attn)
            attn = self.l2norm(attn, 2)
        elif self.raw_feature_norm == "l1norm":
            attn = self.l1norm(attn, 2)
        elif self.raw_feature_norm == "clipped_l1norm":
            attn = nn.LeakyReLU(0.1)(attn)
            attn = self.l1norm(attn, 2)
        elif self.raw_feature_norm == "clipped":
            attn = nn.LeakyReLU(0.1)(attn)
        elif self.raw_feature_norm == "no_norm":
            pass
        else:
            raise ValueError("unknown first norm type:", self.raw_feature_norm)

        if weight is not None:
            attn = attn + weight

        attn_out = attn.clone()

        # --> (batch, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        # --> (batch*queryL, sourceL)
        attn = attn.view(batch_size * queryL, sourceL)
        attn = F.softmax(attn * smooth, dim=1)
        # --> (batch, queryL, sourceL)
        attn = attn.view(batch_size, queryL, sourceL)
        # --> (batch, sourceL, queryL)
        attnT = torch.transpose(attn, 1, 2).contiguous()

        # --> (batch, d, sourceL)
        contextT = torch.transpose(context, 1, 2)
        # (batch x d x sourceL)(batch x sourceL x queryL)
        # --> (batch, d, queryL)
        weightedContext = torch.bmm(contextT, attnT)
        # --> (batch, queryL, d)
        weightedContext = torch.transpose(weightedContext, 1, 2)

        return weightedContext, attn_out

    def cosine_similarity_a2a(self, x1, x2, dim=1):
        # x1: (B, n, d) x2: (B, m, d)
        w12 = torch.bmm(x1, x2.transpose(1, 2))
        # w12: (B, n, m)
        w1 = torch.norm(x1, 2, dim).unsqueeze(2)
        w2 = torch.norm(x2, 2, dim).unsqueeze(1)
        # w1: (B, n, 1) w2: (B, 1, m)
        w12_norm = torch.bmm(w1, w2).clamp(min=self.eps)
        return w12 / w12_norm

    def cosine_similarity(self, x1, x2, dim=1):
        """Returns cosine similarity between x1 and x2, computed along dim."""
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=self.eps))

    def gated_memory_t2i(self, input_0, input_1):
        input_cat = torch.cat([input_0, input_1], 2).cuda()
        input_1 = F.tanh(self.linear_t2i(input_cat))
        gate = torch.sigmoid(self.gate_t2i(input_cat))
        output = input_0 * gate + input_1 * (1 - gate)

        return output

    def gated_memory_i2t(self, input_0, input_1):
        input_cat = torch.cat([input_0, input_1], 2).cuda()
        input_1 = F.tanh(self.linear_i2t(input_cat))
        gate = torch.sigmoid(self.gate_i2t(input_cat))
        output = input_0 * gate + input_1 * (1 - gate)

        return output

    def xattn_score_Text_IMRAM(self, images, captions_all, cap_lens):
        """
    Images: (n_image, n_regions, d) matrix of images
    captions_all: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
        similarities = [[] for _ in range(self.iteration_step)]
        n_image = images.size(0)
        n_caption = captions_all.size(0)
        images = images.float()
        captions_all = captions_all.float()
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            query = cap_i_expand
            context = images
            for j in range(self.iteration_step):
                # "feature_update" by default:
                attn_feat, _ = self.func_attention(query, context, smooth=self.lambda_softmax)

                row_sim = self.cosine_similarity(cap_i_expand, attn_feat, dim=2)
                row_sim = row_sim.mean(dim=1, keepdim=True)
                similarities[j].append(row_sim)

                query = self.gated_memory_t2i(query, attn_feat)

                if not self.no_IMRAM_norm:
                    query = self.l2norm(query, dim=-1)

        # (n_image, n_caption)
        new_similarities = []
        for j in range(self.iteration_step):
            if len(similarities[j]) == 0:
                new_similarities.append([])
                continue
            similarities_one = torch.cat(similarities[j], 1).double()
            if self.training:
                similarities_one = similarities_one.transpose(0, 1)
            new_similarities.append(similarities_one)

        return new_similarities

    def xattn_score_Image_IMRAM(self, images, captions_all, cap_lens):
        """
    Images: (batch_size, n_regions, d) matrix of images
    captions_all: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
        similarities = [[] for _ in range(self.iteration_step)]
        n_image = images.size(0)
        n_caption = captions_all.size(0)
        images = images.float()
        captions_all = captions_all.float()
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            query = images
            context = cap_i_expand
            for j in range(self.iteration_step):
                attn_feat, _ = self.func_attention(query, context, smooth=self.lambda_softmax)

                row_sim = self.cosine_similarity(images, attn_feat, dim=2)
                row_sim = row_sim.mean(dim=1, keepdim=True)
                similarities[j].append(row_sim)

                query = self.gated_memory_i2t(query, attn_feat)

                if not self.no_IMRAM_norm:
                    query = self.l2norm(query, dim=-1)

        # (n_image, n_caption)
        new_similarities = []
        for j in range(self.iteration_step):
            if len(similarities[j]) == 0:
                new_similarities.append([])
                continue
            similarities_one = torch.cat(similarities[j], 1).double()
            if self.training:
                similarities_one = similarities_one.transpose(0, 1)
            new_similarities.append(similarities_one)

        return new_similarities

    def forward(self, img_emb, cap_emb, cap_lens):
        """Compute the loss given pairs of image and caption embeddings
    """
        # compute image-sentence score matrix
        if self.cross_attn == "both":
            scores_t2i = self.xattn_score_Text_IMRAM(img_emb, cap_emb, cap_lens)
            scores_i2t = self.xattn_score_Image_IMRAM(img_emb, cap_emb, cap_lens)
            scores_t2i = torch.stack(scores_t2i, 0).sum(0)
            scores_i2t = torch.stack(scores_i2t, 0).sum(0)
            score = scores_t2i + scores_i2t
        elif self.cross_attn == "i2t":
            scores_i2t = self.xattn_score_Image_IMRAM(img_emb, cap_emb, cap_lens)
            scores_i2t = torch.stack(scores_i2t, 0).sum(0)
            score = scores_i2t
        elif self.cross_attn == "t2i":
            scores_t2i = self.xattn_score_Text_IMRAM(img_emb, cap_emb, cap_lens)

            scores_t2i = torch.stack(scores_t2i, 0).sum(0)
            score = scores_t2i
        else:
            raise NotImplementedError

        import numpy as np
        import random
        score_file = "/youtu_pedestrian_detection/wenzhewang/mmt_modified_ram/scores/" + str(
            random.random()) + ".npy"
        np.save(score_file, torch.stack(score, 0).cpu().detach().numpy())
        print(score_file)

        diagonal = score.diag().view(img_emb.size(0), 1)
        d1 = diagonal.expand_as(score)
        d2 = diagonal.t().expand_as(score)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + score - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + score - d2).clamp(min=0)

        # clear diagonals
        I = (torch.eye(score.size(0)) > .5).cuda()
        cost_s = cost_s.cuda().masked_fill_(I, 0)
        cost_im = cost_im.cuda().masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()

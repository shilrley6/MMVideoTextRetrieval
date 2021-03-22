
from base import BaseModel

from model.bert import BertModel
from model.lstm import LSTMModel
from model.net_vlad import NetVLAD
from model.txt_embeddings import TxtEmbeddings

from transformers.modeling_bert import BertModel as TxtBertModel

import torch as th
import re

class TxtExtract(BaseModel):
    """extract text features from original caption"""

    def __init__(self, modalities, txt_agg, txt_inp, txt_bert_params, vid_bert_params):
        """modalities: all modalities used in video
           txt_agg: txt aggression method for bert: bert+(state: ftn: finetune, frz: freeze), for other: text pooling
           txt_inp: the way to construct the embeddings from word, position and token_type embeddings"""
        super().__init__()

        self.modalities = modalities
        self.txt_agg = txt_agg
        self.txt_inp = txt_inp
        self.vid_bert_params = vid_bert_params

        if self.txt_agg[:4] in ['bert']:
          z = re.match(r'bert([a-z]{3})(\d*)(\D*)', txt_agg)
          assert z
          state = z.groups()[0]
          freeze_until = z.groups()[1]

          # Post aggregation: Use [CLS] token ("cls") or aggregate all tokens
          # (mxp, mnp)
          if z.groups()[2] and z.groups()[2] != 'cls':
            self.post_agg = z.groups()[2]
          else:
            self.post_agg = 'cls'

          if state in ['ftn', 'frz']:
            # State is finetune or frozen, we use a pretrained bert model
            txt_bert_config = 'bert-base-cased'

            # Overwrite config
            if txt_bert_params is None:
              dout_prob = vid_bert_params['hidden_dropout_prob']
              txt_bert_params = {
                  'hidden_dropout_prob': dout_prob,
                  'attention_probs_dropout_prob': dout_prob,
              }
              
            self.txt_bert = TxtBertModel.from_pretrained(txt_bert_config, cache_dir='/youtu_pedestrian_detection/wenzhewang/mmt_data/cache_dir',
                                                         **txt_bert_params)

            if state == 'frz':
              if freeze_until:
                # Freeze only certain layers
                freeze_until = int(freeze_until)
                logger.debug('Freezing text bert until layer %d excluded',
                             freeze_until)
                # Freeze net until given layer
                for name, param in self.txt_bert.named_parameters():
                  module = name.split('.')[0]
                  if name.split('.')[2].isdigit():
                    layer_nb = int(name.split('.')[2])
                  else:
                    continue
                  if module == 'encoder' and layer_nb in range(freeze_until):
                    param.requires_grad = False
                    logger.debug(name)
              else:
                # Freeze the whole model
                for name, param in self.txt_bert.named_parameters():
                  module = name.split('.')[0]
                  if module == 'encoder':
                    param.requires_grad = False
            else:
              assert not freeze_until

          if self.txt_inp == 'bertfrz':
            # Freeze model
            for param in self.txt_bert.embeddings.parameters():
              param.requires_grad = False
          elif self.txt_inp not in ['bertftn']:
            logger.error('Wrong parameter for the text encoder')
          self.text_dim = self.txt_bert.config.hidden_size

        elif self.txt_agg in ['vlad', 'mxp', 'mnp', 'lstm']:
          # Need to get text embeddings
          if self.txt_inp == 'bertfrz':
            ckpt = '/youtu_pedestrian_detection/wenzhewang/mmt_data/word_embeddings/bert/ckpt_from_huggingface.pth'
            self.word_embeddings = TxtEmbeddings(ckpt=ckpt, freeze=True)
          elif self.txt_inp == 'bertftn':
            ckpt = '/youtu_pedestrian_detection/wenzhewang/mmt_data/word_embeddings/bert/ckpt_from_huggingface.pth'
            self.word_embeddings = TxtEmbeddings(ckpt=ckpt)
          elif self.txt_inp == 'bertscr':
            vocab_size = 28996
            emb_dim = 768
            self.word_embeddings = TxtEmbeddings(vocab_size, emb_dim) # return nn.Embedding
          else:
            self.word_embeddings = tokenizer.we_model #word2vec
          emb_dim = self.word_embeddings.text_dim

          if self.txt_agg == 'vlad':
            self.text_pooling = NetVLAD(
                feature_size=emb_dim,
                cluster_size=28,
            )
            self.text_dim = self.text_pooling.out_dim
          elif self.txt_agg == 'mxp':
            self.text_dim = emb_dim
          elif self.txt_agg == 'lstm':
            input_dim = self.word_embeddings.text_dim
            hidden_dim = 512
            layer_dim = 1
            output_dim = hidden_dim
            self.text_pooling = LSTMModel(input_dim, hidden_dim, layer_dim,
                                          output_dim)
            self.text_dim = output_dim

    def forward(self, token_ids, device):
        # Unroll repeated captions into present minibatch
        b, captions_per_video, max_text_words, feat_dim = token_ids.size()
        m = len(self.modalities)

        if self.txt_agg[:4] in ['bert']:
          token_ids = token_ids.view(b * captions_per_video, max_text_words,
                                     feat_dim)

          input_ids_list = []
          token_type_ids_list = []  # Modality id
          position_ids_list = []  # Position
          attention_mask_list = []  # Valid token or not

          ids_size = (b * captions_per_video,)

          for pos_id in range(max_text_words):
            input_ids_list.append(token_ids[:, pos_id, 0].to(dtype=th.long))
            token_type_ids_list.append(th.full(ids_size, 0, dtype=th.long))
            position_ids_list.append(th.full(ids_size, pos_id, dtype=th.long))
            attention_mask_list.append(token_ids[:, pos_id, 1].to(dtype=th.long))

          input_ids = th.stack(input_ids_list, dim=1).to(device)
          token_type_ids = th.stack(token_type_ids_list, dim=1).to(device)
          position_ids = th.stack(position_ids_list, dim=1).to(device)
          attention_mask = th.stack(attention_mask_list, dim=1).to(device)

          txt_bert_output = self.txt_bert(input_ids,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids,
                                          position_ids=position_ids,
                                          head_mask=None)
          last_layer = txt_bert_output[0]

          # the way to obtain final text feature
          if self.post_agg == 'cls':
            text = last_layer[:, 0]

          elif self.post_agg == 'mxp':
            embeddings = last_layer[:, 1:]
            text, _ = th.max(embeddings, 1)

          elif self.post_agg == 'mnp':
            embeddings = last_layer[:, 1:]
            text = th.mean(embeddings, 1)

        elif self.txt_agg in ['vlad', 'mxp', 'mnp', 'lstm']:
          # Need to get text embeddings
          token_ids = token_ids.view(b * captions_per_video, max_text_words,
                                     feat_dim)

          input_ids = token_ids[:, :, 0].to(dtype=th.long)
          attention_mask = token_ids[:, :, 1].to(dtype=th.long)

          word_embs = self.word_embeddings(input_ids)

          if self.txt_agg == 'mxp':
            # max pooling
            word_embs[attention_mask == 0] = -float('Inf')
            text = th.max(word_embs, dim=1)[0]

          elif self.txt_agg == 'vlad':
            text = self.text_pooling(word_embs)

          elif self.txt_agg == 'lstm':
            x_lengths = get_len_sequences(attention_mask)
            text = self.text_pooling(word_embs, x_lengths)

        return text

if __name__ == '__main__':
    # main()
    txt_agg = "bertftn"
    txt_inp = "bertftn"
    modalities = ["face","ocr","rgb","s3d","scene","speech","vggish"]
    vid_bert_params = {"vocab_size_or_config_json_file": 10,"hidden_size": 512,
                   "num_hidden_layers": 4, "num_attention_heads": 4, "intermediate_size": 3072,
                   "hidden_act": "gelu","hidden_dropout_prob": 0.1,"attention_probs_dropout_prob": 0.1,
                   "max_position_embeddings": 32,"type_vocab_size": 19,"initializer_range": 0.02,"layer_norm_eps": 1e-12}
    m = TxtExtract(txt_agg, txt_inp, vid_bert_params, modalities)
    token_ids = torch.zeros(4,3,3,2)
    x = m(token_ids)

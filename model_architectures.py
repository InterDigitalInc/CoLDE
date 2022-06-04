import os, sys
parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(parent_dir)

from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertConfig, BertModel
from transformer import TransformerEncoderLayer, TransformerEncoder

import torch
from torch.nn import LSTM
from torch import nn

import pdb

class DocumentBertLSTM(BertPreTrainedModel):
    """
    BERT output over document in LSTM
    """

    def __init__(self, bert_model_config: BertConfig):
        super(DocumentBertLSTM, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        self.bert_batch_size= self.bert.config.bert_batch_size
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        self.lstm = LSTM(bert_model_config.hidden_size,bert_model_config.hidden_size)

        # For Association Embedding
        self.embed_dim = 768
        self.qw = nn.Linear(self.embed_dim, self.embed_dim)
        self.kw = nn.Linear(self.embed_dim, self.embed_dim)
        self.vw = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.projection = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, 256),
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch: torch.Tensor, document_sequence_lengths: list, device='cuda'):
        
        # contains all BERT sequences
        # bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                        min(document_batch.shape[1],self.bert_batch_size),
                                        self.bert.config.hidden_size), dtype=torch.float, device=device)

        # only pass through bert_batch_size numbers of inputs into bert.
        # this means that we are possibly cutting off the last part of documents.
        for doc_id in range(document_batch.shape[0]):
            bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                                token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                                attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])
        
        # self.lstm.flatten_parameters()
        # output, (_, _) = self.lstm(bert_output.permute(1,0,2))
        # last_layer = output[-1]
        # reduced_dim = self.projection(last_layer)
        # # Shape: 1x768
        # return reduced_dim
        return bert_output

    def to_lstm(self, bert_output1, bert_output2, device, mode):

        if mode == 'train':
            embed_dim = 768

            # Unique embeddings
            # Association Embedding
            q = self.qw(bert_output1)
            k = self.kw(bert_output2)
            v = self.vw(bert_output2)

            multihead_attn = nn.MultiheadAttention(embed_dim, num_heads=6).to(device=device)
            attn_output, attn_output_weights = multihead_attn(q, k, v)
            section1_embed = bert_output1 + attn_output.view(bert_output1.shape)
            section2_embed = bert_output1

        elif mode == 'test':
            section1_embed = bert_output1
            section2_embed = bert_output2

        self.lstm.flatten_parameters()             
        output1, (_, _) = self.lstm(section1_embed.permute(1,0,2))
        last_layer1 = output1[-1]
        reduced_dim1 = self.projection(last_layer1)

        output2, (_, _) = self.lstm(section2_embed.permute(1,0,2))
        last_layer2 = output2[-1]
        reduced_dim2 = self.projection(last_layer2)

        return reduced_dim1, reduced_dim2

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True

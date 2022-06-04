from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertConfig, BertModel
from pytorch_transformers.modeling_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.tokenization_bert import BertTokenizer

from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import torch, math, logging, os

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import ndcg_score, dcg_score, accuracy_score

import pandas as pd
import numpy as np

from tqdm import tqdm 

from operator import itemgetter 

import os, sys
parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(parent_dir)

from model_architectures import DocumentBertLSTM
from loss_supContrast import SupConLoss


class CustomDatasetLoader(Dataset):
    def __init__(self, text1, text1_seq, text2, text2_seq, label):
        """
        Args:
            X (np.array): Features
            y (np.array): Target
        """
        self.text1 = text1
        self.text1_seq = text1_seq
        self.text2 = text2
        self.text2_seq = text2_seq
        self.label = label

    def __len__(self):
        return len(self.text1)

    def __getitem__(self, idx):
        text1 = self.text1[idx]
        text1_seq = self.text1_seq[idx]
        text2 = self.text2[idx]
        text2_seq = self.text2_seq[idx]
        label = self.label[idx]
        return text1, text1_seq, text2, text2_seq, label


def encode_documents(documents: list, tokenizer: BertTokenizer, max_input_length=512):
    """
    Returns a len(documents) * max_sequences_per_document * 3 * 512 tensor where len(documents) is the batch
    dimension and the others encode bert input.

    This is the input to any of the document bert architectures.

    :param documents: a list of text documents
    :param tokenizer: the sentence piece bert tokenizer
    :return:
    """
    tokenized_documents = [tokenizer.tokenize(document) for document in documents]
    max_sequences_per_document = math.ceil(max(len(x)/(max_input_length-2) for x in tokenized_documents))
    # assert max_sequences_per_document <= 20, "Your document is too large, arbitrary size when writing"

    output = torch.zeros(size=(len(documents), max_sequences_per_document, 3, max_input_length), dtype=torch.long)
    all_chunk_ids = torch.zeros(size=(len(documents), max_sequences_per_document, 1, max_input_length), dtype=torch.long)
    document_seq_lengths = [] # number of sequence generated per document
    # Need to use 510 to account for 2 padding tokens
    for doc_index, tokenized_document in enumerate(tokenized_documents):
        max_seq_index = 0
        for seq_index, i in enumerate(range(0, len(tokenized_document), (max_input_length-2))):
            raw_tokens = tokenized_document[i:i+(max_input_length-2)]
            tokens = []
            input_type_ids = []

            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in raw_tokens:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_masks = [1] * len(input_ids)
            chunk_id = [seq_index] * len(input_ids)

            while len(input_ids) < max_input_length:
                input_ids.append(0)
                input_type_ids.append(0)
                attention_masks.append(0)
                chunk_id.append(seq_index)

            assert len(input_ids) == max_input_length and len(attention_masks) == max_input_length and len(input_type_ids) == max_input_length

            # Unique chunk IDS
            import operator
            input_ids = list(map(operator.add, input_ids, chunk_id))
            # input_ids = input_ids + chunk_id

            output[doc_index][seq_index] = torch.cat((torch.LongTensor(input_ids).unsqueeze(0),
                                                           torch.LongTensor(input_type_ids).unsqueeze(0),
                                                           torch.LongTensor(attention_masks).unsqueeze(0)),
                                                           dim=0)
            all_chunk_ids[doc_index][seq_index] = torch.LongTensor(chunk_id)
            max_seq_index = seq_index

        # Unique chunk IDS
        # output = output + all_chunk_ids

        # Randomly shuffle chunks for ROBUSTNESS ANALYSIS
        shuffle_rows = torch.randperm(seq_index+1)
        shuffled_content = output[doc_index][shuffle_rows]
        zero_padding = output[doc_index][seq_index+1: max_sequences_per_document]
        output[doc_index] = torch.cat([shuffled_content, zero_padding])

        document_seq_lengths.append(max_seq_index+1)
    return output, torch.LongTensor(document_seq_lengths)

model_architectures = {
    'DocumentBertLSTM': DocumentBertLSTM,
}

class CoLDE():
    def __init__(self, args=None,
                 labels=None,
                 device='cuda',
                 bert_model_path='bert-base-uncased',
                 architecture="DocumentBertLSTM",
                 batch_size=10,
                 bert_batch_size=7,
                 learning_rate = 5e-5,
                 weight_decay=0.01,
                 use_tensorboard=False,
                 temperature=0.5,
                 use_cosine_similarity=True):

        if args is not None:
            self.args = vars(args)

        if not args:
            self.args = {}
            self.args['bert_model_path'] = bert_model_path
            self.args['device'] = device
            self.args['learning_rate'] = learning_rate
            self.args['weight_decay'] = weight_decay
            self.args['batch_size'] = batch_size
            self.args['bert_batch_size'] = bert_batch_size
            self.args['architecture'] = architecture
            self.args['use_tensorboard'] = use_tensorboard
            self.args['temperature'] = temperature
            self.args['use_cosine_similarity'] = use_cosine_similarity

        if 'fold' not in self.args:
            self.args['fold'] = 0

        self.log = logging.getLogger()
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args['bert_model_path'])

        #account for some random tensorflow naming scheme
        if os.path.exists(self.args['bert_model_path']):
            if os.path.exists(os.path.join(self.args['bert_model_path'], CONFIG_NAME)):
                config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], CONFIG_NAME))
            elif os.path.exists(os.path.join(self.args['bert_model_path'], 'bert_config.json')):
                config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], 'bert_config.json'))
            else:
                raise ValueError("Cannot find a configuration for the BERT based model you are attempting to load.")
        else:
            config = BertConfig.from_pretrained(self.args['bert_model_path'])
        
        # config.__setattr__('num_labels',len(self.args['labels']))
        config.__setattr__('bert_batch_size', self.args['bert_batch_size'])

        # if 'use_tensorboard' in self.args and self.args['use_tensorboard']:
        #     assert 'model_directory' in self.args is not None, "Must have a logging and checkpoint directory set."
        #     from torch.utils.tensorboard import SummaryWriter
        #     self.tensorboard_writer = SummaryWriter(os.path.join(self.args['model_directory'],
        #                                                          "..",
        #                                                          "runs",
        #                                                          self.args['model_directory'].split(os.path.sep)[-1]+'_'+self.args['architecture']+'_'+str(self.args['fold'])))

        # Load the BERT model from 'bert_model_path' - 'bert-base-unacsed' or saved model
        self.bert_doc_classification = model_architectures[self.args['architecture']].from_pretrained(self.args['bert_model_path'], config=config)
        self.bert_doc_classification.freeze_bert_encoder()
        self.bert_doc_classification.unfreeze_bert_encoder_last_layers()

        self.optimizer = torch.optim.Adam(
            self.bert_doc_classification.parameters(),
            weight_decay=self.args['weight_decay'],
            lr=self.args['learning_rate']
        )


    def fit(self, train, val01):
        """
        A list of
        :param documents: a list of documents
        :param labels: a list of label vectors
        :return:
        """

        self.bert_doc_classification.train()
        self.bert_doc_classification.to(device=self.args['device'])
        # Initialize the supervised contrastive loss
        self.loss_function = SupConLoss(temperature=self.args['temperature'])

        # Shape of representations: len_documents x num_chunks_in_entire_text x 3 x 512; 
        # len_documents: chunks for reading the json; num_chunks_in_text: sequence length
        text1_tensors, text1_sequence_lengths = encode_documents(train['text1'], self.bert_tokenizer)
        text2_tensors, text2_sequence_lengths = encode_documents(train['text2'], self.bert_tokenizer)
        label_tensors = torch.tensor(train['label']).to(device=self.args['device'])

        train_data = CustomDatasetLoader(text1_tensors, text1_sequence_lengths,
                                         text2_tensors, text2_sequence_lengths,
                                         label_tensors)

        # Load data into batches using Pytorch's Dataloader
        train_loader = DataLoader(train_data, batch_size=self.args['batch_size'])
        
        epoch_loss = 0.0

        for epoch in tqdm(range(1,self.args['epochs']+1)):
            for text1, text1_seq, text2, text2_seq, label in train_loader:

                #Don't need to add as segment1 embedding = 0
                #seg = torch.zeros(text1.shape)
                #text1 = text1 + seg
                bert_output1 = self.bert_doc_classification(document_batch=text1.to(device=self.args['device']),
                                                                document_sequence_lengths=text1_seq,
                                                                device=self.args['device'])

                # Segment 2
                seg = torch.ones(text2.shape).type(torch.LongTensor)
                text2 = text2 + seg
                bert_output2 = self.bert_doc_classification(document_batch=text2.to(device=self.args['device']),
                                                                document_sequence_lengths=text2_seq,
                                                                device=self.args['device'])

                text1_last_layer, text2_last_layer = self.bert_doc_classification.to_lstm(bert_output1, 
                                                                                            bert_output2,
                                                                                            device=self.args['device'],
                                                                                            mode='test')

                # Shape: batch_size x 2 x projection_layer_dimension
                n_views = 2
                features = torch.cat([text1_last_layer, text2_last_layer], dim=1).view(text1_last_layer.shape[0], 
                                                                            n_views, text1_last_layer.shape[1])
                features = F.normalize(features, p=2, dim=2)
            
                loss = self.loss_function(features, label)[0]

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += float(loss.item())

            epoch_loss /= int(text1_tensors.shape[0] / self.args['batch_size'])  # divide by number of batches per epoch

            # if 'use_tensorboard' in self.args and self.args['use_tensorboard']:
            #     self.tensorboard_writer.add_scalar('Loss/Train', epoch_loss, self.epoch)
            if epoch % 2 == 0:
                self.log.info('Loss after %i epochs: %f' % (epoch, epoch_loss))

            # evaluate on development data
            if epoch % self.args['evaluation_interval'] == 0:
                metric = self.predict_zero_one(val01)
                self.log.info("Prediction 0 or 1: {0}".format(metric))

            if epoch % self.args['checkpoint_interval'] == 0:
                self.save_checkpoint(os.path.join(self.args['model_directory'], "checkpoint_%s" % epoch))
       
        # Parallize the process
        # https://github.com/huggingface/transformers/pull/3842
        # if torch.cuda.device_count() > 1 and not isinstance(self.bert_doc_classification, torch.nn.DataParallel):
        #     self.bert_doc_classification = torch.nn.DataParallel(self.bert_doc_classification)


    def predict_zero_one(self, test):
        self.bert_doc_classification.eval()
        self.bert_doc_classification.to(device=self.args['device'])
        
        # Initialize the supervised contrastive loss
        pairwise_dist = nn.CosineSimilarity(dim=1, eps=1e-6)

        # Shape of representations: len_documents x num_chunks_in_entire_text x 3 x 512; 
        # len_documents: chunks for reading the json; num_chunks_in_text: sequence length
        source_tensors, source_sequence_lengths = encode_documents(test['text1'], self.bert_tokenizer)
        target_tensors, target_sequence_lengths = encode_documents(test['text2'], self.bert_tokenizer)
        label_tensors = torch.tensor(test['label']).to(device=self.args['device'])

        test_data = CustomDatasetLoader(source_tensors, source_sequence_lengths,
                                        target_tensors, target_sequence_lengths,
                                        label_tensors)

        # Load data into batches using Pytorch's Dataloader
        test_loader = DataLoader(test_data, batch_size=self.args['batch_size'])
        
        p_t0, p_t1, p_t2 = [], [], []
        r_t0, r_t1, r_t2 = [], [], []
        f1_t0, f1_t1, f1_t2 = [], [], []
        acc_t0, acc_t1, acc_t2 = [], [], []
        metric = {}

        with torch.no_grad():
            for source, source_seq, target, target_seq, label in test_loader:
                source_bert_output = self.bert_doc_classification(document_batch=source.to(device=self.args['device']),
                                                                document_sequence_lengths=source_seq,
                                                                device=self.args['device'])
                # source_last_layer = self.bert_doc_classification.to_lstm(source_bert_output)
                
                target_bert_output = self.bert_doc_classification(document_batch=target.to(device=self.args['device']),
                                                                document_sequence_lengths=target_seq,
                                                                device=self.args['device'])

                source_last_layer, target_last_layer = self.bert_doc_classification.to_lstm(source_bert_output,
                                                                                            target_bert_output,
                                                                                            device=self.args['device'],
                                                                                            mode='test')

                dist = pairwise_dist(source_last_layer, target_last_layer)

                label = label.cpu()
                
                # Precision, Recall, F1, Accuracy at theta=0.25
                pred_t0 = torch.gt(dist, 0.25).int().cpu()
                p_t0.append(precision_score(pred_t0, label))
                r_t0.append(recall_score(pred_t0, label))
                f1_t0.append(f1_score(pred_t0, label))
                acc_t0.append(accuracy_score(pred_t0, label))

                # Precision, Recall, F1, Accuracy at theta=0.5
                pred_t1 = torch.gt(dist, 0.5).int().cpu()
                p_t1.append(precision_score(pred_t1, label))
                r_t1.append(recall_score(pred_t1, label))
                f1_t1.append(f1_score(pred_t1, label))
                acc_t1.append(accuracy_score(pred_t1, label))

                # Precision, Recall, F1, Accuracy at theta=0.75
                pred_t2 = torch.gt(dist, 0.75).int().cpu()
                p_t2.append(precision_score(pred_t2, label))
                r_t2.append(recall_score(pred_t2, label))
                f1_t2.append(f1_score(pred_t2, label))
                acc_t2.append(accuracy_score(pred_t2, label))
                
        compute = lambda data: float(sum(data)/len(data))

        metric['p_t0'] = compute(p_t0)
        metric['p_t1'] = compute(p_t1)
        metric['p_t2'] = compute(p_t2)

        metric['r_t0'] = compute(r_t0)
        metric['r_t1'] = compute(r_t1)
        metric['r_t2'] = compute(r_t2)

        metric['f1_t0'] = compute(f1_t0)
        metric['f1_t1'] = compute(f1_t1)
        metric['f1_t2'] = compute(f1_t2)

        metric['acc_t0'] = compute(acc_t0)
        metric['acc_t1'] = compute(acc_t1)
        metric['acc_t2'] = compute(acc_t2)

        self.bert_doc_classification.train()
        return metric
    

    def save_checkpoint(self, checkpoint_path: str):
        """
        Saves an instance of the current model to the specified path.
        :return:
        """
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        # else:
        #     raise ValueError("Attempting to save checkpoint to an existing directory")
        self.log.info("Saving checkpoint: %s" % checkpoint_path )

        #save finetune parameters
        net = self.bert_doc_classification
        if isinstance(self.bert_doc_classification, nn.DataParallel):
            net = self.bert_doc_classification.module
        torch.save(net.state_dict(), os.path.join(checkpoint_path, WEIGHTS_NAME))
        #save configurations
        net.config.to_json_file(os.path.join(checkpoint_path, CONFIG_NAME))
        #save exact vocabulary utilized
        self.bert_tokenizer.save_vocabulary(checkpoint_path)

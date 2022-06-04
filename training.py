import os, sys
parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(parent_dir)

from document_matching import CoLDE

from tqdm import tqdm
from pprint import pformat
from sklearn.utils import shuffle

import torch
import torch.nn as nn

import pandas as pd
import multiprocessing
import time, logging, configargparse, os, socket

log = logging.getLogger()

datafiles = {
    'aan': '/data/long_documents/aan/train_pairs_sample.json',
    'wiki': '/data/long_documents/wikipedia/colde/wiki_train_v2.json',
}

valfiles = {
    'aan': '/data/long_documents/aan/ablation/val_samples.json',
    'wiki': '/data/long_documents/wikipedia/wiki_val_samples.json',
}

testfiles = {
    'aan': '/data/long_documents/aan/ablation/test_samples.json',
    'wiki': '/data/long_documents/wikipedia/wiki_test_samples.json',
}


def initialize_arguments(p:configargparse.ArgParser): 
    p.add('--dataset')
    p.add('--topk', help='Evaluation @ k', type=int)
    p.add('--model_storage_directory', help='The directory caching all model runs')
    p.add('--bert_model_path', help='Model path to BERT')
    p.add('--labels', help='Numbers of labels to predict over', type=str)
    p.add('--architecture', help='Training architecture', type=str)
    # p.add('--freeze_bert', help='Whether to freeze bert', type=bool)

    p.add('--batch_size', help='Batch size for training multi-label document classifier', type=int)
    p.add('--bert_batch_size', help='Batch size for feeding 510 token subsets of documents through BERT', type=int)
    p.add('--epochs', help='Epochs to train', type=int)

    p.add('--evaluation_interval', help='Evaluate model on test set every evaluation_interval epochs', type=int)
    p.add('--checkpoint_interval', help='Save a model checkpoint to disk every checkpoint_interval epochs', type=int)

    #Optimizer arguments
    p.add('--learning_rate', help='Optimizer step size', type=float)
    p.add('--weight_decay', help='Adam regularization', type=float)
    p.add('--temperature', help='Temperature for cross-entropy loss', type=float)
    p.add('--use_cosine_similarity', help='Use cosine similarity', type=bool)

    #Non-config arguments
    p.add('--cuda', action='store_true', help='Utilize GPU for training or prediction')
    p.add('--timestamp', help='Run specific signature')
    p.add('--model_directory', help='The directory storing this model run, a sub-directory of model_storage_directory')
    p.add('--use_tensorboard', help='Use tensorboard logging', type=bool)

    #Config file as input
    config_path = os.path.join(parent_dir, 'config.ini')
    args = p.parse_args(['@'+config_path])

    #Set run specific envirorment configurations
    args.timestamp = time.strftime("run_%Y_%m_%d_%H_%M_%S") #+ "_{machine}".format(machine=socket.gethostname())
    args.model_directory = os.path.join(args.model_storage_directory, args.timestamp) #directory
    os.makedirs(args.model_directory, exist_ok=True)
    
    #Handle logging configurations
    log.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(os.path.join(args.model_directory, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(ch)
    log.info(p.format_values())

    #Set global GPU state
    if torch.cuda.is_available() and args.cuda:
        if torch.cuda.device_count() > 1:
            log.info("Using %i CUDA devices" % torch.cuda.device_count() )
        else:
            log.info("Using CUDA device:{0}".format(torch.cuda.current_device()))
        args.device = 'cuda'
    else:
        log.info("Not using CUDA :(")
        args.device = 'cpu'

    return args 


def format_data(args, model, raw_df, mode):
    data = {}
    data['text1'] = raw_df['sec_2'].to_list() # Body Part 1
    data['text2']= raw_df['sec_3'].to_list()  # Body Part 2
    data['label'] = raw_df['label'].to_list() # Label
    data['id'] = raw_df['id'].to_list()
    return data

if __name__ == "__main__":
    p = configargparse.ArgParser(fromfile_prefix_chars='@')
    args = initialize_arguments(p)
    torch.cuda.empty_cache()

    model = CoLDE(args=args)

    # Model Training
    log.info("Supervised Model training...")
    raw_df = pd.read_json(datafiles[args.dataset], lines=True, orient='records')
    train_data = raw_df[['id', 'sec_2', 'sec_3', 'label']]
    train_data.columns = ['id', 'text1', 'text2', 'label']

    val = pd.read_json(valfiles[args.dataset], lines=True, orient='records')
    model.fit(train_data, val)
    log.info("Supervised Model Training Complete.")

    # Model Evaluation
    log.info("Supervised Model Evaluation...")

    # Predict whether the document is relevant or not -- 1 or 0 => 1: relevacnt, 0: not relevant
    test_01 = pd.read_json(testfiles[args.dataset], orient='records', lines=True)
    metric = model.predict_zero_one(test_01)
    log.info("Prediction 0 or 1: {0}".format(metric))

    log.info("Evaluation Complete.")

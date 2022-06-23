# Classifier semantic modeling
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import logging
import os

from argparse import ArgumentParser

from transformers import BertConfig, BertTokenizer, BertForPreTraining, BertTokenizerFast, BertModel
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from transformers import BertTokenizer, BertForMaskedLM
from tokenizers import BertWordPieceTokenizer

from datasets import load_dataset


# constant values for tokenizer and model configurations
VOCAB_SIZE = 13200
LOWER_CASE = True


def load_tokenizer(from_config=False, filepaths=None, savepath=None):
    """load_tokenizer
       Loads a transformers-based tokenizer to train on Hmong data.
       @param from_config (bool) : Indicates whether to create a new tokenizer (True)
           or load the pre-existing transformers-based multilingual tokenizer (False).
       @param filepaths (None or list) : specifies the filenames to use in creating a
           new tokenizer if from_config is set to True.
       @param savepath (None or str) : specifies the location to save the created
           tokenizer to load as a BertTokenizer in transformers.
       returns : a subclass of PreTrainedTokenizer"""
    if from_config:
        tokenizer = BertWordPieceTokenizer(lowercase=LOWER_CASE)
                                      
        tokenizer.train(filepaths,
                        vocab_size=VOCAB_SIZE,
                        min_frequency=5,
                        special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
                        limit_alphabet=1500)
        
        tokenizer_savepath = os.path.join(savepath, 'tokenizer')
        
        tokenizer.save(tokenizer_savepath)
        
        bert_tokenizer = BertTokenizerFast(tokenizer_file=tokenizer_savepath,
                                           do_lower_case=LOWER_CASE,
                                           pad_token = '[PAD]',
                                           unk_token = '[UNK]',
                                           cls_token = '[CLS]',
                                           sep_token = '[SEP]',
                                           mask_token = '[MASK]')
    else:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
                                           
    return bert_tokenizer

# TODO: review to confirm code
def load_model(from_config=False):
    """
    load_model
    Loads a transformers-based model to train on Hmong data.
    @param from_config (bool) : Indicates whether to train from scratch (True)
        or from a pre-existing transformers-based multilingual model (False).
    returns : model of type transformers.BertForMaskedLM
    """
    if from_config:
        config_ = BertConfig(vocab_size=VOCAB_SIZE)
        model = BertForMaskedLM(config_)
    
    else:
        model = BertForMaskedLM.from_pretrained('bert-base-multilingual-uncased')
    
    return model


def generate_filepaths(path):
    """
    generate_filepaths
    Loads the filepaths as a list.
    @param path (str) : Path to the parent directory for the files as a str.
    returns : List containing paths to each file.
    """
    files = [os.path.join(root, name) for root, _, files in os.walk(path) for name in files]
    return files


def load_data(path, tokenizer):
    """
    load_data
    Loads the dataset at the path specified.
    @param path (str) : Path to the parent directory for the dataset.
    @param tokenizer (PreTrainedTokenizer) : tokenizer object corresponding to the model to train.
    returns : 
        transformers.DataCollatorForLanguageModeling object containing the data
        datasets.Dataset containing the training data
        datasets.Dataset containing the test data
    """
    filepaths = generate_filepaths(path)
    dataset_ = load_dataset(path=path, name="hmbert_data", data_files=filepaths, split='train',)
    dataset = dataset_.train_test_split(test_size=0.2)
    
    max_length = 128
    
    def tokenize_data(data_in):
        tokenized = tokenizer(data_in['text'],
                              padding="max_length",
                              max_length=max_length,
                              truncation=True)
        return tokenized
    
    train_data = dataset['train'].map(tokenize_data, batched=True)
    train_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    test_data = dataset['test'].map(tokenize_data, batched=True)
    test_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    # MLM focus only given RoBERTa paper; NSP is excluded
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                               mlm_probability=0.15,
                                               return_tensors='pt')
    
    return collator, train_data, test_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--from_config', action='store_true')
    parser.add_argument('--from_pretrained', dest='from_config', action='store_false')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--save_path', type=str, default='models')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-05)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.set_defaults(from_config=False)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logging.info('Loading tokenizer and model.')
    
    # remove final slash to enable use of datasets.load_dataset
    data_path = args.data_path.rstrip('/')
    
    # TODO: provide full path
    files = generate_filepaths(data_path)
    tokenizer = load_tokenizer(args.from_config, files, data_path)
    
    model = load_model(args.from_config)
    
    logging.info('Loaded tokenizer and model.')
    logging.info('Loading data.')
    
    collator, train_data, test_data = load_data(data_path, tokenizer)
    
    logging.info('Loaded data.')
    
    training_args = TrainingArguments(output_dir=args.save_path,
                                      overwrite_output_dir=True,
                                      evaluation_strategy='epoch',
                                      per_device_train_batch_size=args.batch_size,
                                      per_device_eval_batch_size=args.batch_size,
                                      per_gpu_train_batch_size=args.batch_size,
                                      per_gpu_eval_batch_size=args.batch_size,
                                      gradient_accumulation_steps=args.grad_steps,
                                      learning_rate=args.lr,
                                      weight_decay=args.weight_decay,
                                      max_grad_norm=args.max_grad_norm,
                                      num_train_epochs=args.epochs,
                                      logging_steps=1000,
                                      save_steps=1000,
                                     )

    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=collator,
                      train_dataset=train_data,
                      eval_dataset=test_data,
                     )
    
    logging.info('Beginning training.')
    
    trainer.train()

    logging.info('Training completed.')

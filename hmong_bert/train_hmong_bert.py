# Classifier semantic modeling
# Original code Copyright © 2022 Nathan M. White

# Copyright notice
__copyright__ = "Copyright © 2022 Nathan M. White"

# author
__author__ = "Nathan M. White"
__author_email__ = "nathan.white1@jcu.edu.au"

import os

from transformers import BertConfig, BertTokenizer, BertForPreTraining, BertTokenizerFast, BertModel
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from transformers import BertTokenizer, BertForMaskedLM
from tokenizers import BertWordPieceTokenizer, WordPieceTrainer

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
        wp_trainer = WordPieceTrainer(vocab_size=VOCAB_SIZE,
                                      min_frequency=5,
                                      special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
                                      limit_alphabet=1500)
                                      
        tokenizer.train(filepaths, wp_trainer)
        
        tokenizer.save(savepath)
        
        bert_tokenizer = BertTokenizerFast(tokenizer_file=savepath,
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


def load_dataset(path, tokenizer):
    """
    load_dataset
    Loads the dataset at the path specified.
    @param path (str) : Path to the parent directory for the dataset.
    @param tokenizer (PreTrainedTokenizer) : tokenizer object corresponding to the model to train.
    returns : transformers.DataCollatorForLanguageModeling object containing the data.
    """
    filepaths = generate_filepaths(path)
    dataset_ = load_dataset(path=path, data_files=filepaths, split='train')
    dataset = dataset_.train_test_split(test_size=0.2)
    
    max_length = 128
    
    def tokenize_data(data_in):
        tokenized = tokenizer(data_in['text'],
                              padding="max_length",
                              max_length=max_length,
                              truncation=True)
        return tokenized
    
    train_data = dataset['train'].map(tokenize_data, batched=True, remove_columns=['token_type_ids'])
    train_data.set_format(type='torch')
    
    test_data = dataset['test'].map(tokenize_data, batched=True, remove_columns=['token_type_ids'])
    test_data.set_format(type='torch')
    
    # MLM focus only given RoBERTa paper; NSP is excluded
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                               mlm_probability=0.15,
                                               return_tensors='pt')
    
    return collator


if __name__ == '__main__':
    pass

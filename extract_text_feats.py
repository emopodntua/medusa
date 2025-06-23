# Adapted from https://github.com/zeroQiaoba/MERTools/blob/master/MERBench/feature_extraction/text/extract_text_huggingface.py

# *_*codig:utf-8 *_*
import os
import time
import argparse
import numpy as np
import glob
import torch
from transformers import AutoTokenizer, AutoTokenizer, AutoModelForSequenceClassification # version: 4.5.1, pip install transformers
from transformers import  RobertaTokenizer, RobertaForSequenceClassification


#supported models
ROBERTA_LARGE = 'roberta-large'
MODERN_BERT_LARGE = 'ModernBERT-large'


def find_start_end_pos(tokenizer):
    sentence = 'The weather is very good today.'
    input_ids = tokenizer(sentence, return_tensors='pt')['input_ids'][0]
    start, end = None, None

    # find start, must in range [0, 1, 2]
    for start in range(0, 3, 1):
        outputs = tokenizer.decode(input_ids[start:]).replace(' ', '')
        if outputs == sentence:
            print (f'start: {start};  end: {end}')
            return start, None

        if outputs.startswith(sentence):
            break
   
    # find end, must in range [-1, -2]
    for end in range(-1, -3, -1):
        outputs = tokenizer.decode(input_ids[start:end]).replace(' ', '')
        if outputs == sentence:
            break
    # print(tokenizer.decode(input_ids[start:end]).replace)
    # assert tokenizer.decode(input_ids[start:end]).replace(' ', '') == sentence
    print (f'start: {start};  end: {end}')
    return start, end


# batch_pos and feature_dim
def find_batchpos_embdim(tokenizer, model, gpu):
    sentence = 'The weather is very good today.'
    inputs = tokenizer(sentence, return_tensors='pt')
    if gpu != -1: inputs = inputs.to('cuda')

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True).hidden_states # for new version 4.5.1
        # txt_feats = torch.average(last_four_layers)
        outputs = torch.stack(outputs)[[-1]].sum(dim=0) # sum => [batch, T, D=768]
        outputs = outputs.cpu().numpy() # (B, T, D) or (T, B, D)
        batch_pos = None
        if outputs.shape[0] == 1:
            batch_pos = 0
        if outputs.shape[1] == 1:
            batch_pos = 1
        assert batch_pos in [0, 1]
        feature_dim = outputs.shape[2]
    print (f'batch_pos:{batch_pos}, feature_dim:{feature_dim}')
    return batch_pos, feature_dim


# main process
def extract_embedding(model_name, trans_files, save_dir, feature_level, gpu):

    print('='*30 + f' Extracting "{model_name}" ' + '='*30)
    start_time = time.time()

    # save last four layers
    layer_ids = [-4, -3, -2, -1]

    if model_name == ROBERTA_LARGE:
        model_file = 'FacebookAI/roberta-large'
        model = RobertaForSequenceClassification.from_pretrained(model_file)
        tokenizer = RobertaTokenizer.from_pretrained(model_file)
    elif model_name == MODERN_BERT_LARGE:
        model_file = 'answerdotai/ModernBERT-large'
        model = AutoModelForSequenceClassification.from_pretrained(model_file)
        tokenizer = AutoTokenizer.from_pretrained(model_file)

    if gpu != -1:
        torch.cuda.set_device(gpu)
        model.cuda()
    model.eval()

    print('Calculate embeddings...')
    start, end = find_start_end_pos(tokenizer) # only preserve [start:end+1] tokens
    batch_pos, feature_dim = find_batchpos_embdim(tokenizer, model, gpu) # find batch pos
    
    # Process each file in the list
    for idx, file_path in enumerate(trans_files):
        # Get the filename without extension to use as the identifier
        file_name = os.path.basename(file_path).split('.')[0]
        print(f'File: {file_name}, Index: {idx}')
        
        # Read text from the file
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                sentence = file.read().strip()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue
        
        print(f'Processing {file_name} ({idx+1}/{len(trans_files)})...')
        # extract embedding from sentences       
        embeddings = []
        embeddings = []
        if sentence and len(sentence) > 0:
            inputs = tokenizer(sentence, return_tensors='pt', max_length=512)  # Truncate input length
            if gpu != -1: 
                inputs = inputs.to('cuda')
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True).hidden_states  # for new version 4.5.1
                outputs = torch.stack(outputs)[layer_ids].sum(dim=0)  # sum => [batch, T, D=768]
                outputs = outputs.cpu().numpy()  # (B, T, D)
                
            if batch_pos == 0:
                embeddings = outputs[0, start:end]
            elif batch_pos == 1:
                embeddings = outputs[start:end, 0]
        
        print(f'feature dimension: {feature_dim}')
        # Save to .npy file using the file name
        output_file = os.path.join(save_dir, f'{file_name}.npy')
        
        if feature_level == 'FRAME':
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((1, feature_dim))
            elif len(embeddings.shape) == 1:
                embeddings = embeddings[np.newaxis, :]
            np.save(output_file, embeddings)
        else:
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((feature_dim, ))
            elif len(embeddings.shape) == 2:
                embeddings = np.mean(embeddings, axis=0)
            np.save(output_file, embeddings)
    
    end_time = time.time()
    print(f'Total {len(trans_files)} files done! Time used ({model_name}): {end_time - start_time:.1f}s.')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--model_name', type=str, help='name of pretrained model')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', choices=['UTTERANCE', 'FRAME'], help='output types')
    parser.add_argument('--trans_dir', type=str, help='path to transcriptions directory')
    parser.add_argument('--output_dir', type=str, help='path to output directory')

    args = parser.parse_args()

    model_name = args.model_name
    trans_dir = args.trans_dir
    output_dir = args.output_dir
    feature_level = args.feature_level

    print(f"Extracting {feature_level} features with {model_name} and saving in {output_dir}")

    trans_files = glob.glob(os.path.join(trans_dir, '*.txt'))
    print(f'Find total "{len(trans_files)}" trans files.')

    if not os.path.exists(output_dir): os.makedirs(output_dir)

    extract_embedding(model_name, trans_files, output_dir, feature_level, 1)



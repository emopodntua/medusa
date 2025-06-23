# Adapted from https://github.com/zeroQiaoba/MERTools/blob/master/MERBench/feature_extraction/audio/extract_audio_huggingface.py

import os
import math
import time
import glob
import torch
import argparse
import numpy as np
import soundfile as sf
import sys

from transformers import AutoModel, AutoModelForCTC
from transformers import ClapAudioModel, ClapProcessor, WhisperFeatureExtractor, Wav2Vec2FeatureExtractor, WhisperProcessor

# supported models
HUBERT_LARGE = 'hubert-large' # https://huggingface.co/facebook/hubert-large-ls960-ft
CLAP_GENERAL = 'larger_clap_general' # https://huggingface.co/laion/larger_clap_general
WAVLM_LARGE = 'wavlm-large' # https://huggingface.co/microsoft/wavlm-large
WHISPER_LARGE_V2 = 'whisper-large-v2' # https://huggingface.co/openai/whisper-large-v2
WHISPER_LARGE_V3 = 'whisper-large-v3' # https://huggingface.co/openai/whisper-large-v3

# input_values: [1, wavlen], output: [bsize, maxlen]
def split_into_batch(input_values, maxlen=3000):#16000*10):
    if len(input_values[0]) <= maxlen:
        return input_values
    
    bs, wavlen = input_values.shape
    assert bs == 1
    tgtlen = math.ceil(wavlen / maxlen) * maxlen
    batches = torch.zeros((1, tgtlen))
    batches[:, :wavlen] = input_values
    batches = batches.view(-1, maxlen)
    return batches


def extract(model_name, audio_files, save_dir, feature_level, gpu):

    start_time = time.time()

    if model_name==WHISPER_LARGE_V2:
        model_file = "openai/whisper-large-v2"
        model = AutoModel.from_pretrained(model_file)
        processor = WhisperProcessor.from_pretrained(model_file)
        
    elif model_name==WHISPER_LARGE_V3:
        model_file = "openai/whisper-large-v3"
        model = AutoModel.from_pretrained(model_file)
        processor = WhisperProcessor.from_pretrained(model_file)

    elif model_name == HUBERT_LARGE:
        model_file="facebook/hubert-large-ls960-ft"
        model = AutoModelForCTC.from_pretrained(model_file)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_file) #maybe change this 

    elif model_name == CLAP_GENERAL:
        model_file = "laion/larger_clap_general"
        model = ClapAudioModel.from_pretrained(model_file) #added clap
        processor = ClapProcessor.from_pretrained(model_file, sampling_rate=16000)

    elif model_name == WAVLM_LARGE:
        model_file = "microsoft/wavlm-large"
        model = AutoModel.from_pretrained(model_file)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_file)
    else:
        print("model not supported")
        exit(1)

    if gpu != -1:
        device = torch.device(f'cuda:{gpu}')
        model.to(device)

    model.eval()

    # iterate audios
    for idx, audio_file in enumerate(audio_files, 1):
        file_name = os.path.basename(audio_file)
        vid = file_name[:-4]
        print(f'Processing "{file_name}" ({idx}/{len(audio_files)})...')

        ## process for too short ones
        samples, sr = sf.read(audio_file)
        assert sr == 16000, 'currently, we only test on 16k audio'
        
        ## model inference
        with torch.no_grad():
            if model_name in [WHISPER_LARGE_V2, WHISPER_LARGE_V3, CLAP_GENERAL]:
                layer_ids = [-1]

                input_features = processor.feature_extractor(raw_speech=samples, sampling_rate=sr, return_tensors="pt").input_features # [1, 80, 3000]
                print("model.config.decoder_start_token_id: ",model.config.decoder_start_token_id)
                
                decoder_input_ids = torch.tensor([[1, 1]]) # i added this because above is none
                #decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
                if gpu != -1: input_features = input_features.to(device)
                if gpu != -1: decoder_input_ids = decoder_input_ids.to(device)
                #last_hidden_state = model(input_features, output_hidden_states=True, decoder_input_ids=decoder_input_ids).last_hidden_state
                last_hidden_state = model(input_features, output_hidden_states=True).last_hidden_state #also this
                print("Input Features Shape:", input_features.shape) 
                print("Last Hidden State Shape:", last_hidden_state.shape)
                assert last_hidden_state.shape[0] == 1
                feature = last_hidden_state[0].detach().squeeze().cpu().numpy() # (2, D)
                print(feature.shape)
                print(model.config)
            else:
                layer_ids = [-4, -3, -2, -1]
                input_values = feature_extractor(samples, sampling_rate=sr, return_tensors="pt").input_values # [1, wavlen]
                input_values = split_into_batch(input_values) # [bsize, maxlen=10*16000]
                if gpu != -1: input_values = input_values.to(device)
                hidden_states = model(input_values, output_hidden_states=True).hidden_states # tuple of (B, T, D)
                feature = torch.stack(hidden_states)[layer_ids].sum(dim=0)  # (B, T, D) # -> compress waveform channel
                bsize, segnum, featdim = feature.shape
                feature = feature.view(-1, featdim).detach().squeeze().cpu().numpy() # (B*T, D)

        ## store values
        csv_file = os.path.join(save_dir, f'{vid}.npy')
        if feature_level == 'UTTERANCE':
            feature = np.array(feature).squeeze()
            if len(feature.shape) != 1:
                feature = np.mean(feature, axis=0)
            np.save(csv_file, feature)
        else:
            np.save(csv_file, feature)

    end_time = time.time()
    print(f'Total time used: {end_time - start_time:.1f}s.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--model_name', type=str, default='clap', help='feature extractor')
    parser.add_argument('--feature_level', type=str, default='FRAME', help='FRAME or UTTERANCE')
    parser.add_argument('--audio_dir', type=str, help='path to audio directory')
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    args = parser.parse_args()

    audio_dir = args.audio_dir
    output_dir = args.output_dir
    feature_level = args.feature_level
    model_name = args.model_name

    print(f"Extracting {feature_level} features with {model_name} and saving in {output_dir}")
    
    # audio_files
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    print(f'Find total "{len(audio_files)}" audio files.')


    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # extract features
    extract(model_name, audio_files, output_dir, feature_level , 1)




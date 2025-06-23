import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
from multiprocessing import Pool, cpu_count
import argparse
from tqdm import tqdm


def ASR(wav_folder, output_folder): 

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    torch_dtype = torch.float32  # Force full precision
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    os.makedirs(output_folder, exist_ok=True) 


    wav_files = [os.path.join(wav_folder, f) for f in os.listdir(wav_folder) if f.endswith(".wav")] 
    
    # Process files
    for file_path in tqdm(wav_files, desc="Processing WAV files"):
        try:
            result = pipe(file_path)
            transcription = result["text"]

            # Save transcription to a file in the output directory
            output_filename = os.path.join(output_folder, os.path.splitext(os.path.basename(file_path))[0] + ".txt")
            with open(output_filename, "w") as f:
                f.write(transcription)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe WAV files using Whisper")
    parser.add_argument("input_folder", help="Folder containing input WAV files")
    parser.add_argument("output_folder", help="Folder to save transcriptions")

    args = parser.parse_args()
    
    wav_folder = args.input_folder
    output_folder = args.output_folder

    ASR(wav_folder, output_folder)
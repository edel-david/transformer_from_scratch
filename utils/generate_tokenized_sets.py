import os
from pathlib import Path
import miditok
import numpy as np

from models.GoePT.model_old import Dataset

def hash_to_path(data_dir, hash, suffix = '.bin'):
    return os.path.join(data_dir, 'raw_files', hash[0], hash[1], hash[2], hash + suffix)

ERROR_LOG_PATH = Path("data/tokenized/error_encoding_log.txt")

def log_error(message):
    print(message)
    with ERROR_LOG_PATH.open("a", encoding="utf-8") as err_file:
        err_file.write(message + "\n")

def create_tokenizer():
    # Custom Params for Tokenizer
    TOKENIZER_PARAMS = {
        "pitch_range": (21, 109),
        "beat_res": {(0, 4): 8, (4, 12): 4},
        "num_velocities": 32,
        "special_tokens": ["PAD", "EOS", "MASK", "GENRE"],
        "use_chords": True,
        "use_rests": False,
        "use_tempos": True,
        "use_time_signatures": False,
        "use_programs": True,
        "num_tempos": 32,
        "tempo_range": (40, 250),
    }

    # Creating multitrack tokenizer
    config = miditok.TokenizerConfig(**TOKENIZER_PARAMS)
    tokenizer = miditok.REMI(config)
    print(f"Length of the vocabulary: {tokenizer.len}")
    return tokenizer

def create_tokenized_datasets(data_dir, set_name = 'train'):
    tokenizer = create_tokenizer()
    assert tokenizer.vocab_size <= 2**16, "Vocab size too large for uint16 encoding"

    with open(os.path.join(data_dir, set_name + '.csv'), 'r') as f:
        hashes_and_genres = np.array([line.strip().split(',') for line in f.readlines()])
    
    skipped = 0
    i = 1
    HASHES_TOTAL = len(hashes_and_genres)
    for hash, genre in hashes_and_genres:
        path = hash_to_path(data_dir, hash, suffix='.mid')
        
        try:
            tokens_list = tokenizer.encode(path)
        except (ValueError, RuntimeError) as e:
            log_error(f"Error encoding {path}: {e}. Skipping this file.")
            skipped += 1
            continue
        
        tokenized_path = hash_to_path(data_dir, hash)
        arr = np.array(tokens_list, dtype=np.uint16)
        arr.tofile(tokenized_path)
        print(f"Tokenized {i}/{HASHES_TOTAL} files ({(i / HASHES_TOTAL * 100):.2f}%) - {skipped} skipped ({(skipped / HASHES_TOTAL * 100):.2f}%).")
        i += 1


if __name__ == "__main__":
    data_dir = "data/"
    #create_tokenized_datasets(data_dir, set_name = 'validation')
    dataset = Dataset('validation')
    slices = dataset.get_slices(512)
    for genre in slices.keys():
        print(genre, len(slices[genre]))

    used = set()
    i = 0
    j = 0
    while True:
        slices, genres = dataset.get_batch_from_slices(16, np.random.default_rng())
        for slice in slices:
            sliceAsString = str(list(slice))
            if sliceAsString in used:
                print("Duplicate ", i)
                i += 1
            else:
                used.add(sliceAsString)
                print("Good ", j)
                j += 1

        






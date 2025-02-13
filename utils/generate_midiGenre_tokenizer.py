import csv
from pathlib import Path
import numpy as np
from miditok import REMI, TokenizerConfig
from miditok.utils import split_seq_in_subsequences


# Hash-to-genre mapping from a CSV file
def map_hash_to_genre(csv_path) -> map:
    hash_to_genre = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            midi_hash, genre = row[0], row[1]
            hash_to_genre[midi_hash] = genre
    return hash_to_genre


# Get hashes from the different CSV files (for train, val, test)
# Set datastructure ensures no duplicates
def get_hashes(csv_path):
    hashes = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            midi_hash = row[0]
            hashes.add(midi_hash)
    return hashes


def create_tokenizer(unique_genres):

    # handle spaces or dashes as you please
    # e.g. "Hip-Hop" => "BOS_Hip-Hop" is fine, as it only has one underscore
    # which is required by MIDITok to parse "type_value"
    # If the name itself has underscores, they might be turned into dashes.
    # If you want to avoid confusion, you could replace spaces/dashes yourself.
    genre_special_tokens = [f"GENRE_{genre}" for genre in unique_genres]

    # Custom Params for Tokenizer
    TOKENIZER_PARAMS = {
        "pitch_range": (21, 109),
        "beat_res": {(0, 4): 8, (4, 12): 4},
        "num_velocities": 32,
        "special_tokens": ["PAD", "EOS", "MASK"] + genre_special_tokens,
        "use_chords": True,
        "use_rests": False,
        "use_tempos": True,
        "use_time_signatures": False,
        "use_programs": True,
        "num_tempos": 32,
        "tempo_range": (40, 250),
    }

    # Creating multitrack tokenizer
    config = TokenizerConfig(**TOKENIZER_PARAMS)
    tokenizer = REMI(config)
    print(f"Length of the vocabulary: {tokenizer.len}")
    return tokenizer


ERROR_LOG_PATH = Path("../data/tokenized/error_encoding_log.txt")


def process_dataset(
    midi_paths,
    hash_to_genre,
    tokenizer,
    output_dir,
    dataset_name,
):

    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ERROR_LOG_PATH.open("a", encoding="utf-8") as err_file:
        err_file.write(f"### DATASET: {dataset_name} ###" + "\n")

    csv_sequence_lengths = []

    for midi_path in midi_paths:
        midi_hash = midi_path.stem

        if midi_hash not in hash_to_genre:
            print(f"Warning: {midi_hash} not found in CSV. Skipping.")
            continue

        genre = hash_to_genre[midi_hash]
        genre_token_str = f"GENRE_{genre}"

        # Convert to integer ID from vocubulary
        if genre_token_str not in tokenizer.vocab:
            print(
                f"Warning: genre token {genre_token_str} not found in vocab. Skipping."
            )
            continue

        # genre_token_id = tokenizer.vocab[genre_token_str]

        # Tokenize MIDI file
        try:
            tokens = tokenizer.encode(midi_path)
        except (ValueError, RuntimeError) as e:
            error_message = f"Error encoding {midi_path}: {e}. Skipping this file."
            print(error_message)
            with ERROR_LOG_PATH.open("a", encoding="utf-8") as err_file:
                err_file.write(error_message + "\n")
            continue

        # chunked_sequences = split_seq_in_subsequences(
        #     tokens_list, min_seq_len=max_seq_len - 10, max_seq_len=max_seq_len - 1
        # )

        token_ids = np.array(tokens.ids, dtype=np.uint16)

        # relative_path = midi_path.relative_to("../data/raw_files")
        filename = midi_path.stem + f".bin"
        tokenized_path = Path(output_dir, genre_token_str, filename)
        tokenized_path.parent.mkdir(parents=True, exist_ok=True)

        token_ids.tofile(tokenized_path)

        # Log each md5 token size in the CSV
        csv_sequence_lengths.append(
            {
                "md5": midi_hash,
                "total_tokens": len(tokens.ids),
            }
        )

        # tokens = tokens_list.ids
        # # Insert genre token at the start
        # tokens.insert(0, genre_token_id)

        # train_data = np.array(tokens, dtype=np.uint16)

        # relative_path = midi_path.relative_to("../data/raw_files")
        # midi_tokenized_path = Path(output_dir, relative_path)
        # midi_tokenized_path.parent.mkdir(parents=True, exist_ok=True)

        # train_data.tofile(midi_tokenized_path.with_suffix(".bin"))

    csv_output_path = Path(output_dir, f"{dataset_name}_token_length_summary.csv")
    with csv_output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["md5", "total_tokens"])
        writer.writeheader()
        writer.writerows(csv_sequence_lengths)

    print(f"Finished processing {dataset_name} dataset!")
    print("---  ---")


if __name__ == "__main__":

    csv_path_train = Path("../data/train.csv")
    csv_path_val = Path("../data/validation.csv")
    csv_path_test = Path("../data/test.csv")

    hash_to_genre = {}
    hash_to_genre.update(map_hash_to_genre(csv_path_train))
    hash_to_genre.update(map_hash_to_genre(csv_path_val))
    hash_to_genre.update(map_hash_to_genre(csv_path_test))

    unique_genres = sorted(set(hash_to_genre.values()))

    # Initialize tokenizer
    tokenizer = create_tokenizer(unique_genres)
    tokenizer.save(Path("../models/tokenizers/midi_tokenizer.json"))

    # Get all MIDI paths from raw_files
    midi_paths = list(Path("../data/raw_files").glob("**/*.mid"))

    # Load hashes for train, val, and test
    train_hashes = get_hashes(csv_path_train)
    val_hashes = get_hashes(csv_path_val)
    test_hashes = get_hashes(csv_path_test)

    # Separate paths into train, val, and test
    datasets = {"train": [], "val": [], "test": []}
    removed_files = []

    for midi_path in midi_paths:
        midi_hash = midi_path.stem
        if midi_hash in train_hashes:
            datasets["train"].append(midi_path)
        elif midi_hash in val_hashes:
            datasets["val"].append(midi_path)
        elif midi_hash in test_hashes:
            datasets["test"].append(midi_path)
        else:
            removed_files.append(str(midi_path))

    if removed_files:
        removed_files_path = Path("../data/missing_files.txt")
        with removed_files_path.open("w", encoding="utf-8") as f:
            for file in removed_files:
                f.write(file + "\n")
        print(
            f"Encountered {len(removed_files)} missing files. Stored to: {removed_files_path}"
        )
    else:
        print(f"Encountered no missing files!")

    # Process each dataset (train, val, test)
    for dataset_name, paths in datasets.items():
        output_dir = Path(f"../data/tokenized/{dataset_name}")
        print(f"Processing {dataset_name} dataset with {len(paths)} files ...")
        process_dataset(paths, hash_to_genre, tokenizer, output_dir, dataset_name)


# müssen warhscheinlich files noch in chunks bringen, dieselbe Größe wie die Context size?
# Interferred das Classification Token mit dem Start-of-Sequence Token?
# Bzw. was passiert, wenn es dann immer vorhanden ist beim ersten Chunk, aber bei den restlichen nicht?
# Sollten wir sie dann für alle entfernen?
# Was ist mit Chunks, die weniger Tokens enthalten als die Conxtext size?
# Antwort: Werden nicht hinzugefügt, also gelöscht
# Sollten sie einfach entfernt werden? Dann könnten wir es hier abfangen
# Müssen noch die Implementierung im Model machen, sollte aber recht einfach sein
# tokenizer notwendig, bzw. müssen wir ihn ersetzen durch unseren?
# überlegen wie input übergeben werden soll, da sie ganze Datensätze zusammengefügt haben mit train.bin, val.bin, test.bin

# Decoded Token sequence, wo die Token wieder Strings repräsentieren, kann mit anderem utils Programm gelesen werden

# tokenizer noch trainieren mit BPE? Wird häufig empfohlen
# tokenizer.train(vocab_size=10000, files_paths=midi_paths)
# tokenizer.save(Path("../models/tokenizers/midi_tokenizer_trained.json"))

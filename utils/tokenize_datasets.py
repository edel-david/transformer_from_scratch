#!/usr/bin/env python3

import os
import argparse

import numpy as np
from tokenizers import Tokenizer
from icecream import ic


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="../models/tokenizers/" "goethization_tokenizer.json",
    )

    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)

    # Gather all segmented text files
    files_preprocessed = [f for f in os.listdir(args.input_dir) if f.endswith(".txt")]

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    for file_name in files_preprocessed:
        file_path_preprocessed = os.path.join(args.input_dir, file_name)

        file_name_tokenized = os.path.join(
            args.output_dir, file_name.replace(".txt", ".bin")
        )

        with open(file_path_preprocessed, "r", encoding="utf-8") as in_file:
            data_preprocessed = in_file.read()

            # data_segmented = data_preprocessed.split('<|S|>')
            # data_segmented = [segment.replace('<|E|>', '') for segment in data_segmented]

            # ic(data_segmented)

            # data_tokenized = tokenizer.encode_batch(data_segmented)

            # ic(data_tokenized[7].type_ids)
            # ic(data_tokenized[7].tokens)
            # ic(data_tokenized[7].offsets)
            # ic(data_tokenized[7].attention_mask)
            # ic(data_tokenized[7].special_tokens_mask)
            # ic(data_tokenized[7].overflowing)
            # ic(data_tokenized[7].ids)

            data_tokenized = tokenizer.encode(data_preprocessed)

            # ic(data_tokenized.type_ids)
            # ic(data_tokenized.tokens)
            # ic(data_tokenized.offsets)
            # ic(data_tokenized.attention_mask)
            # ic(data_tokenized.special_tokens_mask)
            # ic(data_tokenized.overflowing)

            # for token, id in zip(data_tokenized.tokens[:128], data_tokenized.ids[:128]):
            #     print(f'{token} {id}')

            # exit()

            data_tokenized = cp.array(data_tokenized.ids, dtype=cp.uint16)

            ic(data_tokenized[:100])

            data_tokenized.tofile(file_name_tokenized)

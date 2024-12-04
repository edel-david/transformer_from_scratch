#!/usr/bin/env python3

import os
import argparse

from tokenizers import Tokenizer, Regex
from tokenizers.models import WordLevel, WordPiece, BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import Split as SplitPreTokenizer
from tokenizers.pre_tokenizers import Whitespace as WhitespacePreTokenizer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.pre_tokenizers import Sequence as PTSequence
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import WordLevelTrainer, WordPieceTrainer, BpeTrainer
from tokenizers.tools import EncodingVisualizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)

    parser.add_argument(
        "--test-string",
        type=str,
        default="""<|S|>
SCHÜLER.
Doch ein Begriff muß bei dem Worte sein.

MEPHISTOPHELES.
Schon gut! Nur muß man sich nicht allzu ängstlich quälen
Denn eben wo Begriffe fehlen,
Da stellt ein Wort zur rechten Zeit sich ein.
Mit Worten läßt sich trefflich streiten,
Mit Worten ein System bereiten,
An Worte läßt sich trefflich glauben,
Von einem Wort läßt sich kein Jota rauben.
<|E|>""",
    )

    args = parser.parse_args()

    # WordLevel tokenizer

    tokenizer = Tokenizer(WordLevel(unk_token="<|UNK|>"))
    trainer = WordLevelTrainer(special_tokens=["<|S|>", "<|E|>", "<|UNK|>"])

    tokenizer.pre_tokenizer = PTSequence(
        [
            WhitespacePreTokenizer(),
            SplitPreTokenizer(Regex("(<\|S\|>|<\|E\|>)"), behavior="isolated"),
        ]
    )

    files = [
        os.path.join(args.input_dir, f"{split}.txt")
        for split in ["test", "train", "val"]
    ]

    tokenizer.train(files, trainer)

    print(f"WordLevel tokenizer vocab size: {tokenizer.get_vocab_size()}")

    # input_text_tokenized = tokenizer.encode(args.test_string)
    input_text_tokenized = tokenizer.encode("Mit Worten läßt sich trefflich streiten,")

    print(input_text_tokenized.ids)
    print(tokenizer.decode(input_text_tokenized.ids))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # with open(os.path.join(args.output_dir, 'visualizations/word_level_visualization.html'), 'w', encoding='utf-8')
    # as out_file:
    #     out_file.write(EncodingVisualizer(tokenizer, default_to_notebook=False)(args.test_string))

    # tokenizer.save(os.path.join(args.output_dir, 'vocabs/word_level.json'))

    # Character-level tokenizer

    tokenizer = Tokenizer(WordPiece(unk_token="<|UNK|>", max_input_chars_per_word=1))
    trainer = WordPieceTrainer(special_tokens=["<|S|>", "<|E|>", "<|UNK|>"])

    tokenizer.pre_tokenizer = PTSequence(
        [
            SplitPreTokenizer(Regex("(<\|S\|>|<\|E\|>)"), behavior="isolated"),
            SplitPreTokenizer("", behavior="isolated"),
        ]
    )

    tokenizer.train(files, trainer)

    print(f"Character-level tokenizer vocab size: {tokenizer.get_vocab_size()}")

    # input_text_tokenized = tokenizer.encode(args.test_string)
    input_text_tokenized = tokenizer.encode("Mit Worten läßt sich trefflich streiten,")

    print(input_text_tokenized.ids)
    print(tokenizer.decode(input_text_tokenized.ids))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # with open(os.path.join(args.output_dir, 'visualizations/char_level_visualization.html'), 'w', encoding='utf-8')
    # as out_file:
    #     out_file.write(EncodingVisualizer(tokenizer, default_to_notebook=False)(args.test_string))

    # tokenizer.save(os.path.join(args.output_dir, 'vocabs/char_level.json'))

    # BPE Tokenizer

    tokenizer = Tokenizer(BPE())

    normalizer = NFC()

    pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    processor = ByteLevelProcessor()
    decoder = ByteLevelDecoder()

    trainer = BpeTrainer(vocab_size=4096, special_tokens=["<|S|>", "<|E|>"])

    tokenizer.normalizer = normalizer

    split_pre_tokenizer = SplitPreTokenizer(
        Regex("(<\|S\|>|<\|E\|>)"), behavior="isolated"
    )

    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.post_processor = processor

    tokenizer.decoder = decoder

    tokenizer.train(files, trainer)

    print(f"BPE tokenizer vocab size: {tokenizer.get_vocab_size()}")

    # input_text_tokenized = tokenizer.encode(args.test_string)
    input_text_tokenized = tokenizer.encode("Mit Worten läßt sich trefflich streiten,")

    print(input_text_tokenized.ids)
    print(tokenizer.decode(input_text_tokenized.ids))

    with open(
        os.path.join(args.output_dir, "visualizations/bpe_visualization.html"),
        "w",
        encoding="utf-8",
    ) as out_file:
        out_file.write(
            EncodingVisualizer(tokenizer, default_to_notebook=False)(args.test_string)
        )

    tokenizer.save(os.path.join(args.output_dir, "vocabs/bpe.json"))

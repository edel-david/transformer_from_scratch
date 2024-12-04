#!/usr/bin/env python3

from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import Split as SplitPreTokenizer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers.tools import EncodingVisualizer


if __name__ == "__main__":

    tokenizer = Tokenizer(BPE())

    normalizer = NFC()

    # pre_tokenizer = ByteLevelPreTokenizer()
    pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    processor = ByteLevelProcessor()
    decoder = ByteLevelDecoder()

    trainer = BpeTrainer(vocab_size=8192, special_tokens=["<|S|>", "<|E|>", "<|PAD|>"])

    tokenizer.normalizer = normalizer

    split_pre_tokenizer = SplitPreTokenizer(
        Regex("(<\|S\|>|<\|E\|>)"), behavior="isolated"
    )

    # tokenizer.pre_tokenizer = PTSequence([split_pre_tokenizer,
    #                                                 pre_tokenizer])

    tokenizer.pre_tokenizer = pre_tokenizer

    # template_processor = TemplateProcessing(
    #     single='<|S|> $A <|E|>',
    #     pair='<|S|> $A <|E|> <|S|> $B:1 <|E|>',
    #     special_tokens=[('<|S|>', 0),
    #                         ('<|E|>', 1)],)

    # tokenizer.post_processor = PRSequence([template_processor, processor])
    tokenizer.post_processor = processor

    tokenizer.decoder = decoder

    tokenizer.enable_padding(pad_id=2, pad_token="<|PAD|>")

    files = [f"../datasets/shuffled/{split}.txt" for split in ["test", "train", "val"]]

    tokenizer.train(files, trainer)

    input_text = """<|S|>
FAUST.
Da steh ich nun, ich armer Tor!
Und bin so klug als wie zuvor;
<|E|>"""

    # input_text_tokenized = tokenizer.encode(input_text)

    # print(input_text_tokenized.ids)
    # print(tokenizer.decode(input_text_tokenized.ids))

    with open("./encoding_visualization.html", "w", encoding="utf-8") as out_file:
        out_file.write(
            EncodingVisualizer(tokenizer, default_to_notebook=False)(input_text)
        )

    tokenizer.save("../models/tokenizers/goe_pt/goe_pt_tokenizer.json")

from miditok import REMI, TokenizerConfig
import numpy as np

# Important: Configuration has to match to what was used the create the file
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


config = TokenizerConfig(**TOKENIZER_PARAMS)
tokenizer = REMI(config)
with open(
    "../data/tokenized/train/GENRE_hits of the 1960s/0a0edd409bfaf23e0364359d959f33db.bin",
    "rb",
) as f:
    token_ids = np.frombuffer(f.read(), dtype=np.uint16)

token_texts = tokenizer._ids_to_tokens(token_ids)
for token in token_texts:
    print(token)

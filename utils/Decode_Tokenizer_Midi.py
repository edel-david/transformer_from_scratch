from miditok import REMI, TokenizerConfig
import numpy as np

# Important: Configuration has to match to what was used the create the file
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["PAD", "EOS", "MASK"]
    + [
        "GENRE_baroque",
        "GENRE_classical",
        "GENRE_country",
        "GENRE_early 20th century",
        "GENRE_hip-hop-rap",
        "GENRE_hits of 2011 2020",
        "GENRE_hits of the 1960s",
        "GENRE_hits of the 1970s",
        "GENRE_hits of the 1980s",
        "GENRE_hits of the 1990s",
        "GENRE_hits of the 2000s",
        "GENRE_instrumental",
        "GENRE_italian%2cfrench%2cspanish",
        "GENRE_jazz",
        "GENRE_latino",
        "GENRE_medley",
        "GENRE_metal",
        "GENRE_modern",
        "GENRE_musical%2cfilm%2ctv",
        "GENRE_oldies",
        "GENRE_pop",
        "GENRE_renaissance",
        "GENRE_rnb-soul",
        "GENRE_rock",
        "GENRE_romantic",
        "GENRE_traditional",
        "GENRE_world",
    ],
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
    "../data/tokenized/train/1/2/0/120a5b2484c24f10677b7d347964b699_chunk0.bin", "rb"
) as f:
    token_ids = np.frombuffer(f.read(), dtype=np.uint16)

token_texts = tokenizer._ids_to_tokens(token_ids)
for token in token_texts:
    print(token)

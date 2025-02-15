import sys
import os
import math
import time
import argparse
from functools import partial
import json

import numpy as np

from tokenizers import Tokenizer
from rich.progress import Progress
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from icecream import ic

sys.path.append(".")

import layers as scr
from loss import cross_entropy_loss
from utils import compress_numpy_array, decompress_numpy_array

import warnings

warnings.filterwarnings("error")


ic.configureOutput(includeContext=True)
ic.disable()


class GoePT:

    def __init__(
        self,
        n_genres: int,
        vocab_size: int = 8192,
        context_length: int = 256,
        batch_size: int = 64,
        n_layer: int = 6,
        n_embd: int = 384,
        dropout: float = 0.2,
        lr: float = 1e-3,
    ) -> None:
        self.n_genres = n_genres
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.dropout = dropout
        self.lr = lr

        self.rng = np.random.default_rng()

        def weight_init(size):
            return self.rng.normal(size=size, loc=0.0, scale=0.02).astype(np.float32)

        def c_proj_weight_init(size):
            return self.rng.normal(
                size=size, loc=0.0, scale=0.02 / math.sqrt(2 * self.n_layer)
            ).astype(np.float32)

        def bias_init(size):
            return np.zeros(shape=size, dtype=np.float32)

        # Define lm_head first so we can pass its
        # weights_transposed property to the wte
        # embedding to implement weight tying

        self.lm_head = scr.Linear(
            self.n_embd,
            # self.vocab_size, OLD
            self.n_genres,
            self.batch_size,
            bias=False,
            lr=self.lr,
            weight_init_func=weight_init,
            bias_init_func=bias_init,
        )

        self.transformer = {
            "wte": scr.Embedding(
                self.vocab_size + 1,
                self.n_embd,
                self.batch_size,
                self.lr,
                init_func=weight_init,
                # weight_external=self.lm_head.weight_transposed, # Old, weight tying not possible with vocab_size != #classes
            ),
            "wpe": scr.Embedding(
                self.context_length,
                self.n_embd,
                self.batch_size,
                self.lr,
                init_func=weight_init,
            ),
            "drop": scr.Dropout(self.dropout),
            "h": [
                scr.Block(
                    self.n_embd,
                    self.context_length,
                    6,
                    self.batch_size,
                    self.lr,
                    self.dropout,
                    weight_init,
                    c_proj_weight_init,
                    bias_init,
                )
                for _ in range(self.n_layer)
            ],
            "ln_f": scr.LayerNorm(self.n_embd, weight_init_func=weight_init),
        }

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate

        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # assert id(self.transformer['wte'].weight) == id(self.lm_head.weight), "wte and lm_head must share the same weights in memory"
    
    def set_lr(self, lr):
        def recursive_set_lr(obj):
            if hasattr(obj, "lr"):
                obj.lr = lr
            if isinstance(obj, list):
                for elem in obj:
                    recursive_set_lr(elem)
            if isinstance(obj, dict):
                for key in obj:
                    recursive_set_lr(obj[key])
            if hasattr(obj, "__dict__"):
                for attr in obj.__dict__:
                    recursive_set_lr(getattr(obj, attr))
        
        recursive_set_lr(self)

    def forward(self, idx, targets=None, train=False):
        b, t = idx.shape
        assert (
            t <= self.context_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.context_length}"
        pos = np.arange(0, t, dtype=np.int64)  # shape (t)

        # replace the first token with the classification token:
        idx[:, 0] = (
            self.vocab_size
        )  # because the normal embedding uses indices 0 to vocab_size-1,
        # we can use vocab_size as the classification token.

        # Forward the GPT model itself
        # Token embeddings of shape (b, t, n_embd)
        tok_emb = self.transformer["wte"].forward(idx)

        # Position embeddings of shape (t, n_embd)
        pos_emb = self.transformer["wpe"].forward(pos)

        # Main transformer
        x = self.transformer["drop"].forward(tok_emb + pos_emb, True)
        for block in self.transformer["h"]:
            x = block.forward(x, train)
        x = self.transformer["ln_f"].forward(x)

        # Compute loss and return
        logits = self.lm_head.forward(
            x[:, [0], :]
        )  # pass the first token, the classification token, to the lm_head.

        if (
            targets is not None
        ):  # branch on knowledge of right answer: If known, calculate loss. else only return logits.
            # if we are given some desired targets also calculate the loss<
            # in both cases, we only apply the lm_head to the first token.
            ic(logits.shape, targets.shape)
            logits_for_loss = logits.reshape(-1, logits.shape[-1])
            targets_for_loss = np.expand_dims(targets.reshape(-1), 1)
            targets_for_loss = scr.one_hot(targets_for_loss, self.n_genres)
            loss = cross_entropy_loss(logits_for_loss, targets_for_loss)
        else:
            loss = None

        return logits, loss

    def backward(self, x):
        grad = self.lm_head.backward(x)
        grad = self.transformer["ln_f"].backward(grad)

        for block in reversed(self.transformer["h"]):
            grad = block.backward(grad)
        grad = self.transformer["drop"].backward(grad)
        self.transformer["wte"].backward(grad)
        self.transformer["wpe"].backward(grad.sum(axis=0))
        return

    def update(self):
        self.lm_head.update()
        self.transformer["ln_f"].update()
        for block in self.transformer["h"]:
            block.update()
        self.transformer["wte"].update()
        self.transformer["wpe"].update()
        return

    def state_dict(self):

        params_all = {
            "lm_head": [
                compress_numpy_array(self.lm_head.weight),
                compress_numpy_array(self.lm_head.bias),
            ],
            "wte": compress_numpy_array(self.transformer["wte"].weight),
            "wpe": compress_numpy_array(self.transformer["wpe"].weight),
            "ln_f": [
                compress_numpy_array(self.transformer["ln_f"].weight),
                compress_numpy_array(self.transformer["ln_f"].bias),
            ],
        }

        for idx, block in enumerate(self.transformer["h"]):
            params_all[f"block_{idx}"] = block.state_dict()

        state_dict = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "batch_size": self.batch_size,
            "n_layer": self.n_layer,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "lr": self.lr,
            "params": params_all,
        }

        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict: dict):

        goe_pt = cls(
            state_dict["vocab_size"],
            state_dict["context_length"],
            state_dict["batch_size"],
            state_dict["n_layer"],
            state_dict["n_embd"],
            state_dict["dropout"],
            state_dict["lr"],
        )

        goe_pt.lm_head.weight = decompress_numpy_array(
            state_dict["params"]["lm_head"][0]
        )
        goe_pt.lm_head.bias = decompress_numpy_array(state_dict["params"]["lm_head"][1])

        goe_pt.transformer["wte"].weight = decompress_numpy_array(
            state_dict["params"]["wte"]
        )
        goe_pt.transformer["wpe"].weight = decompress_numpy_array(
            state_dict["params"]["wpe"]
        )

        goe_pt.transformer["ln_f"].weight = decompress_numpy_array(
            state_dict["params"]["ln_f"][0]
        )
        goe_pt.transformer["ln_f"].bias = decompress_numpy_array(
            state_dict["params"]["ln_f"][1]
        )

        for idx, block in enumerate(goe_pt.transformer["h"]):
            block.load_params(state_dict["params"][f"block_{idx}"])

        return goe_pt

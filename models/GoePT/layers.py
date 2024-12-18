# Transformer-specific layers adapted from numpy-based implementation
# by Rustam Akimov
#
# Implementation available under
# https://github.com/AkiRusProd/numpy-nn-model
#
# and nanoGPT as implemented by Andrej Karpathy
#
# Implementation available under
# https://github.com/karpathy/nanoGPT

import sys
import math
import copy
from types import NoneType
from typing import Union, Callable
import wandb
from utils import log,log_one
import numpy as np
import cupy as cp
#from numpy.typing import ArrayLike
from icecream import ic
from cupy.typing import ArrayLike
sys.path.append(".")

from utils import compress_numpy_array, decompress_numpy_array


class Linear:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        batch_size: int,
        lr: float = 0.1,
        bias: bool = True,
        weight_init_func: Union[Callable, None] = None,
        bias_init_func: Union[Callable, None] = None,
    ) -> None:

        super(Linear, self).__init__()
        self.batch_size = batch_size
        self.lr = lr

        self.use_bias = bias

        self.weight_init_func = weight_init_func
        self.bias_init_func = bias_init_func

        if self.weight_init_func:
            self.weight = cp.asanyarray(
                self.weight_init_func((in_features, out_features))
            )
        else:
            self.weight = cp.random.normal(size=(in_features, out_features)) * cp.sqrt(
                1.0 / in_features
            )

        if self.bias_init_func:
            self.bias = cp.asanyarray(self.bias_init_func((in_features, out_features)))
        else:
            self.weight = cp.random.normal(size=(in_features, out_features)) * cp.sqrt(
                1.0 / in_features
            )

        if self.use_bias:
            if self.bias_init_func:
                self.bias = cp.asanyarray(self.bias_init_func((out_features,)))
            else:
                self.bias = cp.random.normal(size=(out_features,)) * cp.sqrt(
                    1.0 / in_features
                )

        self.grad_weight = cp.zeros((in_features, out_features))
        self.grad_bias = cp.zeros(out_features)

        self.input = cp.zeros((batch_size, in_features))

    def _multi_dim_matmul(
        self,
        mat_a: cp.ndarray,
        mat_b: cp.ndarray,
        transpose_a: bool = False,
        transpose_b: bool = False,
        reshape_output: bool = True,
    ) -> cp.ndarray:
        """
        Replicate torch behavior of flattening all but the
        last dimension of an input of the matrix multiplication
        in linear layers. We implement this for both the first
        and the last matrix in the matrix multiplication to
        provide a unified operation for both the forward and
        the backward pass.
        """

        if (len(mat_a.shape) > 2) or (len(mat_b.shape) > 2):
            # Dimension handling.
            # We should refactor this if we find the time.

            dims_internal_mat_a = (
                mat_a.shape
                if len(mat_a.shape) <= 2
                else (cp.prod(cp.array(mat_a.shape[:-1])).item(), mat_a.shape[-1])
            )

            dims_internal_mat_b = (
                mat_b.shape
                if len(mat_b.shape) <= 2
                else (cp.prod(cp.array(mat_b.shape[:-1])).item(), mat_b.shape[-1])
            )

            mat_a_shape = mat_a.shape[::-1] if transpose_a else mat_a.shape
            mat_b_shape = mat_b.shape[::-1] if transpose_b else mat_b.shape

            dims_out_first = (
                mat_a.shape[:-1]
                if reshape_output
                else (
                    dims_internal_mat_a[1] if transpose_a else dims_internal_mat_a[0],
                )
            )

            dims_out = (*dims_out_first, mat_b_shape[-1])

            def mat_a_transform():
                if transpose_a:
                    return mat_a.reshape(dims_internal_mat_a).T
                else:
                    return mat_a.reshape(dims_internal_mat_a)

            def mat_b_transform():
                if transpose_b:
                    return mat_b.reshape(dims_internal_mat_b).T
                else:
                    return mat_b.reshape(dims_internal_mat_b)

            return cp.matmul(mat_a_transform(), mat_b_transform()).reshape(dims_out)

        else:
            return cp.matmul(mat_a, mat_b.T) if transpose_b else cp.matmul(mat_a, mat_b)

    def forward(self, input: ArrayLike) -> cp.ndarray:

        self.input = cp.asanyarray(input)

        output = self._multi_dim_matmul(self.input, self.weight)
        if self.use_bias:
            output += self.bias
        return output

    def backward(self, grad_output: ArrayLike) -> ArrayLike:

        grad_output = cp.asanyarray(grad_output)

        grad_input = self._multi_dim_matmul(grad_output, self.weight, transpose_b=True)

        # flattened_input_shape = (cp.prod(self.input.shape[:-1]), self.input.shape[-1])
        # flattened_grad_output_shape = (
        #     cp.prod(grad_output.shape[:-1]),
        #     grad_output.shape[-1],
        # )

        self.grad_weight = (1.0 / self.batch_size) * self._multi_dim_matmul(
            self.input, grad_output, transpose_a=True, reshape_output=False
        )

        if self.use_bias:
            self.grad_bias = (1.0 / self.batch_size) * grad_output.sum(0)

        return grad_input

    def update(self) -> None:
        self.weight = self.weight - self.lr * self.grad_weight
        if self.use_bias:
            self.bias = self.bias - self.lr * self.grad_bias

    @property
    def weight_transposed(self):
        return self.weight.T

    @weight_transposed.setter
    def weight_transposed(self, value):
        self.weight = value.T


class Sigmoid:
    def __init__(self, in_features: int, batch_size: int):
        super(Sigmoid, self).__init__()
        self.input = cp.zeros(batch_size)

    def forward(self, input):
        input = cp.asanyarray(input)
        self.input = input
        return 1.0 / (1.0 + cp.exp(-input))

    def backward(self, grad_output):
        # part of template
        grad_output = cp.asanyarray(grad_output)
        grad_input = (
            grad_output * cp.exp(-self.input) / cp.power(1.0 + cp.exp(-self.input), 2)
        )
        return grad_input


class Softmax:
    def __init__(self, axis: int = 1):
        self.input = None
        self.output = None
        self.axis = axis

    def forward(self, input: ArrayLike):
        self.input = cp.asanyarray(input)
        shifted_inp = input - cp.max(input, axis=self.axis, keepdims=True)
        exp_res = cp.exp(shifted_inp)
        output = exp_res / cp.sum(exp_res, axis=self.axis, keepdims=True)
        self.output = output
        return output

    def backward(self, grad: ArrayLike):
        # not part of task, included in template
        grad = cp.asanyarray(grad)
        f_x = self.output
        grad = (grad - (grad * f_x).sum(self.axis, keepdims=True)) * f_x
        return grad


class Dropout:
    def __init__(self, p: float = 0.2):

        self.p = p
        self.scale = 1 / (1 - p)

        self.rng = cp.random.default_rng()

        self.mask = None
        self.input = None

    def forward(self, input: ArrayLike, train: bool = False) -> ArrayLike:

        input = cp.asanyarray(input)

        self.input = input

        if train:
            self.mask = (
                self.rng.binomial(1, 1 - self.p, size=input.shape).astype(input.dtype)
                * self.scale
            )
        else:
            self.mask = 1

        return input * self.mask

    def backward(self, grad):
        grad_out = grad * self.mask  # mask includes the scale!
        return grad_out


def one_hot(a, num_classes):
    return cp.squeeze(cp.eye(num_classes)[a.reshape(-1)])


class LayerNorm:
    def __init__(
        self,
        normalized_shape: Union[int, tuple[int]],
        eps: float = 1e-05,
        lr: float = 1e-3,
        weight_init_func: Union[Callable, None] = None,
        bias_init_func: Union[Callable, None] = None,
    ) -> None:

        self.normalized_shape = (
            (normalized_shape,)
            if isinstance(normalized_shape, int)
            else normalized_shape
        )

        self.eps = eps
        self.lr = lr

        self.weight_init_func = weight_init_func
        self.bias_init_func = bias_init_func

        if self.weight_init_func:
            self.weight = cp.asanyarray(self.weight_init_func((normalized_shape)))
        else:
            self.weight = cp.ones((normalized_shape), dtype=cp.float64)
        if self.bias_init_func:
            self.bias = cp.asanyarray(self.bias_init_func((normalized_shape)))
        else:
            self.bias = cp.zeros((normalized_shape), dtype=cp.float64)

        self.axis = None

        self.input = None

        self.grad_weight = None
        self.grad_bias = None

        self.x_centered = None
        self.stddev_inv = None

    def forward(self, input: ArrayLike) -> cp.ndarray:

        input = cp.asanyarray(input)

        self.input = input

        self.axis = tuple(range(-len(self.normalized_shape), 0))
        #  -n,..., -2 , -1 ohne 0

        mean = cp.mean(input, axis=self.axis, keepdims=True)
        var = cp.var(
            input,
            axis=self.axis,
            keepdims=True,  # mean=mean
        )  # can we pass the mean to the var()?  YES (with newer numpy versions)!
        # the var stays the same after centering. Usefull for gradient calculation (not really)
        self.x_centered = input - mean
        self.stddev_inv = 1 / cp.sqrt(var + self.eps)

        output = self.x_centered * self.stddev_inv

        return self.weight * output + self.bias

    def backward_old(self, grad: ArrayLike) -> cp.ndarray:
        B, T, C = self.input.shape
        self.grad_bias = grad.mean(axis=(0, 1))  # upstream gradient * 1.
        self.grad_weight = (grad * (self.x_centered * self.stddev_inv)).mean(
            axis=(0, 1)
        )  # upstream * centered * invvar

        normalized = self.x_centered * self.stddev_inv
        grad_normalized = grad * self.weight
        grad_x = (
            grad_normalized
            - grad_normalized.mean(-1, keepdims=True)
            - normalized * (grad_normalized * normalized).mean(-1, keepdims=True)
        )
        grad_x = grad_x * self.stddev_inv
        return grad_x

    def backward(self, upstream_grad: ArrayLike) -> ArrayLike:
        self.grad_bias = upstream_grad.mean(axis=(0, 1))  # upstream gradient * 1.
        self.grad_weight = (upstream_grad * (self.x_centered * self.stddev_inv)).mean(
            axis=(0, 1)
        )  # upstream * centered * invvar

        s1 = (upstream_grad * self.weight * self.input).mean(-1, keepdims=True)
        # mean instead of sum, to avoid division by N
        grad_normalized = upstream_grad * self.weight
        lambd = (
            grad_normalized.sum(-1, keepdims=True) * self.input.mean(-1, keepdims=True)
            - s1 * self.stddev_inv**3
        )
        theta = (
            -lambd * self.input.mean(-1, keepdims=True)
            - grad_normalized.mean(-1, keepdims=True) * self.stddev_inv
        )
        grad_x = (
            self.stddev_inv * upstream_grad * self.weight + lambd * self.input + theta
        )
        return grad_x

    def update(self):
        self.weight -= self.lr * self.grad_weight
        self.bias -= self.lr * self.grad_bias
        return


class GELU:
    def __init__(self) -> None:
        self._sqrt_of_2_by_pi = cp.sqrt(2 / cp.pi)
        self.input = None

    def forward(self, input: ArrayLike) -> cp.ndarray:
        self.input = cp.asanyarray(input)
        return (
            0.5
            * input
            * (
                1
                + cp.tanh(
                    self._sqrt_of_2_by_pi * (input + 0.044715 * cp.power(input, 3))
                )
            )
        )

    def backward(self, grad_output: ArrayLike) -> cp.ndarray:
        x = self.input
        m1 = self._sqrt_of_2_by_pi
        m2 = 0.044715
        m3 = m1 * (x + m2 * x**3)
        tanhm3 = cp.tanh(m3)
        first = 0.5 * (1 + tanhm3)
        second = x / 2 * (1 - tanhm3**2) * (m1 + 2 * x**2 * m2 * m1)
        grad_out = (first + second) * grad_output
        return grad_out


class MLP:
    def __init__(
        self,
        d_model,
        batch_size,
        lr,
        dropout,
        c_fc_init_func,
        c_proj_init_func,
        bias_init_func,
    ):

        self.d_model = d_model
        self.batch_size = batch_size
        self.lr = lr

        self.c_fc = Linear(
            d_model,
            4 * d_model,
            batch_size,
            lr,
            weight_init_func=c_fc_init_func,
            bias_init_func=bias_init_func,
        )
        self.gelu = GELU()

        self.c_proj = Linear(
            4 * d_model,
            d_model,
            batch_size,
            lr,
            weight_init_func=c_proj_init_func,
            bias_init_func=bias_init_func,
        )

        self.dropout = Dropout(dropout)

    def forward(self, x: cp.ndarray,train:bool) -> cp.ndarray:
        x = self.c_fc.forward(x)
        x = self.gelu.forward(x)
        x = self.c_proj.forward(x)
        x = self.dropout.forward(x,train)
        return x

    def backward(self, x: cp.ndarray) -> cp.ndarray:
        x = self.dropout.backward(x)
        x = self.c_proj.backward(x)
        x = self.gelu.backward(x)
        x = self.c_fc.backward(x)
        return x

    def update(self) -> None:
        self.c_fc.update()
        self.c_proj.update()
        return

    def get_params(self) -> dict:
        return {
            "c_fc": [
                compress_numpy_array(self.c_fc.weight),
                compress_numpy_array(self.c_fc.bias),
            ],
            "c_proj": [
                compress_numpy_array(self.c_proj.weight),
                compress_numpy_array(self.c_proj.bias),
            ],
        }

    def load_params(self, state_dict: dict) -> None:
        self.c_fc.weight = decompress_numpy_array(state_dict["c_fc"][0])
        self.c_fc.bias = decompress_numpy_array(state_dict["c_fc"][1])
        self.c_proj.weight = decompress_numpy_array(state_dict["c_proj"][0])
        self.c_proj.bias = decompress_numpy_array(state_dict["c_proj"][1])


class MultiHeadAttention:
    def __init__(
        self,
        d_model: int,
        context_size: int,
        n_heads: int,
        batch_size: int,
        lr: float = 0.1,
        dropout: float = 0.1,
        c_attn_weight_init_func: Union[Callable, None] = None,
        c_proj_weight_init_func: Union[Callable, None] = None,
        bias_init_func: Union[Callable, None] = None,
    ) -> None:

        self.d_model = d_model
        self.context_size = context_size
        self.n_heads = n_heads
        self.scale = math.sqrt(d_model)
        self.batch_size = batch_size

        self.attn_dropout = Dropout(dropout)
        self.resid_dropout = Dropout(dropout)
        self.softmax_attn = Softmax(axis=-1)

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.depth = d_model // n_heads

        self.c_attn = Linear(
            d_model,
            3 * d_model,
            batch_size,
            lr,
            weight_init_func=c_attn_weight_init_func,
            bias_init_func=bias_init_func,
        )

        self.c_proj = Linear(
            d_model,
            d_model,
            batch_size,
            lr,
            weight_init_func=c_proj_weight_init_func,
            bias_init_func=bias_init_func,
        )

        self.mask = cp.tril(
            cp.ones((context_size, context_size), dtype=cp.float64)
        ).reshape(1, 1, context_size, context_size)

        self.input = None
        self.v = None
        self.q = None
        self.k = None
        self.attn = None

    def forward(self, input: ArrayLike,train:bool) -> tuple:

        self.input = cp.asanyarray(input)

        B, T, C = self.input.shape

        q, k, v = cp.split(self.c_attn.forward(self.input), 3, axis=2)

        k = k.reshape((B, T, self.n_heads, C // self.n_heads)).transpose(
            0, 2, 1, 3
        )  # (B, nh, T, hs)
        q = q.reshape((B, T, self.n_heads, C // self.n_heads)).transpose(
            0, 2, 1, 3
        )  # (B, nh, T, hs)
        v = v.reshape((B, T, self.n_heads, C // self.n_heads)).transpose(
            0, 2, 1, 3
        )  # (B, nh, T, hs)

        # this works because we use reduced dimensionalality with mutlihead attention,
        # making it similar in complexity to single attention.

        self.k = k
        self.q = q
        self.v = v

        attn = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(k.shape[-1]))
        # k.shape[-1] == C // self.n_heads == multi_head_attention_head_dim == depth

        attn = cp.where(self.mask == 0, -1e9, attn)
        attn = self.softmax_attn.forward(attn)
        attn = self.attn_dropout.forward(attn,train)

        self.attn = attn  # 16 x 6 x 256 x 256
        # v.shape: 16 x 6 x 256 x 64
        x = attn @ v  # x: 16 x 6 x 256 x 64

        x = (
            cp.ascontiguousarray(x)
            .transpose(0, 2, 1, 3)
            .reshape(B, -1, self.n_heads * self.depth)
        )
        x = self.c_proj.forward(x)  # keeps dims
        x = self.resid_dropout.forward(x,train)

        return x, attn

    def backward(self, grad: ArrayLike) -> cp.ndarray:
        # grad: 16 x 256 x ...
        
        B, T, C = self.input.shape
        grad = self.resid_dropout.backward(grad)
        grad = self.c_proj.backward(grad)
        grad = grad.reshape((B, T, self.n_heads, self.depth)).transpose(0, 2, 1, 3)

        v_grad2 = self.attn.transpose(0, 1, 3, 2) @ grad
        # long_grad is gradient for self.attn
        long_grad = grad @ self.v.transpose(0, 1, 3, 2)  # long_grad: 16 x 6 x 256 x 64

        # v.shape: 16 x 6 x 256 x 64
        long_grad = self.attn_dropout.backward(long_grad)
        long_grad = self.softmax_attn.backward(long_grad)
        long_grad = cp.where(self.mask == 0, 0, long_grad)

        long_grad = long_grad * (1 / cp.sqrt(self.depth))
        q_grad = long_grad @ self.k  # insert dimensions swaps
        k_grad = long_grad.transpose(0, 1, 3, 2) @ self.q  #
    

        grad = cp.concatenate(
            (
                q_grad.transpose(0, 2, 1, 3).reshape((B, T, C)),
                k_grad.transpose(0, 2, 1, 3).reshape((B, T, C)),
                v_grad2.transpose(0, 2, 1, 3).reshape((B, T, C)),
            ),
            2,
        )
        down = self.c_attn.backward(grad)
        return down

    def update(self) -> None:
        self.c_proj.update()
        self.c_attn.update()
        return

    def get_params(self) -> dict:
        return {
            "c_attn": [
                compress_numpy_array(self.c_attn.weight),
                compress_numpy_array(self.c_attn.bias),
            ],
            "c_proj": [
                compress_numpy_array(self.c_proj.weight),
                compress_numpy_array(self.c_proj.bias),
            ],
        }

    def load_params(self, state_dict: dict) -> None:
        self.c_attn.weight = decompress_numpy_array(state_dict["c_attn"][0])
        self.c_attn.bias = decompress_numpy_array(state_dict["c_attn"][1])
        self.c_proj.weight = decompress_numpy_array(state_dict["c_proj"][0])
        self.c_proj.bias = decompress_numpy_array(state_dict["c_proj"][1])


class Embedding:
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        batch_size: int,
        lr: float,
        init_func: Union[Callable, None] = None,
        weight_external=None,
    ):

        self.rng = cp.random.default_rng()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.batch_size = batch_size
        self.lr = lr

        self.init_func = init_func

        # If we get external weights passed, use them
        # instead of allocating ones on our own.
        # This is used for implementing weight tying.
        #
        # https://paperswithcode.com/method/weight-tying

        # ??? TODO: check if correct
        if not isinstance(type(weight_external), NoneType):
            if self.init_func:
                self.weight = cp.asanyarray(
                    self.init_func((num_embeddings, embedding_dim))
                )
            else:
                self.weight = self.rng.standard_normal(
                    (num_embeddings, embedding_dim), dtype=cp.float64
                )

        else:
            self.weight = weight_external

        self.gradient_projection_mask = cp.eye(num_embeddings, dtype=cp.uint8)

        self.input = None
        self.grad_weight = cp.zeros((num_embeddings, embedding_dim))

    def forward(self, input: ArrayLike) -> cp.ndarray:
        global step
        log_one(f"Embedding{self.num_embeddings}_max",self.weight.max().item())
        self.input = cp.asanyarray(input)
        return self.weight[self.input.astype(cp.int32), :]

    def backward(self, grad_output: ArrayLike) -> cp.ndarray:
        self.grad_weight.fill(0)
        self.grad_weight[self.input] += grad_output
        return
        # raise NotImplementedError("Implement the Embedding backward path")

    def update(self):
        self.weight -= self.lr * self.grad_weight


class Block:
    def __init__(
        self,
        d_model: int,
        context_size: int,
        n_heads: int,
        batch_size: int,
        lr: float,
        dropout: float,
        weight_init_func: Union[Callable, None],
        c_proj_init_func: Union[Callable, None],
        bias_init_func: Union[Callable, None],
    ) -> None:

        self.d_model = d_model
        self.context_size = context_size
        self.n_heads = n_heads
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout

        self.ln_1 = LayerNorm(d_model, weight_init_func=weight_init_func)

        self.attn = MultiHeadAttention(
            d_model,
            context_size,
            n_heads,
            batch_size,
            lr,
            dropout,
            c_attn_weight_init_func=weight_init_func,
            c_proj_weight_init_func=c_proj_init_func,
            bias_init_func=bias_init_func,
        )

        self.ln_2 = LayerNorm(d_model, weight_init_func=weight_init_func)

        self.mlp = MLP(
            d_model,
            batch_size,
            lr,
            dropout,
            c_fc_init_func=weight_init_func,
            c_proj_init_func=c_proj_init_func,
            bias_init_func=bias_init_func,
        )
    def forward(self, input: ArrayLike,train) -> cp.ndarray:

        input = cp.asanyarray(input)

        x = self.ln_1.forward(input)

        x = self.attn.forward(x,train)[0]

        x = input + x

        residual = copy.deepcopy(x)

        x = self.ln_2.forward(x)
        x = self.mlp.forward(x,train)
        x = residual + x

        return x

    def backward(self, upstream_grad):
        """
        Computes gradients for the transformer block.

        Args:
            upstream_grad: Gradient from the subsequent layer.

        Returns:
            Gradient with respect to the input `x`.
        """
        x = upstream_grad
        # Backward pass for Residual Connection 2
        x = self.mlp.backward(upstream_grad)
        x = self.ln_2.backward(x)
        x+=upstream_grad
        # is correct
        y = self.attn.backward(x)
        y = self.ln_1.backward(y)
        x = x + y
        return x


    def update(self) -> None:
        self.ln_2.update()
        self.ln_1.update()
        self.mlp.update()
        self.attn.update()
        return

    def state_dict(self) -> dict:
        return {
            "ln_1": [
                compress_numpy_array(self.ln_1.weight),
                compress_numpy_array(self.ln_1.bias),
            ],
            "ln_2": [
                compress_numpy_array(self.ln_2.weight),
                compress_numpy_array(self.ln_2.bias),
            ],
            "mlp": self.mlp.get_params(),
            "attn": self.attn.get_params(),
        }

    def load_params(self, state_dict: dict) -> None:
        self.ln_1.weight = decompress_numpy_array(state_dict["ln_1"][0])
        self.ln_1.bias = decompress_numpy_array(state_dict["ln_1"][1])
        self.ln_2.weight = decompress_numpy_array(state_dict["ln_2"][0])
        self.ln_2.bias = decompress_numpy_array(state_dict["ln_2"][1])

        self.mlp.load_params(state_dict["mlp"])
        self.attn.load_params(state_dict["attn"])

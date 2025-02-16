from .layers import Linear, LayerNorm, MultiHeadAttention, Softmax, Sigmoid, GELU, Embedding, one_hot
from .loss import cross_entropy_loss

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from sklearn import metrics
import matplotlib.pyplot as plt

import cupy as xp
import numpy as np
#else:
#    import numpy as xp
#    np = xp

# Only supports CPU execution for now



# Closeness for benchmarking (default in numpy is 1e-05)
comparison_rtol = 1e-04

def set_seeds(seed=42):
    np.random.seed(seed)
    xp.random.seed(seed)
    torch.manual_seed(seed) 

def print_abs_diff_stats(true, pred, title=None):
    abs_diff = np.abs(true - pred)

    rmse = metrics.root_mean_squared_error(true.flatten(), pred.flatten())
    max_error = metrics.max_error(true.flatten(), pred.flatten())

    if title != None:
        print(title)

    print(f'\nAbsdiff:\n\tmean: {abs_diff.mean():.9f}')
    print(f'\tmedian: {np.median(abs_diff):.9f}')
    print(f'\tstdev: {np.std(abs_diff):.9f}')
    print(f'RMSE: {rmse:5%}\tMax residual error: {max_error:.9f}\n')


## NanoGPT MultiHeadAttention reference implementation in PyTorch
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1, bias=False):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

        self.softmax = nn.Softmax(dim=-1)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                    .view(1, 1, block_size, block_size))\

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # manual implementation of attention
        att = q @ k.transpose(-2, -1)

        self.post_scaling_attn = att
        self.post_scaling_attn.retain_grad()

        att = att * (1.0 / math.sqrt(k.size(-1)))

        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        self.masked_attn = att
        self.masked_attn.retain_grad()

        self.softmax_output = att
        self.softmax_output.retain_grad()

        # att = F.softmax(att, dim=-1)
        att = self.softmax(att)

        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att


def test_Linear():
    set_seeds()
    # Settings
    batch_size = 4
    # First test 1D then 2D case
    for (in_shape, out_shape) in [((batch_size, 256), (batch_size, 128)), ((batch_size, 256, 384), (batch_size, 256, 128))]:
        #in_shape = (batch_size, 256)
        #out_shape = (batch_size, 128)
        #in_shape = (batch_size, 256, 384)
        #out_shape = (batch_size, 256, 128)

        # Create inputs
        in_sample_t = torch.randn(in_shape, requires_grad=True)
        in_sample_np = in_sample_t.detach().numpy()

        # Create layers
        operator_t = nn.Linear(in_features=in_shape[-1], out_features=out_shape[-1])
        operator_np = Linear(in_features=in_shape[-1], out_features=out_shape[-1], batch_size=batch_size)

        # Copy weights
        params_t = dict(operator_t.named_parameters())
        operator_np.weight = params_t['weight'].detach().T.numpy()
        operator_np.bias = params_t['bias'].detach().numpy()

        # Forward testing
        out_t = operator_t(in_sample_t)
        out_np = operator_np.forward(in_sample_np)
        xp.testing.assert_allclose(out_t.detach().numpy(), out_np, rtol=2e-2, err_msg="Missmatch in Linear forward path")

        # Backward testing
        # Create some random upstream gradients
        grad_upstream_t = torch.randn(out_t.shape)
        grad_upstream_np = grad_upstream_t.numpy()
        
        # Torch backward
        out_t.backward(grad_upstream_t)
        params_t = dict(operator_t.named_parameters())
        grad_downstream_t = in_sample_t.grad
        grad_weight_t = params_t['weight'].grad
        grad_bias_t = params_t['bias'].grad

        # Own implementation backward
        grad_downstream_np = operator_np.backward(grad_upstream_np)
        grad_weight_np = operator_np.grad_weight
        grad_bias_np = operator_np.grad_bias

        # Compare gradients
        xp.testing.assert_allclose(grad_downstream_t, grad_downstream_np, rtol=comparison_rtol, err_msg="Missmatch in Linear backward path (downstream gradient)")
        xp.testing.assert_allclose(grad_weight_t.T, grad_weight_np, rtol=comparison_rtol, err_msg="Missmatch in Linear backward path (weight)")
        xp.testing.assert_allclose(grad_bias_t, grad_bias_np, rtol=comparison_rtol, err_msg="Missmatch in Linear backward path (bias)")


def test_Sigmoid():
    set_seeds()
    # Settings
    batch_size = 4
    in_shape = (batch_size, 256)
    out_shape = (batch_size, 128)

    # Create inputs
    in_sample_t = torch.randn(in_shape, requires_grad=True)
    in_sample_np = in_sample_t.detach().numpy()

    # Create Layers
    operator_t = nn.Sigmoid()
    operator_np = Sigmoid(in_features=in_shape[-1], batch_size=batch_size)

    # Forward testing
    out_t = operator_t(in_sample_t)
    out_np = operator_np.forward(in_sample_np)
    xp.testing.assert_allclose(out_t.detach().numpy(), out_np, rtol=comparison_rtol, err_msg="Missmatch in Sigmoid forward path")

    # Backward testing
    # Create some random upstream gradients
    grad_upstream_t = torch.randn(out_t.shape)
    grad_upstream_np = grad_upstream_t.numpy()

    # Torch backward
    out_t.backward(grad_upstream_t)
    grad_downstream_t = in_sample_t.grad

    # Own implementation backward
    grad_downstream_np = operator_np.backward(grad_upstream_np)

    # Compare gradients
    xp.testing.assert_allclose(grad_downstream_t, grad_downstream_np, rtol=comparison_rtol, err_msg="Missmatch in Sigmoid backward path")


def test_GELU():
    set_seeds()
    # Settings
    batch_size = 4
    in_shape = (batch_size, 256)
    out_shape = (batch_size, 256)


    # Create inputs
    in_sample_t = torch.randn(in_shape, requires_grad=True)
    in_sample_np = in_sample_t.detach().numpy()

    # Create Layers
    operator_t = nn.GELU(approximate='tanh')
    operator_np = GELU()

    # Forward testing
    out_t = operator_t(in_sample_t)
    out_np = operator_np.forward(in_sample_np)
    # Gelu appears kind weird in torch xD
    xp.testing.assert_allclose(out_t.detach().numpy(), out_np, rtol=1e-3, err_msg="Missmatch in GELU forward path")

    # Backward testing
    # Create some random upstream gradients
    grad_upstream_t = torch.randn(out_t.shape)
    grad_upstream_np = grad_upstream_t.numpy()

    # Torch backward
    out_t.backward(grad_upstream_t)
    grad_downstream_t = in_sample_t.grad

    # Own implementation backward
    grad_downstream_np = operator_np.backward(grad_upstream_np)

    # Compare gradients
    xp.testing.assert_allclose(grad_downstream_t, grad_downstream_np, rtol=2e-3, err_msg="Missmatch in GELU backward path")


def test_LayerNorm():
    set_seeds()
    # Settings
    batch_size = 32
    in_shape = (batch_size, 256, 384)

    # Create inputs
    in_sample_t = torch.randn(in_shape, requires_grad=True)
    in_sample_np = in_sample_t.detach().numpy()

    # Create Layers
    operator_t = nn.LayerNorm(in_shape[-1], bias=False)
    operator_np = LayerNorm(in_shape[-1])

    # Copy weights
    params_t = dict(operator_t.named_parameters())
    operator_np.weight = params_t['weight'].detach().numpy()

    # Forward testing
    out_t = operator_t(in_sample_t)
    out_np = operator_np.forward(in_sample_np)
    xp.testing.assert_allclose(out_t.detach().numpy(), out_np, rtol=3e-2, err_msg="Missmatch in LayerNorm forward path")

    # Backward testing
    # Create some random upstream gradients
    grad_upstream_t = torch.randn(out_t.shape)
    grad_upstream_np = grad_upstream_t.numpy()

    # Torch backward
    out_t.backward(grad_upstream_t)
    params_t = dict(operator_t.named_parameters())
    grad_downstream_t = in_sample_t.grad
    grad_weight_t = params_t['weight'].grad

    # Own implementation backward
    grad_downstream_np = operator_np.backward(grad_upstream_np)
    grad_weight_np = operator_np.grad_weight

    # Compare gradients
    xp.testing.assert_allclose(grad_downstream_t, grad_downstream_np, rtol=8e-2, err_msg="Missmatch in LayerNorm backward path downstream gradient")
    xp.testing.assert_allclose(grad_weight_t, grad_weight_np, rtol=1e-3, err_msg="Missmatch in LayerNorm backward path weight gradient")


def test_Embedding():
    set_seeds()
    # Setting
    batch_size = 4

    for embedding_type in ['position', 'token']:
        # Create inputs
        if embedding_type == 'position':
            # Position embedding
            in_shape = (256,)
            in_sample_t = torch.arange(start=0, end=in_shape[-1], dtype=torch.long, requires_grad=False)
        elif embedding_type == 'token':
            # Token embedding
            in_shape = (batch_size, 256)
            in_sample_t = torch.randint(8192, in_shape, dtype=torch.long, requires_grad=False)
        else:
            raise ValueError(f'No embedding of type {embedding_type}')
        in_sample_np = in_sample_t.detach().numpy()

        # Create Layers
        if embedding_type == 'position':
            # Position embedding
            operator_t = nn.Embedding(256, 384)
            operator_np = Embedding(256, 384, batch_size, lr=0.00001)
        elif embedding_type == 'token':
            # Token embedding
            operator_t = nn.Embedding(8192, 384)
            operator_np = Embedding(8192, 384, batch_size, lr=0.00001)

        # Copy weights
        params_t = dict(operator_t.named_parameters())
        operator_np.weight = params_t['weight'].detach().numpy()

        # Forward testing
        out_t = operator_t(in_sample_t)
        out_np = operator_np.forward(in_sample_np)
        xp.testing.assert_allclose(out_t.detach().numpy(), out_np, rtol=comparison_rtol, err_msg=f"Missmatch in {embedding_type} Embedding forward path")

        if embedding_type == 'position':
            # Position embeding
            out_t = out_t.unsqueeze(0).repeat((batch_size,1,1))

        # Backward testing
        # Create some random upstream gradients
        grad_upstream_t = torch.randn(out_t.shape)
        grad_upstream_np = grad_upstream_t.numpy()

        # Torch backward
        out_t.backward(grad_upstream_t)
        params_t = dict(operator_t.named_parameters())
        grad_weight_t = params_t['weight'].grad

        # Own implementation backward
        operator_np.backward(grad_upstream_np)
        grad_weight_np = operator_np.grad_weight

        xp.testing.assert_allclose(grad_weight_t, grad_weight_np, rtol=comparison_rtol, err_msg=f"Missmatch in {embedding_type} Embedding backward path")


def test_CrossEntropyLoss():
    set_seeds()
    # Settings
    batch_size = 4

    # Create inputs
    in_shape = (batch_size, 256, 8192)
    in_sample_t = torch.randn(in_shape, requires_grad=True)
    in_sample_np = in_sample_t.detach().numpy()

    in_target_t = torch.randint(8192, (in_shape[0], in_shape[1]))
    in_targets_np = in_target_t.numpy()

    # Forward testing
    # Torch forward
    out_t = nn.functional.cross_entropy(in_sample_t.view(-1, in_sample_t.size(-1)), in_target_t.view(-1), ignore_index=-1)

    # Own forward
    soft_m = Softmax(axis=-1)
    logits_np = soft_m.forward(in_sample_np)
    logits_for_loss_np = logits_np.reshape(-1, logits_np.shape[-1])
    targets_for_loss_np = xp.expand_dims(in_targets_np.reshape(-1), 1)
    targets_for_loss_np = one_hot(targets_for_loss_np, 8192)
    out_np = cross_entropy_loss(logits_for_loss_np, targets_for_loss_np)

    # Compare Forward
    xp.testing.assert_allclose(out_t.detach().numpy(), out_np, rtol=comparison_rtol, err_msg="Missmatch in Loss calculation forward path")


def test_Softmax():
    set_seeds()
    # Settings
    batch_size = 1

    # Create inputs
    in_shape = (batch_size, 6, 256, 384)
    in_sample_t = torch.randn(in_shape, requires_grad=True)
    in_sample_t.retain_grad()
    in_sample_np = in_sample_t.detach().numpy()

    # Create Layers
    softmax_axis = -1
    mask_t = nn.Softmax(dim=softmax_axis)
    operator_np = Softmax(axis=softmax_axis)

    # Forward testing
    out_t = mask_t(in_sample_t)
    out_np = operator_np.forward(in_sample_np)
    np.testing.assert_allclose(out_t.detach().numpy(), out_np, rtol=comparison_rtol, err_msg="Missmatch in Softmax forward path")

    # Backward testing
    # Create some random upstream gradients
    set_seeds()
    grad_upstream_t = torch.randn(out_t.shape)
    grad_upstream_np = grad_upstream_t.numpy()

    # Torch backward
    out_t.backward(grad_upstream_t)
    grad_downstream_t = in_sample_t.grad

    # Own implementation backward
    grad_downstream_np = operator_np.backward(grad_upstream_np)

    xp.testing.assert_allclose(grad_downstream_t, grad_downstream_np, rtol=5e-2, err_msg="Missmatch in Softmax backward path downstream gradient")



def test_MultiHeadAttention():
    set_seeds(seed=3)
    # Settings
    batch_size = 4
    d_model = 384
    context_size = 256
    n_heads = 6
    dropout = 0.

    # Create inputs
    in_shape = (batch_size, context_size, d_model)
    in_sample_t = torch.randn(in_shape, requires_grad=True)
    in_sample_np = in_sample_t.detach().numpy()

    # Create Layers
    operator_t = CausalSelfAttention(n_embd=d_model, n_head=n_heads, block_size=context_size, dropout=dropout, bias=False)
    operator_np = MultiHeadAttention(d_model=d_model,
                                    context_size=context_size,
                                    n_heads=n_heads,
                                    batch_size=batch_size,
                                    dropout=dropout,)
    
    # Copy weights
    params_t = dict(operator_t.named_parameters())
    # Since these are linear layers, we need to do the transpose, because PyTorch apperently...
    operator_np.c_attn.weight = xp.asanyarray(params_t['c_attn.weight'].detach().T.numpy())
    operator_np.c_proj.weight = xp.asanyarray(params_t['c_proj.weight'].detach().T.numpy())

    # Forward testing
    out_t, attn_t = operator_t(in_sample_t)
    out_np, attn_np = operator_np.forward(in_sample_np)
    assert xp.allclose(out_t.detach().numpy(), out_np, rtol=7e-2), "Missmatch in MultiHeadAttention forward path"
    #xp.testing.assert_allclose(out_t.detach().numpy(), out_np, rtol=7e-2)

    # Backward testing
    # Create some random upstream gradients
    grad_upstream_t = torch.randn(out_t.shape)
    grad_upstream_np = grad_upstream_t.numpy()

    # Torch backward
    out_t.backward(grad_upstream_t)
    params_t = dict(operator_t.named_parameters())
    grad_downstream_t = in_sample_t.grad.numpy()
    grad_in_weight_t = params_t['c_attn.weight'].grad.T.numpy()
    grad_out_weight_t = params_t['c_proj.weight'].grad.T.numpy()

    # Own implementation backward
    grad_downstream_np = operator_np.backward(grad_upstream_np)
    grad_in_weight_np = operator_np.c_attn.grad_weight
    grad_out_weight_np = operator_np.c_proj.grad_weight

    # Fine-grained gradient comparison

    grad_softmax_t = operator_t.softmax_output.grad.detach().numpy()
    grad_softmax_np = operator_np.softmax_grad_output

    print_abs_diff_stats(grad_softmax_t, grad_softmax_np, 'Attention Softmax grad')

    grad_masked_attn_t = operator_t.masked_attn.grad.detach().numpy()
    grad_masked_attn_np = operator_np.grad_attn_masked

    print_abs_diff_stats(grad_masked_attn_t, grad_masked_attn_np, 'Masked attention grad')

    grad_scaled_attn_t = operator_t.post_scaling_attn.grad.detach().numpy()
    grad_scaled_attn_np = operator_np.grad_attn_scaled

    print_abs_diff_stats(grad_scaled_attn_t, grad_scaled_attn_np, 'Post-scaling attention grad')

    # exit()

    # Compare gradients

    print_abs_diff_stats(grad_downstream_t, grad_downstream_np, 'Downstream grad')
    print_abs_diff_stats(grad_in_weight_t, grad_in_weight_np, 'Attention weight grad')
    print_abs_diff_stats(grad_out_weight_t, grad_out_weight_np, 'Projection weight grad')

    try:
        xp.testing.assert_allclose(grad_downstream_t, grad_downstream_np, rtol=5e-2, err_msg="Missmatch in MultiHeadAttention backward path (downstream grad)")
    except AssertionError as e:

        absdiff_flattened = (grad_downstream_t - grad_downstream_np).flatten()

        display_indices = np.argsort(absdiff_flattened)[::-1][:10]

        display_vals_absdiff = absdiff_flattened[display_indices]
        display_vals_t = grad_downstream_t.flatten()[display_indices]
        display_vals_np = grad_downstream_np.flatten()[display_indices]

        print('Downstream grad test assertion failed. Highest contributing values:')

        for index, val_t, val_np, absdiff in\
                zip(display_indices, display_vals_t, display_vals_np, display_vals_absdiff):
            print(f'Index: {index} target: {val_t:.8f} actual: {val_np:.8f}  absdiff: {absdiff:.8f}')
        
        # Re-raise error for PyTest
        raise e

    try:
        np.testing.assert_allclose(grad_in_weight_t, grad_in_weight_np, rtol=5e-2, err_msg="Missmatch in MultiHeadAttention backward path (c_attn.grad_weight)")
    except AssertionError as e:

        absdiff_flattened = (grad_in_weight_t - grad_in_weight_np).flatten()

        display_indices = np.argsort(absdiff_flattened)[::-1][:10]

        display_vals_absdiff = absdiff_flattened[display_indices]
        display_vals_t = grad_downstream_t.flatten()[display_indices]
        display_vals_np = grad_downstream_np.flatten()[display_indices]

        print('c_attn.grad_weight test assertion failed. Highest contributing values:')

        for index, val_t, val_np, absdiff in\
                zip(display_indices, display_vals_t, display_vals_np, display_vals_absdiff):
            print(f'Index: {index} target: {val_t:.8f} actual: {val_np:.8f}  absdiff: {absdiff:.8f}')
        
        # Re-raise error for PyTest
        raise e

    try:
        np.testing.assert_allclose(grad_out_weight_t, grad_out_weight_np, rtol=5e-2, err_msg="Missmatch in MultiHeadAttention backward path (c_proj.grad_weight)")
    except AssertionError as e:
        absdiff_flattened = (grad_out_weight_t - grad_out_weight_np).flatten()

        display_indices = np.argsort(absdiff_flattened)[::-1][:10]

        display_vals_absdiff = absdiff_flattened[display_indices]
        display_vals_t = grad_downstream_t.flatten()[display_indices]
        display_vals_np = grad_downstream_np.flatten()[display_indices]

        print('c_prod.grad_weight test assertion failed. Highest contributing values:')

        for index, val_t, val_np, absdiff in\
                zip(display_indices, display_vals_t, display_vals_np, display_vals_absdiff):
            print(f'Index: {index} target: {val_t:.8f} actual: {val_np:.8f}  absdiff: {absdiff:.8f}')
        
        # Re-raise error for PyTest
        raise e

    for t, n, name in zip(np.split(grad_in_weight_t, 3, axis=1), np.split(grad_in_weight_np, 3, axis=1), ['Q grad', 'K grad', 'V grad']):
        print_abs_diff_stats(t, n, name)

    plt.matshow((grad_in_weight_t - grad_in_weight_np))
    plt.savefig('../grad_attention.png')


















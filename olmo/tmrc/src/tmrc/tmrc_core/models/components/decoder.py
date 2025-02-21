import torch

import torch.nn as nn
from torch.nn import functional as F

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from tmrc.tmrc_core.models.components import ACTIVATION_REGISTRY, MASK_REGISTRY


class CausalSelfAttention(nn.Module):
    """
    A simple adapation of vanilla self attention head with a causal mask.
    Minor changes from Karpathy's GPT implementation.
    """

    def __init__(
            self, 
            d_model: int, 
            n_head: int, 
            attn_bias: bool, 
            proj_bias: bool, 
            dropout_p: float, 
            context_length: int, 
            flash: bool, 
            flex: bool, 
            compile_flex: bool
        ):
        """
        Args:
            d_model: Dimension of the embedding used in attention layer
            n_head: Number of heads of attention layer
            attn_bias: Is there is a bias for the attention layer
            proj_bias: Is there is a bias for the projection at the end of attention
            dropout_p: Percentage of parameters to dropout during training
            context_length: Maximum number of tokens for attention
            flash: Whether attention layer uses FlashAttention or not
            flex: Whether attention layer uses FlexAttention or not
            compile_flex: Is FlexAttention layer compiled
        """
        super().__init__()
        assert d_model % n_head==0
        self.c_attn = nn.Linear(d_model, 3*d_model, bias=attn_bias) # easier to do K,V,Q at once
        self.c_proj = nn.Linear(d_model, d_model, bias=proj_bias)
        self.attn_dropout = nn.Dropout(dropout_p)
        self.proj_dropout = nn.Dropout(dropout_p)
        
        self.d_model = d_model
        self.n_heads = n_head

        self.flash = flash #hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flex = flex

        self.compile_flex = compile_flex

        if self.flex and self.compile_flex:
            self.flex_attention = torch.compile(torch.nn.attention.flex_attention.flex_attention, dynamic=False)
        if not (self.flash or self.flex):
            print("Warning: flash attention not found (torch >= 2.0)")
            self.register_buffer("causal_mask",
                                 torch.tril(torch.ones(context_length, context_length)).view(1, 1, context_length, context_length))
 

    def forward(self, x, created_block_mask=None):
        # x: (B, T, C)

        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2) # (B, n_h, T, d_head)
        k = k.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2)
        v = v.view(B, T, self.n_heads, C//self.n_heads).transpose(1,2)

        if self.flash:
            y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        elif self.flex:
            if self.compile_flex:
                y = self.flex_attention(q, k, v, block_mask=created_block_mask)
            else:
                y = flex_attention(q, k, v, block_mask=created_block_mask)
        else:
            w = (q @ k.transpose(-2, -1))*k.size(-1)**(-0.5)
            w = w.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
            w = F.softmax(w, dim=-1)
            w = self.attn_dropout(w)
            y = w @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_dropout(self.c_proj(y))
        return y

    
class MLP(nn.Module):

    def __init__(
        self, 
        d_model: int, 
        mlp_scale_factor: int, 
        mlp_bias: bool, 
        dropout_p: float, 
        activation: str
    ):
        """
        Args:
            d_model: Dimension of the embedding used in attention layer
            mlp_scale_factor: How many times bigger is the intermediate layer in the MLP
            mlp_bias: Is there bias at the end of the MLP
            dropout_p: Percentage of parameters to dropout during training
            activation: Activation function used for MLP
        """
        super().__init__()
        self.c_fc    = nn.Linear(d_model, mlp_scale_factor * d_model, bias=mlp_bias)
        self.activation = ACTIVATION_REGISTRY.get(activation)
        self.c_proj  = nn.Linear(mlp_scale_factor * d_model, d_model, bias=mlp_bias)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    """ Basic decoder block. """

    def __init__(
        self, 
        d_model,
        n_head: int, 
        attn_bias: bool, 
        proj_bias: bool, 
        context_length: int, 
        flash: bool, 
        flex: bool, 
        compile_flex: bool,
        ln_bias: bool, 
        mlp_scale_factor: int, 
        mlp_bias: bool, 
        activation: str, 
        dropout_p: float,
        **kwargs
    ):
        """
        Args:
            d_model: Dimension of the embedding used in attention layer
            n_head: Number of heads of attention layer
            attn_bias: Is there is a bias for the attention layer
            proj_bias: Is there is a bias for the projection at the end of attention
            context_length: Maximum number of tokens for attention
            flash: Whether attention layer uses FlashAttention or not
            flex: Whether attention layer uses FlexAttention or not
            compile_flex: Is FlexAttention layer compiled
            ln_bias: Is there bias applied to the layer norm layers
            mlp_scale_factor: How many times bigger is the intermediate layer in the MLP
            mlp_bias: Is there bias at the end of the MLP
            activation: Activation function used for MLP
            dropout_p: Percentage of parameters to dropout during training
        """
        super().__init__(**kwargs)
        self.ln_1 = nn.LayerNorm(d_model, bias=ln_bias)
        self.attn = CausalSelfAttention(d_model=d_model, n_head=n_head, attn_bias=attn_bias, proj_bias=proj_bias, dropout_p=dropout_p, context_length=context_length, flash=flash, flex=flex, compile_flex=compile_flex)
        self.ln_2 = nn.LayerNorm(d_model, bias=ln_bias)
        self.mlp = MLP(d_model=d_model, mlp_scale_factor=mlp_scale_factor, mlp_bias=mlp_bias, dropout_p=dropout_p, activation=activation)
    
    def forward(self, x, created_block_mask=None):
        x = x + self.attn(self.ln_1(x), created_block_mask)
        x = x + self.mlp(self.ln_2(x))
        return x 
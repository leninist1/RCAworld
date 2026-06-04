"""Flamingo-style Gated Cross-Attention for LLM integration.

Implements the gated cross-attention mechanism from the Flamingo paper
(Alayrac et al., 2022). The key insight is that cross-attention outputs
are gated by a learnable tanh gate before being added to the residual
stream, allowing the model to smoothly learn when to use visual/world-model
context.

For RCAWorld-Foundation, this module bridges the diagnostic world model
tokens to a frozen/partially-frozen LLM, enabling the LLM to reason over
compressed system-state representations.

Reference:
  Flamingo: a Visual Language Model for Few-Shot Learning
  https://arxiv.org/abs/2204.14198
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class GatedCrossAttentionBlock(nn.Module):
    """Single Flamingo-style gated cross-attention block.

    Inserted between existing transformer layers in the LLM.
    The gate starts near zero, allowing the LLM to initially ignore
    the world-model context and gradually learn to incorporate it.

    Architecture:
        x = x + tanh(alpha) * CrossAttn(LN(x), context)
        x = x + FFN(LN(x))

    Attributes:
        hidden_dim: Dimension of LLM hidden states.
        num_heads: Number of attention heads.
        context_dim: Dimension of world-model diagnostic tokens.
        dropout_rate: Dropout rate for cross-attention.
    """

    hidden_dim: int
    num_heads: int = 8
    context_dim: Optional[int] = None
    dropout_rate: float = 0.0

    def setup(self):
        ctx_dim = self.context_dim or self.hidden_dim

        self.attn_norm = nn.LayerNorm(name="attn_norm")
        self.ffn_norm = nn.LayerNorm(name="ffn_norm")

        # Cross-attention: LLM hidden states attend to world-model tokens
        self.cross_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            name="cross_attn",
        )

        # Feed-forward network (same as standard transformer FFN)
        self.ffn = nn.Sequential([
            nn.Dense(self.hidden_dim * 4, name="ffn_up"),
            nn.gelu,
            nn.Dense(self.hidden_dim, name="ffn_down"),
        ])

        # Learnable gate parameter (alpha in Flamingo paper)
        # Initialized to zero so the model starts by ignoring context
        self.alpha = self.param(
            "alpha",
            nn.initializers.zeros_init(),
            (),
            jnp.float32,
        )

    def __call__(self,
                 x: jnp.ndarray,
                 context: jnp.ndarray,
                 context_mask: Optional[jnp.ndarray] = None,
                 deterministic: bool = True,
                 ) -> jnp.ndarray:
        """Apply gated cross-attention.

        Args:
            x: [B, S, D] LLM hidden states (S = sequence length).
            context: [B, C, D_ctx] world-model diagnostic tokens (C = num tokens).
            context_mask: [B, C] or [B, 1, C] mask for context tokens.
            deterministic: If False, apply dropout.

        Returns:
            x: [B, S, D] updated LLM hidden states.
        """
        # Cross-attention with gate
        x_norm = self.attn_norm(x)
        # Project context to same dim if needed
        if self.context_dim is not None and self.context_dim != self.hidden_dim:
            context = nn.Dense(self.hidden_dim, name="ctx_proj")(context)

        attn_out = self.cross_attn(
            inputs_q=x_norm,
            inputs_kv=context,
            mask=context_mask,
            deterministic=deterministic,
        )

        # Gated residual: tanh(alpha) * attn_out
        gate = jnp.tanh(self.alpha)
        x = x + gate * attn_out

        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))

        return x


class GatedCrossAttentionLayer(nn.Module):
    """A block of N gated cross-attention layers inserted into the LLM.

    Typically inserted every K transformer layers (e.g., every 4th layer
    in a 32-layer LLM). Each instance contains num_layers cross-attention
    blocks.

    Attributes:
        hidden_dim: LLM hidden dimension.
        num_heads: Number of attention heads per block.
        num_layers: Number of gated cross-attention blocks in this module.
        context_dim: Dimension of world-model tokens.
        dropout_rate: Dropout rate.
    """

    hidden_dim: int
    num_heads: int = 8
    num_layers: int = 1
    context_dim: Optional[int] = None
    dropout_rate: float = 0.0

    def setup(self):
        self.layers = [
            GatedCrossAttentionBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                context_dim=self.context_dim,
                dropout_rate=self.dropout_rate,
                name=f"gca_block_{i}",
            )
            for i in range(self.num_layers)
        ]

    def __call__(self,
                 x: jnp.ndarray,
                 context: jnp.ndarray,
                 context_mask: Optional[jnp.ndarray] = None,
                 deterministic: bool = True,
                 ) -> Tuple[jnp.ndarray, list]:
        """Apply all gated cross-attention blocks sequentially.

        Args:
            x: [B, S, D] LLM hidden states.
            context: [B, C, D_ctx] diagnostic tokens.
            context_mask: [B, C] optional mask.
            deterministic: If False, apply dropout.

        Returns:
            x: [B, S, D] updated hidden states.
            gate_values: List of current gate values (for monitoring).
        """
        gate_values = []
        for layer in self.layers:
            x = layer(x, context, context_mask, deterministic)
            gate_values.append(jnp.tanh(layer.alpha))
        return x, gate_values


class DiagnosticProjector(nn.Module):
    """Projects world-model diagnostic tokens into LLM-compatible embeddings.

    Takes the world model's diagnostic token representation and produces
    a sequence of context tokens that can be attended to by the LLM via
    gated cross-attention.

    The diagnostic tokens include:
      - Global system state summary
      - Per-entity onset/candidate embeddings
      - Propagation chain embeddings
      - Mechanism candidate embeddings
      - Counterfactual repair gain estimates

    Attributes:
        num_diagnostic_tokens: Number of output diagnostic tokens.
        world_token_dim: Dimension of world model tokens.
        llm_hidden_dim: Dimension of LLM hidden states.
    """

    num_diagnostic_tokens: int = 16
    world_token_dim: int = 256
    llm_hidden_dim: int = 4096

    def setup(self):
        # Learnable query tokens (like BLIP-2 Q-Former queries)
        self.diag_queries = self.param(
            "diag_queries",
            nn.initializers.normal(stddev=0.02),
            (self.num_diagnostic_tokens, self.world_token_dim),
            jnp.float32,
        )

        # Cross-attention to compress world tokens into diagnostic tokens
        self.compress_cross_attn = nn.MultiHeadDotProductAttention(
            num_heads=8,
            qkv_features=self.world_token_dim,
            out_features=self.world_token_dim,
            name="compress_attn",
        )

        # Project to LLM space
        self.llm_projection = nn.Sequential([
            nn.Dense(self.llm_hidden_dim, name="proj_hidden"),
            nn.gelu,
            nn.Dense(self.llm_hidden_dim, name="proj_out"),
        ])

    def __call__(self,
                 world_tokens: jnp.ndarray,
                 world_token_mask: Optional[jnp.ndarray] = None,
                 ) -> jnp.ndarray:
        """Compress world model tokens into LLM-compatible diagnostic tokens.

        Args:
            world_tokens: [B, N_world, world_token_dim] raw world model tokens.
            world_token_mask: [B, N_world] optional mask.

        Returns:
            diagnostic_tokens: [B, num_diagnostic_tokens, llm_hidden_dim]
        """
        B = world_tokens.shape[0]

        # Expand learnable queries to batch
        queries = jnp.tile(
            self.diag_queries[None, :, :], (B, 1, 1))  # [B, Q, world_token_dim]

        # Compress world tokens via cross-attention
        compressed = self.compress_cross_attn(
            inputs_q=queries,
            inputs_kv=world_tokens,
            mask=world_token_mask,
        )  # [B, Q, world_token_dim]

        # Project to LLM dimension
        diagnostic_tokens = self.llm_projection(compressed)  # [B, Q, llm_hidden_dim]

        return diagnostic_tokens

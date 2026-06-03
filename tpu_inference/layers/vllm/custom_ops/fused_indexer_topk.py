# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" TPU implementation of the StreamIndex algorithm for the 
DeepSeek-V4 Lightning Indexer. This implementation is intended to be used
with vLLM and TorchAX, but it should be usable as a standalone JAX kernel.
"""

import jax
import jax.numpy as jnp


@jax.jit(static_argnames=["k", "compression_ratio", "c_s", "c_t"])
def streamindex_chunked_topk(
    query_projection: jax.Array,  # Shape: [B, T, H_I, D]
    compressed_keys: jax.Array,   # Shape: [B, S, H_I, D]
    indexer_weights: jax.Array,   # Shape: [B, T, H_I]
    k: int,
    compression_ratio: int,
    c_s: int = 8196,
    c_t: int = 2048
) -> jax.Array:
    """
    Executes a chunked, streaming top-k selection over long sequences natively in JAX.
    Prevents the O(B * S * H_I * T) memory explosion by bounding evaluation tiles.
    """
    B, S, H_I, D = compressed_keys.shape
    _, T, _ = indexer_weights.shape

    init_values = jnp.full((B, T, H_I, k), -jnp.inf, dtype=jnp.float32)
    init_indices = jnp.full((B, T, H_I, k), -1, dtype=jnp.int32)

    num_s_chunks = S // c_s

    def s_loop_cond(state):
        s_idx, _, _ = state
        return s_idx < num_s_chunks

    def s_loop_body(state):
        s_idx, current_values, current_indices = state

        keys_chunk = jax.lax.dynamic_slice_in_dim(compressed_keys, s_idx * c_s, c_s, axis=1)
        num_t_chunks = T // c_t

        def t_loop_cond(t_state):
            t_idx, _, _ = t_state
            return t_idx < num_t_chunks

        def t_loop_body(t_state):
            t_idx, val_acc, idx_acc = t_state

            queries_chunk = jax.lax.dynamic_slice_in_dim(query_projection, t_idx * c_t, c_t, axis=1)
            weights_chunk = jax.lax.dynamic_slice_in_dim(indexer_weights, t_idx * c_t, c_t, axis=1)

            inner_dot = jnp.einsum("bthi,bshi->bths", queries_chunk, keys_chunk)

            scores_tile = jax.nn.relu(inner_dot) * jnp.expand_dims(weights_chunk, axis=-1)

            s_absolute = jnp.arange(c_s, dtype=jnp.int32)[None, :] + (s_idx * c_s)
            t_absolute = jnp.arange(c_t, dtype=jnp.int32)[:, None] + (t_idx * c_t)

            causal_mask = (s_absolute * compression_ratio) <= t_absolute
            causal_mask = jnp.expand_dims(causal_mask, axis=(0, 2))
            scores_tile = jnp.where(causal_mask, scores_tile, -jnp.inf)

            scores_flat = scores_tile.reshape(-1, c_s)
            local_vals, local_ids = jax.lax.top_k(scores_flat, k)

            local_vals = local_vals.reshape(B, c_t, H_I, k)
            local_ids = local_ids.reshape(B, c_t, H_I, k)

            global_ids_tile = local_ids + (s_idx * c_s)

            old_vals_tile = jax.lax.dynamic_slice_in_dim(val_acc, t_idx * c_t, c_t, axis=1)
            old_ids_tile = jax.lax.dynamic_slice_in_dim(idx_acc, t_idx * c_t, c_t, axis=1)

            merged_vals = jnp.concatenate([old_vals_tile, local_vals], axis=-1)
            merged_ids = jnp.concatenate([old_ids_tile, global_ids_tile], axis=-1)

            merged_vals_flat = merged_vals.reshape(-1, 2 * k)
            merged_ids_flat = merged_ids.reshape(-1, 2 * k)

            filtered_indices = jnp.argsort(merged_vals_flat, axis=-1)[:, -k:]
            filtered_indices = filtered_indices[:, ::-1]

            row_idx = jnp.arange(merged_vals_flat.shape[0])[:, None]
            new_vals_tile = merged_vals_flat[row_idx, filtered_indices].reshape(B, c_t, H_I, k)
            new_ids_tile = merged_ids_flat[row_idx, filtered_indices].reshape(B, c_t, H_I, k)

            val_acc = jax.lax.dynamic_update_slice_in_dim(val_acc, new_vals_tile, t_idx * c_t, axis=1)
            idx_acc = jax.lax.dynamic_update_slice_in_dim(idx_acc, new_ids_tile, t_idx * c_t, axis=1)

            return t_idx + 1, val_acc, idx_acc

        _, current_values, current_indices = jax.lax.while_loop(
            t_loop_cond, t_loop_body, (0, current_values, current_indices)
        )

        return s_idx + 1, current_values, current_indices

    # Execute outer S loop
    _, _, final_indices = jax.lax.while_loop(
        s_loop_cond, s_loop_body, (0, init_values, init_indices)
    )

    return final_indices


# Copyright 2022 Cerebras Systems.
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

import tensorflow as tf


def gather_last_tokens(hidden_states, input_seq_lens):
    """
    Assuming hidden_states has dimensions [B, S, H],
    gather the particular hidden states slicing in the
    sequence dimension at input_seq_len-1, so that it
    is the hidden state vector after the last time step.

    We have to reshape/gather/reshape because there is
    an issue with using gather_nd through LAIR.

    :param Tensor hidden_states: 3-d tensor (B,S,H).
    :param Tensor input_seq_lens: 1-d tensor sequence lengths.
    :returns: 2-d Tensor of last hidden states (B, H).
    """
    [batch_size, max_seq_len, hidden_size] = hidden_states.get_shape()

    # flatten hidden_states from [B, S, H] to [B*S, H]
    flat_hidden_states = tf.reshape(
        hidden_states, [batch_size * max_seq_len, hidden_size],
    )

    # [0, 1, ... B-1]*S = [0, S, 2*S, ... (B-1)*S]
    flat_offsets = tf.range(0, batch_size, dtype=tf.int32) * max_seq_len

    # [L[0]-1, L[1]-1, ... L[B-1]-1] + [0, S, 2*S, ... (B-1)*S]
    # = [L[0]-1, L[1]-1 + S, L[2]-1 + 2*S, ... L[B-1]-1 + (B-1)*S]
    flat_indices = input_seq_lens - 1 + flat_offsets

    # gather last tokens and reshape to [B, H]
    h_last = tf.reshape(
        tf.gather(flat_hidden_states, flat_indices, axis=0),
        [batch_size, hidden_size],
    )

    return h_last

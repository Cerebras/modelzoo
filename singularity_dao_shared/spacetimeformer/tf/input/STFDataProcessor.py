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

import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from modelzoo.common.tf.input.utils import transform_dataset
from modelzoo.common.tf.model_utils.shard_dataset import shard_dataset


class STFDataProcessor:
    def __init__(self, params):
        # Modeling Parameters
        self.time_resolution = params["time_resolution"]
        self.context_length = params["context_length"]
        self.target_length = params["target_length"]
        self.dec_start_token_len = params.get("start_token_length", 3)
        self.max_wavelength = params.get("max_wavelength", 10000)
        self.time_dim = params.get(
            "time_dimension", 10
        )  # number of sines and cosines
        assert self.time_dim % 2 == 0, "Time dimension must be divisible by 2"

        # Data/Training Parameters
        self.batch_size = params["batch_size"]
        self.data_path = params["data_path"]
        assert os.path.exists(
            params["data_path"]
        ), f"Invalid data path: {params['data_path']}"
        self.date_format = (
            "%Y-%m-%d HH:MM" if params["contains_hr_min"] else "%Y-%m-%d"
        )
        self.shuffle = params.get("shuffle", True)
        self.shuffle_buffer = params.get("shuffle_buffer", 10 * self.batch_size)
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.repeat = params.get("repeat", True)
        self.num_buckets = params.get("num_buckets", 10)
        self.use_multiple_workers = params.get("use_multiple_workers", False)

        self.should_invert_norm = 'scaler_stats_path' in params
        if self.should_invert_norm:
            with open(params['scaler_stats_path']) as scaler_stats_file:
                norm_stats = json.load(scaler_stats_file)
                self.scales = np.tile(norm_stats['scale'], self.target_length)
                self.means = np.tile(norm_stats['mean'], self.target_length)

        self.verbose = params.get("verbose", False)

    def create_tf_dataset(
        self, mode=tf.estimator.ModeKeys.TRAIN, input_context=None
    ):
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        self.qprint("Reading from file...")
        self.df = pd.read_csv(self.data_path)
        assert (
            "Datetime" in self.df.columns
        ), "Dataframe must contain 'Datetime' column"
        norm_tgts = self.df.loc[:, self.df.columns != 'Datetime'].values
        norm_tgts = np.expand_dims(norm_tgts, -1)
        self.qprint("Extracting time representation...")
        time_vecs = self.time2vec_pos_enc(self.df['Datetime'])
        time_vecs = np.repeat(
            np.expand_dims(time_vecs, 1), norm_tgts.shape[1], axis=1
        )
        val_time = np.concatenate((norm_tgts, time_vecs), axis=-1)
        self.qprint("Generating batches...")
        return self.generate_batches(val_time, is_training, input_context)

    def normalize_columns(self, target_df):
        self._scaler = StandardScaler()
        return self._scaler.fit_transform(target_df.values)

    def time2vec_pos_enc(self, datetime):
        '''
        Input: df - Pandas series of length N representing datetimes as a string
        Output: Pandas dataframe of shape NxD, input times --> corresponding row in df
        '''
        _time_features = self._construct_time_features(datetime).values
        N, K = _time_features.shape

        def periodic(x, i):
            periodic_input = x / (
                self.max_wavelength ** ((2 * (i // 2)) / self.time_dim)
            )
            return (
                np.sin(periodic_input) if i % 2 == 0 else np.cos(periodic_input)
            )

        pos_enc = np.zeros((N, K * self.time_dim))
        for i in range(self.time_dim):
            pos_enc[:, i * K : (i + 1) * K] = periodic(_time_features, i)
        return pos_enc

    def _construct_time_features(self, datetime):
        '''
        Input: df - Pandas series of length N representing datetimes as a string
        Output: New Pandas dataframe with time-based features
        '''
        out_df = pd.DataFrame({})
        dates = pd.to_datetime(datetime, format=self.date_format)
        years = dates.apply(lambda row: row.year)
        max_year = years.max()
        min_year = years.min()
        out_df["Year"] = dates.apply(
            lambda row: (row.year - min_year) / max(1.0, (max_year - min_year))
        )
        out_df["Month"] = dates.apply(
            lambda row: 2.0 * ((row.month - 1) / 11.0) - 1.0, 1
        )
        out_df["Day"] = dates.apply(
            lambda row: 2.0 * ((row.day - 1) / 30.0) - 1.0, 1
        )
        out_df["Weekday"] = dates.apply(
            lambda row: 2.0 * (row.weekday() / 6.0) - 1.0, 1
        )
        out_df["Hour"] = dates.apply(
            lambda row: 2.0 * ((row.hour) / 23.0) - 1.0, 1
        )
        out_df["Minute"] = dates.apply(
            lambda row: 2.0 * ((row.minute) / 59.0) - 1.0, 1
        )
        return out_df

    def generate_batches(self, val_time, is_training, input_context):
        '''
        Input: val_time - numpy tensor of value + time vector concatenation 
                Shape - (time steps, variables, temporal embedding dimension)
        Out: TF Dataset with batched slices
        '''
        TS, V, TED = val_time.shape
        val_time = tf.convert_to_tensor(val_time, dtype=tf.float32)

        spatial_ind_enc = tf.tile(
            tf.range(V, dtype=tf.int32), (self.context_length,)
        )
        spatial_ind_dec = tf.tile(
            tf.range(V, dtype=tf.int32),
            (self.target_length + self.dec_start_token_len,),
        )

        givens_enc = tf.ones((self.context_length * V,), dtype=tf.int32)
        givens_dec = tf.concat(
            (
                tf.ones((self.dec_start_token_len * V,), dtype=tf.int32),
                tf.zeros((self.target_length * V,), dtype=tf.int32),
            ),
            axis=0,
        )

        enc_in_local_pos = tf.expand_dims(
            tf.range(self.context_length * V, dtype=tf.float32), -1
        )
        dec_in_local_pos = tf.expand_dims(
            tf.range(
                (self.target_length + self.dec_start_token_len) * V,
                dtype=tf.float32,
            ),
            axis=-1,
        )

        slice_range = tf.cast(
            self.time_resolution * (self.context_length + self.target_length),
            tf.int32,
        )

        def extract_slice(start):
            samp = tf.identity(
                val_time[start : start + slice_range : self.time_resolution]
            )  # C_T, V, TED
            samp.set_shape((self.context_length + self.target_length, V, TED))
            labels = tf.identity(samp[self.context_length :, :, 0])

            enc_in = samp[: self.context_length, :, :]
            enc_in = tf.reshape(enc_in, (self.context_length * V, TED))
            enc_in = tf.concat((enc_in, enc_in_local_pos), axis=-1)

            dec_in_start_tokens = samp[
                (
                    self.context_length - self.dec_start_token_len
                ) : self.context_length,
                :,
                :,
            ]
            dec_in_target_tokens = tf.concat(
                (
                    tf.zeros((self.target_length, V, 1), dtype=tf.float32),
                    samp[self.context_length :, :, 1:],
                ),
                axis=-1,
            )
            dec_in = tf.concat(
                (dec_in_start_tokens, dec_in_target_tokens), axis=0
            )
            dec_in = tf.reshape(
                dec_in,
                ((self.dec_start_token_len + self.target_length) * V, TED),
            )
            dec_in = tf.concat((dec_in, dec_in_local_pos), axis=-1)

            features = {
                "enc_in": enc_in,
                "spatial_ind_enc": spatial_ind_enc,
                "givens_enc": givens_enc,
                "dec_in": dec_in,
                "spatial_ind_dec": spatial_ind_dec,
                "givens_dec": givens_dec,
            }

            if self.should_invert_norm:
                features["means"] = self.means.astype(np.float32)
                features["scales"] = self.scales.astype(np.float32)

            return (features, labels)

        num_samples = tf.cast(TS - slice_range, tf.int64)
        dataset = tf.data.Dataset.range(num_samples, output_type=tf.int32)
        dataset = shard_dataset(
            dataset, self.use_multiple_workers, input_context
        )

        return transform_dataset(
            dataset=dataset,
            map_fn=extract_slice,
            batch_size=self.batch_size,
            is_training=is_training,
            shuffle=self.shuffle,
            shuffle_buffer=self.shuffle_buffer,
            repeat=self.repeat,
            seed=self.shuffle_seed,
            map_before_batch=True,
        )

    def qprint(self, *args):
        if self.verbose:
            print(*args)

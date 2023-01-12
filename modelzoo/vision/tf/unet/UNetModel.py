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

"""
UNet model to be used with TF Estimator
"""
import tensorflow as tf
from tensorflow.compat.v1.losses import Reduction
from tensorflow.python.keras.layers import Flatten, concatenate

from modelzoo.common.tf.layers.ActivationLayer import ActivationLayer
from modelzoo.common.tf.layers.Conv2DLayer import Conv2DLayer
from modelzoo.common.tf.layers.Conv2DTransposeLayer import Conv2DTransposeLayer
from modelzoo.common.tf.layers.MaxPool2DLayer import MaxPool2DLayer
from modelzoo.common.tf.metrics.dice_coefficient import dice_coefficient_metric
from modelzoo.common.tf.optimizers.Trainer import Trainer
from modelzoo.common.tf.TFBaseModel import TFBaseModel
from modelzoo.vision.tf.unet.utils import color_codes


class UNetModel(TFBaseModel):
    """
    UNet model to be used with TF Estimator
    """

    def __init__(self, params):
        super(UNetModel, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )

        self.num_classes = params["train_input"]["num_classes"]
        assert (
            self.num_classes == 2
        ), "Currently only binary classification is supported!"
        self.num_output_channels = 1

        self.logging_dict = {}

        ### Model params
        mparams = params["model"]
        self.skip_connect = mparams["skip_connect"]
        self.eval_ignore_classes = mparams.get("eval_ignore_classes", [])

        self.data_format = mparams["data_format"]
        self.features_axis = 1 if self.data_format == "channels_first" else -1
        self.downscale_method = mparams.get("downscale_method", "max_pool")

        self.enable_bias = mparams["enable_bias"]
        self.nonlinearity = mparams["nonlinearity"]
        self.nonlinearity_params = mparams.get("nonlinearity_params", dict())
        self.nonlinearity = getattr(tf.keras.layers, self.nonlinearity)(
            **{**self.nonlinearity_params, **dict(dtype=self.policy)},
        )
        self.encoder_filters = mparams["encoder_filters"]
        self.decoder_filters = mparams["decoder_filters"]

        self.initial_conv_filters = mparams.get("initial_conv_filters")
        self.convs_per_block = mparams.get(
            "convs_per_block", ["3x3_conv", "3x3_conv"]
        )

        self.eval_metrics = mparams.get(
            "eval_metrics", ["mIOU", "DSC", "MPCA", "Acc"]
        )
        assert (
            len(self.encoder_filters) == len(self.decoder_filters) + 1
        ), "Number of encoder filters should be equal to number of decoder filters + 1 (bottleneck)"

        self.initializer = mparams["initializer"]
        self.initializer_params = mparams.get("initializer_params")
        if self.initializer_params:
            self.initializer = getattr(
                tf.compat.v1.keras.initializers, self.initializer
            )(**self.initializer_params)

        self.bias_initializer = mparams["bias_initializer"]
        self.bias_initializer_params = mparams.get("bias_initializer_params")
        if self.bias_initializer_params:
            self.bias_initializer = getattr(
                tf.compat.v1.keras.initializers, self.bias_initializer
            )(**self.bias_initializer_params)

        # CS util params for layers
        self.boundary_casting = mparams["boundary_casting"]
        self.tf_summary = mparams["tf_summary"]

        self.output_dir = params["runconfig"]["model_dir"]
        self.log_image_summaries = mparams.get("log_image_summaries", False)
        self.mixed_precision = mparams["mixed_precision"]

        # Model trainer
        self.trainer = Trainer(
            params=params["optimizer"],
            tf_summary=self.tf_summary,
            mixed_precision=self.mixed_precision,
        )

    def _unet_block(self, x, block_idx, n_filters, encoder=True):
        with tf.compat.v1.name_scope(f"block{block_idx}"):
            skip_connection = None
            for conv_idx, conv_type in enumerate(self.convs_per_block):
                if conv_type == "3x3_conv":
                    x = Conv2DLayer(
                        filters=n_filters,
                        kernel_size=3,
                        strides=(1, 1),
                        padding="same",
                        name=("enc_" if encoder else "dec_")
                        + f"conv{block_idx}_{conv_idx}",
                        data_format=self.data_format,
                        use_bias=self.enable_bias,
                        kernel_initializer=self.initializer,
                        bias_initializer=self.bias_initializer,
                        boundary_casting=self.boundary_casting,
                        tf_summary=self.tf_summary,
                        dtype=self.policy,
                    )(x)
                elif conv_type == "1x1_conv":
                    x = Conv2DLayer(
                        filters=n_filters,
                        kernel_size=1,
                        strides=(1, 1),
                        padding="same",
                        name=("enc_" if encoder else "dec_")
                        + f"1x1_conv{block_idx}_{conv_idx}",
                        data_format=self.data_format,
                        use_bias=self.enable_bias,
                        kernel_initializer=self.initializer,
                        bias_initializer=self.bias_initializer,
                        boundary_casting=self.boundary_casting,
                        tf_summary=self.tf_summary,
                        dtype=self.policy,
                    )(x)
                else:
                    raise ValueError(
                        f"Unsupported convolution type: {conv_type}"
                    )
                x = ActivationLayer(
                    self.nonlinearity,
                    boundary_casting=self.boundary_casting,
                    tf_summary=self.tf_summary,
                    dtype=self.policy,
                )(x)

            return x, x

    def build_model(self, features, mode):
        # Get input image.
        x = features

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        if self.skip_connect:
            skip_connections = []

        if self.initial_conv_filters:
            x = Conv2DLayer(
                filters=self.initial_conv_filters,
                kernel_size=3,
                activation="relu",
                padding="same",
                name="initial_conv",
                data_format=self.data_format,
                use_bias=self.enable_bias,
                kernel_initializer=self.initializer,
                bias_initializer=self.bias_initializer,
                boundary_casting=self.boundary_casting,
                tf_summary=self.tf_summary,
                dtype=self.policy,
            )(x)

        ##### Encoder
        with tf.compat.v1.name_scope("encoder"):
            for block_idx in range(len(self.encoder_filters) - 1):
                x, skip_connection = self._unet_block(
                    x, block_idx, self.encoder_filters[block_idx], encoder=True
                )
                if self.skip_connect:
                    skip_connections.append(skip_connection)

                if self.downscale_method == "max_pool":
                    x = MaxPool2DLayer(
                        pool_size=2,
                        strides=2,
                        name=f"pool{block_idx}",
                        data_format=self.data_format,
                        boundary_casting=self.boundary_casting,
                        tf_summary=self.tf_summary,
                        dtype=self.policy,
                    )(
                        x
                    )  # W/(2^(block_idx+1)) x H/(2^(block_idx+1))

        ##### Bottleneck
        with tf.compat.v1.name_scope("bottleneck"):
            x, skip_connection = self._unet_block(
                x,
                len(self.encoder_filters) - 1,
                self.encoder_filters[-1],
                encoder=False,
            )

        ##### Decoder
        with tf.compat.v1.name_scope("decoder"):
            for block_idx in range(len(self.decoder_filters)):
                with tf.compat.v1.name_scope(f"block{block_idx}"):
                    x = Conv2DTransposeLayer(
                        filters=self.decoder_filters[block_idx],
                        kernel_size=2,
                        strides=2,
                        padding="same",
                        name=f"dec_convt{block_idx}",
                        data_format=self.data_format,
                        use_bias=self.enable_bias,
                        kernel_initializer=self.initializer,
                        bias_initializer=self.bias_initializer,
                        boundary_casting=self.boundary_casting,
                        tf_summary=self.tf_summary,
                        dtype=self.policy,
                    )(x)

                    if self.skip_connect:
                        x = concatenate(
                            [x, skip_connections[-1 - block_idx]],
                            axis=self.features_axis,
                            dtype=self.policy,
                        )

                    x, _ = self._unet_block(
                        x,
                        block_idx,
                        self.decoder_filters[block_idx],
                        encoder=False,
                    )

        ##### Output
        logits = Conv2DLayer(
            filters=self.num_output_channels,
            kernel_size=1,
            activation="linear",
            padding="same",
            name="output_conv",
            data_format=self.data_format,
            use_bias=self.enable_bias,
            kernel_initializer=self.initializer,
            bias_initializer=self.bias_initializer,
            boundary_casting=self.boundary_casting,
            tf_summary=self.tf_summary,
            dtype=self.policy,
        )(x)

        return logits

    def build_total_loss(self, logits, features, labels, mode):
        # Get input image and corresponding gt mask.
        input_image = features
        reshaped_mask_image = labels

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # Flatten the logits
        flatten = Flatten(
            dtype="float16" if self.mixed_precision else "float32"
        )
        reshaped_logits = flatten(logits)

        # Binary Cross-Entropy loss
        loss = tf.compat.v1.losses.sigmoid_cross_entropy(
            reshaped_mask_image,
            reshaped_logits,
            loss_collection=None,
            reduction=Reduction.SUM_OVER_BATCH_SIZE,
        )

        if self.log_image_summaries and is_training:
            self._write_image_summaries(
                logits, input_image, reshaped_mask_image, is_training=True,
            )

        return loss

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self.trainer.build_train_ops(total_loss)

    def _write_image_summaries(
        self, logits, input_image, mask_image, is_training=True
    ):
        def _get_image_summary(img):
            """
            Make an image summary for 4d tensor image.
            """
            V = img - tf.reduce_min(input_tensor=img)
            V = V / tf.reduce_max(input_tensor=V)
            V *= 255
            V = tf.cast(V, tf.uint8)

            return V

        def _convert_mask_to_rgb(image):
            color_tensors = []
            for i in range(self.num_classes):
                color_tensors.append(
                    tf.concat(
                        [
                            (color_codes[i][0] / 255) * tf.ones_like(image),
                            (color_codes[i][1] / 255) * tf.ones_like(image),
                            (color_codes[i][2] / 255) * tf.ones_like(image),
                        ],
                        axis=-1,
                    )
                )

            image = tf.tile(image, tf.constant([1, 1, 1, 3], tf.int32))
            image_int32 = tf.cast(image, tf.int32)
            for i in range(self.num_classes):
                image = tf.where(
                    tf.math.equal(
                        image_int32,
                        tf.constant(i, shape=mask_image.shape, dtype=tf.int32),
                    ),
                    color_tensors[i],
                    image,
                )

            return image

        if is_training:
            eval_suffix = ""
        else:
            eval_suffix = "_eval"

        # Display original input image.
        input_image = tf.transpose(a=input_image, perm=[0, 2, 3, 1])

        mask_image = tf.reshape(mask_image, input_image.shape[0:3] + [1])
        mask_image = tf.cast(
            mask_image, tf.float16 if self.mixed_precision else tf.float32
        )

        if input_image.shape[-1] != 3:
            input_image = tf.tile(
                input_image, tf.constant([1, 1, 1, 3], tf.int32)
            )

        tf.compat.v1.summary.image(
            "Input_image" + eval_suffix, _get_image_summary(input_image), 3,
        )

        tf.compat.v1.summary.image(
            "Original_mask" + eval_suffix,
            _get_image_summary(_convert_mask_to_rgb(mask_image)),
            3,
        )

        tf.compat.v1.summary.image(
            "Input_image_mask_overlayed" + eval_suffix,
            _get_image_summary(
                0.6 * input_image + 0.4 * _convert_mask_to_rgb(mask_image)
            ),
            3,
        )

        if self.num_output_channels == 1:
            logits = tf.concat(
                [tf.ones(logits.shape, dtype=logits.dtype) - logits, logits,],
                axis=-1,
            )

        preds = tf.argmax(input=logits, axis=3)
        preds = tf.expand_dims(preds, -1)
        preds = tf.cast(preds, mask_image.dtype)

        # Display reconstructed mask from U-Net.
        tf.compat.v1.summary.image(
            "Reconstruction_mask" + eval_suffix,
            _get_image_summary(_convert_mask_to_rgb(preds)),
            3,
        )

        tf.compat.v1.summary.image(
            "Reconstruction_mask_overlayed" + eval_suffix,
            _get_image_summary(
                0.6 * input_image + 0.4 * _convert_mask_to_rgb(preds)
            ),
            3,
        )

    def build_eval_metric_ops(self, logits, labels, features):
        """
        Evaluation metrics
        """
        reshaped_mask_image = labels

        reshaped_mask_image = tf.cast(reshaped_mask_image, dtype=tf.int32)

        # Ensure channels are the last dimension for the rest of eval
        # metric calculations. Otherwise, need to do the rest of ops
        # according to the channels dimension
        if self.data_format == "channels_first":
            logits = tf.transpose(a=logits, perm=[0, 2, 3, 1])

        pred = tf.reshape(
            logits, [tf.shape(input=logits)[0], -1, self.num_output_channels],
        )

        if self.num_output_channels == 1:
            pred = tf.concat(
                [tf.ones(pred.shape, dtype=pred.dtype) - pred, pred], axis=-1
            )

        pred = tf.argmax(pred, axis=-1)

        # ignore void classes
        ignore_classes_tensor = tf.constant(
            False, shape=reshaped_mask_image.shape, dtype=tf.bool
        )
        for ignored_class in self.eval_ignore_classes:
            ignore_classes_tensor = tf.math.logical_or(
                ignore_classes_tensor,
                tf.math.equal(
                    reshaped_mask_image,
                    tf.constant(
                        ignored_class,
                        shape=reshaped_mask_image.shape,
                        dtype=tf.int32,
                    ),
                ),
            )

        weights = tf.where(
            ignore_classes_tensor,
            tf.zeros_like(reshaped_mask_image),
            tf.ones_like(reshaped_mask_image),
        )

        metrics_dict = dict()

        if "DSC" in self.eval_metrics:
            metrics_dict["eval/dice_coefficient"] = dice_coefficient_metric(
                labels=reshaped_mask_image,
                predictions=pred,
                num_classes=self.num_classes,
                weights=weights,
            )

        if "mIOU" in self.eval_metrics:
            metrics_dict["eval/mean_iou"] = tf.compat.v1.metrics.mean_iou(
                labels=reshaped_mask_image,
                predictions=pred,
                num_classes=self.num_classes,
                weights=weights,
            )

        if "MPCA" in self.eval_metrics:
            metrics_dict[
                "eval/mean_per_class_accuracy"
            ] = tf.compat.v1.metrics.mean_per_class_accuracy(
                labels=reshaped_mask_image,
                predictions=pred,
                num_classes=self.num_classes,
                weights=weights,
            )

        if "Acc" in self.eval_metrics:
            metrics_dict["eval/accuracy"] = tf.compat.v1.metrics.accuracy(
                labels=reshaped_mask_image, predictions=pred, weights=weights,
            )

        return metrics_dict

    def get_evaluation_hooks(self, logits, labels, features):
        """ As a result of this TF issue, need to explicitly define summary
        hooks to able to log image summaries in eval mode
        https://github.com/tensorflow/tensorflow/issues/15332
        """
        if self.log_image_summaries:
            input_image = features
            reshaped_mask_image = labels
            reshaped_mask_image = tf.cast(reshaped_mask_image, dtype=tf.int32)
            self._write_image_summaries(
                logits, input_image, reshaped_mask_image, is_training=False,
            )
            summary_hook = tf.estimator.SummarySaverHook(
                save_steps=1,
                output_dir=self.output_dir,
                summary_op=tf.compat.v1.summary.merge_all(),
            )
            return [summary_hook]
        else:
            return None

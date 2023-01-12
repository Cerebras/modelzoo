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
AUC (Area under the curve) metric for PyTorch.
"""
import torch

from modelzoo.common.pytorch.metrics.cb_metric import CBMetric
from modelzoo.common.pytorch.metrics.metric_utils import divide_no_nan


class AUCMetric(CBMetric):
    """
    The AUC (area under the curve) of the ROC (Receiver operating characteristic;
    default) or PR (Precision Recall) curves are quality measures of binary
    classifiers. Unlike the accuracy, and like cross-entropy losses, ROC-AUC
    and PR-AUC evaluate all the operating points of a model.

    This class approximates AUCs using a Riemann sum. During the metric
    accumulation phase, predictions are accumulated within predefined buckets
    by value. The AUC is then computed by interpolating pre-bucket averages.
    These buckets define the evaluated operational points.

    Internally, we keep track of four local variables, `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` that are used to
    compute the AUC.

    To discretize the AUC curve, a linearly spaced set of thresholds is used to
    compute pairs of recall and precision values. The area under the ROC-curve
    is therefore computed using the height of the recall values by the false
    positive rate, while the area under the PR-curve is computed using the
    height of the precision values by the recall. This value is ultimately
    returned as `auc`.

    The `num_thresholds` variable controls the degree of discretization with
    larger numbers of thresholds more closely approximating the true AUC. The
    quality of the approximation may vary dramatically depending on
    `num_thresholds`. The `thresholds` parameter can be used to manually specify
    thresholds which split the predictions more evenly.

    For a best approximation of the real AUC, `predictions` should be distributed
    approximately uniformly in the range [0, 1] and not peaked around 0 or 1.
    The quality of the AUC approximation may be poor if this is not the case.
    Setting `summation_method` to 'minoring' or 'majoring' can help quantify the
    error in the approximation by providing lower or upper bound estimate of
    the AUC.

    If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

    Args:
        labels: A `Tensor` of ground truth labels of type `int32` or `int64`
            and shape matching `predictions`. Will be cast to `bool`.
        predictions: A floating point `Tensor` of size (num_predictions,)
            and whose values are in the range `[0, 1]`.
        weights: (Optional) `Tensor` whose rank is either 0, or the
            same rank as `labels`, and must be broadcastable to `labels`
            (i.e., all dimensions must be either `1` or the same as the
            corresponding `labels` dimension).
        num_thresholds: (Optional) Defaults to 200. The number of thresholds to
            use when discretizing the roc curve. Value must be > 1.
        curve: (Optional) Specifies the name of the curve to be computed, `ROC`
            [default] or 'PR' for the Precision-Recall-curve.
        name: (Optional) string name of the metric instance. If None or empty
            string, it defaults to the name of the class.
        summation_method: (Optional) Specifies the Riemann summation method
            (https://en.wikipedia.org/wiki/Riemann_sum) used.
            'interpolation' (default) applies mid-point summation scheme for
                `ROC`. For PR-AUC, interpolates (true/false) positives but not
                the ratio that is precision (see Davis & Goadrich 2006 for details);
            'minoring' applies left summation for increasing intervals and
                right summation for decreasing intervals;
            'majoring' that does the opposite of `minoring`.
        thresholds: (Optional) A list of floating point values to use as the
            thresholds for discretizing the curve. If set, the `num_thresholds`
            parameter is ignored. Values should be in [0, 1]. Endpoint thresholds
            equal to {-epsilon, 1+epsilon} for a small positive epsilon value
            will be automatically included with these to correctly handle
            predictions equal to exactly 0 or 1.

    Returns:
        auc: A float representing the approximated value of area under the curve.

    Raises:
        ValueError: If `predictions` and `labels` have mismatched shapes, or if
            `weights` is not `None` and its shape doesn't match `predictions`.
    """

    def __init__(
        self,
        num_thresholds=200,
        curve="ROC",
        name=None,
        summation_method="interpolation",
        thresholds=None,
    ):
        if curve not in ("ROC", "PR"):
            raise ValueError(
                f"Curve must be either ROC or PR. Curve {curve} is unknown."
            )
        self.curve = curve

        if summation_method not in ("interpolation", "minoring", "majoring"):
            raise ValueError(
                f"Invalid AUC summation method value: '{summation_method}'. "
                "Expected values are ['interpolation', 'majoring', 'minoring']"
            )
        self.summation_method = summation_method

        if thresholds is not None:
            self.num_thresholds = len(thresholds) + 2
            thresholds = sorted(thresholds)
        else:
            if num_thresholds <= 1:
                raise ValueError(
                    f"num_thresholds must be > 1. Got {num_thresholds}"
                )
            self.num_thresholds = num_thresholds
            thresholds = [
                (i + 1) * 1.0 / (num_thresholds - 1)
                for i in range(num_thresholds - 2)
            ]

        # Add an endpoint "threshold" below zero and above one for either
        # threshold method to account for floating point imprecisions.
        epsilon = 1e-7
        self.thresholds = torch.tensor(
            [0.0 - epsilon] + thresholds + [1.0 + epsilon], dtype=torch.float
        ).reshape(-1, 1)

        super(AUCMetric, self).__init__(name=name)

    def init_state(self):
        self.reset_state()

    def update_on_host(self, labels, predictions, weights=None):
        if labels.shape != predictions.shape:
            raise ValueError(
                f"`labels` and `predictions` have mismatched shapes of "
                f"{labels.shape} and {predictions.shape} respectively."
            )
        if weights is not None:
            if weights.shape != labels.shape:
                raise ValueError(
                    f"`labels`={labels.shape} and ",
                    f"`weights`={weights.shape} have mismatched shapes",
                )
            weights = weights.detach()
            if len(weights.shape) > 1:
                weights = torch.flatten(weights)

        labels = labels.detach()
        predictions = predictions.detach()

        if len(labels.shape) > 1:
            labels = torch.flatten(labels)
            predictions = torch.flatten(predictions)

        if torch.amin(labels) < 0:
            raise ValueError(
                f"Negative values in `labels` tensor is not allowed"
            )
        if torch.amin(predictions) < 0 or torch.amax(predictions) > 1:
            raise ValueError(
                f"Values in `predictions` tensor must be in [0, 1]"
            )

        # Tile predictions to have shape (self.num_thresholds, num_predictions)
        preds_tiled = torch.tile(predictions, (self.num_thresholds, 1))

        # Tile thresholds to have shape (self.num_thresholds, num_predictions)
        thresh_tiled = self.thresholds.expand(-1, predictions.shape[0])

        # Compare predictions and threshold
        pred_is_pos = torch.gt(preds_tiled, thresh_tiled)
        pred_is_neg = torch.logical_not(pred_is_pos)
        label_is_pos = labels > 0
        label_is_neg = torch.logical_not(label_is_pos)

        def weighted_assign_add(label, pred, var, weights=None):
            label_and_pred = torch.logical_and(label, pred)
            if weights is not None:
                label_and_pred = label_and_pred.mul(weights)
            var.add_(torch.sum(label_and_pred, axis=-1))

        # Update values for TP, FP, FN, TN
        weighted_assign_add(
            label_is_pos, pred_is_pos, self.true_positive, weights
        )
        weighted_assign_add(
            label_is_neg, pred_is_pos, self.false_positive, weights
        )
        weighted_assign_add(
            label_is_pos, pred_is_neg, self.false_negative, weights
        )
        weighted_assign_add(
            label_is_neg, pred_is_neg, self.true_negative, weights
        )

    def interpolate_pr_auc(self):
        """
        Interpolation formula inspired by section 4 of Davis & Goadrich 2006.
        https://www.biostat.wisc.edu/~page/rocpr.pdf
        Note here we derive & use a closed formula not present in the paper
        as follows:

        Precision = TP / (TP + FP) = TP / P

        Modeling all of TP (true positive), FP (false positive) and their sum
        P = TP + FP (predicted positive) as varying linearly within each interval
        [A, B] between successive thresholds, we get:

        Precision slope = dTP / dP
                        = (TP_B - TP_A) / (P_B - P_A)
                        = (TP - TP_A) / (P - P_A)
        Precision = (TP_A + slope * (P - P_A)) / P

        The area within the interval is (slope / total_pos_weight) times

        int_A^B{Precision.dP} = int_A^B{(TP_A + slope * (P - P_A)) * dP / P}
        int_A^B{Precision.dP} = int_A^B{slope * dP + intercept * dP / P}

        where intercept = TP_A - slope * P_A = TP_B - slope * P_B, resulting in

        int_A^B{Precision.dP} = TP_B - TP_A + intercept * log(P_B / P_A)

        Bringing back the factor (slope / total_pos_weight) we'd put aside, we get

        slope * [dTP + intercept *  log(P_B / P_A)] / total_pos_weight

        where dTP == TP_B - TP_A.

        Note that when P_A == 0 the above calculation simplifies into

        int_A^B{Precision.dTP} = int_A^B{slope * dTP} = slope * (TP_B - TP_A)

        which is really equivalent to imputing constant precision throughout the
        first bucket having >0 true positives.
        """
        dtp = (
            self.true_positive[: self.num_thresholds - 1]
            - self.true_positive[1:]
        )
        p = self.true_positive + self.false_positive
        dp = p[: self.num_thresholds - 1] - p[1:]
        prec_slope = divide_no_nan(dtp, torch.clamp(dp, min=0))
        intercept = self.true_positive[1:] - torch.multiply(prec_slope, p[1:])

        safe_p_ratio = torch.where(
            torch.logical_and(p[: self.num_thresholds - 1] > 0, p[1:] > 0),
            divide_no_nan(
                p[: self.num_thresholds - 1], torch.clamp(p[1:], min=0)
            ),
            torch.ones_like(p[1:], dtype=torch.float),
        )

        pr_auc_increment = divide_no_nan(
            prec_slope * (dtp + intercept * torch.log(safe_p_ratio)),
            torch.clamp(
                self.true_positive[1:] + self.false_negative[1:], min=0
            ),
        )

        return torch.sum(pr_auc_increment)

    def compute(self):
        """Returns the AUC as a float"""
        if self.curve == "PR" and self.summation_method == "interpolation":
            return float(self.interpolate_pr_auc())

        # Set 'x' and 'y' values for the curves
        recall = divide_no_nan(
            self.true_positive, self.true_positive + self.false_negative
        )
        if self.curve == "ROC":
            fp_rate = divide_no_nan(
                self.false_positive, self.false_positive + self.true_negative
            )
            x = fp_rate
            y = recall
        else:  # curve == "PR"
            precision = divide_no_nan(
                self.true_positive, self.true_positive + self.false_positive
            )
            x = recall
            y = precision

        # Find the rectangle heights based on `summation_method`
        if self.summation_method == "interpolation":
            heights = (y[: self.num_thresholds - 1] + y[1:]) / 2
        elif self.summation_method == "minoring":
            heights = torch.minimum(y[: self.num_thresholds - 1], y[1:])
        else:  # self.summation_method == "majoring"
            heights = torch.maximum(y[: self.num_thresholds - 1], y[1:])

        # Sum up the areas of all the rectangles
        return float(
            torch.sum(torch.mul(x[: self.num_thresholds - 1] - x[1:], heights))
        )

    def reset_state(self):
        self.true_positive = torch.zeros(
            self.num_thresholds, dtype=torch.float32
        )
        self.true_negative = torch.zeros(
            self.num_thresholds, dtype=torch.float32
        )
        self.false_positive = torch.zeros(
            self.num_thresholds, dtype=torch.float32
        )
        self.false_negative = torch.zeros(
            self.num_thresholds, dtype=torch.float32
        )

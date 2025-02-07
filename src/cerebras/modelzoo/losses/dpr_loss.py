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

from torch import nn


class DPRLoss(nn.Module):
    def __init__(self, mutual_information, softmax_temperature):
        super(DPRLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.mutual_information = mutual_information
        self.softmax_temperature = softmax_temperature

    def forward(self, q2c_scores, labels, c2q_scores=None, context_labels=None):
        '''
        Args:
        q2c_scores: question-to-context scores
                    (batch_size, batch_size * num_context)
        labels: labels for question-to-context scores
                    (batch_size, )

        Optional Args:
        c2q_scores: context_to_question scores
                    (batch_size, batch_size)
        context_labels: labels for context_to_question scores
                    (batch_size, )

        Please see the comment in dpr_model.py for more details.

        '''
        num_context = int(q2c_scores.shape[1] / q2c_scores.shape[0])
        # By setting mutual information to True, we are adding additional
        # context-to-question loss to the original question-to-context loss,
        # which makes the loss value scale different from the case
        # where mutual information is OFF. Therefore, we add a distribution_factor,
        # based on the number of contexts, to keep loss value at the same scale.
        distribution_factor = 1 / (1 + num_context)

        if not self.mutual_information:
            loss = self.loss_fn(q2c_scores / self.softmax_temperature, labels)
        else:
            question_to_context_loss = self.loss_fn(
                q2c_scores / self.softmax_temperature, labels
            )
            context_to_question_loss = self.loss_fn(
                c2q_scores / self.softmax_temperature, context_labels
            )
            loss = (
                (1 - distribution_factor) * question_to_context_loss
                + distribution_factor * context_to_question_loss
            )
        return loss

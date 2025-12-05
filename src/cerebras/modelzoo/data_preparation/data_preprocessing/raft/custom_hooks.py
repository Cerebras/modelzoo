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


def raft_toy_bigger_split_context_hook(example, kwargs):
    contexts = example['context']

    if isinstance(contexts, str):
        contexts = [contexts]

    return {'contexts': contexts}


def raft_toy_bigger_split_question_hook(example, kwargs):

    return {'id': example['id'], 'question': example['question']}

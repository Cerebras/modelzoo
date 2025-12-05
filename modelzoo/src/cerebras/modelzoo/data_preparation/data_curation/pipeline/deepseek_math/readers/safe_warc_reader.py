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

import logging

from datatrove.pipeline.readers.warc import WarcReader


class SafeWarcReader(WarcReader):
    def read_file(self, filepath):
        try:
            for doc in super().read_file(filepath):
                yield doc
        except EOFError as e:
            logging.warning(f"EOFError in file {filepath}: {e}")
        except Exception as e:
            logging.warning(f"Error in file {filepath}: {e}")

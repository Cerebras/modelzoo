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

from glob import glob
from itertools import repeat, takewhile


def count_lines(filename_pattern):
    """
    Returns linecount in the given filename or sum over all of
    the filenames matching the pattern.

    Takes a filename pattern and globs the pattern
    to get all matching filenames. In order, this function reads all the
    files in raw format for a fixed buffer size till EOF is reached. It
    then counts the number of ``\n``
    in each buffer and returns the sum. Using ``takewhile`` and ``repeat``
    provides inbuilt speedup compared to writing custom while/for loop to
    handle EOF.

    `Note`: The size of the buffer is currently set to 1024*1024, but this
    is not optimized for all files. Some files can be read in faster by
    modifying the buffer size. This value is suboptimal to memory usage on
    a local dev instance.

    :param str filename_pattern: filename glob pattern (or filename)
    :returns: integer number of lines in the file
    """
    line_count = 0
    for filename in glob(filename_pattern):
        with open(filename, "rb") as f:
            bufgen = takewhile(
                lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None))
            )
            line_count += sum(buf.count(b'\n') for buf in bufgen)

        # check if last line has "\n"; if not increment line count by 1
        with open(filename, "r") as f:
            # seek to the end of the file
            f.seek(0, 2)
            # seek the last character of file
            f.seek(f.tell() - 1, 0)
            if f.read() != "\n":
                line_count += 1

    return line_count

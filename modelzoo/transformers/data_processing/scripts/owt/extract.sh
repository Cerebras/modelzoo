#!/bin/bash

# Copyright 2020 Cerebras Systems.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


## Currently, there is a server side issue with directly downloading the
## files using the link. Once the team managing the dataset have addressed these
## issues, we can reuse the below code.
# pip install gdown
# gdown https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx

tar -xvf openwebtext.tar.xz
cd openwebtext
for f in *.xz; do mkdir ${f%.xz}; tar -Jvxf $f -C ${f%.xz}; rm $f; done

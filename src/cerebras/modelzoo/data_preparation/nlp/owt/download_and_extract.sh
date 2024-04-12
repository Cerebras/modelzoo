#!/bin/bash

pip install gdown
gdown https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx

tar -xvf openwebtext.tar.xz
cd openwebtext
for f in *.xz; do mkdir ${f%.xz}; tar -Jvxf $f -C ${f%.xz}; rm $f; done

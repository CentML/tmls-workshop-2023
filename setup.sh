#!/bin/bash

LOCAL=${1:-$PWD}
python3 -m virtualenv $LOCAL/centml_tools
source $LOCAL/centml_tools/bin/activate
pip install torch deepview-profile numpy packaging
git clone https://github.com/NVIDIA/apex
pushd $LOCAL/apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
popd
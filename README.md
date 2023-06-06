# TMLS Workshop 2023

This repository contains the necesary files for the workshop.

If you wish to run locally, you will need:

- Pytorch >=2.0.0
- Nvidia Apex package (https://github.com/NVIDIA/apex)
- Deepview Profile (https://pypi.org/project/deepview-profile/)
- Deepview Explore (https://marketplace.visualstudio.com/items?itemName=CentML.deepview-explore)

Follow these steps: <br>

```
1. Create a python virtual environment and activate it (recommended)
   - python3 -m virtualenv ./env
   - source env/bin/activate
2. pip install torch deepview-profile numpy packaging
3. For Apex:
    - git clone https://github.com/NVIDIA/apex
    - cd apex
    - pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Alternative installation

```
1. clone this repository
2. cd tmls-workshop-2023
3. bash setup.sh
```

## Notes:

Independently of the installation method, for Apex you will need to install local cuda libraries. You can follow these [instructions](https://towardsdatascience.com/installing-multiple-cuda-cudnn-versions-in-ubuntu-fcb6aa5194e2)

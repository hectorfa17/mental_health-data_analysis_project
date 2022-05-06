![](images/banner.jpg)

## Overview

Data Analytics project about Mental Health at work. The project deep dives on attitudes from both Tech employers and employees towards mental health and mental health dissorders at work.

The Dataset I used belongs to "OPEN SOURCING MENTAL ILLNESS, LTD", you can find it on [this link](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)

## Contents

- Notebooks
1. `Main.ipynb` - Main Notebook with a more straight-to-the-point path to the resolution, all functions are called from their respective module, all scalers/models/transformers are loaded from pkl files.
2. `extended_da.ipynb` - Original Notebook with all the code I used, included some additional notes, all functions are written here, before being reallocated to their respective module.
3. `functions.py` - Separate module with all the functions I used.
4. `src/4.song_recomender_function.ipynb` - Created the function "user friendly"

- Libraries
1. `src/spoty_jzar.py` - This is the class that is used as a Spotify interface
2. `src/music_jzar.py` - this lib is used to scrap and get the raw lists
3. `src/cluster_jzar.py` - this lib is used for clustering functions

## Datasets

-  `data/survey` - Thanks to "OPEN SOURCING MENTAL ILLNESS, LTD" for creating this dataset, you can find it on [this link](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
-  `data/survey_updated` - Same Dataset, after being cleaned and processed.

## Requirements

appnope==0.1.2
argon2-cffi==21.3.0
argon2-cffi-bindings==21.2.0
asttokens==2.0.5
attrs==21.4.0
autopep8==1.6.0
backcall==0.2.0
beautifulsoup4==4.11.1
bleach==4.1.0
certifi==2021.10.8
cffi==1.15.0
cycler==0.11.0
debugpy==1.5.1
decorator==5.1.1
defusedxml==0.7.1
entrypoints==0.4
executing==0.8.3
fastjsonschema==2.15.1
fonttools==4.33.3
ipykernel==6.9.1
ipython==8.3.0
ipython-genutils==0.2.0
jedi==0.18.1
Jinja2==3.0.3
joblib==1.1.0
jsonschema==4.4.0
jupyter-client==7.2.2
jupyter-contrib-core==0.3.3
jupyter-core==4.10.0
jupyter-nbextensions-configurator==0.4.1
jupyterlab-pygments==0.1.2
kiwisolver==1.4.2
MarkupSafe==2.0.1
matplotlib==3.5.2
matplotlib-inline==0.1.2
mistune==0.8.4
nbclient==0.5.13
nbconvert==6.5.0
nbformat==5.3.0
nest-asyncio==1.5.5
notebook==6.4.11
numpy==1.22.3
packaging==21.3
pandas==1.4.2
pandocfilters==1.5.0
parso==0.8.3
pexpect==4.8.0
pickleshare==0.7.5
Pillow==9.1.0
pip==21.2.4
prometheus-client==0.13.1
prompt-toolkit==3.0.20
ptyprocess==0.7.0
pure-eval==0.2.2
pycodestyle==2.8.0
pycparser==2.21
Pygments==2.11.2
pyparsing==3.0.4
pyrsistent==0.18.0
python-dateutil==2.8.2
pytz==2022.1
PyYAML==6.0
pyzmq==22.3.0
scikit-learn==1.0.2
scipy==1.8.0
seaborn==0.11.2
Send2Trash==1.8.0
setuptools==61.2.0
six==1.16.0
sklearn==0.0
soupsieve==2.3.2
stack-data==0.2.0
terminado==0.13.1
testpath==0.5.0
threadpoolctl==3.1.0
tinycss2==1.1.1
toml==0.10.2
tornado==6.1
traitlets==5.1.1
typing_extensions==4.1.1
wcwidth==0.2.5
webencodings==0.5.1
wheel==0.37.1

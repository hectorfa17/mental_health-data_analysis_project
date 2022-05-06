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

- beautifulsoup4==4.10.0
- matplotlib==3.3.4
- numpy==1.19.5
- pandas==1.4.1
- requests==2.25.1
- scikit_learn==1.0.2
- scipy==1.6.0
- seaborn==0.11.2
- spotipy==2.19.0
- streamlit==1.8.0


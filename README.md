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

## Installation:

1. Clone this repo
2. Install all the requirements from the requirements.txt file
3. Navigate through the Notebooks




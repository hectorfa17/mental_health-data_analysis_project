![](images/banner.jpg)

## Overview

Data Analytics project about Mental Health at work. The project deep dives on attitudes from both Tech employers and employees towards mental health and mental health dissorders at work.

The project consists of 3 parts: 

1. Machine Learning applying 2 Logistic Regression models to predict whether an employee would sick for mental health help or not.
2. Hypothesis testing to find out whether the mean of age of people who seek for mental health help is similar or not.
3. Data Visualization with Tableau, where I propose and answer some questions interesting insights after a thorough data exploration process.

## Contents

- Notebooks
1. `Main.ipynb` - Main Notebook with a more straight-to-the-point path to the resolution, all functions are called from their respective module, all scalers/models/transformers are loaded from pkl files.
2. `extended_da.ipynb` - Original Notebook with all the code I used, included some additional notes, all functions are written here, before being reallocated to their respective module.
3. `datavisualization_tableau.txt`- File containing the link to the Data Visualization presentation in Tableau, for a quicker access, click on [this link] (https://public.tableau.com/app/profile/hector.fontenla/viz/MentalHealth_16473455406840/Story1?publish=yes)

- Custom Modules
1. `functions.py` - Separate module with all the functions I used.

- Other folders
1. data - Contains all the Datasets used in this project
2. encoders - Contains all the fitted Encoders stored in pkl files.
3. models - Contains all the fitted Models stored in pkl files.
4. transformers - Contains all the fitted Transformers/Scalers stored in pkl files.

## Datasets

-  `data/survey` - Thanks to "OPEN SOURCING MENTAL ILLNESS, LTD" for creating this dataset, you can find it on [this link](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
-  `data/survey_updated` - Same Dataset, after being cleaned and processed.

## Installation:

1. Clone this repo
2. Install all the requirements from the requirements.txt file
3. Navigate through the Notebooks




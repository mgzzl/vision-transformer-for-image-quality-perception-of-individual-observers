# Vision Transformer For Image Quality Prediction Of Individual Observers - Dataset

### Description
This directory contains all the utilized datasets. Under `Dataset`, you can find both all the images and all the evaluations used for training the AI models. The directory `Obs_iterative` contains all the evaluations from the participants, but split into the conducted sessions (1-4). Sessions 1 and 4 include the same images but with different evaluations. Example: `assets/Obs_iterative/Obs1/Obs1_1.csv` == `assets/Obs_iterative/Obs1/Obs1_4_selfacc.csv`. Under `Test` you will find the test data set `DSX` and the ratings for this from the perspective of the respective subject.

### Note
There is both a data set DS4.0 and ratings Obs4.0. The reason for this is that an error occurred during the execution of subject 4 (the first session was not repeated here), and therefore all files or directories with the extension 4.0 contain this error. Both DS4 and Obs4 are free of errors.
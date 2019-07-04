# Classical-neural Recommendation System

This is the codebase for the team `Project Passline` on Project: Collaborative Filtering in the course Computational Intelligence Lab of ETH Zurich in Spring 2019. 

Our classical-neural approach blends neural networks based methods and classical non-neural collaborative filtering methods. Our solution achieves 0.97102 RMSE(Root Mean Square Error) on private test set and stands at the second place.

## Set environments on Leonhard

### Login to Leonhard cluster  
1. Connect to ETH VPN.  
2. In your terminal:`ssh [user_name]@login.leonhard.ethz.ch`

### Set up the environments  
load some modules  
```
module load python_gpu/3.6.4 hdf5 eth_proxy
module load cudnn/7.2
```
Clone the git repo
```
cd ~
git clone https://github.com/lavanova/CIL_project.git
```
Install virtualenvwrapper if you haven't done so yet
```
pip3 install --user virtualenvwrapper
source $HOME/.local/bin/virtualenvwrapper.sh
```
Create a virtual environment
```
mkvirtualenv "cil"
```
The virtual environment is by default activated. You can disable or enable it by using
```
deactivate
workon "cil"
```







Recommendation system

Data is not in the repository
Data available: https://inclass.kaggle.com/c/cil-collab-filtering-2019

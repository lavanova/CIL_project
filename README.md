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
pip3 install --user virtualenv
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
With the virtual environment activated, go to our project folder and install the denpendencies
```
cd CIL_project
python setup.py install
```
## Note
After you set up the environment for first time, you have to do following to recover environments everytime you login to leonhard.
```
module load python_gpu/3.6.4 hdf5 eth_proxy
module load cudnn/7.2
source $HOME/.local/bin/virtualenvwrapper.sh
workon "cil"
```

## Initialization and download the dataset

```
sh init.sh
sh download_data.sh
```
Then you can see there are `data_train.csv` and `sampleSubmission.csv` under directory `./data`. They are downloaded from my polybox. You can also download the dataset from kaggle, and move them to leonhard, put them under directory `~/CIL_project/data/`.
## Repository structure
After doing initialization in previous section, you can see our repository has following structure.
* `data/` - dataset files and some other data files generated by our program.
* `cache/` - use to store predictions on our 10% validation set of all our single models. They will be used to decide the weight of every model in blending. For example, you will see `cache/ALS_big`, `cache/KNN_item`, `cache/encoder` etc after training models.
* `test/` - use to store predictions of all our single models on Kaggle test set. They will be blended to get the blending results. For example, you will see `test/ALS_big`, `test/KNN_item`, `test/encoder` etc after training models. You can submit these files to Kaggle to see the performance of our single model(not the blending results).
* `blend_result/` - store the result of our blending method, you will find `blend_result/out.csv` after blending models. You can submit it to Kaggle to see the performance of our blending results.
* `blender_classical/` - implementations of the blending algorithm and classical non-neural methods.
* `encoder/` - implementation of Auto-encoder
* `neucf/` - implementation of neural network based models except Auto-encoder. Auto-encoder is implemented in `encoder/`.
* `rbm/` - implementation of Restricted Bolzmann Machine(RBM). But we don't use RBM.
* `blender.sh` - this script will run `blender_classical/blender.py` to do blending. After running this script, you will find `out.csv` under directory `blend_result`. Then you can submit it to Kaggle to see the performance of our blending method. 
* `download_data.sh` - download dataset.
* `download_results.sh` - this is used when you want to reproduce our blending results without training models. This script will download predictions on Kaggle's test set and our validation set of all our single models into directories `test/` and `cache/` respectively. 
* `init.sh` - do some initialization
* `setup.py` - download dependencies
* `train_classical.sh` - train all our classical non-neural models and store models' predictions on our validation set and Kaggle's test set under directory `cache` and `test` respectively. This script will run `blender_classical/baseline1_main.py`.
* `train_neural.sh` - train all our neural network based models and store models' predictions on our validation set and Kaggle's test set under directory `cache` and `test` respectively. This scripy will run `encoder/training.py` and `neucf/train.sh`.

## Reproduce blending results without training models

Our blending method blends 16 classical models and 14 neural network models, it takes several hours to train models from scratch. So here we provide a way to reproduce our blending results without training models. To reproduce results from scratch, please go to next section `Reproduce results from scratch` where you can train all our models from scratch.

To reproduce blending results without training models, first download all the single models' predictions on validation set and Kaggle's test set. 
```
sh download_results.sh
```
Then blend single models' predictions on Kaggle's test set based on their predictions on validation set.
```
sh blender.sh
```
Then you can find a `out.csv` under directory `blend_result/`. Use `scp` to copy the `out.csv` to your own machine, and submit it to Kaggle. You will see 0.97102 RMSE on private test set and 0.96886 RMSE on public test set.

## Reproduce blending results from scratch

If you just go through section `Reproduce blending results without training models`, before going to reproduce blending results from scratch, you have to do following cleaning work (You can't skip this cleaning process, otherwise there will be 60 models. In our blending, there are 16 classical models and 14 neural models summing up to 30 models).
```
rm cache/*
rm test/*
rm blend_result/*
```
If it prompt something like `rm: remove write-protected regular file 'filename'?`, type `y` to remove that file. Otherwise that file won't get removed.  
Then you can go to train our models.

### Train non-neural classical models
```
sh train_classical.sh
```
or
```
bsub -n 4 -W 24:00 -R "rusage[mem=4096, ngpus_excl_p=1]" sh train_classical.sh
```
Then you will find classical models' predictions on validation set and Kaggle's test set in directories `cache/` and `test/` respectively.  
Before going to train neural models, if you want to see the performance of blending non-neural classical models, you can run `blender.sh`.
```
sh blender.sh
```
Then you can find a `out.csv` under directory `blend_result/`. This is the predictions of blending classical models on Kaggle's test set. Use `scp` to copy the `out.csv` to your own machine, and submit it to Kaggle. You will see ~0.97808 RMSE on private test set and ~0.97631 on public test set(result won't be exactly same, but it will be close to what I post here)

### Train neural models
First train locally linear embedding, factor analysis embedding, spectral embedding and non-negative matrix factorization embedding which are used as external embeddings of neural network.
```
cd neucf
python embedding.py
cd ..
```
Or you can download these four embeddings directly which have already been trained.
```
cd neucf
sh download_embedding.sh
cd ..
```
Then you can start training neural models.
```
bsub -n 4 -W 24:00 -R "rusage[mem=4096, ngpus_excl_p=1]" sh train_neural.sh
```
Then you will find neural models' predictions on validation set and Kaggle's test set in directories `cache/` and `test/` respectively.  
Then you can blend neural models and non-neural classical models to reproduce our best performance.
```
sh blender.sh
```
Then you can find a `out.csv` under directory `blend_result/`. This is the predictions on Kaggle's test set of blending classical models and neural models. Use `scp` to copy the `out.csv` to your own machine, and submit it to Kaggle. You will see ~0.97102 RMSE on private test set and ~0.96886 on public test set(result won't be exactly same, but it will be close to what I post here)

## Authors
CIL Team: Project Passline  
Xinyuan Huang, Chengyuan Yao, Qifan Guo, Hanxue Liang  
ETH Zurich

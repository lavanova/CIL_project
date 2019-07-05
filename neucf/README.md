# Neural models

## Framework structure
* `data.py` - prepare data for training, validation and test for `NeuMF` model and `MLP` model
* `download_embedding.sh` - download locally linear embedding, factor analysis embedding, spectral embedding and non-negative
matrix factorization embedding for my polybox. They serve as external embedding of `MLP_origin` model. You can also train
these embeddings from scratch by running `embedding.py`.
* `embedding.py` - train the above embeddings from scratch.
* `init.sh` - initialization, called by repo root's `init.sh`.
* `model.py` - implementation of variants of `NeuMF` model and `MLP` model. `class NeuCF` implements `NeuMF_8` and `NeuMF_16`.
`class NeuCF2` implements `MLP_origin` and `MLP_ngcfemb`. Other classes are not used in our final solution.
* `move.sh` - deprecated. my experiments file, I use it during project. you don't have to use it. 
* `neucf.py` - training script for `NeuMF` model and `MLP` model. It will put the predictions on validation set and Kaggle's
test set of best model which has the lowest validation RMSE under repo root's directories `cache` and `test` respectively.
* `ngcf.py` - implementation of `NGCF` model, also serve as training script for `NGCF` model. when training end-to-end model
`NGCF_endtoend0` and `NGCF_endtoend1`, it will put the predictions on validation set and Kaggle's
test set of best model which has the lowest validation RMSE under repo root's directories `cache` and `test` respectively.
* `ngcf_data.py` - prepare data for `NGCF` model
* `train.sh` - train 14 neural models which will be used in blending. It will put
predictions on validation set and Kaggle's test set under repo root's directories `cache` and `test` respectively.
* `train2.sh` - deprecated. other experiments, you don't have to use it.
* `utils.py` - utility functions(early_stopping)
## Output
After running `train.sh`, you will find 14 models' predictions on validation set and Kaggle's test set under repo root's directories
`cache` and `test` respectively. For one specific model, it's output files' names under `cache` and `test` are same, for example, `repo_root/cache/normal_decay1500` and `repo_root/test/normal_decay1500`. The one to one corresponding between output file names and model names in report are listed here:
* `MLP_origin` with different training hyper parameters 
  * `normal` - `MLP_origin` without learning rate(lr) decay during training
  * `normal_decay1500` - `MLP_origin` with lr decay step 1500
  * `normal_decay2500` - `MLP_origin` with lr decay step 2500
  * `normal_decay3500` - `MLP_origin` with lr decay step 3500
* `MLP_ngcfemb` with different training hyper parameters
  * `normal_ngcfemb` - `MLP_ngcfemb` without lr decay during training
  * `normal_ngcfemb_decay1500` - `MLP_ngcfemb` with lr decay step 1500
  * `normal_ngcfemb_decay2500` - `MLP_ngcfemb` with lr decay step 2500
  * `normal_ngcfemb_decay3500` - `MLP_ngcfemb` with lr decay step 3500
* `basenn_8` - `NeuMF_8`
* `basenn_16` - `NeuMF_16`
* `ngcf_endtoend0` - `NGCF_endtoend0`
* `ngcf_endtoend1` - `NGCF_endtoend1`
 
## Reference
Our `NGCF` model's implementation is based on [link](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)


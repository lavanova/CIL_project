# AutoEncoder
## Setup
Simply run
```
sh init_encoder.sh
```
## Framework Structure
* `training.py` - the invoking script
* `train_model.py & base_model.py` - model implementations
* `tf_record_writer.py` -writing the training data (data_train.csv) into binary TF_Records
* `dataset.py` -buildind the input pipeline for training data, validation data and test data.
* `utils.py` - utility methods
* `parameters.py` - parameters
* `init_encoder.sh` - creating data dir

## Training Process
Simply run 
```
python tf_record_writer.py
python training.py
```
## Output
The output will be in upper level directory:test and cache. The result will then be used for blending. 
## Reference
Our `AutoEncoder` model's implementation is based on [link](https://github.com/artem-oppermann/Deep-Autoencoders-For-Collaborative-Filtering)


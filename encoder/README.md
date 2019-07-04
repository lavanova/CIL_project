# AutoEncoder
## Setup
Simply run
```
sh init_encoder.sh
```
## Framework Structure
* `training.py` - the invoking script
* `train_model.py & base_model` - model implementations
* `tf_record_writer.py` -writing the csv file to tfrecord 
* `dataset.py` -reading the tfrecord

## Training Process
Simply run 
```
python training.py
```
## Output
The output will be in upper level directory:test and cache. The result will then be used for blending. 

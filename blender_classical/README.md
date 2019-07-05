# Classical blending
## Setup
Firstly need to setup the virtual environment stated in the README under root

## Framework Structure
* `baseline1_main.py` main for running classical models
* `blender.py` - blender function implementation
* `ALS.py` - ALS models
* `surprise_models.py` - surprise models
* `svd_baseline.py` - svd models
* `cross_validate.py` -buildind the input pipeline for training data, validation data and test data.
* `parameters.py` - defining macros/paths
* `utils.py` - utility functions for I/O
* `mean.py` - fill up the CF matrix with mean methods (deprecated)

## Run models and blend
```
python3 baseline1_main.py
python3 blender.py
```
## Output
The output for individual models will be in upper level directory:test and cache. The blending result will be in root out.csv.

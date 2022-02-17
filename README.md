# Modeling cortical network dynamics with CB1-mediated inhibition

> Simulation and analysis code 

## Requirements / Installation

Using `git`:
```
git clone https://github.com/yzerlaut/CB1_ntwk_modeling
cd CB1_ntwk_modeling
git submodule init
git submodule update
```
Note that the code relies on custom modules for [spiking network simulation] (based on Brian2) and [data visualization] (based on matplotlib).

## Reproducing the results of the study

### 1) Optimization of model parameters

#### L23-only model: connectivity parameters
```
python code/L23_connec_params.py scan # N.B. some runs might fail when using multiprocessing, use: scan-fix-missing, scan-with-repeat-fix-missing
```
and analyze with:
```
python code/L23_connec_params.py scan-analysis
```

#### Adding the L4-to-L23 pathway

### 2) Effect of CB1-specific in 

#### N.B. usage

The module uses "relative import" so everything should be launched from the root directory (`CB1_ntwk_modeling/`):
- notebooks: `jupyter notebook notebooks/notebook_of_interest.ipynb`
- scripts: `python code/the_script_of_interest.py --arguments bla bla`


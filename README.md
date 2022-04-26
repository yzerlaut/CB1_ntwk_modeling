# Modeling cortical dynamics with CB1-mediated inhibition

> Simulation and analysis code for the modelling work of the study:
> 

## Requirements / Installation

Using `git`:
```
git clone https://github.com/yzerlaut/CB1_ntwk_modeling
cd CB1_ntwk_modeling
git submodule init
git submodule update
```
Note that the code relies on custom modules for [spiking network simulation](based on Brian2) and [data visualization](based on matplotlib).

## Reproducing the results of the study

All results are reproduced by launching the following set of instructions:
```
# --- Layer23-circuit connectivity optimization --- #
python src/L23_connec_params.py V1 &
python src/L23_connec_params.py V2
# sleep 1s
python src/L23_connec_params.py plot
# --- Layer4-L23 connectivity optimization --- #
python src/L4.py test-run
python src/L4.py test-analysis
# --- look at gain curves --- #
python src/gain.py V1 with-repeat
python src/gain.py V2 with-repeat
python src/gain.py V2-CB1-KO with-repeat
python src/gain.py analysis
# --- psyn dep on spontaneous activity --- #
$python_path src/bg_act.py L23-psyn-scan
$python_path src/bg_act.py L23-psyn-analysis
# --- final spontaneous activity --- #
python src/Model.py V1 &
python src/Model.py V2 &
python src/Model.py V2-no-CB1-L4 &
python src/Model.py  V2-CB1-KO 
# sleep 20s
python src/Model.py plot
# --- final temporal dynamics --- #
python src/input-processing.py V1 &
python src/input-processing.py V2 &
# python src/input-processing.py V2-no-CB1-L4 &
python src/input-processing.py V2-CB1-KO 
# sleep 10s
python src/input-processing.py plot
```
They are listed in the (bash.sh)[./bash.sh] script (comment/uncomment part of the files to launch it step-by-step).

#### N.B. usage

The module uses "relative import" so everything should be launched from the root directory (`CB1_ntwk_modeling/`):
- notebooks: `jupyter notebook notebooks/notebook_of_interest.ipynb`
- scripts: `python src/the_script_of_interest.py --arguments bla bla`


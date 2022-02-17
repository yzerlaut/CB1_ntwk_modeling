python_path='/home/yann.zerlaut/miniconda3/bin/python'
# --- Layer23-circuit connectivity optimization --- #
# $python_path code/L23_connec_params.py V1 &
# $python_path code/L23_connec_params.py V2
# sleep 1s
# $python_path code/L23_connec_params.py plot
# --- Layer4-L23 connectivity optimization --- #
# $python_path code/L4.py test-run
# $python_path code/L4.py test-analysis
# --- final spontaneous activity --- #
# $python_path code/Model.py V1 &
# $python_path code/Model.py V2 &
# $python_path code/Model.py V2-no-CB1-L4 &
# $python_path code/Model.py  V2-CB1-KO 
# sleep 20s
# $python_path code/Model.py plot
# --- final temporal dynamics --- #
$python_path code/input-processing.py V1 &
$python_path code/input-processing.py V2 &
# $python_path code/input-processing.py V2-no-CB1-L4 &
$python_path code/input-processing.py V2-CB1-KO 
sleep 10s
$python_path code/input-processing.py plot

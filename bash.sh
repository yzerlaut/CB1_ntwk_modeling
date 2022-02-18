python_path='/home/yann.zerlaut/miniconda3/bin/python'
# --- Layer23-circuit connectivity optimization --- #
# $python_path code/L23_connec_params.py scan
# $python_path code/L23_connec_params.py scan-analysis
# --- Layer4-L23 connectivity optimization --- #
# $python_path code/L4.py test-run
# $python_path code/L4.py test-analysis
# --- look at gain curves --- #
# $python_path code/gain.py V1 with-repeat
# $python_path code/gain.py V2 with-repeat
# $python_path code/gain.py V2-CB1-KO with-repeat
# $python_path code/gain.py analysis
# --- psyn dep on spontaneous activity --- #
# $python_path code/bg_act.py L23-psyn-scan
# $python_path code/bg_act.py L23-psyn-analysis
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
$python_path code/input-processing.py V2-no-CB1-L4 &
$python_path code/input-processing.py V2-CB1-KO 
sleep 10s
$python_path code/input-processing.py plot

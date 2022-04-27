python_path='/home/yann.zerlaut/miniconda3/bin/python'
# --- Layer23-circuit connectivity optimization --- #
# $python_path src/L23_connec_params.py scan
# $python_path src/L23_connec_params.py scan-analysis
# --- Layer4-L23 connectivity optimization --- #
# $python_path src/L4.py test-run
# $python_path src/L4.py test-analysis
# --- compute gain curves --- #
#$python_path src/gain.py L23-ntwk V1 &
#$python_path src/gain.py L23-ntwk V2 &
$python_path src/gain.py L4L23-ntwk V1 &
$python_path src/gain.py L4L23-ntwk V2 &
$python_path src/gain.py L4L23-ntwk V2-CB1-KO &
# $python_path src/gain.py V2 with-repeat
# $python_path src/gain.py V2-CB1-KO with-repeat
# $python_path src/gain.py analysis
# $python_path src/gain.py V1 with-repeat
# $python_path src/gain.py V2 with-repeat
# $python_path src/gain.py V2-CB1-KO with-repeat
# $python_path src/gain.py analysis
# --- psyn dep on spontaneous activity --- #
# $python_path src/bg_act.py L23-psyn-scan
# $python_path src/bg_act.py L23-psyn-analysis
# ---
# $python_path src/bg_act.py L4-L23-psyn-pconn-scan
# --- final spontaneous activity --- #
# $python_path src/Model.py V1 &
# $python_path src/Model.py V2 &
# $python_path src/Model.py V2-no-CB1-L4 &
# $python_path src/Model.py  V2-CB1-KO 
# sleep 20s
# $python_path src/Model.py plot
# --- final temporal dynamics --- #
# $python_path src/input-processing.py V1 &
# $python_path src/input-processing.py V2 &
# $python_path src/input-processing.py V2-CB1-KO 
# sleep 10s
# $python_path src/input-processing.py plot
# xdg-open dnoc/full_dynamics_raw.png &
# xdg-open doc/full_dynamics_summary.png

#$python_path src/input-processing.py seed-input-scan
#$python_path src/input-processing.py seed-input-scan
#$python_path src/input-processing.py seed-input-scan
#$python_path src/input-processing.py seed-input-scan
#$python_path src/input-processing.py seed-input-scan
#$python_path src/input-processing.py seed-input-scan
#$python_path src/input-processing.py seed-input-scan

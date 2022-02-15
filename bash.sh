python_path='/home/yann.zerlaut/miniconda3/bin/python'
$python_path code/input-processing.py V1 &
$python_path code/input-processing.py V2 &
$python_path code/input-processing.py V2-CB1-KO &
$python_path code/input-processing.py V2-no-CB1-L4 &
sleep 180s
$python_path code/input-processing.py plot

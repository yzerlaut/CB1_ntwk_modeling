source $HOME/.bashrc
python code/input-processing.py V1 &
python code/input-processing.py V2 &
python code/input-processing.py V2-CB1-KO &
python code/input-processing.py V2-no-CB1-L4 &
#sleep 300s; python code/input-processing.py plot

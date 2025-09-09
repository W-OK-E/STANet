## To begin training STANet in a tmux session:
```bash
tmux new-session -t StanLee  #StanLee is just the name of the session your code will run in, can be anything
cd ~/PS_10/sat_change/STANet/
source .venv/bin/activate 


uv run ../utils/monitor.py 'uv run train.py --dataset_mode changedetection   --dataset_type DCDD   --dataroot /home/ubuntu/data/PS_10/DCDD   --split test'
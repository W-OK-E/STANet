## To begin training STANet in a tmux session:
```bash
tmux new-session -t StanLee  #StanLee is just the name of the session your code will run in, can be anything
cd ~/PS_10/sat_change/STANet/
source .venv/bin/activate 


uv run ../utils/monitor.py 'uv run train.py --dataset_mode changedetection   --dataset_type DCDD   --dataroot /home/ubuntu/data/PS_10/DCDD   --split test'
```

With this setup, your training runs even if you are disconnected from the server and when the training finishes you will be notified on slack

## Possible Errors and their solutions:

Failed to send slack message cannot find url -  Possibly your slack webhook has expired, so:
- Google slack incoming webhooks 
- Create a new webhook after choosing the desired channel
- Then put the url you get in the SLACK_WEBHOOK variable in the monitor script.
 
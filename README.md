# Image editing with Diffusion Transformers


## Installation
First, create a python environment with python 3.9 or higher, and install the required packages:
    
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage (with izar cluster)
EPFL SCITAS izar cluster provides nodes with Nvidia V100 GPUs (32GB). Hereafter is a quick tutorial to set up the environment for development.

1. **Start a job:**
    ```bash
    # from izar login node
    sbatch scripts/izar/remote.run

    # or, if you want to use N GPUs
    sbatch --gres=gpu:N scripts/izar/remote.run
    ```
2. **Connect to the node:**
    ```bash
    # from izar login node
    srun --pty --jobid <job_id> /bin/bash
    ```
3. **Start jupyter server:**
    ```bash
    # from izar compute node
    source .venv/bin/activate
    module load tmux
    tmux
    hostname -i
    jupyter-notebook --no-browser --port=8888 --ip=$(hostname -i)
    ```

    You will then need to forward the port from the compute node to your local machine. Open a new terminal on **your local machine** and run the following command:

    ```bash
    # from your local machine
    ssh -L 8888:<compute_node_ip>:8888 <your_username> izar.epfl.ch -f -N
    ```
    In case you can't see the ip of the compute node correctly in the output of the jupyter server, you should see it as an output of the command `hostname -i` run in the compute node.
    
    Troubleshooting: you might encounter the following error when running the command above:
    ```bash
    bind [::1]:8888: Address already in use
    channel_setup_fwd_listener_tcpip: cannot listen to port: 8888
    Could not request local forwarding.
    ```
    In such case you can identify a running process using the port 8888 and kill it:
    ```bash
    lsof -i :8888
    kill <pid>
    ```
    where `<pid>` is the process id of the process using the port 8888.

4. **Open jupyter notebook:** 
        
    Open your browser and go to `http://127.0.0.1:8888/tree?token=<token>`, where `<token>` is the token printed in the output of the jupyter server command.

⚠️ Remember to stop the job when you are done. You can do this by running the following command from the izar login node:
```bash
scancel <job_id>
```


## Quick start

Then, the notebook [`notebooks/example.ipynb`](notebooks/example.ipynb) shows a hello-world example of how to run different diffusion models from the `diffusers` library.
# DiT-Edit: Image and Text Guided Image Composition with Diffusion Transformers 🖼️✨
### TODOs:
- [ ] TODO: move KV-Edit and TF-ICON code into a separate folder with git submodules
- [ ] TODO: Clean notebooks folder

Welcome to **DiT-Edit**! This project explores cutting-edge techniques for **exemplary-based image editing**, also known as **image composition**, using the power of Diffusion Transformers (DiTs).

Ever wanted to seamlessly transfer an object from one image into another, making it look like it truly belongs? That\'s exactly what DiT-Edit aims to achieve! 🎯

**The Core Idea:**

Given a background image (`bg`) and a foreground image (`fg`), DiT-Edit intelligently transfers a selected element from the `fg` image to a specific location in the `bg` image. The magic lies in its ability to:

*   **Preserve Content**: Keep the original background intact.
*   **Adapt Seamlessly**: Adjust the transferred element for differences in size, perspective, and style, making it blend naturally into the new scene.
*   **Training-Free**: Achieve this without requiring model retraining, leveraging the capabilities of pre-trained DiT models like FLUX.
*   **Text Guidance**: Optionally use text prompts to further guide and refine the composition.

![Task Explanation](assets/task_explanation.png)
*Fig 1: Visual explanation of the exemplary-based image editing task.*

Our method builds upon the FLUX model family, one of the state-of-the-art Diffusion Transformers, and uses clever techniques like QKV (Query/Key/Value) injection and image inversion to achieve high-quality results.

![Examples of DiT-Edit](assets/examples.png)
*Fig 2: Examples showcasing the capabilities of DiT-Edit in various scenarios.*

This repository contains the code for our DiT-Edit implementation, tools for benchmarking, and scripts to reproduce our experiments. Dive in to explore how we\'re pushing the boundaries of image editing! 🚀


## Installation
First, create a python environment with python 3.9 or higher, and install the required packages:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Setting Up Pre-Commit Hooks
After cloning the repository and installing dependencies as described above (which includes `pre-commit` via `setup.py`), you need to install the git hooks:

```bash
pre-commit install
```

### Download Scoring Models
If you want to run the quantitative benchmarking, you'll need to download the aesthetic score predictor and place it in the root of the directory.

```bash
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/refs/heads/main/sac+logos+ava1-l14-linearMSE.pth
```

### Third Party Baselines
To compare our results with other SoTA methods, we cloned their code and applied minor modifications for compatibility. The third-party code is located in the `third_party` folder, which contains submodules for each method. To initialize the submodules, run:

TODO: check this works and is correct once the submodules are added
```bash
git submodule update --init --recursive
```

## Codebase Structure

The project is organized as follows:

-   `src/dit_edit/`: Contains the core logic for the DiT-Edit method.
    -   `core/`: Core components like the FLUX pipeline modifications (`flux_pipeline.py`), inversion logic (`inversion.py`), and attention/feature injection processors (`dit_edit_processor.py`).
    -   `data/`: Data loading utilities, including benchmark data handling (`bulk_load.py`, `benchmark_data.py`).
    -   `evaluation/`: Evaluation scripts and metrics (`eval.py`).
    -   `utils/`: Utility functions for inference, logging, etc.
    -   `config.py`: Centralized configuration management using a dataclass (`DitEditConfig`) for hyperparameters shared across scripts.
    -   `run.py`: Script for running DiT-Edit on a single pair of background/foreground images with a bounding box.
    -   `run_on_benchmark.py`: Script for running DiT-Edit over the entire benchmark dataset or a subset of it.
-   `notebooks/`: Jupyter notebooks for experimentation, visualization, and examples.
-   `scripts/`: Utility scripts, potentially for data processing, aggregation, or specific tasks like ablation studies.
-   `benchmark_images_generations/`: Default directory where samples from benchmark are stored, together with the output images and metrics from `run_on_benchmark.py`.
-   `KV-Edit/`, `third_party/`: (Potentially) Directories containing code for other methods for comparison.
-   `assets/`: Static assets, possibly including sample images or plots for the README.
-   `setup.py`: Python package setup script.
-   `README.md`: This file.

## Running the Main Scripts

The two primary scripts for using DiT-Edit are `src/dit_edit/run.py` and `src/dit_edit/run_on_benchmark.py`. Both scripts share a common set of hyperparameters defined in `src/dit_edit/config.py`, which can be overridden via command-line arguments.

After installing the package with `pip install -e .`, you can also run these scripts as commands:
- `dit-run` (equivalent to `python src/dit_edit/run.py`)
- `dit-run-benchmark` (equivalent to `python src/dit_edit/run_on_benchmark.py`)

### 1. Single Image Composition

This script allows you to compose a foreground image onto a background image given a bounding box mask.

**Basic Usage:**
Using the command:
```bash
dit-run \
    --bg_path /path/to/your/background_image.jpg \
    --fg_path /path/to/your/foreground_image.png \
    --bbox_path /path/to/your/bounding_box_mask.png \
    --output_path /path/to/save/output_composed_image.png \
    --prompt "An optional text prompt describing the desired composition"
```

**Key Arguments:**

*   `--bg_path`: Path to the background image.
*   `--fg_path`: Path to the foreground image (preferably with an alpha channel or a clear subject).
*   `--bbox_path`: Path to a binary mask image (black with a white rectangle) indicating the target placement area for the foreground object in the background image's coordinate space.
*   `--output_path`: Where to save the resulting composed image.
*   `--prompt`: (Optional) A text prompt to guide the composition.
*   `--debug`: Enable debug mode to save intermediate images.

**Hyperparameters:**

You can override the default hyperparameters defined in `DitEditConfig`. For a full list, run:
Example of overriding hyperparameters:
```bash
dit-run \
    --bg_path ... \
    --fg_path ... \
    --bbox_path ... \
    --output_path ... \
    --tau-alpha 0.3 \
    --timesteps 40 \
    --guidance-scale 2.5 \
    --no-inject-v # Example of a boolean flag
```

### 2. Benchmark Evaluation
[This script](src/dit_edit/run_on_benchmark.py) runs the DiT-Edit method over a predefined benchmark dataset (expected to be in `benchmark_images_generations/`). It generates images and calculates evaluation metrics.

**Basic Usage:**

Using the script directly:
```bash
dit-run-benchmark # or
python src/dit_edit/run_on_benchmark.py
```

**Key Arguments:**

*   `--run-on-first N`: Process only the first N images from each category in the benchmark. Set to -1 to run on all (default).
*   `--random-samples`: If used with `--run-on-first N` (where N > 0), randomly sample N images from each category.
*   `--random-samples-seed SEED`: Seed for the random sampling.
*   `--skip-available`: If set, skips processing for image/parameter combinations where output files already exist.
*   `--no-save-output-images`: Disable saving of the generated images (metrics will still be saved). By default, images are saved.

**Hyperparameters:**

Similar to `run.py`, you can override the default hyperparameters. For a full list, run:
```
Example of overriding hyperparameters:
```bash
dit-run-benchmark \
    --tau-beta 0.7 \
    --alpha-noise 0.03 \
    --layers-for-injection vital \
    --seed 123
```

This will run the benchmark evaluation using the specified hyperparameter values. Output images and JSON files with metrics will be saved in subdirectories within `benchmark_images_generations/`.


## Running on izar cluster
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

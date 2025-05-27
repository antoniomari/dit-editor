## SoTA Baselines

To achieve a fair comparison with state-of-the-art methods, we had to slightly adapt their original codebases. This directory contains two submodules that provide the exact code we executed when computing results from other state-of-the-art methods. 

The SoTA baselines we considered are:
- **TF-ICON:**
    It is a method for image-guided image composition which used old U-Net based diffusion models. As this method already tackles our task, very few changes were needed to run it on the benchmark
    - Official Paper: [here](https://arxiv.org/abs/2307.12493)
    - Original repository: [`Shilin-LU/TF-ICON`](https://github.com/Shilin-LU/TF-ICON)
    - Our fork: [`matteosantelmo/TF-ICON`](https://github.com/matteosantelmo/TF-ICON)
- **KV-Edit:**
    This method is instead for text-guided image editing on recent DiT models. Although this method needed more adaptation to work with our task, therefore we provide the code here. In particular, in order to define precise prompts for the editing, we've used Google Gemini API.
    - Official Paper: [here](https://arxiv.org/abs/2502.17363)
    - Original repository: [`Xilluill/KV-Edit`](https://github.com/Xilluill/KV-Edit)
    - Our fork:[`matteosantelmo/KV-Edit`](https://github.com/matteosantelmo/KV-Edit)
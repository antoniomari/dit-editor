from setuptools import setup, find_packages

setup(
    name="dit_edit",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "Pillow",
        "numpy",
        "pandas",
        "tqdm",
        "diffusers",
        "transformers",
        "einops",
        "accelerate",
        "scikit-image",
        "lpips",
        "rembg",
    ],
    python_requires=">=3.10",
)
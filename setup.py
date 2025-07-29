from setuptools import setup, find_packages
import os

# Get the directory where this setup.py is located
setup_dir = os.path.dirname(os.path.abspath(__file__))
version_file = os.path.join(setup_dir, "verl", "version", "version")

try:
    with open(version_file, "r") as f:
        version = f.read().strip()
except FileNotFoundError:
    print(f"Version file not found at: {version_file}")
    version = "0.4.0.dev"

readme_path = os.path.join(setup_dir, "README.md")
try:
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "VERL - Reinforcement Learning for Large Language Models"

setup(
    name="verl",
    version=version,
    author="Bytedance Ltd.",
    description="Reinforcement Learning for Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers>=4.44.0",
        "tokenizers",
        "accelerate",
        "datasets",
        "numpy",
        "packaging",
        "pyyaml",
        "ray",
        "wandb",
        "omegaconf",
        "hydra-core",
        "sentencepiece",
        "protobuf",
        "pandas",
        "tqdm",
        "psutil",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
        ],
        "modelscope": [
            "modelscope",
        ],
        "training": [
            "deepspeed",
            "megatron-core",
            "flash-attn",
        ],
        "inference": [
            "vllm",
            "sglang",
        ],
        "full": [
            "deepspeed",
            "megatron-core",
            "flash-attn",
            "vllm",
            "sglang",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    include_package_data=True,
    package_data={
        "verl": ["version/version"],
    },
    entry_points={
        "console_scripts": [
            "verl-train=verl.trainer.main_ppo:main",
            "verl-eval=verl.trainer.main_eval:main",
            "verl-generate=verl.trainer.main_generation:main",
        ],
    },
)
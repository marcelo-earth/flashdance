from setuptools import setup, find_packages

setup(
    name="flashdance",
    version="0.2.0",
    description="Attention benchmarking: Flash Attention, GQA, MLA, RoPE, ALiBi, and more",
    author="Marcelo",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "matplotlib",
        "tabulate",
    ],
    extras_require={
        "dev": ["pytest", "pandas"],
    },
)

"""setup.py — installable package configuration."""
from setuptools import setup, find_packages

setup(
    name="object-detection-ai",
    version="2.0.0",
    description="Advanced Object Detection using YOLO11/YOLOv8 with tracking, segmentation & analytics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Aranya2801",
    url="https://github.com/Aranya2801/Object-detection-using-AI",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    install_requires=[
        "ultralytics>=8.3.0",
        "opencv-python>=4.10.0",
        "numpy>=1.26.0",
        "PyYAML>=6.0.2",
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": ["pytest>=8.3.0", "pytest-cov>=5.0.0"],
        "viz": ["matplotlib>=3.9.0", "seaborn>=0.13.0", "plotly>=5.23.0"],
        "export": ["onnx>=1.16.0", "onnxruntime>=1.18.0"],
        "notebook": ["jupyter>=1.1.0", "ipywidgets>=8.1.0"],
    },
    entry_points={
        "console_scripts": [
            "detect=detect:main",
            "od-train=train:main",
            "od-bench=benchmark:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)

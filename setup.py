from setuptools import setup, find_packages

setup(
    name="slam_toolbox",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "questionary>=2.0.0",
        "numpy>=1.20.0",
        "open3d>=0.15.0",
        "pyyaml>=6.0",
        "rich>=12.0.0",
    ],
    entry_points={
        "console_scripts": [
            "slam_toolbox=slam_toolbox.cli:main",
        ],
    },
    author="FineNav",
    description="A Python CLI tool for interactive SLAM map processing.",
)

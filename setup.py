from setuptools import find_packages, setup

# import torch
with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="aide-ds",
    version="0.1.4",
    author="Asim Osmany",
    author_email="asim@aims.ac.za",
    description="Open source Autonomous AI for Data Science and Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Asimawad/aide-agent",
    packages=find_packages(),
    package_data={
        "aide": [
            "../utils/competition_template.json",
            "../requirements.txt",
            "utils/config.yaml",
            "utils/viz_templates/*",
            "example_tasks/bitcoin_price/*",
            "example_tasks/house_prices/*",
            "example_tasks/*",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "aide = aide.run:run",
        ],
    },
)

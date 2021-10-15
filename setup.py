import os
from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pddlgym_textworld", # Replace with your own username
    version=open(os.path.join("pddlgym_textworld", "version.py")).read().split("=")[-1].strip("' \n"),
    author="Michiaki Tatsubori",
    author_email="mich@jp.ibm.com",
    description="PDDLGym interface for TextWorld",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.ibm.com/IBM-Research-AI/pddlgym_textworld",
    #packages=setuptools.find_packages(),
    packages=['pddlgym_textworld'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=open('requirements.txt').readlines()
)

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="adaptive-shots",
    version="0.1.0",
    author="Gökhan Mete Ertürk",
    author_email="8rlvjfxsh@mozmail.com",
    description="A Python package for a simple adaptive few-shots prompting engine using a contextual combinatorial bandits algorithm.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gokhanmeteerturk/adaptive-shots",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)

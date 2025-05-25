#!/usr/bin/env python3
"""
Setup script for Llama 3 Travel Assistant Fine-tuning
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llama3-travel-assistant",
    version="1.0.0",
    author="Travel AI Team",
    description="Fine-tune Llama 3 8B to create a specialized travel assistant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llama3-travel-assistant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    keywords="llama, llama3, travel, assistant, fine-tuning, ai, nlp, chatbot",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llama3-travel-assistant/issues",
        "Source": "https://github.com/yourusername/llama3-travel-assistant",
    },
) 
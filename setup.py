from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="random_allocation",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Random Allocation for Differential Privacy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/random_allocation",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "numba>=0.60.0",
        "dp_accounting>=0.4.4",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
) 
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sfiralium-core",
    version="0.1.0",
    author="O.S. Basargin, S. Chernenko",
    author_email="sfiraliumlab@gmail.com",
    description="Topological Computing Framework based on Sfiral Principle",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sfiraliumlab-ux/Sfiralium-Core",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
)

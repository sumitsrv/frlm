"""FRLM package setup."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"


def read_requirements() -> list:
    """Parse requirements.txt, ignoring comments and blank lines."""
    lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    reqs = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
    return reqs


setup(
    name="frlm",
    version="0.1.0",
    description=(
        "Factual Retrieval Language Model — separates factual knowledge "
        "retrieval from linguistic generation in biomedical LLMs"
    ),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="FRLM Team",
    python_requires=">=3.10",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "pytest-asyncio>=0.23",
            "ruff>=0.1",
            "mypy>=1.8",
            "black>=23.12",
            "isort>=5.13",
        ],
        "notebook": [
            "jupyter>=1.0",
            "ipykernel>=6.28",
            "matplotlib>=3.8",
            "seaborn>=0.13",
        ],
    },
    entry_points={
        "console_scripts": [
            "frlm-train=scripts.09_train_joint:main",
            "frlm-serve=src.inference.server:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
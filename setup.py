from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent


def read_requirements(filename: str) -> list:
    """
    Read pinned requirements, skipping comments and pip directives.

    Lines beginning with "-" (-r, --index-url, ...) are pip options rather than
    requirement specifiers, and setuptools rejects them.
    """
    lines = (ROOT / filename).read_text(encoding="utf-8").splitlines()
    return [
        line.split("#")[0].strip()
        for line in lines
        if line.strip() and not line.strip().startswith(("#", "-"))
    ]


setup(
    name="advanced-fraud-detection",
    version="2.9.0",
    author="Aaron Emmanuel Xavier Sequeira",
    description=(
        "Credit card fraud detection with device fingerprinting, payment "
        "source validation and behavioural analysis"
    ),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/aaron-seq/advanced-fraud-detection-system",
    project_urls={
        "Bug Tracker": "https://github.com/aaron-seq/advanced-fraud-detection-system/issues",
        "Source Code": "https://github.com/aaron-seq/advanced-fraud-detection-system",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    packages=find_packages(include=["app*", "src*"]),
    python_requires=">=3.11",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        # Gradient boosting, for producing trained artefacts to serve from
        # MODEL_PATH. Not needed to run the API.
        "ml": [
            "scikit-learn>=1.5",
            "xgboost>=2.0",
            "lightgbm>=4.3",
            "imbalanced-learn>=0.12",
            "optuna>=3.6",
        ],
        # Model explainability.
        "xai": ["shap>=0.46"],
        # Kafka streaming pipeline.
        "streaming": ["kafka-python>=2.0.2"],
        # Streamlit monitoring dashboard.
        # streamlit >= 1.37 caps numpy<3 rather than <2, so it co-exists
        # with the numpy 2.x needed for Python 3.13.
        "dashboard": ["streamlit>=1.37", "plotly>=5.20"],
        "dev": read_requirements("requirements-dev.txt"),
    },
    # The previous console_scripts pointed at five modules, three of which
    # (quantum_fraud_detector, explainable_ai_fraud, ensemble_optimizer) do not
    # exist, so installing the package produced commands that failed on import.
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "fraud detection",
        "device fingerprinting",
        "machine learning",
        "financial security",
        "fintech",
    ],
)

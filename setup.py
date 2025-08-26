from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="advanced-fraud-detection",
    version="2.0.0",
    author="Aaron Emmanuel Xavier Sequeira",
    author_email="aaron.sequeira@fraud-detection.ai",
    description="Advanced Credit Card Fraud Detection System with Quantum ML, GNNs, and Federated Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aaronseq12/CreditCardFraudDetection",
    project_urls={
        "Bug Tracker": "https://github.com/aaronseq12/CreditCardFraudDetection/issues",
        "Documentation": "https://github.com/aaronseq12/CreditCardFraudDetection/docs",
        "Source Code": "https://github.com/aaronseq12/CreditCardFraudDetection",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Financial",
        "Topic :: Security",
    ],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "quantum": [
            "qiskit>=0.45.0",
            "qiskit-machine-learning>=0.7.0",
            "pennylane>=0.33.0",
            "pennylane-lightning>=0.33.0",
        ],
        "gnn": [
            "torch-geometric>=2.4.0",
            "dgl>=1.1.0",
            "networkx>=3.2.0",
        ],
        "federated": [
            "flwr>=1.6.0",
            "cryptography>=41.0.0",
        ],
        "streaming": [
            "kafka-python>=2.0.2",
            "streamlit>=1.29.0",
            "redis>=5.0.0",
        ],
        "monitoring": [
            "mlflow>=2.9.0",
            "evidently>=0.4.18",
            "prometheus-client>=0.19.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
            "pre-commit>=3.5.0",
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "all": [
            # Quantum
            "qiskit>=0.45.0",
            "qiskit-machine-learning>=0.7.0",
            "pennylane>=0.33.0",
            "pennylane-lightning>=0.33.0",
            # Graph Neural Networks
            "torch-geometric>=2.4.0",
            "dgl>=1.1.0",
            "networkx>=3.2.0",
            # Federated Learning
            "flwr>=1.6.0",
            "cryptography>=41.0.0",
            # Streaming
            "kafka-python>=2.0.2",
            "streamlit>=1.29.0",
            "redis>=5.0.0",
            # Monitoring
            "mlflow>=2.9.0",
            "evidently>=0.4.18",
            "prometheus-client>=0.19.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fraud-detector=src.advanced_fraud_detector:main",
            "fraud-stream=src.streaming_fraud_detector:main",
            "fraud-federated=src.federated_learning_fraud:main",
            "fraud-gnn=src.graph_neural_network_fraud:main",
            "fraud-quantum=src.quantum_fraud_detector:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.txt"],
    },
    zip_safe=False,
    keywords=[
        "fraud detection",
        "machine learning",
        "quantum computing",
        "graph neural networks",
        "federated learning",
        "credit card fraud",
        "financial security",
        "artificial intelligence",
        "deep learning",
        "ensemble methods",
        "real-time analytics",
        "streaming",
        "explainable ai",
        "fintech",
    ],
)

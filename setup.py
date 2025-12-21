from setuptools import setup, find_packages

setup(
    name="enterprise-mlops-platform",
    version="1.0.0",
    description="Enterprise MLOps Platform with Feature Stores",
    author="techdevworks",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "mlflow>=2.9.0",
        "databricks-sdk>=0.12.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.1.0",
        "pyspark>=3.4.0",
        "fastapi>=0.104.0",
        "apache-airflow>=2.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "flake8>=6.1.0",
            "black>=23.12.0",
            "mypy>=1.7.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

from setuptools import setup, find_packages

setup(
    name="ai-co-scientist",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langchain-ollama>=0.0.1",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "networkx>=3.0",
        "loguru>=0.7.0",
        "typing-extensions>=4.5.0"
    ],
) 
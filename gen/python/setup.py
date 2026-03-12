"""Setup file for runner gRPC package."""

from setuptools import setup, find_packages

setup(
    name="runner-grpc",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "grpcio>=1.78.0",
        "grpcio-tools>=1.78.0",
        "protobuf>=4.25.0",
    ],
)
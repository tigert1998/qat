import setuptools

VERSION = "0.0.1"

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="QAT",
    version=VERSION,
    author="Xiaohu Tang",
    author_email="tigertang.zju@outlook.com",
    description="A manually implemented PyTorch quantization aware training example.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tigert1998/qat",
    packages=["qat", "qat.networks", "qat.data", "qat.export"],
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=[],
    python_requires='>=3.5',
)

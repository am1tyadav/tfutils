import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="tfutils",
    version="0.0.1",
    author="Amit Yadav",
    author_email="amit.yadav.iitr@gmail.com",
    description="TensorFlow and Keras Utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/am1tyadav/tfutils.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
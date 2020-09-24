import setuptools

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

setuptools.setup(
    name="probs",
    version="0.0.2",
    author="Tyler Yep",
    author_email="tyep@cs.stanford.edu",
    description="Probability library for Python",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/tyleryep/probs",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("Readme.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open("requirements.txt", encoding="utf-8") as req_file:
    requirements = req_file.read().splitlines()

test_requirements = [
    "pytest>=3",
]

setup(
    author="Johannes Seiffarth",
    author_email="j.seiffarth@fz-juelich.de",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
    ],
    description="The dmc-masking library provides real-time microfluidic chamber masking capabilities",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="dmc-masking",
    name="dmc-masking",
    packages=find_packages(include=["dmc_masking", "dmc_masking.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/JojoDevel/dmc-masking",
    version="0.0.1",
    zip_safe=False,
)

#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Low Ching Nam Raymond",
    author_email='iamraymondlow@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="This repo contains all of the source code for the stop activity prediction study conducted in Singapore.",
    entry_points={
        'console_scripts': [
            'stop_activity_prediction=stop_activity_prediction.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='stop_activity_prediction',
    name='stop_activity_prediction',
    packages=find_packages(include=['stop_activity_prediction', 'stop_activity_prediction.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/iamraymondlow/stop_activity_prediction',
    version='0.1.0',
    zip_safe=False,
)

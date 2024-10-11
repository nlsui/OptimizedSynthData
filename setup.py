from setuptools import setup, find_packages

setup(
    name='OptimizedSynthData',  # Project name
    version='0.1.0',  # Initial version
    description='A package for synthesizing and analyzing data using machine learning models.',
    author='nlsui',  # Replace with your name
    author_email='mariusfaust27@example.com',  # Replace with your email
    url='https://github.com/nlsui/OptimizedSynthData',  # Replace with your project’s repo URL if applicable
    packages=find_packages(),  # Automatically find all modules and packages
    py_modules=['main'],  # Exports main.py as a module
    install_requires=[
        'numpy',
        'torch',
        'transformers',
        'langchain',
        'scipy',
    ],  # List all dependencies here
    entry_points={
        'console_scripts': [
            'optimizedsynthdata = main:main',  # Adds command line script to execute main.py's main function
        ],
    },
)

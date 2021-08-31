from setuptools import setup
from setuptools import setup, find_packages
from setuptools.extension import Extension

setup(
   name='email_classifier_tflite',
   version='0.0.1',
   description='A module for email classifier for the tflite module',
   author='vmujadia',
   author_email='Aqmar Hussain',
   packages=find_packages()
)

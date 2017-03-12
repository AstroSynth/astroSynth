from setuptools import setup
from os import path

HERE = path.abspath(path.dirname(__file__))

setup(name='astroSynth',
      version='0.3.2',
      description='Very Basic Astrophysics Synthetic Generation Suite',
      url='https://github.com/tboudreaux/astroSynth.git',
      author='Thomas Boudreaux',
      author_email='thomas@boudreauxmail.com',
      license='MIT',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3'
      ],
      install_requires=[
          'numpy>=1.12.0',
          'pandas>=0.19.2',
          'tqdm>=4.11.2',
          'scipy>=0.19.0'
      ],
      packages=['astroSynth'],
      zip_safe=False)

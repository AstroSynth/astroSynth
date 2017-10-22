from setuptools import setup
from os import path

HERE = path.abspath(path.dirname(__file__))

setup(name='astroSynth',
      version='0.6.1.5',
      description='Very Basic Astrophysics Synthetic Generation Suite',
      url='https://github.com/tboudreaux/astroSynth.git',
      author='Thomas Boudreaux',
      author_email='thomas@boudreauxmail.com',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3'
      ],
      install_requires=[
          'numpy>=1.12.0',
          'pandas>=0.19.2',
          'tqdm>=4.11.2',
          'scipy>=0.19.0',
          'astropy>=1.3.2',
          'names>=0.3.0',
      ],
      packages=['astroSynth', 'astroSynth.Objects'],
      zip_safe=False)

# astroSynth
Version: 0.5.1 - Beta <br>
A very basic synthetic generation suite

# Installation
Two ways to install

## Pip
```bash
pip install astroSynth
```
## Github
or to enusure you have the most up-to-date version clone the repository
```bash
git clone https://github.com/tboudreaux/astroSynth
cd astroSynth
python setup.py install
```

# Usage
## Main Ideas
The fundamental structure in astroSynth is the observation, an observation is a PVS() type objects with can iteract with light curve data. An observation can make new data, load previously made data, save data, and serve as an access suite for the data.
## Creating data with an observation
```python
import astroSynth
obs_1 = astroSynth.PVS(Number=10, noise_range=[0.1, 1], numpoints=200, name='TestOne')
obs_1.build(amp_range=[1, 10], freq_range=[1, 10], phase_range=[1, 2], L_range=[1, 3])
obs_1.generate(pfrac=0.5)
print(obs_1)
obs_1.save
```
## Loading Data with an observation
Assuming you have aready have some sythetic data directory 'TestOne/'
```python
import astroSynth
obs_2 = astroSynth.PVS()
obs_2.load(directory='TestOne')
print(obs_2)
```
## Accessing Data stored in an observation
Once data is stored (either loaded or generated) it can all be accessed in the same manner
```python
import astroSynth
import matplotlib.pyplot as plt
obs_3 = astroSynth.PVS()
obs_3.load(directory='TestOne')
for Time, Flux, Classification in obs_3:
  plt.plot(Time, Flux, 'o--')
  if Classification == 0:
    plt.title('Non Variable')
  elif Classification == '1':
    plt.title('Variable')
  plt.show()
```
Note that when retriving elements from an observation a three element tuple will be returned, the first element being the Time array of the observation, the second element being the Flux array of the observation, and the third element being the classification of the light curve (variable - 1, non variable - 0)

## Getting Batches of data
astroSynth can retrive batches of Light curves or Fourier transforms at a time, this can be helpful if training an algorithm with a data set larger than can fit in memory.
### Get Batches of Light Curves
```python
import astroSynth
obs_4 = astroSynth.PVS()
obs_4.load(directory='TestOne')
for data in obs_4.batch_get(batch_size=5):
    bar = foo(data)
```
### Get Batches of Fourier Transform
```python
import astroSynth
obs_5 = astroSynth.PVS()
obs_5.load(directory='TestOne')
for data in obs_5.batch_get(batch_size=5, ft=True):
    bar = foo(data)
```
Both of these will return 5 elemements at a time (that number can be made n length such that n <= len(obs)) or <'mem_size'> where the batch will expand to fill the avalible memory

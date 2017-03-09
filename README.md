# astroSynth
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
obs_1 = astroSynth.PVS(Number=150, noise_range=[0.1, 1], numpoints=200, name='TestOne')
obs_1.build(amp_range=[1, 10], freq_range=[1, 10], phase_range=[1, 2], L_range=[1, 3])
obs_1.generate(pfrac=0.5)
print(obs_1)
obs_1.save
```
## Loading Data with an observation
Assuming you have aready have some sythetic data directory 'TestOne\'
```python
import astroSynth
obs_2 = astroSynth.PVS()
obs_2.load(directory='TestOne')
print(obs_2)
```

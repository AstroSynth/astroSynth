# astroSynth
Version: 0.5.3.5 - Beta <br>
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
for Time, Flux, Classification, number, pparams in obs_3:
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

### Using POS to generate surveys
astroSynth has the ability to generate what I call surveys, these are datasets where there are discrete targets, each with multiple light curves (visits or observations) spread over time. The module used to generate surveys is POS. This is build onto of PVS - essentiall each target is a PVS object where one light curve was generated, then parts of that light curve are thrown away, leaving multiple observations behind. 

Use POS to generate data in the following manner
```python
import astroSynth
survey = astroSynth.POS(number = 100, name='survey_1')
survey.build(amp_range=[0.01, 0.03])
survey.generate(pfrac=0.5)

survey.save()
```

Data can then be loaded 

```python
from astroSynth import POS
survey = POS()
survey.load(directory='survey_1')
```

To access light curves stored inside survey there are multiple approacheds. One can access each individual PVS object and then use the methods from PVS, for example

```python
from astroSynth import POS
survey = POS()
survey.load(directory='survey_1')

time = survey[0][0][0]
flux = survey[0][0][1]
classificaion = survey[0][0][2]
Target_ID = survey[0][0][3]
pulsation_parameters = survey[0][0][4]
```

there is also an iterator to get objects, that behaves in a very similar manner

```python
from astroSynth import POS
survey = POS()
survey.load(directory='survey_1')

for target in survey.xget_object():
    num_visits = len(target)
    time = target[0][0]
    flux = target[0][1]
    classificaion = target[0][2]
    Target_ID = target[0][3]
    pulsation_parameters = target[0][4]
```
Note both that target in the above example is a PVS object, and that I am only accessing the first visit (thus the 0 subscript), to access other visits
```python
from astroSynth import POS
survey = POS()
survey.load(directory='survey_1')

for target in survey.xget_object():
    num_visits = len(target)
    time = target[visit_num][0]
    flux = target[visit_num][1]
    classificaion = target[visit_num][2]
    Target_ID = target[visit_num][3]
    pulsation_parameters = target[visit_num][4]
```
Place the visit number you are interested in (indexed from 0) in visit_num.
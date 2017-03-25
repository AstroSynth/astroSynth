import astroSynth
from sys import getsizeof

print('INSTANTIATDE')
survey = astroSynth.Objects.POS(number=10000, numpoints=200) 

print('BUILDING')
survey.build(obs_range=[10, 50])

print('GENERATING')
survey.generate()
print('DONE')

print('Size of Survey: {}'.format(getsizeof(survey)))
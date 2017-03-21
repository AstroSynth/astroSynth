import os
import names
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from sys import getsizeof
from astroSynth import PVS
from astroSynth import SDM
from tempfile import TemporaryFile


class POS():
	def __init__(self, prefix='SynthStar', mag_range=[10, 20], noise_range=[0.1, 1.1],
		         number=1000, numpoints=300, verbose=0, name=None):
		if name is None:
			name = prefix
		self.name = name
		self.prefix = prefix
		self.mag_range = mag_range
		self.size = number
		self.depth = numpoints
		self.verbose = 0
		self.noise_range = noise_range
		self.targets = dict()
		self.int_name_ref = dict()
		self.classes = dict()

	@staticmethod
	def __load_spec_class__(path):
		file_data = open(path, 'rb')
		file_data = file_data.decode('utf-8')
		file_data = file_data.readlines()
		file_data = [x.rstrip().split(':') for x in file_data]
		amp_range = None
		phase_range = None
		freq_range = None
		L_range = None
		for i, e in enumerate(file_data):
			if i[0] == 'amp_range':
				amp_range = i[1]
			elif i[0] == 'freq_range':
				freq_range = i[1]
			elif i[0] == 'phase_range':
				phase_range = i[1]
			elif i[0] == 'L_range':
				L_range = i[1]
			else:
				print('Warning! Unkown line encountered -> {}:{}'.format(i[0], i[1]))
		return amp_range, phase_range, freq_range, L_range

	@staticmethod
	def __seed_generation__(seed=1):
		np.random.seed(seed)

	def __build_survey__(self, amp_range=[0, 0.2],freq_range = [1, 100],
	  		  			 phase_range=[0, np.pi], L_range=[1, 3],
    		  			 amp_varience=0.01, freq_varience=0.01, 
    		  			 phase_varience=0.01,  obs_range=[10, 100]):
		for i in tqdm(range(self.size)):
			pulsation_modes = np.random.randint(L_range[0],
				                                L_range[1] + 1)

			pulsation_amp = np.random.uniform(amp_range[0],
				                              amp_range[1],
				                              pulsation_modes)
			pap = amp_varience * pulsation_amp

			pulsation_frequency = np.random.uniform(freq_range[0],
				                                    freq_range[1],
				                                    pulsation_modes)
			pfp = freq_varience * pulsation_frequency

			pulsation_phase = np.random.uniform(phase_range[0],
		    	                                phase_range[1],
		    	                                pulsation_modes)
			ppp = phase_varience * pulsation_phase

			observations = np.random.randint(obs_range[0],
											 obs_range[1])

			target_name = names.get_full_name().replace(' ', '-')
			target_id = "{}_{}".format(self.prefix, target_name)
			self.targets[target_id] = PVS(Number=observations, numpoints=self.depth, 
				                          verbose=self.verbose, noise_range=self.noise_range, 
				                          mag_range=self.mag_range, name=target_id, lpbar=False)
			self.targets[target_id].build(amp_range=[pulsation_amp - pap, pulsation_amp + pap],
										  freq_range=[pulsation_frequency - pfp, pulsation_frequency + pfp],
										  phase_range=[pulsation_phase - ppp, pulsation_phase + ppp],
										  L_range=[pulsation_modes, pulsation_modes])


	def build(self, load_from_file = False, path=None, amp_range=[0, 0.2], 
		      freq_range = [1, 100], phase_range=[0, np.pi], L_range=[1, 3], 
		      amp_varience=0.01, freq_varience=0.01, phase_varience=0.01, 
		      seed=1, obs_range=[10, 100]):
		if load_from_file is True:
			try:
				assert path is not None
			except AssertionError as e:
				e.arge += ('Error! No Path to file given', 'Did you specify path?')

				raise

		self.__seed_generation__(seed=seed)

		if load_from_file is True:
			ar, pr, fr, lr = self.__load_spec_class__(path)
			if ar is not None:
				amp_range = ar
			if pr is not None:
				phase_range = pr
			if fr is not None:
				freq_range = fr
			if lr is not None:
				L_range = lr

		self.__build_survey__(amp_range=amp_range, L_range=L_range, freq_range=freq_range,
							  phase_range=phase_range, amp_varience=amp_varience,
							  phase_varience=phase_varience, freq_varience=freq_varience,
							  obs_range=obs_range)

	def generate(self, pfrac=0.5):
		for i in tqdm(self.targets, desc='Geneating Survey Data', total=self.size):
			rand_pick = np.random.uniform(0, 10)
			if rand_pick < pfrac * 10:
				self.classes[i] = 1
			else:
				self.classes[i] = 0

			self.targets[i].generate(pfrac=self.classes[i])
		print('Size of targets is: {}'.format(getsizeof(self.targets)))
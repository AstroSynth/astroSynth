import os
import time
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
		self.name_int_ref = dict()
		self.classes = dict()
		self.target_ref = dict()
		self.dumps = dict()
		self.state = -1

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
				                              amp_range[1])#,
				                              #pulsation_modes)
			pap = amp_varience * pulsation_amp

			pulsation_frequency = np.random.uniform(freq_range[0],
				                                    freq_range[1])#,
				                                    #pulsation_modes)
			pfp = freq_varience * pulsation_frequency

			pulsation_phase = np.random.uniform(phase_range[0],
		    	                                phase_range[1])#,
		    	                                #pulsation_modes)
			ppp = phase_varience * pulsation_phase

			observations = np.random.randint(obs_range[0],
											 obs_range[1])

			target_name = names.get_full_name().replace(' ', '-')
			target_id = "{}_{}".format(self.prefix, target_name)
			self.targets[target_id] = PVS(Number=observations, numpoints=self.depth, 
				                          verbose=self.verbose, noise_range=self.noise_range, 
				                          mag_range=self.mag_range, name=target_id, 
				                          lpbar=False, ftemp=True)

			self.targets[target_id].build(amp_range=[pulsation_amp - pap, pulsation_amp + pap],
										  freq_range=[pulsation_frequency - pfp, pulsation_frequency + pfp],
										  phase_range=[pulsation_phase - ppp, pulsation_phase + ppp],
										  L_range=[pulsation_modes, pulsation_modes])
			self.int_name_ref[i] = target_id
			self.name_int_ref[target_id] = i


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

	def generate(self, pfrac=0.5, target_in_mem=100):
		dumpnum = 0
		lastdump = 0
		for j, i in tqdm(enumerate(self.targets), desc='Geneating Survey Data', total=self.size):
			rand_pick = np.random.uniform(0, 10)
			if rand_pick < pfrac * 10:
				self.classes[i] = 1
			else:
				self.classes[i] = 0
			self.targets[i].generate(pfrac=self.classes[i])
			if j-lastdump >= target_in_mem:
				path_a = "{}/.{}_temp".format(os.getcwd(), self.prefix)
				if os.path.exists(path_a):
					shutil.rmtree(path_a)
				os.mkdir(path_a)
				path = "{}/.{}_temp/{}_dump".format(os.getcwd(), self.prefix, dumpnum)
				os.mkdir(path)
				for k, x in enumerate(self.targets):
					if k < j-lastdump: 
						star_path = "{}/{}".format(path, x)
						os.mkdir(star_path)
						self.targets[x].save(path=star_path)
						self.targets[x] = None
				self.target_ref[dumpnum] = [lastdump, target_in_mem + lastdump]
				self.dumps[dumpnum] = path
				dumpnum += 1
				lastdump = j

	def __load_dump__(self, n=0, state_change=True):
		if state_change is True:
			self.targets = dict()
		else:
			ttargets = dict
		try:
			assert n >= -1 and n < len(self.dumps)
		except AssertionError as e:
			e.args += ('ERROR! Dump index {} out of range for dump of size {}'.format(n, len(self.dumps)))
			raise
		dump_path = self.dumps[n]
		load_targets = os.listdir(dump_path)
		for target in load_targets:
			load_path = "{}/{}".format(dump_path, target)
			if state_change is True:
				self.targets[target] = PVS()
				self.targets[target].load(load_path)
			else:
				ttargets[target] = PVS()
				ttargets[target].load(load_path)
		if state_change is False:
			return ttargets

	def __get_target_lc__(self, target=0, n=0, state_change=False):
		if isinstance(target, int):
			try:
				assert target < len(self.targets)
			except AssertionError as e:
				e.args += ('ERROR!, Target Index Out Of Range')
				raise
			try:
				assert n < len(self.targets)
			except AssertionError as e:
				e.args += ('ERROR! Target Light Curve index out of range')
				raise

			target_id = self.int_name_ref[n]
			target_num = target
		elif isinstance(target, str):
			try:
				assert target in self.targets
			except AssertionError as e:
				e.args('Error! Target Index not found in targets reference')
				raise
			try:
				assert n < len(self.targets)
			except AssertionError as e:
				e.args += ('ERROR! Target Light Curve index out of range')
				raise

			target_id = target
			target_num = self.name_int_ref[target_id]

		dump_num = -1
		for k in self.item_ref:
			if int(self.target_ref[k][0]) <= target_num <= int(self.target_ref[k][1]):
				dump_num = int(k)
				break

		if dump_num != self.state:
			self.__load_dump__(n=dump_num)



	def __del__(self):
		path = "{}/.{}_temp".format(os.getcwd(), self.prefix)
		# shutil.rmtree(path)
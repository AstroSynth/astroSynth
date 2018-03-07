import os
import names
import shutil
# import numba
from ..SDM import *
import numpy as np
from tqdm import tqdm
from sys import getsizeof
from astroSynth import PVS
from astroSynth import SDM
from astropy import units as u
from tempfile import TemporaryFile
from scipy.signal import spectrogram
from multiprocessing import Pool
from scipy import misc
from contextlib import closing
from warnings import warn

class POS():
	def __init__(self, prefix='SynthStar', mag_range=[10, 20], noise_range=[0.05, 0.1],
		         number=100, numpoints=100000, verbose=0, name=None, DEBUG=False,pbar=True):
		if name is None:
			name = prefix
		self.name = name
		self.prefix = prefix
		self.mag_range = mag_range
		self.size = number
		self.depth = numpoints
		self.verbose = verbose
		self.noise_range = noise_range
		self.targets = dict()
		self.int_name_ref = dict()
		self.name_int_ref = dict()
		self.classes = dict()
		self.target_ref = dict()
		self.dumps = dict()
		self.save_exists = False
		self.state = -1
		self.current = 0
		self.comp_q_s = 0
		self.DEBUG = DEBUG
		self.logfile = 'POS_{}.log'.format(self.name)
		self.absolute_ref = dict()
		self.pbar = not pbar

	@staticmethod
	def __load_spec_class__(path):
		"""
		"""
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

	def __list_check__(self, l):
		if isinstance(l, list):
			return l
		else:
			return [l, l]

	def __build_survey__(self, amp_range=[0, 0.2],freq_range = [1, 100],
	  		  			 phase_range=[0, np.pi], L_range=[1, 3],
    		  			 amp_varience=0.01, freq_varience=0.01, 
    		  			 phase_varience=0.01,  obs_range=[10, 100]):
		for i in tqdm(range(self.size), disable=self.pbar):
			L_range = self.__list_check__(L_range)
			amp_range = self.__list_check__(amp_range)
			phase_range = self.__list_check__(phase_range)
			freq_range = self.__list_check__(freq_range)

			pulsation_amp = np.random.uniform(amp_range[0],
				                              amp_range[1])

			pulsation_frequency = np.random.uniform(freq_range[0],
				                                    freq_range[1])

			pulsation_phase = np.random.uniform(phase_range[0],
		    	                                phase_range[1])
			magnitude = np.random.uniform(self.mag_range[0],
										  self.mag_range[1])

			target_name = names.get_last_name().replace(' ', '-')
			cont = False

			while cont is False:
				target_id = "{}-{}".format(target_name, np.random.randint(0, 5001))
				if target_id not in self.targets:
					cont = True

			self.targets[target_id] = PVS(Number=1, numpoints=self.depth, 
				                          verbose=self.verbose, noise_range=self.noise_range, 
				                          mag=magnitude, name=target_id, 
				                          dpbar=True, ftemp=True, single_object=True)
			self.targets[target_id].build(amp_range=amp_range,
										  freq_range=freq_range,
										  phase_range=phase_range,
										  L_range=L_range)
			self.int_name_ref[i] = target_id
			self.name_int_ref[target_id] = i

	def build(self, load_from_file = False, path=None, amp_range=[0, 0.2], 
		      freq_range = [1, 100], phase_range=[0, np.pi], L_range=[1, 3], 
		      amp_varience=0.01, freq_varience=0.01, phase_varience=0.01, 
		      seed=1, visits=[10, 100]):
		if load_from_file is True:
			try:
				assert path is not None
			except AssertionError as e:
				e.arge += ('Error! No Path to file given', 'Did you specify path?')

				raise

		# self.__seed_generation__(seed=seed)

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
							  obs_range=visits)

	def generate(self, pfrac=0.5, target_in_mem=100, vtime_units=u.hour,
				 btime_units=u.day, exposure_time=30, visit_range=[1, 10],
				 visit_size_range=[0.5, 2], break_size_range=[10, 100],
				 etime_units=u.second,af=lambda x: 0):
		self.save_exists = False
		dumpnum = 0
		lastdump = 0
		refs = list()
		last = 0
		for j, i in tqdm(enumerate(self.targets), desc='Geneating Survey Data', total=self.size,
						disable=self.pbar):
			refs.append(i)
			rand_pick = np.random.uniform(0, 10)
			if rand_pick < pfrac * 10:
				self.classes[i] = 1
			else:
				self.classes[i] = 0

			self.targets[i].generate(pfrac=self.classes[i], vtime_units=vtime_units,
				                     btime_units=btime_units, exposure_time=exposure_time,
				                     visit_range=visit_range, visit_size_range=visit_size_range,
				                     break_size_range=break_size_range, etime_units=etime_units,
				                     af=af)
			for k in range(len(self.targets[i])):
				self.absolute_ref[last+k] = [j, k]
			last += len(self.targets[i]) - 1
			if j-lastdump >= target_in_mem:
				path_a = "{}/.{}_temp".format(os.getcwd(), self.prefix)
				if self.save_exists is False:
					if os.path.exists(path_a):
						shutil.rmtree(path_a)
					os.mkdir(path_a)
					self.save_exists = True
				path = "{}/.{}_temp/{}_dump".format(os.getcwd(), self.prefix, dumpnum)
				os.mkdir(path)
				for k, x in enumerate(refs):
					#if k < j-lastdump: 
					star_path = "{}/{}".format(path, x)
					if self.targets[x] is not None:
						os.mkdir(star_path)
						self.targets[x].save(path=star_path)
						self.targets[x] = None
				self.target_ref[dumpnum] = [lastdump, target_in_mem + lastdump]
				self.dumps[dumpnum] = path
				dumpnum += 1
				lastdump = j
		self.target_ref[-1] = [lastdump, self.size - 1]
		if self.save_exists is True:
			path = "{}/.{}_temp".format(os.getcwd(), self.prefix)
			self.__save_survey__(path)

	def __get_target_id__(self, n):
		if isinstance(n, int) or isinstance(n, np.integer):
			target_id = self.int_name_ref[n]
		if isinstance(n, str):
			target_id = n
		return target_id

	def __get_target_number__(self, n):
		if isinstance(n, int) or isinstance(n, np.integer):
			target_id = n
		if isinstance(n, str):
			target_id = self.name_int_ref[n]
		return target_id

	def __get_lc__(self, n=0, full=True, sn=0, start=0, stop=None, state_change=False):
		target_id = self.__get_target_id__(n)
		target_num = self.__get_target_number__(n)

		dump_num = -1
		for k in self.target_ref:
			if int(self.target_ref[k][0]) <= target_num <= int(self.target_ref[k][1]):
				dump_num = int(k)
				break

		if dump_num != self.state:
			pull_from = self.__load_dump__(n=dump_num, state_change=state_change)
			if state_change is False:
				try:
					assert sn < len(pull_from[target_id])
				except AssertionError as e:
					e.args += ('ERROR!', 'Target Light Curve index out of range')
					raise
				if stop == None:
					stop = len(pull_from[target_id])
				if full is True:
					times = list()
					fluxs = list()
					for Time, Flux, classes, _, p in pull_from[target_id].xget_lc(start=start,
						                                                       stop=stop):
						times.extend(Time)
						fluxs.extend(Flux)
						c = classes
						pp = p
				else:
					times = pull_from[target_id][sn][0]
					fluxs = pull_from[target_id][sn][1]
					c = pull_from[target_id][sn][2]
					pp = pull_from[target_id][sn][4]
			else:
				try:
					assert sn < len(self.targets[target_id])
				except AssertionError as e:
					e.args += ('ERROR!', 'Target Light Curve index out of range')
					raise
				if stop == None:
					stop = len(self.targets[target_id])
				if full is True:
					times = list()
					fluxs = list()
					for Time, Flux, classes, _, p in self.targets[target_id].xget_lc(start=start,
						                                                          stop=stop):
						times.extend(Time)
						fluxs.extend(Flux)
						c = classes
						pp = p
				else:
					times = self.targets[target_id][sn][0]
					fluxs = self.targets[target_id][sn][1]
					c = self.targets[target_id][sn][2]
					pp = self.targets[target_id][sn][4]
		else:
			if stop == None:
				stop = len(self.targets[target_id])
			if full is True:
				times = list()
				fluxs = list()
				for Time, Flux, classes, _, p in self.targets[target_id].xget_lc(start=start,
					                                                          stop=stop):
					times.extend(Time)
					fluxs.extend(Flux)
					c = classes
					pp = p
			else:
				times = self.targets[target_id][sn][0]
				fluxs = self.targets[target_id][sn][1]
				c = self.targets[target_id][sn][2]
				pp = self.targets[target_id][sn][4]

		return times, fluxs, c, target_id, pp

	def get_lc(self, n=0, full=True, sn=0, start=0, stop=None, state_change=False):
		return self.__get_lc__(n=n, full=full, sn=sn, start=start, stop=stop,
			                   state_change=state_change)

	def get_full_lc(self, n=0, state_change=False):
		return self.__get_lc__(n=n, full=True, state_change=state_change)

	def PVS_get_lc(self, n=0, state_change=False):
		refernce = self.absolute_ref[n]
		return self.get_lc_sub(n=refernce[0], sub_element=refernce[1], 
			                   state_change=state_change)

	def PVS_xget_lc(self, start=0, stop=None, state_change=True):
		if stop is None:
			stop = self.size
		if stop > self.size:
			raise IndexError('Error! Stop Value cannot excede Size of PVS Object')
		for i in range(start, stop):
			refernce = self.absolute_ref[i]
			yield self.get_lc_sub(n=refernce[0], sub_element=refernce[1], 
			                      state_change=state_change)

	def xget_lc(self, start=0, stop=None, state_change=True):
		if stop is None:
			stop = self.size
		if stop > self.size:
			stop = self.size
		for i in range(start, stop):
			yield self.__get_lc__(n=i, state_change=state_change)

	def get_lc_sub(self, n=0, sub_element=0, state_change=False):
		return self.__get_lc__(n=n, full=False, sn=sub_element,
			                   state_change=False)

	def __compress_spect__(self, spect):
		spect = np.array(spect)
		return spect/abs(np.max(spect))

	def __Debug_log__(self, i, arg='NPA', udfile=True):
		if self.DEBUG:
			if udfile:
				with open(self.logfile, 'a') as f:
					if i == 0:
						f.write('------\n')
					else:
						f.write('[DEBUG:{}]: {} \n'.format(i, arg))
			else:
				print('DEBUG: {}-- {}'.format(i, arg))

	def __get_spect__(self, n=0, s=500, dim=50,
					  power_spec=True, state_change=True,
					  Normalize=False):
		target_id = self.__get_target_id__(n)
		target_num = self.name_int_ref[target_id]
		LD_stretch = 1
		dump_num = -1
		self.count = n
		for k in self.target_ref:
			if int(self.target_ref[k][0]) <= target_num <= int(self.target_ref[k][1]):
				dump_num = int(k)
				break
		Amps = list()
		if dump_num != self.state:
			pull_from = self.__load_dump__(n=dump_num, state_change=state_change)
			if state_change is True:
				UD_stretch = float(len(self.targets[target_id])/dim)
				if UD_stretch < 1:
					UD_stretch = int(1/UD_stretch)
				for Freq, Amp, Class, Index, kwarg in self.targets[target_id].xget_ft(power_spec=True):
					Amps.append(Amp)
				out_tuple = (np.repeat(np.repeat(Amps, LD_stretch, axis=1),UD_stretch, axis=0),
					         Freq, self.targets[target_id][0][2], target_id, kwarg)
			else:
				UD_stretch = float(len(pull_from[target_id])/dim)
				if UD_stretch < 1:
					UD_stretch = int(1/UD_stretch)
				for Freq, Amp, Class, Index, kwarg in pull_from[target_id].xget_ft(power_spec=True):
					Amps.append(Amp)
				out_tuple = (np.repeat(np.repeat(Amps, LD_stretch, axis=1),UD_stretch, axis=0),
					         Freq, pull_from[target_id][0][2], target_id, kwarg)
		else:
			UD_stretch = float(len(self.targets[target_id])/dim)
			if UD_stretch < 1:
				UD_stretch = 1/UD_stretch
			for Freq, Amp, Class, Index, kwarg in self.targets[target_id].xget_ft(power_spec=True):
				Amps.append(Amp)
			out_tuple = (np.repeat(np.repeat(Amps, LD_stretch, axis=1),UD_stretch, axis=0),
				         Freq, self.targets[target_id][0][2], target_id, kwarg)	
		orig_max = out_tuple[0].max()
		orig_min = out_tuple[0].min()
		orig_range = orig_max - orig_min
		out_img = misc.imresize(out_tuple[0], (dim, s), interp='cubic')
		out_img = ((out_img * orig_range)/255.0)+orig_min
		if Normalize is True:
			out_img = out_img/(np.mean(out_img) - 1)
		out_tuple = (out_img, out_tuple[1], out_tuple[2], out_tuple[3], out_tuple[4])
		return out_tuple

	def get_spect(self, n=0, s=500, dim=50, power_spec=True, 
				  state_change=True, Normalize=False):
		return self.__get_spect__(n=n, s=s, dim=dim, power_spec=power_spec,
								  state_change=state_change, Normalize=Normalize)

	def xget_spect(self, start=0, stop=None, s=500, dim=50,
			       power_spec=True, state_change=True):
		if stop is None:
			stop = self.size
		if stop > self.size:
			stop = self.size
		for i in range(start, stop):
			yield self.__get_spect__(n=i, s=s, dim=dim, power_spec=False,
									 state_change=state_change)

	def PVS_get_ft(self, n=0, s=300, state_change=False, power_spec=False, ct1=False):
		refernce = self.absolute_ref[n]
		return self.get_ft_sub(n=refernce[0], sub_element=refernce[1],
							   state_change=state_change, power_spec=power_spec,
							   ct1=ct1, s=s)

	def PVS_xget_ft(self, start=0, stop=None, s=300, state_change=True,
				power_spec=False, ct1=False):
		if stop is None:
			stop = len(self.absolute_ref)
		if stop > len(self.absolute_ref):
			raise IndexError('Error! Stop Value is Larger than size of PVS Object')
		for i in range(start, stop):
			refernce = self.absolute_ref[i]
			yield self.get_ft_sub(n=refernce[0], sub_element=refernce[1],
							   state_change=state_change, power_spec=power_spec,
							   ct1=ct1, s=s)
	def get_full_ft(self, n=0, s=500, state_change=False, power_spec=False, ct1=False, frange=[None],nymult=1):
		lc = self.get_full_lc(n=n, state_change=state_change)
		if frange[0] == None:
			avg_sample_rate = (max(lc[0])-min(lc[0]))/len(lc[0])
			nyquist = 1/(2*avg_sample_rate)
			res = 1/(max(lc[0])-min(lc[0]))
			f = np.linspace(0.1*res, nymult*nyquist, s)
		else:
			f = np.linspace(frange[0], frange[1], s)

		pgram = lombscargle(lc[0], lc[1], f, normalize=True)
		return f, pgram

	def get_ft_sub(self, n=0, sub_element=0, s=300, state_change=False, power_spec=False, ct1=False):
		target_id = self.__get_target_id__(n)
		target_num = self.__get_target_number__(n)

		dump_num = -1
		for k in self.target_ref:
			if int(self.target_ref[k][0]) <= target_num <= int(self.target_ref[k][1]):
				dump_num = int(k)
				break
		if dump_num != self.state:
			pull_from = self.__load_dump__(n=dump_num, state_change=state_change)
			if state_change is True:
				out_tuple = self.targets[target_id].get_ft(n=sub_element, s=s, power_spec=power_spec)
			else:
				out_tuple = pull_from[target_id].get_ft(n=sub_element, s=s, power_spec=power_spec)
		else:
			out_tuple = self.targets[target_id].get_ft(n=sub_element, s=s, power_spec=power_spec)
		if ct1 is True:
			comp_As = compress_to_1(out_tuple[1])
		else:
			comp_As = out_tuple[1]
		out_tuple = (out_tuple[0], comp_As, out_tuple[2], self.int_name_ref[n], out_tuple[4])
		return out_tuple

	def __load_dump__(self, n=0, state_change=True):
		if state_change is True:
			self.targets = dict()
			self.state = n
		else:
			ttargets = dict()
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
		else:
			return 0

	def save(self, path=None):
		if path is None:
			path = "{}/{}".format(os.getcwd(), self.name)
		if os.path.exists(path):
			shutil.rmtree(path)
		os.mkdir(path)
		for dump in self.dumps:
			group_path = "{}/Group_{}".format(path, dump)
			if os.path.exists(group_path):
				shutil.rmtree(group_path)
			os.mkdir(group_path)
			data = self.__load_dump__(n=dump, state_change=False)
			for target in data:
				target_save_path = "{}/{}".format(group_path, target)
				if os.path.exists(target_save_path):
					shutil.rmtree(target_save_path)
				os.mkdir(target_save_path)
				data[target].save(path=target_save_path)
		mem_path = "{}/Group_-1".format(path)
		if os.path.exists(mem_path):
			shutil.rmtree(mem_path)
		os.mkdir(mem_path)	
		for target in tqdm(self.targets, total=self.size, desc='Saving {} to Disk'.format(self.name),
						   disable=self.pbar):
			if self.targets[target] is not None:
				target_save_path = "{}/{}".format(mem_path, target)
				if os.path.exists(target_save_path):
					shutil.rmtree(target_save_path)
				os.mkdir(target_save_path)
				self.targets[target].save(path=target_save_path)
		self.__save_survey__(path)
		return path

	def __save_survey__(self, path):
		with open('{}/Object_Class.POS'.format(path), 'w') as f:
			out = list()
			for target in self.classes:
				out.append("{}:{}".format(target, self.classes[target]))
			out = '\n'.join(out)
			f.write(out)
		with open('{}/Item_Ref.POS'.format(path), 'w') as f:
			out = list()
			for target_id in self.int_name_ref:
				out.append("{}:{}".format(target_id, self.int_name_ref[target_id]))
			out = '\n'.join(out)
			f.write(out)
		with open("{}/Item_Loc.POS".format(path), 'w') as f:
			out = list()
			for dump in self.target_ref:
				out.append("{}:{}:{}".format(dump, self.target_ref[dump][0],
					                         self.target_ref[dump][1]))
			out = '\n'.join(out)
			f.write(out)
		with open('{}/Absolute_Ref.POS'.format(path), 'w') as f:
			out = list()
			for key in self.absolute_ref:
				out.append("{}:{}:{}".format(key, self.absolute_ref[key][0],
										     self.absolute_ref[key][1]))
			out = '\n'.join(out)
			f.write(out)
		with open("{}/Object_Meta.POS".format(path), 'w') as f:
			out = list()
			out.append('Size:{}'.format(self.size))
			out.append('Name:{}'.format(self.name))
			out.append('Prefix:{}'.format(self.prefix))
			out.append('Depth:{}'.format(self.depth))
			out.append('Verbose:{}'.format(self.verbose))
			out.append('Noise:{}:{}'.format(self.noise_range[0],
										    self.noise_range[1]))
			out.append('MagRange:{}:{}'.format(self.mag_range[0],
											   self.mag_range[1]))
			out = '\n'.join(out)
			f.write(out)

	def load(self, directory='.', start=-1, pbar=True):
		files = os.listdir(directory)
		NTG = False
		try:
			assert 'Item_Loc.POS' in files
		except AssertionError as e:
			e.args += ('Error! Corrupted POS object', 'Cannot find "Item_Loc.POS"')
			raise
		try:
			assert 'Item_Ref.POS' in files
		except AssertionError as e:
			e.args += ('Error! Corrupted POS object', 'Cannot find "Item_Red.POS"')
			raise
		try:
			assert 'Object_Class.POS' in files
		except AssertionError as e:
			e.args += ('Error! Corrupted POS object', 'Cannot find "Object_Class.POS"')
			raise
		try:
			assert 'Object_Meta.POS' in files
		except AssertionError as e:
			e.args += ('Error! Corrupted POS object', 'Cannot find "Object_Meta.POS"')
			raise
		if 'Absolute_Ref.POS' not in files:
			warn('Unable To Locate Absolute_Ref.POS, '\
				 'Will Attempt to generate and save to'\
				 'directory')
			NTG = True
		if directory[-1] == '/':
			directory = directory[:-1]
		dumps = [x.split('_')[1] for x in files if not '.POS' in x and 'Group' in x]
		dumps = [int(x) for x in dumps]

		dump_dirs = ["{}/Group_{}".format(directory, x) for x in dumps]
		for d, p in zip(dumps, dump_dirs):	
			self.dumps[d] = p
		with open('{}/Item_Loc.POS'.format(directory), 'r') as f:
			for line in tqdm(f.readlines(), desc="Item Loc", disable=self.pbar):
				data = line.split(':')
				self.target_ref[int(data[0])] = [int(data[1]), int(data[2])]
		with open('{}/Item_Ref.POS'.format(directory), 'r') as f:
			for line in tqdm(f.readlines(), desc="Item Ref", disable=self.pbar):
				data = line.split(':')
				self.int_name_ref[int(data[0])] = data[1].rstrip()
				self.name_int_ref[data[1].rstrip()] = int(data[0])
		with open('{}/Object_Class.POS'.format(directory), 'r') as f:
			for line in tqdm(f.readlines(), desc="Object Class", disable=self.pbar):
				data = line.split(':')
				self.classes[data[0].rstrip()] = int(data[1])

		with open('{}/Object_Meta.POS'.format(directory), 'r') as f:
			for line in tqdm(f.readlines(), desc='Object Meta', disable=self.pbar):
				data = line.split(':')
				if data[0] == 'Size':
					self.size = int(data[1])
				elif data[0] == 'Name':
					self.name = data[1]
					self.logfile = 'POS_{}.log'.format(self.name)
				elif data[0] == 'Prefix':
					self.prefix = data[1]
				elif data[0] == 'Depth':
					self.depth = int(data[1])
				elif data[0] == 'Verbose':
					self.verbose = int(data[1])
				elif data[0] == 'Noise':
					self.noise_range = [float(data[1]), float(data[2])]
				elif data[0] == 'MagRange':
					self.mag_range = [float(data[1]), float(data[2])]
		dumps = [int(x) for x in dumps]
		dumps = sorted(dumps)
		self.__load_dump__(n=-1)
		self.state = -1

		if NTG is True:
			self.absolute_ref = self.__gen_absolute_ref__(save=True, directory=directory, pbar=pbar)
		else:
			with open('{}/Absolute_Ref.POS'.format(directory), 'r') as f:
				for line in tqdm(f.readlines(), desc="Absolute Path", disable=self.pbar):
					data = line.split(':')
					self.absolute_ref[int(data[0].rstrip())] = [int(data[1].rstrip()),
															    int(data[2].rstrip())]

	def __gen_absolute_ref__(self, save=False, directory=None, pbar=True):
		iaf = dict()
		c = 0
		for n, target in tqdm(enumerate(self.xget_object()), 
							  desc='Generating Absolute Refernce',
							  disable=self.pbar):
			for i in range(len(target)):
				iaf[c] = [n, i]
				c += 1
		if save is True:
			assert directory is not None

			with open('{}/Absolute_Ref.POS'.format(directory), 'w') as f:
				out = list()
				for key in iaf:
					out.append("{}:{}:{}".format(key, iaf[key][0], iaf[key][1]))
				out = '\n'.join(out)
				f.write(out)
		return iaf

	def names(self):
		return list(self.targets.keys())

	def xget_object(self, start=0, stop=None, state_change=True):
		if stop is None:
			stop = self.size
		if stop > self.size:
			stop = self.size
		for i in range(start, stop):
			yield self.get_object(n=i, state_change=state_change)

	def get_object(self, n=0, state_change=False):
		target_id = self.__get_target_id__(n)
		target_num = self.__get_target_number__(n)
		dump_num = -1
		for k in self.target_ref:
			if int(self.target_ref[k][0]) <= target_num <= int(self.target_ref[k][1]):
				dump_num = int(k)
				break
		if dump_num != self.state:
			pull_from = self.__load_dump__(n=dump_num, state_change=state_change)
			if state_change is True:
				out_obj = self.targets[target_id]
			else:
				out_obj = pull_from[target_id]
		else:
			out_obj = self.targets[target_id]
		return out_obj

	def __batch_get_lc__(self, start=0, mem_size=1e9, step=1,
						 stop=None):
		if stop is None:
			stop = self.size
		mem_use_single = getsizeof(self.__get_lc__(n=start))
		num = int(mem_size / mem_use_single)
		if stop < start + (num * step):
			num = stop
		else:
			num *= step
			num += start
		out_times = list()
		out_fluxs = list()
		out_class = list()
		out_tarid = list()
		for j in range(start, num, step):
			Time, Flux, Class, TID = self.__get_lc__(n=j, state_change=True)
			out_times.append(Time)
			out_fluxs.append(Flux)
			out_class.append(Class)
			out_tarid.append(TID)
		return out_times, out_fluxs, out_class, out_tarid

	def __batch_get_spect__(self, start=0, mem_size=1e9, step=1,
							stop=None, s=500, dim=50,
							power_spec=True, n_threds=4,
							Normalize=False):
		if stop is None:
			stop = self.size
		use_target = self.targets[list(self.targets)[0]].name
		mem_use_single = getsizeof(self.__get_spect__(n=use_target, s=s, dim=dim,
													  power_spec=power_spec))
		num = int(mem_size / mem_use_single)
		if stop < start + (num * step):
			num = stop
		else:
			num *= step
			num += start
		with closing(Pool(4, maxtasksperchild=1000)) as p:
			data_inputs = range(start, num, step)
			params = [data_inputs, s, dim, power_spec, Normalize]
			pool_params = np.array(self.__gen_pool_params__(params)).T
			pool_output = p.starmap(self.__spect_thread_retive__, pool_params)
			pool_output = np.array(pool_output)
			out_imigs = pool_output[:, 0]
			out_freqs = pool_output[:, 1]
			out_class = pool_output[:, 2]
			out_tarid = pool_output[:, 3]
			out_kwarg = pool_output[:, 4]
		return out_imigs, out_freqs, out_class, out_tarid, out_kwarg	

	def __gen_pool_params__(self, parameters):
		r = parameters[0]
		s = [parameters[1]] * len(r)
		dim = [parameters[2]] * len(r)
		power_spec = [parameters[3]] * len(r)
		Normalize = [parameters[4]] * len(r)
		return [r, s, dim, power_spec, Normalize]

	def __spect_thread_retive__(self, n, s, dim, power_spec, Normalize):
		return self.__get_spect__(n=n, s=s,dim=dim,
								  power_spec=power_spec, Normalize=Normalize)

	def get_spects(self, start=0, mem_size=1e9, step=1,
				   stop=None, s=500, dim=50,
				   power_spec=True, n_threds=4,
				   Normalize=False):
		return self.__batch_get_spect__(start=start, mem_size=mem_size,
							            step=step, stop=stop, s=s, dim=dim,
							            power_spec=power_spec,
							            n_threds=n_threds, Normalize=Normalize)

	def batch_get(self, batch_size=10, spect=False, s=None, dim=50, mem_size=1e9,
				  power_spec=True, NormalizeSpect=False):
		if isinstance(batch_size, str):
			try:
				assert batch_size == 'mem_size'
			except AssertionError as e:
				e.args += ('Error Unrecognizer argumenent: <"{}">'.format(batch_size),
						   'Please either set batch_size to an integer st. 0 < batchsize <= len(PVS())',
						   'or set batch batch_size equal to <"mem_size"> where the batch will fill the defined memory',
						   'this is defaulted to 1GB but can be adjusted (in byte space) with the mem_size parameter')
				raise
		if isinstance(batch_size, int):
			try:
				assert 0 < batch_size <= self.size
			except AssertionError as e:
				e.args += ('Error, Invalid batch size', 'Please make sure batch_size parameter is greater than 0', 
						   'please also make sure batch size parameter is less than or equal to len(PVS())')
				raise
		if batch_size == 'mem_size':
			if spect is False:
				mem_use_single = getsizeof(self.targets[0])
				batch_size = int(mem_size / mem_use_single)
			else:
				mem_use_single = getsizeof(self.get_spect(s=s, dim=dim,
														  power_spec=power_spec))
				batch_size = int(mem_size / mem_use_single)
		if spect is True and s is None:
			s = 500
		if spect is False:
			for i in range(int(self.size / batch_size)):
				yield self.__batch_get_lc__(start=i * batch_size,
											stop=(i * batch_size) + batch_size,
											mem_size=mem_size)
		else:
			for i in range(int(self.size / batch_size)):
				yield self.__batch_get_spect__(start = i * batch_size,
											   stop = (i * batch_size) + batch_size,
											   s=s, mem_size=mem_size, dim=dim,
											   Normalize=NormalizeSpect)

	def __repr__(self):
		out = list()
		out.append("Survey Name: {prefix}".format(prefix=self.prefix))
		out.append("Survey Size: {size}".format(size=self.size))
		out.append("Survey Object Name: {name}".format(name=self.name))
		if self.verbose >= 1:
			out.append('Noise Range: {}->{}'.format(self.noise_range[0], self.noise_range[1]))

		return '\n'.join(out)

	def __getitem__(self, key):
		if isinstance(key, int):
			try:
				assert key < self.size
			except AssertionError as e:
				e.args += ('Index Error!', '{} Not in range for POS of size {}'.format(key, self.size))
				raise
			return self.get_object(n=key)
		elif isinstance(key, str):
			return self.get_object(n=key)

	def __len__(self):
		return self.size

	def __del__(self):
		path = "{}/.{}_temp".format(os.getcwd(), self.prefix)
		if os.path.exists(path):
			shutil.rmtree(path)
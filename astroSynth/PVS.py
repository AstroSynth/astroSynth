from .SDM import Gen_FT, NyApprox, Normalize, Make_Syth_LCs
import numpy as np
from tqdm import tqdm
from sys import getsizeof
from tempfile import TemporaryFile
import os
import shutil


"""
PVS (Pulsating Variable Star)- Class

Description:
    Class to generate, load, acess, store, and do very basic analysis
        Synthetically generated Light curves

"""

class PVS:
    def __init__(self, Number=100, noise_range=[0.1, 1.1], vmod=True,
                 f=lambda x: np.sin(x), numpoints=100, mag_range=[6, 20],
                 verbose=0, name=None):
        """
        PVS Initilization

        Params:
            Number: Number of light curves to generate in the PVS object (int)
            noise_range: noise range to use with [0] being the lowest and [1]
                         being the largest (2-element float list)
            vmod: Use a continuous range of functions or one function defined
                  in f (bool)
            f: Function to define pulsation, used onley if vmod is true 
               (python function)
            numpoints: The number of points to generate per light curve (int)
            mag_range: The magnitude range of stars to generate light curves 
                       from (2-element floar list) [Not currently used]
            verbose: the verbosity which with to use when representing the 
                     object (0 - default, 1 - add dump info, 2 - add stored 
                     data)
            name: Name of object to use as directory name when saving object
        Returnes:
            Fully formed PVS() type object, ready to build-generate or to 
            load data
        """
        self.size = Number
        self.noise_range = noise_range
        self.depth = numpoints
        self.mag_range = mag_range
        self.verbose = verbose
        self.lcs = np.zeros((0, self.depth, 2))
        self.dumps = dict()
        self.class_dumps = dict()
        self.generated = False
        self.kwargs = dict()
        self.item_ref = dict()
        self.classification = np.zeros((0))
        self.temp_file = True
        self.state = -1
        if name is not None:
            self.name = name.rstrip()
        else:
            self.name = name
        if vmod is True:
            self.vmod = True
            self.built = False
            self.f = dict()
        else:
            self.f = f
            self.built = True
            self.vmod = False

    @staticmethod
    def _seed_generation_(seed=1):
        """
        descriptpion:
            Seed the random generator with the same seed each time 
            to achive comparable results.
        params:
            seed: Seed to use in np.random.seed() (int)
        returns:
            N/A
        pre-state:
            Random unseeded
        post-state:
            Random seeded to param:seed

        """
        np.random.seed(seed)

    def __debug_check__(self):
        """
        description:
            print the current development version
        params:
            self: PVS() object
        Returns:
            N/A
        """
        print('Version 0.3.4 Development')

    def __build_func__(self, phase_range=[0, np.pi], amp_range=[0, 1],
                       freq_range=[1e-7, 1], L_range=[1, 3]):
        """
        description:
            hidden function to build the continuous set of pulsation 
            charectaristic functions
        params:
            self: PVS() objects
            phase_range: range of phases to use (randomly select 
                         between them inclusive) where [0] is the
                         smallest phase and [1] is the largest phase 
                         (2-element float list)
            amp_range: range of amplitudes to use (randomly select
                       between them inlusive) where [0] is the smallest 
                       amplitude and [1] is the largest amplitude 
                       (2-element float list)
            freq_range: range of frequencies to use (randomly select 
                        between them inclusive) where [0] is the smallest 
                        frequency and [1] is the largest frequency 
                        (2-element float list)
            L_range: range of pulsation modes to use (randomly select 
                     between them inclusive) where [0] is the smallest 
                     number of pulsation modes and [1] is the largest 
                     number of pulsation modes (2-element int list)
        returns:
            N/A
        pre-state:
            if param:self.vmod is true:
                param:self.kwargs empty dictionary
                param:self.f empty dictionary
            if param:self.vmod is False:
                param:self.kwargs empty dictionary
                param:self.f empty dictionary
        post-state:
            if param:self.vmod is true:
                param:self.kwargs dicitonary filled with parameters for funational form
                param:self.f dictionaty filled with functional forms
        """
        if self.vmod is True:
            for i in tqdm(range(self.size), desc='Building Light Curve Functional Form'):
                kwargs = dict()
                kwargs['num'] = np.random.randint(L_range[0],
                                                  L_range[1] + 1)
                kwargs['phase'] = np.random.uniform(phase_range[0],
                                                    phase_range[1],
                                                    kwargs['num'])
                kwargs['amp'] = np.random.uniform(amp_range[0],
                                                  amp_range[1],
                                                  kwargs['num'])
                kwargs['freq'] = np.random.uniform(freq_range[0],
                                                   freq_range[1],
                                                   kwargs['num'])
                self.kwargs[i] = kwargs
                self.f[i] = lambda x, d: self.__mode_addition__(x, **d)
    @staticmethod
    def __mode_addition__(x, num=1, phase=[0], amp=[1], freq=[1]):
        """
        description:
            combine multiple modes of pulsation into one function
        params:
            x: semi-continuous array of values to evalueate 
               function over (__getitem__ numerical type)
            num: number of pulsation modes to consider (int)
            phase: phases to use in pulsation modes 
                  (float list of size num)
            amp: amplutudes to use in pulsation modes:
                 (float list of size num)
            freq: frequencies to use in pulsation modes:
                  (float list of size num)
        returns:
            fout: evaluated over x sum of sin functions (ndarray)
        raises:
            AssertationError: If num = 0 then Assertation error is raised
            AssertationError: If num is not equal to the length of all
                              three parameter lists then an Assertation
                              error is rasised
        """
        try:
            assert num is not 0
        except AssertionError as e:
            e.args += ('Error: num is 0', 'Cannot have 0 Pulsations mode')
            raise

        try:
            assert len(phase) == len(amp) == len(freq) == num
        except AssertionError as e:
            e.args += ('Error: Pulsation mode lengh inconsistent', 
                      'length of phase, amp, freq, and size of n are inconsistent', 
                      'these must always match')
            raise

        phase = np.array(phase)
        amp = np.array(amp)
        freq = np.array(freq)

        fout = amp[0] * np.sin(2 * np.pi * freq[0] * x + phase[0])
        for i in range(1, num):
            fout += amp[i] * np.sin(2 * np.pi * freq[i] * x + phase[i])

        return fout

    def build(self, phase_range=[0, np.pi], amp_range=[0, 1],
              freq_range=[1e-7, 1], L_range=[1, 3], seed=1):
        """
        description:
            user facing build function to seed, build and then store 
            the built state
        Params:
            self: PVS() object
            phase_range: range of phases to use (randomly select 
                         between them inclusive) where [0] is the
                         smallest phase and [1] is the largest phase 
                         (2-element float list)
            amp_range: range of amplitudes to use (randomly select
                       between them inlusive) where [0] is the smallest 
                       amplitude and [1] is the largest amplitude 
                       (2-element float list)
            freq_range: range of frequencies to use (randomly select 
                        between them inclusive) where [0] is the smallest 
                        frequency and [1] is the largest frequency 
                        (2-element float list)
            L_range: range of pulsation modes to use (randomly select 
                     between them inclusive) where [0] is the smallest 
                     number of pulsation modes and [1] is the largest 
                     number of pulsation modes (2-element int list)
            seed: Seed to use in np.random.seed() (int)
        Returns:
            N/A
        pre-state:
            PVS() object is unseeded
            if param:self.vmod is true:
                param:self.kwargs empty dictionary
                param:self.f empty dictionary
            if param:self.vmod is False:
                param:self.kwargs empty dictionary
                param:self.f empty dictionary
            param:self.built is False
        post-state:
            if param:self.vmod is true:
                param:self.kwargs dicitonary filled with parameters for funational form
                param:self.f dictionaty filled with functional forms
            param:self.built is True
        """
        self._seed_generation_(seed)
        self.__build_func__(phase_range=phase_range, amp_range=amp_range,
                            freq_range=freq_range, L_range=L_range)
        self.built = True

    def generate(self, pfrac=0.1):
        """
        description:
            generate the data given an already build PVS() object 
            (where param:self.built is true)
        params:
            self: PVS() object
            pfrac: Pulstion fraction - fraction of generated 
                   targets which will show a pulsation (float)
        returnes:
            N/A
        Raises:
            AssertationError: if the PVS() object has not been built
        pre-state:
            param:self.generated is False
            param:self.classification is empty ndarray
            param:self.lcs is empty list
            param:self.temp_file is true
            param:self.dumps is empty dictionary
            param:self.class_dumps is empty dictionary
            param:self.item_ref is empty dictionary
            No file are save to disk
        post-state:
            param:self.generated is True
            param:self.temp_file is true
            param:classification is 1D ndarray of size 
                  param:self.size
            param:lcs is 3D array of size 
                  (param:self.size x param:self.depth x 2)
            param:self.dumps may be filled
            param:self.class_dumps may be filled
            param:self.item_ref may be filld
            Files are saved to disk as temp files
        """
        try:
            assert self.built is True
        except AssertionError as e:
            e.args += ('PVS objects functional form not built',
                       'have you run PVS.build()?')
            raise

        dump_num = 0
        last_dump = 0
        self.generated = True
        list_lcs = list()
        for i in tqdm(range(self.size), desc='Geneating Light Curves'):
            rand_pick = np.random.uniform(0, 10)
            if rand_pick < pfrac * 10:
                pulsator = True
                self.classification = np.append(self.classification, 1)
            else:
                pulsator = False
                self.classification = np.append(self.classification, 0)
            if self.vmod is True:
                tlc = Make_Syth_LCs(f=lambda x: self.f[i](x, self.kwargs[i]), pulsator=pulsator,
                                        numpoints=self.depth,
                                        noise_range=self.noise_range)
            else:
                tlc = Make_Syth_LCs(f=self.f, pulsator=pulsator,
                                        numpoints=self.depth,
                                        noise_range=self.noise_range)
            list_lcs.append(tlc)
            if getsizeof(list_lcs) > 1e5:
                self.dumps[dump_num] = TemporaryFile()
                self.class_dumps[dump_num] = TemporaryFile()
                self.item_ref[dump_num] = [last_dump, len(list_lcs) + last_dump]
                np.save(self.dumps[dump_num], np.array(list_lcs))
                np.save(self.class_dumps[dump_num], self.classification)
                dump_num += 1
                last_dump = i
                list_lcs = list()
                self.classification = np.zeros((0))
            self.item_ref[-1] = [last_dump + 1, len(list_lcs) + last_dump]
        self.lcs = np.array(list_lcs)
        self.temp_file = True

    def __get_lc__(self, n=0, state_change=False):
        """
        desctription:
            Hidden function to retrieve the nth light curve from the PVS() 
            object with the possibilty existing to change the data loaded 
            into memory
        Params:
            self: PVS() object
            n: index of light curve to retrieve (int)
            state_change: whether to allow the object to change what is 
                          loaded into param:self.lcs and 
                          param:self.classification in order that future 
                          retrivals may not take so many np.load calls (bool)
        Returns:
            Four element Tuple
                0: Light curve time array
                1: Light curve flux array
                2: Light curve classification (0.0 - non variable, 1.0 - variable)
                3: index of retived light curve
        Raises:
            AssertationError: if param:self.generated is False
        pre-state:
            param:self.lcs is some data
            param:self.classifcation is some data
            param:self.state is some integer
        post-state:
            if param:state_change is True:
                param:self.lcs may change to represent the data location retrived
                param:self.classification may change to represent the data 
                location retrived
                if those changes then param:self.state will update to represent that
        """

        try:
            assert self.generated is True
        except AssertionError as e:
            e.args += ('PVS objects Light Curves are not generated',
                       'have you run PVS.generate()?')
            raise
        file_num = -1
        base = 0
        for k in self.item_ref:
            if int(self.item_ref[k][0]) <= n <= int(self.item_ref[k][1]):
                file_num = int(k)
                base = int(self.item_ref[k][0])
                break

        if file_num != self.state:
            if self.temp_file is True:
                self.dumps[file_num].seek(0)
                self.class_dumps[file_num].seek(0)
            tlcs = np.load(self.dumps[file_num])
            tclass = np.load(self.class_dumps[file_num])
            if state_change is True:
                self.lcs = tlcs
                self.classification = tclass
                self.state = file_num

            if self.temp_file is True:
                self.dumps[file_num].seek(0, os.SEEK_END)
                self.class_dumps[file_num].seek(0, os.SEEK_END)
            return tlcs[n - base].T[1], tlcs[n - base].T[0], tclass[n - base], n
        else:
            return self.lcs[n - base].T[1], self.lcs[n - base].T[0], self.classification[n - base], n

    def xget_lc(self, stop=None, start=0):
        if stop is None:
            stop = self.size
        for i in range(start, stop):
            yield self.__get_lc__(n=i)

    def save(self):
        try:
            assert self.generated is True
        except AssertionError as e:
            e.args += ('Light Curves have not been generated as of yet',
                       'have you run PVS.generate()?')
            raise

        path = "{}/{}".format(os.getcwd(), self.name)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        for dump, cdump in zip(self.dumps, self.class_dumps):
            self.dumps[dump].seek(0)
            self.class_dumps[cdump].seek(0)
            tlc = np.load(self.dumps[dump])
            tclass = np.load(self.class_dumps[cdump])
            self.dumps[dump].seek(0, os.SEEK_END)
            self.class_dumps[cdump].seek(0, os.SEEK_END)
            if self.name is not None:
                np.save("{}/LightCurve_{}.npy".format(path, dump), tlc)
                np.save("{}/LightCurve_Class_{}.npy".format(path, dump), tclass)
            else:
                np.save("{}/LightCurve_{}.npy".format(os.getcwd(), dump), tlc)
                np.save("{}/LightCurve_Class_{}.npy".format(os.getcwd(), dump), tclass)

        if len(self.lcs > 0):
            if self.name is not None:
                np.save("{}/LightCurve_{}.npy".format(path, -1), self.lcs)
                np.save("{}/LightCurve_Class_{}.npy".format(path, -1), self.classification)
            else:
                np.save("{}/LightCurve_{}.npy".format(os.getcwd(), -1), self.lcs)
                np.save("{}/LightCurve_Class_{}.npy".format(os.getcwd(), -1), self.classification)

        self._save_model_()

    def _save_model_(self):
        if self.name is not None:
            path = "{}/{}".format(os.getcwd(), self.name)
            if not os.path.exists(path):
                os.mkdir(path)
            with open('{}/item_loc_meta.PVS'.format(path), 'w') as f:
                out = list()
                for i in self.item_ref:
                    out.append("{}:{}:{}".format(i, self.item_ref[i][0], self.item_ref[i][1]))
                out = '\n'.join(out)
                f.write(out)
            with open('{}/object_meta.PVS'.format(path), 'w') as f:
                out = list()
                out.append('Size:{}'.format(self.size))
                out.append('Depth:{}'.format(self.depth))
                out.append('Name:{}'.format(self.name))
                out.append('Verbose:{}'.format(self.verbose))
                out.append('Noise:{}:{}'.format(self.noise_range[0], self.noise_range[1]))
                out = '\n'.join(out)
                f.write(out)
        else:
            path = os.getcwd()
            with open('{}/item_loc_meta.PVS'.format(path), 'w') as f:
                out = list()
                for i in self.item_ref:
                    out.append("{}:{}:{}".format(i, self.item_ref[i][0], self.item_ref[i][1]))
                out = '\n'.join(out)
                f.write(out)
            with open('{}/object_meta.PVS'.format(path), 'w') as f:
                out = list()
                out.append('Size:{}'.format(self.size))
                out.append('Depth:{}'.format(self.depth))
                out.append('Name:{}'.format(self.name))
                out.append('Verbose:{}'.format(self.verbose))
                out.append('Noise:{}:{}'.format(self.noise_range[0], self.noise_range[1]))
                out = '\n'.join(out)
                f.write(out)

    def load(self, directory='.', start=-1):
        files = os.listdir(directory)
        if directory[-1] == '/':
            directory = directory[:-1]

        try:
            assert 'item_loc_meta.PVS' in files
        except AssertionError as e:
            e.args += ('Cannot locate item meta data file',
                       'get_lc will not work',
                       'is the file present?')
            raise

        try:
            assert 'object_meta.PVS' in files
        except AssertionError as e:
            e.args += ('Cannot locate object meta data file',
                       'save will not work unil this is located',
                       'is the file present?')
            raise

        self.dumps = dict()
        self.class_dumps = dict()
        self.item_ref = dict()
        lcs = [x for x in files if 'LightCurve' in x and 'LightCurve_Class' not in x]
        lclass = [x for x in files if 'LightCurve_Class' in x]

        try:
            assert len(lcs) == len(lclass) and len(lcs) > 0
        except AssertionError as e:
            e.args += ('Invalid dataset dimsneions',
                       'No Files found or number of class files not in agreement with number of datafiles',
                       'Have you entered the correct path? default is current working directory.')
            raise

        try:
            assert 'LightCurve_Class_{}.npy'.format(start) in lclass and 'LightCurve_{}.npy'.format(start) in lcs
        except AssertionError as e:
            e.args += ('No First File to load into memory',
                       'Have you specified a valid start location (default = 0)')
            raise
        self.lcs = np.load('{}/LightCurve_{}.npy'.format(directory, start))
        self.classification = np.load('{}/LightCurve_Class_{}.npy'.format(directory, start))
        other_lcs = [x for x in lcs]
        other_lclass = [x for x in lclass]
        for i, j in zip(other_lcs, other_lclass):
            num_lcs = int(i.split('_')[1].split('.')[0])
            num_lclass = int(j.split('_')[2].split('.')[0])
            self.dumps[num_lcs] = "{}/{}".format(directory, i)
            self.class_dumps[num_lclass] = "{}/{}".format(directory, j)
        with open('{}/item_loc_meta.PVS'.format(directory), 'r') as f:
            lines = [x.split(':') for x in f.readlines()]
            for i in lines:
                self.item_ref[i[0]] = [i[1], i[2]]
        with open('{}/object_meta.PVS'.format(directory), 'r') as f:
            lines = [x.split(':') for x in f.readlines()]
            for i in lines:
                if i[0] == 'Size':
                    self.size = int(i[1].rstrip())
                elif i[0] == 'Depth':
                    self.depth = int(i[1].rstrip())
                elif i[0] == 'Name':
                    self.name = i[1].rstrip()
                elif i[0] == 'Verbose':
                    self.verbose = int(i[1].rstrip())
                elif i[0] == 'Noise':
                    self.noise_range[0] = float(i[1].rstrip())
                    self.noise_range[1] = float(i[2].rstrip())
        self.generated = True
        self.temp_file = False

    def __repr__(self):
        l = list()
        l.append('Size: {s}'.format(s=self.size))
        l.append('Noise Range: {n}'.format(n=self.noise_range))
        l.append('Magnitude Range: {m}'.format(m=self.mag_range))
        l.append('Depth: {d}'.format(d=self.depth))
        if self.verbose >= 1:
            if self.generated is True:
                l.append('Paths to dumps: {d}'.format(d=self.dumps))
                l.append('Memory Size: {s} MB'.format(s=(getsizeof(self.lcs) + getsizeof(self.classification)) * 1e-6))
                l.append('Item Reference: {r}'.format(r=self.item_ref))
        if self.verbose >= 2:
            if self.generated is True:
                l.append('Stored Data: {d}'.format(d=self.lcs))
                l.append('Classification Array: {c}'.format(c=self.classification))
            if self.built is True:
                l.append('Functions: {f}'.format(f=self.f))
        out = '\n'.join(l)
        return out

    def get_ft(self, n=0, s=300, state_change=False):
        Time, Flux, Classification, o = self.__get_lc__(n, state_change=state_change)
        FT = Gen_FT(Time, Flux, NyApprox(Time), s)
        return FT['Freq'], FT['Amp'], Classification, n

    def xget_ft(self, stop=None, s=300):
        if stop is None:
            for i in range(self.size):
                yield self.get_ft(n=i, s=s)
        else:
            for i in range(stop):
                yield self.get_ft(n=i, s=s)

    def batch_get(self, batch_size=10, ft=False, s=None, mem_size=1e9):
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
        if ft is True and s is None:
            s = 300
        if batch_size == 'mem_size':
            if ft is False:
                mem_use_single = getsizeof(self.lcs[0])
                batch_size = int(mem_size / mem_use_single)
            else:
                mem_use_single = getsizeof(self.get_ft(s=s))
                batch_size = int(mem_size / mem_use_single)
        if ft is False:
            for i in range(int(self.size / batch_size)):
                yield self.__batch_get_lc__(start=i * batch_size,
                                            stop=(i * batch_size) + batch_size,
                                            mem_size=mem_size)
        else:
            for i in range(int(self.size / batch_size)):
                yield self.__batch_get_ft__(start = i * batch_size,
                                            stop=(i * batch_size) + batch_size,
                                            s=s, mem_size=mem_size)

    def __batch_get_ft__(self, start=0, mem_size = 1e9, step=1,
                         stop=None, s=300):
        if stop is None:
            stop = self.size
        mem_use_single = getsizeof(self.get_ft(s=s))
        num = int(mem_size / mem_use_single)
        if stop < start + (num * step):
            num = stop
        else:
            num *= step
            num += start
        out_freq = list()
        out_amp = list()
        out_class = list()
        out_number = list()
        for i in range(start, num, step):
            Freq, Amp, Class, Number = self.get_ft(n=i, s=s, state_change=True)
            out_freq.append(Freq)
            out_amp.append(Amp)
            out_class.append(Class)
            out_number.append(Number)
        return out_freq, out_amp, out_class, out_number

    def __batch_get_lc__(self, start=0, mem_size=1e9, step=1,
                         stop=None):
        if stop is None:
            stop = self.size
        mem_use_single = getsizeof(self.lcs[0])
        num = int(mem_size / mem_use_single)
        if stop < start + (num * step):
            num = stop
        else:
            num *= step
            num += start
        out_time = list()
        out_flux = list()
        out_class = list()
        out_number = list()
        for j in range(start, num , step):
            Time, Flux, Class, Number = self.__get_lc__(n=j, state_change=True)
            out_time.append(Time)
            out_flux.append(Flux)
            out_class.append(Class)
            out_number.append(Number)
        j = 0
        return out_time, out_flux, out_class, out_number

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.__get_lc__(n=key)
        elif isinstance(key, slice):
            tup_cut = key.indices(len(self))
            return self.__batch_get_lc__(start=tup_cut[0],
                                         stop=tup_cut[1],
                                         step=tup_cut[2])
        else:
            raise TypeError("index must be int or slice")

    def __len__(self):
        return self.size

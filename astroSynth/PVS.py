from .SDM import *
from tqdm import tqdm
from sys import getsizeof
from tempfile import TemporaryFile
import os
import shutil
import time
import pickle
import math
from warnings import warn

"""
PVS (Pulsating Variable Star)- Class

Description:
    Class to generate, load, acess, store, and do very basic analysis
        Synthetically generated Light curves

"""

class PVS:
    def __init__(self, Number=100, noise_range=[0.1, 1.1], vmod=True,
                 f=lambda x: np.sin(x), numpoints=100, mag=10,
                 verbose=0, name=None, dpbar=False, lpbar=True, ftemp=False,
                 match_phase=False, single_object=False, T0=0):
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
            mag: The magnitude range stars to generate light curves 
                       from (float) 
            verbose: the verbosity which with to use when representing the 
                     object (0 - default, 1 - add dump info, 2 - add stored 
                     data)
            name: Name of object to use as directory name when saving object
            dpbar: Diable progress bars class wide (bool)
            lpbar: leave progress bars after completion class wide (bool)
            ftemp: Turn the object into a fully temporary object (bool)
            match_phase: Match the pases of all observations based on 
                         intervisit time (bool)
            single_object: Generate one data set with multiple observations
                           for a single object (bool)
        Returnes:
            Fully formed PVS() type object, ready to build-generate or to 
            load data
        """
        self.size = Number
        self.noise_range = noise_range
        self.depth = numpoints
        self.mag = mag
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
        self.max_amp = 0.1
        self.dpbar = dpbar
        self.lpbar = lpbar
        self.ftemp = ftemp
        self.match_phase = match_phase
        self.phasing = dict()
        self.dumps_are_temp = True
        self.single = single_object
        self.T0 = T0
        self.current = 0
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

        if self.match_phase is True:
            self.ivist = self.__list_check__(ivist)
        else:
            self.ivist = None

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
        print('Version 0.6.1.5 Development')

    def __nu_0__(self, n, l):
        raise NotImplementedError

    def __rot_split_freq__(self, l, n, p_rot):
        nu_0 = self.__nu_0__(n, l)
        ms = np.arange(-l, l, 1)
        freqs = list()
        for m in ms:
            nu = nu_0 + (m/p_rot) * (1 - (1 / (l * (l + 1))))
            freqs.append(nu)
        return freqs


    def __build_single__(self, phase_range=[0, np.pi], amp_range=[0, 1],
                         freq_range=[1e-7, 1], L_range=[1, 3]):
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
        self.kwargs[0] = kwargs
        self.f = lambda x, d: self.__mode_addition__(x, **d)

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
        if self.match_phase is True:
            break_size = np.random.uniform(self.ivist[0],
                                           self.ivist[1],
                                           self.size)
        if self.vmod is True:
            for i in tqdm(range(self.size), desc='Building Light Curve Functional Form',
                          leave=self.lpbar, disable=self.dpbar):
                kwargs = dict()
                kwargs['num'] = np.random.randint(L_range[0],
                                                  L_range[1] + 1)
                if self.match_phase is True:
                    if i == 0:
                        kwargs['phase'] = np.random.uniform(phase_range[0],
                                                            phase_range[1],
                                                            kwargs['num'])
                    else:
                        cont_phase = list()
                        for p, f in zip(prev_phasing, prev_frequency):
                            cont_phase.append(p + np.pi * math.modf((break_size[i] * f))[0])
                        kwargs['phase'] = cont_phase
                else:
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
                # prev_phasing = kwargs['phase']
                # prev_frequency = kwargs['freq']
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

    def __list_check__(self, l):
        if isinstance(l, list):
            return l
        else:
            return [l, l]

    def __params_2_list__(self, phase_values, amp_values,
                        freq_values, L_values):
        phase = self.__list_check__(phase_values)
        amp = self.__list_check__(amp_values)
        freq = self.__list_check__(freq_values)
        L = self.__list_check__(L_values)

        return phase, amp, freq, L

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
        self.max_amp = amp_range[1]
        phase_range, amp_range, freq_range, L_range = self.__params_2_list__(phase_values=phase_range,
                                                                             freq_values=freq_range,
                                                                             amp_values=amp_range,
                                                                             L_values=L_range)
        # self._seed_generation_(seed)
        if self.single is False:
            self.__build_func__(phase_range=phase_range, amp_range=amp_range,
                                freq_range=freq_range, L_range=L_range)
        else:
            self.__build_single__(phase_range=phase_range, amp_range=amp_range,
                                  freq_range=freq_range, L_range=L_range)
        self.built = True

    def __dump_data__(self, src, last_dump=0, dump_num=0):
        """
        description:
            Internal Function which handels passing data from memory to disk
                during the data generation process
        Params:
            self: PVS() object
            src: list to write to disk
            last_dump: The index at the start of the last dump
            dump_num: The index of the dump currently happening
        Returns:
            N/A
        Post-State:
            self.dumps_are_temp is set to true
            self.dumps[dump_num] is filled with the src list
            self.class_dumps[dump_bnum] is filled with the classification array for that dump
            self.classifications is reset to the zero array 
        """
        self.dumps_are_temp = True
        self.dumps[dump_num] = TemporaryFile()
        self.class_dumps[dump_num] = TemporaryFile()
        self.item_ref[dump_num] = [last_dump, len(src) + last_dump]
        np.save(self.dumps[dump_num], np.array(src))
        np.save(self.class_dumps[dump_num], self.classification)
        self.classification = np.zeros((0))

    def __pick_pulsator__(self, pfrac=0.1):
        """
        description:
            given some pulsation fraction (decimal probability) will return a 1
                1 with a pfrac * 100 likely hood and a 0 with a (pfrac * 100) - 100
                likelyhood
        Params:
            pfrac: pulsation fraction, float, deceimal probability that a 1 will be returnes
        Returns:
            pulsator: 1 or 0
        Post-State:
            self.classification has 1 or 0 appened to it
        """
        rand_pick = np.random.uniform(0, 10)
        if rand_pick < pfrac * 10:
            pulsator = True
            self.classification = np.append(self.classification, 1)
        else:
            pulsator = False
            self.classification = np.append(self.classification, 0)
        return pulsator

    def __generate_multi__(self, pfrac=0.1, exposure_time=30, af=lambda x:0):
        """
        description:
            generation function for generating non continuous epherarities
        Params:
            pfrac: the pulsation fraction of returned objects [float]
            exposure_time: the time in seconds between the center of one exposre
                            and the center of the next exposure [float]
        Returns:
            N/A
        Pre-State:
            Light Curves Non-generates
        Post-State:
            Light Curves Generated and accessable to the user
        """
        dump_num = 0
        last_dump = 0
        list_lcs = list()
        obs_time = self.depth * exposure_time
        for i in tqdm(range(self.size), desc='Geneating Light Curves',
                      leave=self.lpbar, disable=self.dpbar):
            pulsator = self.__pick_pulsator__(pfrac=pfrac)
            if self.vmod is True:
                tlc = Make_Syth_LCs(f=lambda x: self.f[i](x, self.kwargs[i]), pulsator=pulsator,
                                    numpoints=self.depth,
                                    noise_range=self.noise_range, start_time=self.T0,
                                    end_time=self.T0 + obs_time, magnitude=self.mag,
                                    af=af)
            else:
                tlc = Make_Syth_LCs(f=self.f, pulsator=pulsator,
                                    numpoints=self.depth,
                                    noise_range=self.noise_range, start_time=self.T0,
                                    end_time=self.T0 + obs_time, magnitude=self.mag,
                                    af=af)
            list_lcs.append(tlc)
            if getsizeof(list_lcs) > 1e5:
                self.__dump_data__(list_lcs, last_dump=last_dump, dump_num=dump_num)
                dump_num += 1
                last_dump = i
                list_lcs = list()
            self.item_ref[-1] = [last_dump + 1, len(list_lcs) + last_dump]
        self.lcs = np.array(list_lcs)
        self.temp_file = True

    def __generate_single__(self, visit_range=[1, 10], visit_size_range=[10, 100],
                            pfrac=0.1, exposure_time=30, break_size_range=[5, 25],
                            etime_units=u.second, btime_units=u.day, vtime_units=u.hour,
                            af=lambda x: 0):
        """
        description:
            Generation routine for generating a light curves based on a shared epherities
        Params:
            Visit_range: integer list defining the minimum number of visits and the maximum
                         number of visits
            Visit_size_range: float list defining the minimum length of a visit and 
                              The maximum size of a visit
            pfrac: Float defining the likelyhood that a target will be a pulsator
            exposure_time: the time in seconds between the center of one exposre
                            and the center of the next exposure [float]
            break_size_range: float list defining the minimum and maximum length of breaks
                              between visits
            etime_units: The unit (astroPy.units) defining how the expoure time is defined
            btime_units: the unit (astroPy.units) defining how the break time is defined
            vtime_units: the unit (astroPy.units) defining how the visit time unit is defined 
        """
        pulsator = self.__pick_pulsator__(pfrac=pfrac)
        if pulsator:
            classification = 1
        else:
            classification = 0
        obs_time = self.depth * exposure_time
        tlc = Make_Syth_LCs(f=lambda x: self.f(x, self.kwargs[0]), pulsator=pulsator,
                            numpoints=self.depth, noise_range=self.noise_range,
                            start_time=self.T0, end_time=self.T0 + obs_time,
                            magnitude=self.mag,af=af)
        tlc = np.array(tlc).T
        times, fluxs, integration_time = Make_Visits(tlc, visit_range=visit_range,
                                                     visit_size_range=visit_size_range,
                                                     break_size_range=break_size_range,
                                                     exposure_time=exposure_time,
                                                     vtime_units=vtime_units,
                                                     btime_units=btime_units,
                                                     etime_units=etime_units,
                                                     time_col=1, flux_col=0,
                                                     pbar=self.dpbar)
        if len(times) == 1:
            self.lcs = np.array([[times[0], fluxs[0]]])
        else:
            self.lcs = np.array([fluxs, times]).T
        self.size = len(self.lcs)
        kwargs = self.kwargs[0]
        for i in range(self.size):
            self.kwargs[i] = kwargs
        for index, _ in enumerate(self.lcs):
            self.classification = np.append(self.classification, classification)

        self.item_ref[-1] = [0, len(self.lcs)]

    def generate(self, pfrac=0.1, vtime_units=u.hour,
                 btime_units=u.day, exposure_time=30,
                 visit_range=[1, 10], visit_size_range=[0.5, 2],
                 break_size_range=[10, 100], etime_units=u.second,
                 af=lambda x:0):
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
        self.generated = True
        if self.single is False:
            self.__generate_multi__(pfrac=pfrac, exposure_time=exposure_time,af=af)
        else:
            self.__generate_single__(pfrac=pfrac, exposure_time=exposure_time,
                                     visit_range=visit_range, visit_size_range=visit_size_range,
                                     break_size_range=break_size_range, vtime_units=vtime_units,
                                     btime_units=btime_units, etime_units=etime_units,af=af)

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
        if self.size != 1:
            file_num = -1
            base = 0
            if n == self.size - 1:
                base = int(self.item_ref[-1][0])
                file_num = -1
            else:
                for k in self.item_ref:
                    if int(self.item_ref[k][0]) <= n < int(self.item_ref[k][1]):
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
                return tlcs[n - base].T[1], tlcs[n - base].T[0], tclass[n - base], n, self.kwargs[n]
            else:
                return self.lcs[n - base].T[1], self.lcs[n - base].T[0], self.classification[n - base], n, self.kwargs[n]
        else:
            return self.lcs[0][0], self.lcs[0][1], self.classification[0], 0, self.kwargs[0]

    def xget_lc(self, stop=None, start=0):
        """
        description:
            iterator over the light curves in a PVS object
        Params:
            stop: the maximum index to iterate to before stoping
            start: the index to start iteration from
        Yeilds:
            Current light curve including
                Flux Array [list]
                Time Array [list]
                Classification as Pulsator or NOV [int]
                Absolute Index Within PVS() [int]
                parameters defining target [dict]
        """
        if stop is None:
            stop = self.size
        if stop > self.size:
            stop = self.size
        for i in range(start, stop):
            yield self.__get_lc__(n=i)

    def save(self, path=None, ftemp_override=False):
        """
        description:
            save the current PVS object to disk for latter use
        Params:
            path: the path to save the object to
            ftemp_override: allows the user to prevent overwriting of files when saving
                            In that case no files will be saved
        Returns:
            Path to the saved files
        Post-State:
            All data needed to recrate PVS object written to disk

        """
        try:
            assert self.generated is True
        except AssertionError as e:
            e.args += ('Light Curves have not been generated as of yet',
                       'have you run PVS.generate()?')
            raise

        if path is None:
            if self.name is not None:
                if self.ftemp is False or ftemp_override is True:
                    path = "{}/{}".format(os.getcwd(), self.name)
                    if os.path.exists(path):
                        shutil.rmtree(path)
                    os.mkdir(path)
                elif self.ftemp is True and ftemp_override is False:
                    path = "{}/.{}_temp".format(os.getcwd(), self.name)
                    if os.path.exists(path):
                        shutil.rmtree(path)
                    os.mkdir(path)
            else:
                if self.ftemp is False or ftemp_override is True:
                    path = os.getcwd()
                elif self.ftemp is True and ftemp_override is False:
                    path = "{}/.{}_temp".format(os.getcwd(), time.asctime().replace(' ', '_'))
                    if os.path.exists(path):
                        shutil.rmtree(path)
                    os.mkdir(path)
        for dump, cdump in zip(self.dumps, self.class_dumps):
            if dump != -1:
                if self.dumps_are_temp is True:
                    self.dumps[dump].seek(0)
                    self.class_dumps[cdump].seek(0)
                tlc = np.load(self.dumps[dump])
                tclass = np.load(self.class_dumps[cdump])
                if self.dumps_are_temp is True:
                    self.dumps[dump].seek(0, os.SEEK_END)
                    self.class_dumps[cdump].seek(0, os.SEEK_END)
                np.save("{}/LightCurve_{}.npy".format(path, dump), tlc)
                np.save("{}/LightCurve_Class_{}.npy".format(path, dump), tclass)
        if len(self.lcs) > 0:
            np.save("{}/LightCurve_{}.npy".format(path, -1), self.lcs)
            np.save("{}/LightCurve_Class_{}.npy".format(path, -1), self.classification)

        self._save_model_(path=path)
        self.dumps_are_temp = False
        return path

    @staticmethod
    def __pickle_object__(obj, filename):
        """
        description:
            General Helper function to pickle an object 
        Params:
            obj: object to be pickled
            filename: Name to call pickeled object on disk
        Post-State:
            object obj saved as a pickle on disk
        """
        with open(filename, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def __open_pickle_jar__(filename):
        """
        description:
            general helper function to unpickle and read object from
                disk
        Params:
            filename: the file to read from
        Returns:
            pobject: the object that was unpickeld
        """
        with open(filename, 'rb') as input:
            pobject = pickle.load(input)
        return pobject

    def _save_model_(self, path = None):
        """
        description:
            General helper function to save all the meta data assocuated with PVS() object
                to disk
        Params:
            path: the path to save to
        Post-state:
            All metadata about PVS stored to disk at path path
        """
        try:
            assert path is not None
        except AssertionError as e:
            e.args += ('No Save path set in _save_model_', 'Has path been set?')
            raise

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
            out.append('MAmp:{}'.format(self.max_amp))
            out.append('Magnitude:{}'.format(self.mag))
            out = '\n'.join(out)
            f.write(out)

        self.__pickle_object__(self.kwargs, '{}/pparams.pkl'.format(path))

    def load(self, directory='.', start=-1):
        """
        description:
            Loads a saved PVS object from disk and initlalizes it in such a way as to be
                useful to the user
        params:
            directory: the path to load from
            start: the first dump number to use (-1 will place the scanner at the end of the
                                                PVS object)
        """
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
                self.item_ref[int(i[0])] = [i[1], i[2]]
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
                elif i[0] == 'MAmp':
                    self.max_amp = float(i[1].rstrip())
                elif i[0] == 'Magnitude':
                    self.mag = float(i[1].rstrip())
        go = False
        try:
            assert "pparams.pkl" in os.listdir(directory)
            go = True
        except AssertionError as e:
            warn('Warning!, unable to locate pulsation parameters file',
                  'pparams.pkl missing in {}'.format(directory),
                  'pulsation parameters will be intialized to empty parameter matrix')
            self.kwargs = [{'num': None, 'amp':[None], 'freq':[None], 'phase':[None]}] * self.size

        if go is True:
            self.kwargs = self.__open_pickle_jar__("{}/pparams.pkl".format(directory))

        self.generated = True
        self.temp_file = False

    def __repr__(self):
        """
        description:
            generate a string representation of the PVS object
        Returns:
            string representation of the PVS object
        """
        l = list()
        l.append('Name: {n}'.format(n=self.name))
        l.append('Size: {s}'.format(s=self.size))
        l.append('Noise Range: {n}'.format(n=self.noise_range))
        l.append('Magnitude: {m}'.format(m=self.mag))
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

    def get_ft(self, n=0, s=300, state_change=False, power_spec=False):
        """
        description:
            retuens the Lomb-Scargle Periodigram of a spesific light curve in PVS
                object's memory
        Params:
            n: Index of light curve to take the LSP of
            s: Number of frequency bins ot use in the LSP
            state_change: Should PVS dump all current data in order to focus on the 
                        region where n light curve is or just grab that one
            power_spec: return a power spec or a amplitude normalized LSP
        Retuens:
            Freuqncy List
            Amplitude List
            Classification [int]
            Index [int]
            paramertes defining light curve [dict]
        """
        Time, Flux, Classification, o, pp = self.__get_lc__(n, state_change=state_change)
        try:
            FT = Gen_FT(Time, Normalize(Flux, df=False), NyApprox(Time), s, power_spec=power_spec)
        except ValueError as e:
            e.args += ('Error! Division By Zero Error', self.name)
            raise
        return FT['Freq'], FT['Amp'], Classification, n, pp

    def xget_ft(self, start=0, stop=None, s=300, power_spec=False,
                state_change=True):
        """
        description:
            iterator for LSPs associated with light curves in PVS
        Params:
            Start: Value to start iteration from
            Stop: Value to stop iteration at
            s: number of frequency bins to use in LSP
            power_spec: return a power spec or a amplitude normalized LSP
            state_change: Should PVS dump all current data in order to focus on the 
                        region where n light curve is or just grab that one
        Yeilds:
            For the current LSP:
                Freuqncy List
                Amplitude List
                Classification [int]
                Index [int]
                paramertes defining light curve [dict]
        """
        if stop is None:
            for i in range(start, self.size):
                yield self.get_ft(n=i, s=s, power_spec=power_spec,
                                  state_change=state_change)
        else:
            for i in range(start, stop):
                yield self.get_ft(n=i, s=s, power_spec=power_spec,
                                  state_change=state_change)

    def batch_get(self, batch_size=10, ft=False, s=None, mem_size=1e9):
        """
        description:
            Get batches of data either Light curves of LSPs
        """
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
        out_pparams = list()
        for i in range(start, num, step):
            Freq, Amp, Class, Number, pparams = self.get_ft(n=i, s=s, state_change=True)
            out_freq.append(Freq)
            out_amp.append(Amp)
            out_class.append(Class)
            out_number.append(Number)
            out_pparams.append(pparams)
        return out_freq, out_amp, out_class, out_number, out_pparams

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
        out_pparams = list()
        for j in range(start, num , step):
            Time, Flux, Class, Number, pparams = self.__get_lc__(n=j, state_change=True)
            out_time.append(Time)
            out_flux.append(Flux)
            out_class.append(Class)
            out_number.append(Number)
            out_pparams.append(pparams)
        j = 0
        return out_time, out_flux, out_class, out_number, out_pparams

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

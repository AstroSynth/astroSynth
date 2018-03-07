import numpy as np
import pandas as pd

import math
import random as r

import astropy.units as u
from astropy.stats import LombScargle

from scipy.signal import lombscargle
from scipy.interpolate import interp1d
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.stats import chisquare
from scipy.optimize import curve_fit

from tqdm import tqdm
from warnings import warn
# import numba
"""
Description:
    General Helper functions for light curve and fourier analysis
    Used heavily in the astroSynth Modules

"""

def compress_to_1(data):
    if max(data) != 1:
        return do_compress(data, min(data), max(data))
    else:
        return data

# @numba.jit(nopython=True)
def do_compress(data, old_min, old_max):
    f = lambda x, y, z:((x - y) / (z - y))
    out = np.array(data)
    out = np.apply_along_axis(lambda x: f(x, old_min, old_max), 0, out)
    return out

def Mag_2_Flux(m, F0):
    """
    Description:
        Converts Magnitude to flux based on some given zero point
    Args:
        m - Magnitude to be converted (float)
        F0 - 0 Point Magnitude (float)
    returns:
        Flux value (float)
    Raises:
        N/A
    """
    return np.exp(-0.9065295642 * m) * F0


def initialize_dict(data, values):
    """
    Description:
        intialiezed all values in a given list in a dict to 0
    Args:
        data - the dictionary to work on (dict)
        values - the keys to intialize to 0 (list)
    Returns:
        A dictionary with all keys in values set equal to 0
    Raises:
        N/A
    """
    for i, e in enumerate(values):
        try:
            temp = data[e]
        except KeyError:
            data[e] = 0
    return data

# @numba.jit(nopython=True)
def Normalize(frame, key_col='Flux', df=True):
    """
    Description:
        Mean Normalized a time string
    Args:
        frame - Data to normalize (DataFrame, list)
        key_col = if frame is a DataFrame then tells which element to look under (iterm refernece)
        df - if frame a dataframe or a list (bool)
    returns:
        a structure of type frame which has had its or the values in its key_coloum table mean normalized
    Raises:
        N/A
    """
    mean = np.mean(frame)
    new = []
    for i, e in enumerate(frame):
        new.append((e / mean) - 1)
    return new

# @numba.jit()
def NyApprox(serise, start=0):
    """
    Decription:
        Appriximate the Nyquist frequency of a given data set
    Args:
        serise - time string to have the Nyquist approximated (numeric 1D iteratable)
        start - first element in serise to approximate Nyquist by
    Returns:
        Approximate Nyquist frequency of the time string passed in (float)
    Raises:
        N/A
    """
    Ny = 0
    number = 0
    if len(serise < 15):
        T = (serise[-1]-serise[0])/len(serise)
        return 1.0/(2*T)
    else:
        for i, e in enumerate(serise[:10]):
            if i > 1:
                NyGuess = 1.0 / ((e - serise[(i + start) - 1]) * 2.0)
                if not math.isnan(NyGuess):
                    Ny += NyGuess
                    number += 1
        return Ny / float(number)

# @numba.jit()
def Gen_flsp(time, flux, NyFreq, s):
    if len(time) != 1:
        frequency = np.linspace(0, NyFreq, s)
        power = LombScargle(time, flux).power(frequency, method='fast')
        return{'Freq': frequency, 'Amp': power}
    else:
        return{'Freq': np.linspace(0, 1, s), 'Amp': np.ones(s)}

def Gen_fft(time, flux, NyFreq):
    N = len(flux)
    yf = fft(flux)
    xf = np.linspace(0, NyFreq, N//2)
    yf = 2.0/N * np.abs(yf[0:N//2])
    return {'Freq': xf, 'Amp': yf}

def Periodigram(x, y, frequency_range=[0.5, 1], samples=100):
    def sine_nv(x, f, phase, A):
        return A*np.sin((2*np.pi*f*x)+phase)
    sine = np.vectorize(sine_nv)
    frequencies = np.linspace(frequency_range[0], frequency_range[1], samples)
    chi2 = np.zeros(samples)
    for i, frequency in enumerate(frequencies):
        fit_func = lambda xf, phase, A: sine(xf, frequency, phase, A)
        fit, covar = curve_fit(fit_func, x, y)
        freq_fit = sine(x, frequency, *fit)
        ddof = (len(y)-2)**2
        chi2[i] = (chisquare(freq_fit, f_exp=y)[0])/ddof
    return frequencies, chi2

# @numba.jit()
def Gen_FT(time, flux, NyFreq, s, power_spec=False, periodigram=False):
    """
    Description:
        Generate a Fourier Transform (Lomb Scargle Periodigram techniacally) given a time 
        and value set. Can also generate a power spectrum.
    Args:
        time - Time string to pass in (numeric 1D iteratable)
        flux - value string to pass in (numeric 1D iteratable)
        NyFreq - Nyquist frequency of the time string (float)
        s - number of samples in frequency array (int)
        power_spec - generate a power spectrum of not (bool)
    returns:
        dicionay:
            'Freq' - Frequency array in Hz (float)
            'Amp' - Amplitude array for FT (float)
    Raises:
        HARD FAIL - Raised on exessive NaN values in Time Array
    """
    # return Gen_flsp(time, flux, NyFreq, s)
    # return Gen_fft(time, flux, NyFreq)
    timeSansNAN = []
    for checksum in time:
        if math.isnan(checksum) is False:
            timeSansNAN.append(checksum)

    # Power Spectrum
    if len(timeSansNAN) > 0:
        res = 1 / (max(timeSansNAN) - min(timeSansNAN))
    else:
        warn('NON FATAL ERROR (time Not Build Correctly) |'
              'ATTEMPTING RECOVERING WITH DEFAULT SETTINGS')
        try:
            res = 1.0 / 30.0
            warn('RECOVERY [OKAY]')
        except:
            warn('RECOVERY [FAILED] | HARD FAIL')
            return -1
    # samples = int(NyFreq/res) * 10
    samples = s
    f = np.linspace(res / 10, NyFreq, samples)

    xuse = np.asarray(time)
    yuse = np.asarray(flux)

    if not periodigram:
        try:
            pgram = lombscargle(xuse, yuse, f * 2 * np.pi)
        except ZeroDivisionError:
            warn('Zero division encountered in GEN_FT')
            pgram = np.linspace(0, 1, samples)
        normval = xuse.shape[0]
        if power_spec is False:
            pgramgraph = np.sqrt(4 * (pgram / normval))
        else:
            pgramgraph = pgram
        fgo = f
        return {'Freq': fgo.tolist(), 'Amp': pgramgraph.tolist()}
    else:
        frequency, chi2 = periodigram()


def Gen_Spect(data, break_size, samples, time_col='Time', flux_col='Flux',
              spread_UD=10, spred_LR=2, pbar=False):
    """
    Description:
        Generate a spectrogram (or Tailing FT) given some keyed data set
    Args:
        data - a keyed dataset (such as a DataFrame or dictionary) containing at the least
                time data and value data data
        break_size - size of individual input data (i.e. how many points to pass into FT 
                     Generation) (int)
        samples - length of frequency array (int)
        time_col - key for the element containg time data (item reference)
        flux_col - key for the element containg value data (item reference)
        spread_UD - how much each ft should be streched up and down in the final 
                    spectrogram (int)
        spread_LR - how much each ft should be streched left and right in the 
                    final specttrogram (int)
    Returns:
        a 2D spectrogram which can be viewed as an image (2D nd.float64 array)
    Raises:
        N/A

    """
    depth = 0
    for i in tqdm(data.axes[0], disable=pbar):
        check = len(data[i][time_col]) // break_size
        if check > 0:
            for j in range(check):
                depth += 1
    spect = np.zeros((depth, samples))
    count = 0
    for i in tqdm(data.axes[0], disable=pbar):
        check = len(data[i][time_col]) // break_size
        if check > 0:
            for j in range(check):
                bottom = j * break_size
                top = break_size * (j + 1)
                ft_temp = Gen_FT(data[i][time_col][bottom:top],
                                 Normalize(data[i],
                                 key_col=flux_col)[flux_col][bottom:top],
                                 NyApprox(data[i][time_col][bottom:top],
                                 start=bottom), samples)
                spect[count] = Normalize(ft_temp['Amp'], df=False)

                count += 1
    return np.repeat(np.repeat(spect.T, spread_UD, axis=1), spred_LR,
                     axis=0)


def Make_LC(noise_level=0, f=lambda x: np.sin(x), magnitude=10, numpoints=100,
            start_time=0, end_time=0, af=lambda x: 0):
    """
    Description:
        Builds a syntehtic light curve for a pulsating star given some functioanal form
    Args:
        noise_level - noise to introduce into the light curve (introduced via a 
                      normal noise disstribution) (float)
        f - function to introduce into light curve (function of one argument)
        mag - magnitude of the star (used in noise calculations) (float)
        numpoints - number of points to generate for the light curve (int)
        start_time - start time of the light curve (float)
        end_time - end time of the light curve (float)
        af - alias function, used to indriduced alias signals into *all* light curves (function)
    Returns:
        DataFrame of two colums:
            'Time' -  the time the light curve happened over
            'Flux' - the flux vaues for that light curve over that time
    Raises:
        N/A
    """
    key = np.linspace(start_time, end_time, numpoints)
    if end_time == 0:
        end_time = numpoints
    if noise_level != 0:
        temp_lc = np.random.normal(magnitude, noise_level, numpoints)
        signal = f(key)
        temp_lc += signal
        alias_signal = af(key)
        temp_lc += alias_signal
    else:
        temp_lc = f(key)
        alias_signal = af(key)
        temp_lc += alias_signal
    data = {'Time': key,
           'Flux': temp_lc}
    return pd.DataFrame(data=data)


def Insert_Break(data, break_size_range=[0.1, 10], break_period_range=[1, 25],
                 time_col='Time', Flux_col='Flux'):
    """
    Description:
        Insert breaks into a data set of a defined size to better 
        emulate observing patterns of telescopes.
    Args:
        data - keyed data to be split (keyd data like dict, DataFrame)
        break_size - size of breaks to introduce (float)
        break_period - time between breaks to introduce (float)
        break_size_randomizer - upper and lower additional limit to 
                                the break_size to introduce (float)
        break_period_randomizer - upper and lower additional limit 
                                  to the break_period to introduce 
                                  (float)
        time_col - Column key for Time (item reference)
        flux_col - Column key for Flux (item reference)
    Returns:
        data set with requested breaks introduced as a list of 
        discrete 2D dats sets
    Raises:
        N/A
    """
    break_size = np.random.uniform(break_size_range[1], break_size_range[0])
    break_period = np.random.uniform(break_period_range[0], break_period_range[1])
    key_index = data[time_col][0]
    in_break = False
    starts = [0]
    ends = []
    for i, e in enumerate(data[time_col]):
        if in_break is False and e - key_index >= break_period:
            key_index = e
            in_break = True
            ends.append(i)
            break_size = np.random.uniform(break_size_range[1], break_size_range[0])
            break_period = np.random.uniform(break_period_range[0], break_period_range[1])
        if in_break is True and e - key_index >= break_size:
            key_index = e
            in_break = False
            starts.append(i)
    if len(starts) != len(ends):
        ends.append(len(data[time_col]))
    visits = [None] * len(starts)
    times = [None] * len(starts)
    for k, (i, j) in enumerate(zip(starts, ends)):
        visits[k] = data[Flux_col][i:j]
        times[k] = data[time_col][i:j]
    return visits, times, starts, ends

def Make_Visits(data, visit_range=[0, 10], visit_size_range=[0.5, 2],
                break_size_range=[1, 10], exposure_time=30, etime_units=u.second,
                btime_units=u.day, vtime_units=u.hour, time_col=0, flux_col=0,pbar=False):
    unorm_break_size_range = [(x*btime_units).to(etime_units).value for x in break_size_range]
    unorm_visit_size_range = [(x*vtime_units).to(etime_units).value for x in visit_size_range]
    break_size_range = [int(x/exposure_time) for x in unorm_break_size_range]
    visit_size_range = [int(x/exposure_time) for x in unorm_visit_size_range]
    num_visits = np.random.randint(visit_range[0], visit_range[1])
    num_breaks = num_visits - 1

    visit_length = np.random.randint(visit_size_range[0], visit_size_range[1], num_visits)
    break_length = np.random.randint(break_size_range[0], break_size_range[1], num_visits)


    integration_time = sum(visit_length)
    values = list()
    times = list()
    prev = 0
    if num_breaks != 0:
        for visit, lbreak in tqdm(zip(visit_length, break_length), total=num_breaks, disable=pbar):
            if not visit+prev >= len(data[time_col])*exposure_time:
                values.append(data[flux_col][prev:prev+visit])
                times.append(data[time_col][prev:prev+visit])
                prev = prev+visit+(int(lbreak/exposure_time))
            else:
                values.append(data[flux_col][prev:])
                times.append(data[time_col][prev:])
    else:
        values.append(data[flux_col][:visit_length[0]])
        times.append(data[time_col][:visit_length[0]])

    return [x for x in times if len(x) is not 0], [x for x in values if len(x) is not 0], integration_time

def Make_Syth_LCs(noise_range=[0.1, 1.1], f=lambda x: np.sin(x),
                  pulsator=True, numpoints=100, start_time=0, end_time=0,
                  magnitude=10, af=lambda x: 0):
    """
    Description:
        More abstract caller for Make_lc() wich allows for non 
        pulsating lcs to be generated mor simply
    Args:
        noise_range - the possible range of noised to be passed 
        to make_lc (2 element list of floats)
        f - function to be introduced into light curve only 
        used if pulsator is Triue (function)
        pulsator - Shoulf the light curve be a pulsator (use f) 
        or stable (f=lambda x: 0) (bool)
        numpoints - how many points should the returned light 
        curve contain
    Returns:
        2D list of light curve
    Raises:
        N/A
    """
    noise = r.uniform(noise_range[0], noise_range[1])
    if pulsator is True:
        lcs = Make_LC(noise_level=noise, f=f, numpoints=numpoints,
                      start_time=start_time, end_time=end_time,
                      magnitude=magnitude, af=af)
    else:
        lcs = Make_LC(noise_level=noise, f=lambda x: 0,
                      numpoints=numpoints, start_time=start_time,
                      end_time=end_time, magnitude=magnitude, af=af)
    return lcs.as_matrix().tolist()


def Make_Transit(data, phase=True, noise=False, noise_level=0):
    """
    EXPERIMENTAL
    Desctiption:
        Built a transiet light curve using the python mobule eb
        This method attempts to wrap eb in a more pythonic interface
    Args:
        data - data dictionary to be used in eb calls (dict)
        phase - should the retunes be in phase space or time space (bool)
                ** Currently only phase space is working **
        noise - Should noise be introcued (bool)
        noise_level - amount of noise to introduce into the light curve
    Returns:
        Dataframe of the light curve:
            Phase / **time** - x array
            flux - y array
    Raises:
        N/A
    """
    import eb
    keys = ['J', 'RASUM', 'RR', 'COSI', 'Q', 'KTOT', 'LDLIN1', 'LDNON1', 'GD1',
            'REFL1','ROT1', 'FSPOT1', 'OOE1O', 'OOE11A', 'OOE11B',
            'LDLIN2','LDNON2', 'GD2', 'REFL2', 'ECOSW', 'ESINW',
            'LOGG1', 'LOGG2', 'P']
    data = initialize_dict(data, keys)
    parm = np.zeros(eb.NPAR, dtype=np.double)
    parm[eb.PAR_J]      =  data['J']
    parm[eb.PAR_RASUM]  =  data['RASUM']
    parm[eb.PAR_RR]     =  data['RR']
    parm[eb.PAR_COSI]   =  data['COSI']
    parm[eb.PAR_Q]      =  data['Q']
    parm[eb.PAR_CLTT]   =  1000.0 * data['KTOT'] / eb.LIGHT
    parm[eb.PAR_LDLIN1] =  data['LDLIN1']
    parm[eb.PAR_LDNON1] =  data['LDNON1']
    parm[eb.PAR_GD1]    =  data['GD1']
    parm[eb.PAR_REFL1]  =  data['REFL1']
    parm[eb.PAR_ROT1]   =  data['ROT1']
    parm[eb.PAR_FSPOT1] =  data['FSPOT1']
    parm[eb.PAR_OOE1O]  =  data['OOE1O']  # Those are Os not zeros
    parm[eb.PAR_OOE11A] =  data['OOE11A']
    parm[eb.PAR_OOE11B] =  data['OOE11B']
    parm[eb.PAR_LDLIN2] =  data['LDLIN2']
    parm[eb.PAR_LDNON2] =  data['LDNON2']
    parm[eb.PAR_GD2]    =  data['GD2']
    parm[eb.PAR_REFL2]  =  data['REFL2']
    parm[eb.PAR_ECOSW]  =  data['ECOSW']
    parm[eb.PAR_ESINW]  =  data['ESINW']
    parm[eb.PAR_P]      =  data['P']
    parm[eb.PAR_T0]     =  data['TO']
    parm[eb.PAR_LOGG1]  =  data['LOGG1']
    parm[eb.PAR_LOGG2]  =  data['LOGG2']

    (ps, pe, ss, se) = eb.phicont(parm)
    if ps > 0.5:
        ps -= 1.0

    pdur = pe - ps
    sdur = se - ss
    if pdur > sdur:
        mdur = pdur
    else:
        mdur = sdur

    pa = 0.5 * (ps + pe)
    sa = 0.5 * (ss + se)

    phi = np.empty([3, data['N']], dtype=np.double)
    phi[0] = np.linspace(pa - mdur, pa + mdur, phi.shape[1])
    phi[1] = np.linspace(sa - mdur, sa + mdur, phi.shape[1])
    phi[2] = np.linspace(data['Pstart'], data['Pend'], phi.shape[1])

    typ = np.empty_like(phi, dtype=np.uint8)
    typ.fill(eb.OBS_MAG)

    y = eb.model(parm, phi, typ, eb.FLAG_PHI)
    if noise is True:
        Flux = np.random.normal(y[2], noise_level, data['N'])
    else:
        Flux = y[2]
    data = {'Phase': phi[2], 'Flux': Flux}
    return pd.DataFrame(data=data)

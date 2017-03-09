import numpy as np
import pandas as pd
from scipy.signal import lombscargle
import math
import random as r
from tqdm import tqdm
"""
Description:
    General Helper functions for light curve and fourier analysis
    Used heavily in the astroSynth Modules

"""


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
    if df is True:
        mean = frame[key_col].mean()
        new = []
        for i, e in enumerate(frame[key_col]):
            new.append((e / mean) - 1)
            # new.append(e-mean)
        new_df = frame.drop(key_col, 1)
        new_df[key_col] = pd.Series(new, index=new_df.index)
        return new_df
    else:
        mean = np.mean(frame)
        new = []
        for i, e in enumerate(frame):
            new.append((e / mean) - 1)
        return new


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
    for i, e in enumerate(serise[:10]):
        if i > 1:
            NyGuess = 1.0 / ((e - serise[(i + start) - 1]) * 2.0)
            # print "NyGuess is {}".format(NyGuess)
            if not math.isnan(NyGuess):
                Ny += NyGuess
                number += 1
    return Ny / float(number)


def Gen_FT(time, flux, NyFreq, s, power_spec=False):
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
    timeSansNAN = []
    for checksum in time:
        if math.isnan(checksum) is False:
            timeSansNAN.append(checksum)

    # Power Spectrum
    if len(timeSansNAN) > 0:
        res = 1 / (max(timeSansNAN) - min(timeSansNAN))
    else:
        print('NON FATAL ERROR (time Not Build Correctly) |'
              'ATTEMPTING RECOVERING WITH DEFAULT SETTINGS')
        try:
            res = 1.0 / 30.0
            print('RECOVERY [OKAY]')
        except:
            print('RECOVERY [FAILED] | HARD FAIL')
            return -1
    # samples = int(NyFreq/res) * 10
    samples = s
    f = np.linspace(res / 10, NyFreq, samples)
    xuse = np.asarray(time)
    yuse = np.asarray(flux)
    try:
        pgram = lombscargle(xuse, yuse, f * 2 * np.pi)
    except ZeroDivisionError:
        pgram = np.linspace(0, 1, 300)
    normval = xuse.shape[0]
    if power_spec is False:
        pgramgraph = np.sqrt(4 * (pgram / normval))
    else:
        pgramgraph = pgram
    fgo = f
    return {'Freq': fgo.tolist(), 'Amp': pgramgraph.tolist()}


def Gen_Spect(data, break_size, samples, time_col='Time', flux_col='Flux',
              spread_UD=10, spred_LR=2):
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
    for i in tqdm(data.axes[0]):
        check = len(data[i][time_col]) // break_size
        if check > 0:
            for j in range(check):
                depth += 1
    spect = np.zeros((depth, samples))
    count = 0
    for i in tqdm(data.axes[0]):
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


def Make_LC(noise_level=0, f=lambda x: np.sin(x), mag=10, numpoints=100,
            start_time=0, end_time=0):
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
    Returns:
        DataFrame of two colums:
            'Time' -  the time the light curve happened over
            'Flux' - the flux vaues for that light curve over that time
    Raises:
        N/A
    """
    key = np.linspace(0, numpoints, numpoints)
    if end_time == 0:
        end_time = numpoints
    if noise_level != 0:
        temp_lc = np.random.normal(Mag_2_Flux(mag, 10),
                                   noise_level, numpoints)
        signal = f(key)
        temp_lc += signal
    else:
        temp_lc = f(key)
    data = {'Time': np.linspace(start_time, end_time, numpoints),
           'Flux': temp_lc}
    return pd.DataFrame(data=data)


def Insert_Break(data, break_size=0, break_period=0,
                 break_size_randomizer=0.5, break_period_randomizer=0.5,
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
    if break_period != 0 and break_size != 0:
        bs_random = r.uniform(0, break_size_randomizer)
        bs_random -= break_size_randomizer / 2.0
        break_size += bs_random
        bp_random = r.uniform(0, break_period_randomizer)
        bp_random -= break_period_randomizer / 2.0
        break_period += bp_random
        key_index = data[time_col][0]
        in_break = False
        starts = [0]
        ends = []
        for i, e in enumerate(data[time_col]):
            if in_break is False and e - key_index >= break_period:
                key_index = e
                in_break = True
                ends.append(i)
            if in_break is True and e - key_index >= break_size:
                key_index = e
                in_break = False
                starts.append(i)
        if len(starts) != len(ends):
            ends.append(len(data[time_col]))
        visits = [None] * len(starts)
        for k, (i, j) in enumerate(zip(starts, ends)):
            visits[k] = data.iloc[i:j]
        return visits
    else:
        return [data]


def Make_Syth_LCs(noise_range=[0.1, 1.1], f=lambda x: np.sin(x),
                  pulsator=True, numpoints=100):
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
        lcs = Make_LC(noise_level=noise * 2.5, f=f, numpoints=numpoints)
    else:
        lcs = Make_LC(noise_level=noise * 2.5, f=lambda x: 0,
                      numpoints=numpoints)
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

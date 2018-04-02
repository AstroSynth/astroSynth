import numpy as np
import pandas as pd

from vectorpy import vector

from scipy.interpolate import interp1d

import astropy.coordinates as coord
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from tqdm import tqdm

class star:
    def __init__(self, mag, ra, dec, prefix='Synth'):
        self.mag = mag
        self.ra = ra % 360
        self.dec = dec
        self.prefix = prefix
        self.t = 0
        self.name = "{}_{:0.2f}_{:0.2f}".format(self.prefix, self.ra, self.dec)
        
    def light(self, t):
        outupt = np.sin(t + self.t)
        outupt += np.random.normal(scale=1.0, size=len(t))
        self.t += t[-1]
        return outupt

    def __repr__(self):
        out_str = list()
        out_str.append('Name: {}'.format(self.name))
        out_str.append('RA: {:0.3f}, Dec: {:0.3f}'.format(self.ra, self.dec))
        out_str.append('Magnitude: {:0.2f}'.format(self.mag))
        return '\n'.join(out_str)

class celestial_sphere:
    def __init__(self, number_stars, posfile):
        self.number_stars = number_stars
        self.stars = list()
        self.target_list = np.zeros((number_stars, 4))
        self.posfile = posfile
        self.populate(self.posfile)
        
    def populate(self, posfile):
        df = pd.read_csv(posfile, nrows=self.number_stars)
        for i in tqdm(df.T, total=self.number_stars):
            self.target_list[i] = np.array([i, df.iloc[i]['MAG'], df.iloc[i]['RA'], df.iloc[i]['DEC']])
            self.stars.append(star(self.target_list[i][1], self.target_list[i][2], self.target_list[i][3]))
            
    def get_field(self, center, FOV):
        field = list()
        max_dec = center[0] + FOV/2
        min_dec = center[0] - FOV/2
        max_ra = center[1] + FOV/2
        min_ra = center[1] - FOV/2
        for star in self.stars:
            if (min_ra <= star.ra <= max_ra) and (min_dec <= star.dec <= max_dec):
                field.append(star)
        return field

    def __repr__(self):
        out_str = list()
        out_str.append('Celestial Sphere')
        out_str.append('Number of Stars: {}'.format(self.number_stars))
        out_str.append('Populated From: {}'.format(self.posfile))
        return '\n'.join(out_str)

    def __get__(self, key):
        assert isinstance(key, int)
        return self.stars[key]

class planet:
    def __init__(self, rotperiod, sphere, R, orbperiod=np.pi*1e7, A=149.60e9):
        self.rotperiod = rotperiod
        self.orbperiod = orbperiod
        self.sphere = sphere
        self.R = R
        self.raLonOffset = 0 # degrees
        self.orbitdeg = 0 # degrees
        self.V = 2*np.pi*self.R/(self.rotperiod)
        self.rotperiodMap = interp1d([0, self.rotperiod], [0, 360])
        self.orbperiodMap = interp1d([0, self.orbperiod], [0, 360])
        self.A = A
        
    def can_observe(self, alt, lat, lon, ra, dec, outputPlot=False):
        telR = alt+self.R
        distance_2_star = 10*self.A
        telescope = vector(telR*np.cos((lat)*0.0174533)*np.cos((lon+self.raLonOffset)*0.0174533),
                      telR*np.cos((lat)*0.0174533)*np.sin((lon+self.raLonOffset)*0.0174533),
                      telR*np.sin((lat)*0.0174533))
        s = vector(distance_2_star*np.cos(dec*0.0174533)*np.cos(ra*0.0174533),
                      distance_2_star*np.cos(dec*0.0174533)*np.sin(ra*0.0174533),
                      distance_2_star*np.sin(dec*0.0174533))

        sun = vector(np.cos(self.orbitdeg * 0.0174533), np.sin(self.orbitdeg * 0.0174533), 0)
        local_telescope = telescope
        telescope = telescope + self.A*sun
        pointing = (s-telescope)/abs((s-telescope))
        sunpointing = np.pi - np.arccos((sun.dot(pointing))/(abs(sun)*abs(pointing)))
        tooNearSun = sunpointing <= 0.174533 # 10 degrees away from the sun no observations
        tooNearHorizon = np.pi - np.arccos((local_telescope.dot(pointing))/(abs(local_telescope)*abs(pointing)))
        tooNearHorizon = tooNearHorizon < np.pi/2
        
        return (not tooNearSun) and (not tooNearHorizon)
    
    def rotate(self, mc):
        self.raLonOffset = self.rotperiodMap(mc%self.rotperiod)
        
    def orbit(self, mc):
        self.orbitdeg = self.orbperiodMap(mc%self.orbperiod)

    def get_field(self, center, FOV, lat, lon, alt):
        if self.can_observe(alt, lat, lon, center[1], center[0]):
            return self.sphere.get_field(center, FOV)
        else:
            return list()

    def __repr__(self):
        out_str = list()
        out_str.append('Planet')
        out_str.append('Radius: {}'.format(self.R))
        out_str.append('Rotational Period: {}'.format(self.rotperiod))
        out_str.append('Orbital Period: {}'.format(self.orbperiod))
        out_str.append('Stars Visible: {}'.format(self.sphere.number_stars))
        return '\n'.join(out_str)

class telescope:
    def __init__(self, lat, lon, FOV, planet, alt=90, azm=0, elv=2000, minslew=0, starttime = '2000-01-01 12:00:00'):
        self.lat = lat
        self.lon = lon
        self.FOV = FOV
        self.alt = alt
        self.azm = azm % 360
        self.planet = planet
        self.elv = elv
        self.DB = LCDB()
        self.slewrate = 13 # seconds/degree (approximate PTF slew rate for the 5 day experiment)
        self.up_time = 0.05 # months
        self.up_time *= 2.628e+6
        self.cycle_time = 0.1 * 2.628e+6
        self.last_off = 0
        self.on = True
        self.tbs = 0
        self.start_obs = 20 * 3600
        self.end_obs = 6 * 3600
        self.day_length = 24 * 3600
        self.minslew = minslew
        self.loc = EarthLocation(lat = lat*u.deg, lon = lon*u.deg, height=elv*u.m)
        self.start_time = Time(starttime)
        self.master_clock = Time(starttime)
        self.AltAz = SkyCoord(alt = self.alt*u.degree, az = self.azm*u.degree, obstime = self.master_clock, frame = 'altaz', location = self.loc)
        icrscoors = self.AltAz.icrs
        self.pointing = [icrscoors.icrs.dec.deg, icrscoors.icrs.ra.deg]
        
    def kill_time(self, dt):
        self.master_clock += dt*u.second
        self.time_update()
        
    def time_update(self):
        dsec = (self.master_clock-self.start_time).sec
        self.planet.rotate(dsec)
        self.planet.orbit(dsec)
        self.AltAz = SkyCoord(alt = self.alt*u.degree, az = self.azm*u.degree, obstime = self.master_clock, frame = 'altaz', location = self.loc)
        if self.on is True and dsec - self.last_off >= self.up_time:
            self.tbs = dsec
            self.on = False
        if self.on is False and dsec - self.tbs >= self.cycle_time-self.up_time:
            self.on = True
            self.last_off = dsec

    def slew2(self, alt=None, azm=None):
        if alt != None and not self.minslew <= alt <= 90:
            raise ValueError('Unable to slew telescope to {}, bounds are 0 -> 90 degrees'.format(alt))
        if alt is None:
            alt = self.alt
        if azm is None:
            azm = self.azm
        dalt = abs(alt-self.alt)
        dazm = abs(azm-self.azm) % 360
        self.master_clock += (self.slewrate*np.sqrt(dalt**2+dazm**2))*u.second
        self.alt = alt
        self.azm = azm % 360
        self.time_update()
        
    def slewBy(self, dalt=0, dazm=0):
        self.master_clock += (self.slewrate*np.sqrt(dalt**2+dazm**2))*u.second
        if not self.minslew <= self.alt + dalt <= 90:
            raise ValueError('Unable to slew telescope to {}, bounds are 0 -> 90 degrees'.format(self.alt + dalt))
        self.alt += dalt
        self.azm = (self.azm + dazm)%360
        self.time_update()
                
    def get_field(self, outputPlot=False):
        icrscoors = self.AltAz.icrs
        self.pointing = [icrscoors.icrs.dec.deg, icrscoors.icrs.ra.deg]
        return self.planet.get_field(self.pointing, self.FOV, self.lat, self.lon, self.elv, outputPlot=outputPlot)
    
    def basic_observe(self, field, nobs, exptime, n=0):
        dt = exptime*nobs
        elapsed_time = (self.master_clock-self.start_time).sec
        t = np.arange(elapsed_time, elapsed_time+(exptime*nobs), exptime)
        self.master_clock += dt*u.second
        self.time_update()
        if len(field) != 0 and self.on is True:
            return field[n], t, field[n].light(t)
        else:
            return None, None, None
    
    def observe(self, nobs, exptime, n=0):
        field = self.get_field()
        return self.basic_observe(field, nobs, exptime, n=n)
            
    def xobserver(self, nobs, exptime):
        field = self.get_field()
        for i, _ in enumerate(field):
            yield self.basic_observe(field, nobs, exptime, n=i)
            
    def recorde_field(self, nobs, exptime):
        for s, time, flux in self.xobserver(nobs, exptime):
            if s:
                self.DB.insert_data(s.name, time.tolist(), flux.tolist())

    def __repr__(self):
        out_str = list()
        out_str.append('Telescope')
        out_str.append('Latitude: {}'.format(self.lat))
        out_str.append('Longitude: {}'.format(self.lon))
        out_str.append('Elevation: {}'.format(self.elv))
        out_str.append('Pointing: Ra -> {:0.3f}, Dec -> {:0.3f}'.format(self.pointing[1], self.pointing[0]))
        out_str.append('Azm: {:0.2f}, Alt: {:0.2f}'.format(self.azm, self.alt))
        return '\n'.join(out_str)

class LCDB:
    def __init__(self, name='TestDB'):
        self.name = name
        self.starData = dict()
    
    def insert_data(self, name, time, flux):
        if name in self.starData:
            self.starData[name][0].extend(time)
            self.starData[name][1].extend(flux)
        else:
            self.starData[name] = list()
            self.starData[name].append(time)
            self.starData[name].append(flux)
            
    def retrive_data(self, name):
        return self.starData[name]
    
    def dump_data(self):
        return self.starData
    
    def records(self):
        return self.starData.keys()

if __name__ == '__main__':
    sphere = celestial_sphere(5000, '../Scrambled_RA_Dec_Mag.csv')
    earth = planet(86400, sphere, 6378100, orbperiod=25000)
    Scope = telescope(0, 0, 10, earth, elv=100)
    print(Scope)
    print('\n\n')
    print(earth)
    print('\n\n')
    print(sphere)
        
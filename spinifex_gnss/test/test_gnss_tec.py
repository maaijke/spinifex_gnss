from gnss_tec import get_electron_density_gnss
from spinifex.geometry import get_ipp_from_skycoord
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from lofarantpos.db import LofarAntennaDatabase
import astropy.units as u
import numpy as np

mydb = LofarAntennaDatabase()

def test_get_electron_density_gnss():
    stat_pos = EarthLocation.from_geocentric(*mydb.phase_centres['CS002LBA'], unit="m")
    source = SkyCoord.from_name("3C380")
    times = Time(60478.875, format = "mjd") + np.arange(12*4) * 5 * u.min 
    height_array = np.arange(100,1500,30) * u.km
    ipp = get_ipp_from_skycoord(source=source, loc=stat_pos, times=times, height_array=height_array)
    return get_electron_density_gnss(ipp)
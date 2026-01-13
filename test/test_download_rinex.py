from gnss_tec import select_gnss_stations
from download_gnss import download_rinex
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from spinifex.geometry import get_ipp_from_skycoord
import astropy.units as u
from datetime import date
import numpy as np
from lofarantpos.db import LofarAntennaDatabase
mydb = LofarAntennaDatabase()


def test_download_rinex():
    stat_pos = EarthLocation.from_geocentric(*mydb.phase_centres['CS002LBA'], unit="m")
    source = SkyCoord.from_name("3C380")
    times = Time(60478.875, format = "mjd") + np.arange(2) * 5 * u.min 
    height_array = np.arange(100,1500,30) * u.km
    ipp = get_ipp_from_skycoord(source=source, loc=stat_pos, times=times, height_array=height_array)
    gnss_list = select_gnss_stations(ipp.loc)
    gnss_file_list = download_rinex(date=ipp.times[0].to_datetime(), stations=gnss_list)
    return gnss_file_list
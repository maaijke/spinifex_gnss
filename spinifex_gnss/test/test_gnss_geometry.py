from gnss_geometry import get_sat_pos_object, get_sat_pos, get_stat_sat_ipp
from pathlib import Path
import glob
from astropy.time import Time
import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation

datapath = Path("./data/")

def test_get_sat_pos_object():
    sp3_files = [Path(i) for i in sorted(glob.glob(datapath.as_posix() +  "/*SP3.gz"))]
    sat_pos_object = get_sat_pos_object(sp3_files = sp3_files)
    return sat_pos_object


def test_get_sat_pos():
    sat_pos_object = test_get_sat_pos_object()
    times  = Time("2025-06-26T12:05:00") + np.arange(4) * 13 * u.min
    sat_pos = get_sat_pos(obs=sat_pos_object, times=times, sat_name="G02")
    return sat_pos

def test_get_stat_sat_ipp():

    satpos = test_get_sat_pos()
    gnsspos= EarthLocation.from_geodetic(6 * u.deg, 52 *u.deg, 0 *u.m)
    times = Time("2025-06-26T12:05:00") + np.arange(4) * 13 * u.min
    return get_stat_sat_ipp(satpos=satpos, gnsspos=gnsspos, times=times)
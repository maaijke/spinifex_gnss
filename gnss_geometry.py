import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation, ITRS, AltAz
from spinifex.geometry import IPP, get_ipp_from_altaz
from pathlib import Path
from scipy.interpolate import CubicSpline
import georinex as gr
import xarray
from typing import Any


def interpolate_satellite(sat_data, time_target, method: str = "cubicspline"):
    x_values = Time(sat_data.time).mjd
    target_x = time_target.mjd
    x, y, z = sat_data.position.values.T
    x[np.isnan(x)] = 0
    y[np.isnan(y)] =0
    z[np.isnan(z)] =0
    if method == "linear":
        x_interp = np.interp(target_x, x_values, x)
        y_interp = np.interp(target_x, x_values, y)
        z_interp = np.interp(target_x, x_values, z)
    else:
        spl = CubicSpline(x_values, x)
        x_interp = spl(target_x)
        spl = CubicSpline(x_values, y)
        y_interp = spl(target_x)
        spl = CubicSpline(x_values, z)
        z_interp = spl(target_x)
    return np.array((x_interp, y_interp, z_interp)).T * 1000 * u.m


def get_sat_pos(sp3_files, times:Time, sat_name: str):  # TODO: correct for time of transmission
    sp3s = [gr.load(i) for i in sp3_files[:3]]
    obs = xarray.merge(sp3s)
    return EarthLocation(
            *(interpolate_satellite(obs.sel(sv=sat_name), times).T)
        )


def get_azel_sat(satpos: EarthLocation, gnsspos: EarthLocation, times: Time):
    itrs_geo = satpos.itrs
    topo_itrs_repr = itrs_geo.cartesian.without_differentials() - gnsspos.itrs.cartesian
    itrs_topo = ITRS(topo_itrs_repr, obstime=times, location=gnsspos)
    aa = itrs_topo.transform_to(AltAz(obstime=times, location=gnsspos))
    return aa

def get_stat_sat_ipp(
    satpos: EarthLocation,
    gnsspos: EarthLocation,
    times: Time,
    height_array: u.Quantity = np.array([300]) * u.km,
):

    azel = get_azel_sat(satpos, gnsspos, times)
    return get_ipp_from_altaz(gnsspos, azel, height_array)


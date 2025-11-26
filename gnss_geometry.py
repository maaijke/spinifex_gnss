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
import subprocess


def interpolate_satellite(sat_data:xarray, time_target:Time, method: str = "cubicspline")->u.Quantity:
    """Interpolate satellite positions to requested times

    Parameters
    ----------
    sat_data : xarray
        array with ephemeris data from sp3 files
    time_target : Time
        times to interpolate to
    method : str, optional
        method of interpolation, ["linear","cubicspline"] by default "cubicspline"

    Returns
    -------
    u.Quantity
        interpolated ITRF positions
    """    
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


def get_sat_pos_object(sp3_files:list[Path]) -> xarray:
    # TODO: remove georinex dependency, parse textfiles directly
    """read ephemeris files

    Parameters
    ----------
    sp3_files : list[Path]
        list of files, must containt 3 days around the requested date

    Returns
    -------
    xarray
        array with sattelite ephermeris data
    """    
    sp3_unzipped = []
    for sp3 in sp3_files[:3]:
        if sp3.suffix == ".gz":
            subprocess.run(["gunzip", "-f", str(sp3)])
            sp3_unzipped.append(sp3.parent / sp3.stem)
        else:
            sp3_unzipped.append(sp3)
    print("unpacked sp3", sp3_unzipped, sp3_files)
    sp3s = [gr.load(i) for i in sp3_unzipped[:3]]
    obs = xarray.merge(sp3s)
    for sp3 in sp3_unzipped:
        print("gzipping back", sp3, str(sp3))
        subprocess.run(["gzip", "-f", str(sp3)])
    return obs

def get_sat_pos(obs:xarray, times:Time, sat_name:str) ->EarthLocation:
    """get satellite positions at requested time of specific satellite

    Parameters
    ----------
    obs : xarray
        ephemeris data
    times : Time
        times to interpolate to
    sat_name : str
        satellite prn

    Returns
    -------
    EarthLocation
        location of satellite
    """    
    return EarthLocation(
            *(interpolate_satellite(obs.sel(sv=sat_name), times).T)
        )


def get_azel_sat(satpos: EarthLocation, gnsspos: EarthLocation, times: Time) -> AltAz:
    """Calculate alt az of satellite

    Parameters
    ----------
    satpos : EarthLocation
        satellite position
    gnsspos : EarthLocation
        receiver positions
    times : Time
        times

    Returns
    -------
    AltAz
        azimuth and elevations at times
    """    
    itrs_geo = satpos.itrs
    topo_itrs_repr = itrs_geo.cartesian.without_differentials() - gnsspos.itrs.cartesian
    itrs_topo = ITRS(topo_itrs_repr, obstime=times, location=gnsspos)
    aa = itrs_topo.transform_to(AltAz(obstime=times, location=gnsspos))
    return aa

def get_stat_sat_ipp(
    satpos: EarthLocation,
    gnsspos: EarthLocation,
    times: Time,
    height_array: u.Quantity = np.array([350]) * u.km,
) -> IPP:
    """Get ionospheric pierce point of satellite receiever combination

    Parameters
    ----------
    satpos : EarthLocation
        satellite position
    gnsspos : EarthLocation
        receiver position
    times : Time
        times # note should be in gps time
    height_array : u.Quantity, optional
        altitudes of ionospheric piercepoints, by default np.array([350])*u.km

    Returns
    -------
    IPP
        _description_
    """
    azel = get_azel_sat(satpos, gnsspos, times)
    return get_ipp_from_altaz(gnsspos, azel, height_array)


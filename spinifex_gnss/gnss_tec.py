import numpy as np
from astropy.time import Time
import astropy.units as u

from astropy.coordinates import EarthLocation
from spinifex.geometry import IPP
from spinifex.times import get_unique_days, get_indexlist_unique_days
from spinifex.ionospheric.tec_data import ElectronDensity
from spinifex_gnss.download_gnss import download_dcb, download_rinex, download_satpos_files
from spinifex_gnss.parse_gnss import (
    parse_dcb_sinex,
    process_all_rinex_parallel,
)
from spinifex_gnss.proces_gnss_data import gnss_pos_dict, get_ipp_density
from spinifex_gnss.gnss_geometry import get_sat_pos_object
from typing import Any


MIN_DISTANCE_SELECT = 1000 * u.km
DEFAULT_IONO_HEIGHT = 450 * u.km

def get_min_distance(ipp: IPP, gnss_pos: EarthLocation):
    return (
        np.min(
            np.sqrt(
                (ipp.x.value - gnss_pos.x.value) ** 2
                + (ipp.y.value - gnss_pos.y.value) ** 2
                + (ipp.z.value - gnss_pos.z.value) ** 2
            )
        )
        * u.m
    )


def select_gnss_stations(ipp_location: EarthLocation):
    hidx = np.argmin(np.abs(ipp_location[0].height - DEFAULT_IONO_HEIGHT))
    ipp_earth = EarthLocation(
        lon=ipp_location[:, hidx].lon, lat=ipp_location[:, hidx].lat, height=0 * u.m
    )
    gnss_list = []
    for gnss, gnss_pos in gnss_pos_dict.items():
        if get_min_distance(ipp_earth, gnss_pos) < MIN_DISTANCE_SELECT:
            gnss_list.append(gnss)
    return gnss_list


def _select_times_from_ipp(ipp: IPP, indices: np.ndarray) -> IPP:
    return IPP(
        loc=ipp.loc[indices],
        times=ipp.times[indices],
        los=ipp.los[indices],
        airmass=ipp.airmass[indices],
        altaz=ipp.altaz[indices],
        station_loc=ipp.station_loc,
    )


def get_electron_density_gnss(ipp: IPP, select_constellations=None):
    unique_days = get_unique_days(ipp.times)
    unique_days_indices = get_indexlist_unique_days(unique_days, ipp.times)
    all_data = []
    for day, indices in zip(unique_days, unique_days_indices):
        selected_ipp = _select_times_from_ipp(ipp, indices)
        gnss_list = select_gnss_stations(selected_ipp.loc)
        dcb = parse_dcb_sinex(download_dcb(date=day.to_datetime())[0])
        gnss_file_list = sorted(download_rinex(date=day.to_datetime(), stations=gnss_list))
        st_list = sorted([i.name[:9] for i in gnss_file_list]) # the stations for which data was found
        gnss_file_list_next_day = sorted(download_rinex(date=(day + 1*u.day).to_datetime(), stations=st_list))
        st_list2 = sorted([i.name[:9] for i in gnss_file_list_next_day]) # the stations for which data was found
        if not st_list==st_list2:
            gnss_file_list = sorted([i for i in gnss_file_list if i.name[:9] in st_list2])
        gnss_file_list = [(i,j) for (i,j) in zip(gnss_file_list,gnss_file_list_next_day)]# pairs of files
        gnss_data_list = process_all_rinex_parallel(
            gnss_file_list, dcb=dcb, select_constellations=select_constellations
        )
        gnss_data_list = [i for i in gnss_data_list if i.is_valid]
        sp3_files = download_satpos_files(date=day.to_datetime())
        sat_pos_object = get_sat_pos_object(sp3_files=sp3_files[:3])
        # sat_clk, gnss_clk = parse_clk_data(sp3_files[-1])

        # electron_density  = interpolate_to_ipp(stec_gnss_data, ipp.loc[indices], ipp.times[indices])
        all_data.append(
            get_ipp_density(
                gnss_data_list=gnss_data_list,
                ipp_target=selected_ipp,
                dcb=dcb,
                sat_pos_object=sat_pos_object,
            )
        )
    return ElectronDensity(
        electron_density=np.concatenate([i.electron_density for i in all_data], axis=0),
        electron_density_error=np.concatenate(
            [i.electron_density_error for i in all_data], axis=0
        ),
    )

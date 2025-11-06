import numpy as np
from astropy.time import Time
import astropy.units as u

from astropy.coordinates import EarthLocation, GCRS
from spinifex.geometry import IPP
from spinifex.ionospheric import get_density_ionex_single_layer, tec_data
from spinifex.times import get_unique_days, get_indexlist_unique_days
from download_gnss import download_dcb, download_rinex, download_satpos_files
from parse_gnss import (
    parse_dcb_sinex,
    process_all_rinex_parallel,
    parse_clk_data,
    get_tec_gnss,
    GNSSTECData,
)
from gnss_geometry import get_sat_pos, get_stat_sat_ipp
from typing import Any

MIN_DISTANCE_SELECT = 600 * u.km
HEIGHT_ARRAY = np.arange(50, 2000, 10) * u.km

euref_station_file = "data/data_euref_pos.ssc2"
# TODO: get more gnss stations/databases
gnss_pos_dict = {}
with open(euref_station_file) as myf:
    for line in myf:
        pos = [float(i) for i in line.strip().split()[1:]]
        gnss_pos_dict[line[:9]] = EarthLocation.from_geocentric(*pos, unit=u.m)


def get_min_ipp_distance(ipp_ss: list[IPP], ipp_data: IPP):

    return np.min(
        np.linalg.norm(
            u.Quantity(ipp_ss.geocentric).to(u.m).value
            - u.Quantity(ipp_data.geocentric).to(u.m).value[..., np.newaxis],
            axis=0,
        ),
        axis=-1,
    )


def get_tec_sat_stat(ipp_s: IPP):  # needed for extra bias corrections
    tec = get_density_ionex_single_layer(ipp_s)
    return tec


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
    hidx = np.argmin(np.abs(ipp_location[0].height - 300 * u.km))
    ipp_earth = EarthLocation(
        lon=ipp_location[:, hidx].lon, lat=ipp_location[:, hidx].lat, height=0 * u.m
    )
    gnss_list = []
    for gnss, gnss_pos in gnss_pos_dict.items():
        if get_min_distance(ipp_earth, gnss_pos) < MIN_DISTANCE_SELECT:
            gnss_list.append(gnss)
    return gnss_list


def remove_gnss_bias_gim(
    stec_list: list[GNSSTECData], ipp_list: list[IPP]
) -> GNSSTECData:
    dtec = []
    default_options = tec_data.IonexOptions()
    for ipp,stec in zip(ipp_list,stec_list):
        select = ipp.altaz.alt.deg > 65
        h_idx = np.argmin(np.abs(ipp.loc[0].height.to(u.km).value - default_options.height.to(u.km).value))
        print(np.sum(select))
        if not np.sum(select):
            continue
        ipp_select = IPP(
            altaz=ipp.altaz[select],
            airmass=ipp.airmass[select],
            loc=ipp.loc[select],
            times=ipp.times[select],
            los=ipp.los[select],
            station_loc=ipp.station_loc,
        )
        gim_tec = get_density_ionex_single_layer(ipp=ipp_select)
        dtec.append(np.sum(gim_tec.electron_density,axis=1) - stec.tec_phase[select] / ipp.airmass[select,h_idx]) 
    #TODO : flag station if varaince of dtec is too large
    return np.mean(np.concatenatr(dtec))

def get_electron_density_gnss(ipp: IPP):
    unique_days = get_unique_days(ipp.times)
    unique_days_indices = get_indexlist_unique_days(unique_days, ipp.times)
    for day, indices in zip(unique_days, unique_days_indices):
        gnss_list = select_gnss_stations(ipp.loc[indices])
        dcb = parse_dcb_sinex(download_dcb(date=day.to_datetime())[0])
        gnss_file_list = download_rinex(date=day.to_datetime(), stations=gnss_list)
        gnss_data_list = process_all_rinex_parallel(
            gnss_file_list, times=ipp.times[indices], dcb=dcb
        )
        sp3_files = download_satpos_files(date=day.to_datetime())
        sat_clk, gnss_clk = parse_clk_data(sp3_files[-1])
        stec_list = {}
        # TODO: do this in parallel
        stec_gnss_data = {}
        for gnss_data in gnss_data_list:
            if gnss_data.is_valid:
                if gnss_data.station in gnss_clk.keys():
                    clock_correction = (
                        np.mean([i["bias"] for i in gnss_clk[gnss_data.station]]) * u.s
                    )
                else:
                    clock_correction = 0 * u.s
                stec_list[gnss_data.station] = get_tec_gnss(
                    gnss_data, dcb, clock_correction
                )
                stec_ipp_list = []
                for stec in stec_list[gnss_data.station]:
                    sat_pos = get_sat_pos(
                        sp3_files[:3], stec.times_of_transmission, stec.prn
                    )
                    clock_correction = (
                        np.mean([i["bias"] for i in sat_clk[str(stec.prn)]]) * u.s
                    )
                    sat_stat_ipp = get_stat_sat_ipp(
                        satpos=sat_pos,
                        gnsspos=gnss_pos_dict[gnss_data.station],
                        times=stec.times + clock_correction,
                        height_array=HEIGHT_ARRAY,
                    )
                    stec_ipp_list.append(sat_stat_ipp)
                correction = remove_gnss_bias_gim(stec_list=stec_list[gnss_data.station], ipp_list=stec_ipp_list[-1])
                stec_gnss_data[gnss_data.station] = (
                    stec_list[gnss_data.station],
                    stec_ipp_list,
                    correction
                )
        electron_density  = interpolate_to_ipp(stec_gnss_data, ipp.loc[indices], ipp.times[indices])
        return stec_gnss_data
    
    def interpolate_to_ipp(stec_data:dict[Any], ipp:EarthLocation, times:Time) :
        pass

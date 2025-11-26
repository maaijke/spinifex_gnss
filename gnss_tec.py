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
    GNSSData,
)
from gnss_geometry import get_sat_pos, get_stat_sat_ipp, get_sat_pos_object
from typing import Any
from concurrent.futures import as_completed, ProcessPoolExecutor

MIN_DISTANCE_SELECT = 1000 * u.km
HEIGHT_ARRAY = np.arange(50, 2000, 10) * u.km

euref_station_file = "data/data_euref_pos.ssc2"
gnss_station_file = "data/data_gnss_pos.txt"
# TODO: get more gnss stations/databases
gnss_pos_dict = {}
with open(gnss_station_file) as myf:
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
    stec_list: list[GNSSTECData], ipp_list: list[IPP], timestep:int=10
) -> GNSSTECData:
    dtec = []
    default_options = tec_data.IonexOptions()
    for ipp,stec in zip(ipp_list,stec_list):
        if type(ipp)==bool and not ipp:
            continue
        select = ipp.altaz.alt.deg[::timestep] > 65
        h_idx = np.argmin(np.abs(ipp.loc[0].height.to(u.km).value - default_options.height.to(u.km).value))
        if not np.sum(select):
            continue
        ipp_select = IPP(
            altaz=ipp.altaz[::timestep][select],
            airmass=ipp.airmass[::timestep][select],
            loc=ipp.loc[::timestep][select],
            times=ipp.times[::timestep][select],
            los=ipp.los[::timestep][select],
            station_loc=ipp.station_loc,
        )
        gim_tec = get_density_ionex_single_layer(ipp=ipp_select)
        dtec.append(np.sum(gim_tec.electron_density,axis=1) - stec.tec_phase[::timestep][select] / ipp_select.airmass[:,h_idx]) 
    #TODO : flag station if varaince of dtec is too large
    return np.nanmean(np.concatenate(dtec))

def interpolate_to_ipp(sat_stat_ipp, ipp_loc:EarthLocation, times:Time) :
    timeselect =np.argmin(np.abs(times.mjd -sat_stat_ipp.times.mjd[:,np.newaxis]), axis=0 )
    distance = np.linalg.norm(
            u.Quantity(sat_stat_ipp.loc[timeselect].geocentric).to(u.m).value
            - u.Quantity(ipp_loc.geocentric).to(u.m).value,
            axis=0,
        )
    return timeselect,distance,sat_stat_ipp.airmass[timeselect], sat_stat_ipp.altaz.alt.deg[timeselect]
    
def _get_stec_info(gnss_data:GNSSData, gnss_clk,dcb, sat_pos_object, sat_clk, ipp, indices):
    if gnss_data.station in gnss_clk.keys():
        clock_correction = (
            np.mean([i["bias"] for i in gnss_clk[gnss_data.station]]) * u.s
        )
    else:
        clock_correction = 0 * u.s
    stec_list = get_tec_gnss(
        gnss_data, dcb, clock_correction, timestep=1
    )
    stec_ipp_list = []
    distance_list = []
    airmass_list = []
    elevation_list = []
    stec_ipp = []
    for stec in stec_list:
        try:
            clock_correction = (
                np.mean([i["bias"] for i in sat_clk[str(stec.prn)]]) * u.s
            )
            sat_pos = get_sat_pos(
                sat_pos_object, stec.times_of_transmission - clock_correction, stec.prn
            )
            sat_stat_ipp = get_stat_sat_ipp(
                satpos=sat_pos,
                gnsspos=gnss_pos_dict[gnss_data.station],
                times=stec.times - clock_correction,
                height_array=ipp.loc[indices][0].height,
            )           
            stec_ipp_list.append(sat_stat_ipp)
            timeselect, distance, airmass, elevation = interpolate_to_ipp(sat_stat_ipp, ipp.loc[indices], ipp.times[indices])
            distance_list.append(distance)
            airmass_list.append(airmass)
            elevation_list.append(elevation)
            stec_ipp.append(stec.tec_phase[timeselect])
        except:
            stec_ipp_list.append(False)
            continue
    correction = remove_gnss_bias_gim(stec_list=stec_list, ipp_list=stec_ipp_list, timestep=10)

    #stec_ipp = [select_stec_times(i,timeselect) for i in stec_list]
    return (
        stec_ipp,
        distance_list,
        airmass_list,
        elevation_list,
        correction
    )

def select_stec_times(gnsstec:GNSSTECData, timeselect:list[int]):
    return GNSSTECData(
            tec_phase=gnsstec.tec_phase[timeselect],
            tec_pseudorange=gnsstec.tec_pseudorange[timeselect],
            station=gnsstec.station,
            prn=gnsstec.prn,
            times=gnsstec.times[timeselect],
            times_of_transmission=gnsstec.times_of_transmission[timeselect],
        )

def get_electron_density_gnss(ipp: IPP):
    unique_days = get_unique_days(ipp.times)
    unique_days_indices = get_indexlist_unique_days(unique_days, ipp.times)
    all_data = []
    for day, indices in zip(unique_days, unique_days_indices):
        gnss_list = select_gnss_stations(ipp.loc[indices])
        dcb = parse_dcb_sinex(download_dcb(date=day.to_datetime())[0])
        gnss_file_list = download_rinex(date=day.to_datetime(), stations=gnss_list)
        gnss_data_list = process_all_rinex_parallel(
            gnss_file_list, times=ipp.times[indices], dcb=dcb
        )
        gnss_data_list = [i for i in gnss_data_list if i.is_valid]
        sp3_files = download_satpos_files(date=day.to_datetime())
        sat_pos_object = get_sat_pos_object(sp3_files=sp3_files[:3])
        sat_clk, gnss_clk = parse_clk_data(sp3_files[-1])
        stec_gnss_data = {}
        with ProcessPoolExecutor(max_workers=6) as executor:
        # Submit all tasks
            future_to_station = {
                executor.submit(_get_stec_info, gnss_data, gnss_clk,dcb, sat_pos_object, sat_clk, ipp, indices) : gnss_data.station
                for gnss_data in gnss_data_list
         }

        # Collect results as they complete
        for future in as_completed(future_to_station):
            station = future_to_station[future]
            try:
                result = future.result()
                stec_gnss_data[station] = result
            except Exception as e:
                stec_gnss_data[station] = f"Error: {e}"

        #electron_density  = interpolate_to_ipp(stec_gnss_data, ipp.loc[indices], ipp.times[indices])
        all_data.append(stec_gnss_data)
    return all_data
    



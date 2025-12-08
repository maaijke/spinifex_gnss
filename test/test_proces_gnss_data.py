from proces_gnss_data import (
    get_gnss_station_density,
    _get_dcb_value,
    _get_phase_corrected,
    get_gim_correction,
    getphase_tec,
    getpseudorange_tec,
    _get_cycle_slips,
    get_transmission_time,
    gnss_pos_dict,
    get_interpolated_tec,
    get_ipp_density
)

from parse_gnss import parse_dcb_sinex, get_gnss_data
from gnss_geometry import get_sat_pos_object, get_sat_pos, get_stat_sat_ipp
from pathlib import Path
import glob
import astropy.units as u
import numpy as np
from spinifex.geometry import get_ipp_from_skycoord
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

datapath = Path("./data/")

def get_test_data():
    gnss_file = datapath / "IJMU00NLD_R_20251770000_01D_30S_MO.crx.gz"
    dcb_data = datapath / "CAS0MGXRAP_20251770000_01D_01D_DCB.BSX.gz"
    
    dcb = parse_dcb_sinex(dcb_data)
    gnss_data_list = get_gnss_data(gnss_file, dcb=dcb, station="IJMU00NLD")
    sp3_files = [
        Path(i)
        for i in sorted(glob.glob(datapath.as_posix() + "/*20251*0000*SP3.gz"))
    ]
    sat_pos_object = get_sat_pos_object(sp3_files=sp3_files)
    return gnss_data_list, dcb, sat_pos_object


def get_test_data_single_constellation():
    gnss_data_list, dcb, sat_pos_object = get_test_data()
    return [i for i in gnss_data_list if i.constellation=="G"][0], dcb, sat_pos_object

def test_getpsuedorange_tec():
    gnss_data, _, _ = get_test_data_single_constellation()

    prn = "G02"
    return getpseudorange_tec(
        c1=gnss_data.gnss[prn][:, 0],
        c2=gnss_data.gnss[prn][:, 1],
        dcb_sat=0,
        dcb_stat=0,
        constellation="G",
    )


def test_getphase_tec():
    gnss_data, _, _ = get_test_data_single_constellation()
    prn = "G02"
    return getphase_tec(
        l1=gnss_data.gnss[prn][:, 2], l2=gnss_data.gnss[prn][:, 3], constellation="G"
    )


def test_get_cycle_slips():
    phase_tec = test_getphase_tec()
    return _get_cycle_slips(phase_tec=phase_tec)


def test_get_gim_correction():
    gnss_data, dcb, sat_pos_object = get_test_data_single_constellation()

    prns = sorted(gnss_data.gnss.keys())
    stec_values = []
    ipp_sat_stat = []
    dcb_stat = 0
    for prn in prns:
        try:
            sat_data = gnss_data.gnss[prn]
            dcb_sat = _get_dcb_value(dcb, gnss_data.c1_str, gnss_data.c2_str, prn)
            transmission_time = get_transmission_time(
                sat_data[:, 1], gnss_data.times, dcb_sat=dcb_sat, dcb_stat=dcb_stat
            )
            pseudo_stec = getpseudorange_tec(
                sat_data[:, 0],  # c1 data
                sat_data[:, 1],  # c2 data
                dcb_sat=dcb_sat,
                dcb_stat=dcb_stat,
                constellation=gnss_data.constellation,
            )
            phase_stec = getphase_tec(
                sat_data[:, 2], sat_data[:, 3], constellation=gnss_data.constellation
            )
            phase_stec = _get_phase_corrected(
                phase_stec, pseudo_stec
            )  # correct bias per cycle slip
            sat_pos = get_sat_pos(sat_pos_object, transmission_time, prn)
            ipp_sat_stat.append(
                get_stat_sat_ipp(
                    satpos=sat_pos,
                    gnsspos=gnss_pos_dict[gnss_data.station],
                    times=gnss_data.times,
                    height_array=np.array(
                        [
                            350,
                        ]
                    )
                    * u.km,
                )
            )
            stec_values.append(phase_stec)
        except:
            print("Fail for", prn)
    # correction probably needs to be per satellite constellation, this can be solved by creating a gnss_data object
    # per constellation
    return stec_values, get_gim_correction(
        stec_data=stec_values, ipp_sat_stat=ipp_sat_stat, timestep=1
    )


def test_get_gnss_station_density():
    gnss_data, dcb, sat_pos_object = get_test_data_single_constellation()
    times = Time("2025-06-26T12:05:00") + np.arange(3) * 13 * u.min
    station_pos = EarthLocation.from_geodetic(6 * u.deg, 52 * u.deg, 0 * u.m)
    height_array = np.array([300, 500]) * u.km
    profiles = np.zeros(times.shape + height_array.shape)
    profiles[:, 0] = 0.7
    profiles[:, 1] = 0.3
    ipp = get_ipp_from_skycoord(
        source=SkyCoord.from_name("CAS A"),
        times=times,
        loc=station_pos,
        height_array=height_array,
    )
    return get_gnss_station_density(
        gnss_data=gnss_data,
        dcb=dcb,
        ipp_target=ipp,
        profiles=profiles,
        sat_pos_object=sat_pos_object,
    )


def test_get_interpolated_tec():
    input_data = np.zeros((4,3))
    input_data[0] = np.array((10,1,1))
    input_data[1] = np.array((8,1,-1))
    input_data[2] = np.array((5,-1.5,-1.1))
    input_data[3] = np.array((9,-0.5,1))
    return get_interpolated_tec([[input_data]])

def test_get_ipp_density():
    gnss_data_list, dcb, sat_pos_object = get_test_data()
    times = Time("2025-06-26T12:05:00") + np.arange(3) * 13 * u.min
    station_pos = EarthLocation.from_geodetic(6 * u.deg, 52 * u.deg, 0 * u.m)
    height_array = np.array([300, 500]) * u.km
    profiles = np.zeros(times.shape + height_array.shape)
    profiles[:, 0] = 0.7
    profiles[:, 1] = 0.3
    ipp = get_ipp_from_skycoord(
        source=SkyCoord.from_name("CAS A"),
        times=times,
        loc=station_pos,
        height_array=height_array,
    )
    return get_ipp_density(ipp_target=ipp, gnss_data_list = gnss_data_list, dcb=dcb, sat_pos_object=sat_pos_object)
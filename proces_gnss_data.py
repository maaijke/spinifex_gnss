from parse_rinex import RinexData
from parse_gnss import DCBdata
from typing import NamedTuple
from astropy.time import Time
import numpy as np
from spinifex.geometry import IPP
from gnss_geometry import get_stat_sat_ipp

class GNSSData(NamedTuple):
    """Object containing data dictionary and some metadata for gnss"""

    gnss: RinexData | None
    """the data"""
    c1_str: str
    """label for pseudorange 1"""
    c2_str: str
    """label for pseudorange 2"""
    l1_str: str
    """label for phase 1"""
    l2_str: str
    """label for phase 2"""
    station: str
    """name of the gnss station"""
    has_dcb: bool
    """flag if dcb data available"""
    is_valid: bool
    """Does this have valid gnss data"""
    times: Time


FREQ = {
    "G": {"f1": 1575.42e6, "f2": 1227.60e6},  # GPS L1/L2
    "R": {
        "f1": 1602.00e6 + 9 * 0.5625e6,
        "f2": 1246.00e6 + 9 * 0.4375e6,
    },  # nominal GLONASS; per-slot ideally
    "E": {"f1": 1575.42e6, "f2": 1191.795e6},  # Galileo E1/E5
    "C": {"f1": 1561.098e6, "f2": 1207.14e6},  # BeiDou B1/B2
    "J": {"f1": 1575.42e6, "f2": 1227.60e6},  # QZSS same as GPS
}


class STEC_GNSS(NamedTuple):
    stec_values: np.ndarray
    ipp: list[IPP]
    bias: float




def get_transmission_time(c1:np.ndarray, c2:np.ndarray, times:Time) -> Time:
    pass

def  getpseudorange_tec(c1:np.ndarray, c2:np.ndarray):
    pass

def getphase_tec(l1:np.ndarray, l2: np.ndarray):
    pass

def get_cycle_slips(phase_tec: np.ndarray):
    pass

def get_phase_corrected(phase_tec:np.ndarray, pseudo_tec:np.ndarray, cycle_slips):
    pass

def get_gim_correction(prn:list[str], station:str, times:Time, phasestec:np.ndarray):
    pass

def get_distance_ipp(ipp_sat_stat:IPP, ipp_target:IPP):
    pass

def get_dcb_value(dcb, c1_str, c2_str, prn):
    pass

def get_gnss_density(gnss_data:GNSSData, ipp: IPP, dcb: DCBdata):
    dcb_stat = get_dcb_value(dcb, gnss_data.c1_str, gnss_data.c2_str, gnss_data.station)
    prns = sorted(gnss_data.gnss.keys())
    stec_values = []
    ipp_sat_stat = []
    for prn in prns:
        sat_data = gnss_data.gnns[prn] 
        dcb_sat = get_dcb_value(dcb, gnss_data.c1_str, gnss_data.c2_str, prn)
        transmission_time = get_transmission_time(sat_data.c1, sat_data.c2, gnss_data.times )
        pseudo_stec = getpseudorange_tec(sat_data.c1, sat_data.c2, dcb_sat, dcb_stat)
        phase_stec = getphase_tec(sat_data.l1, sat_data.l2)
        cycle_slips = get_cycle_slips(phase_stec)
        phase_stec = get_phase_corrected(phase_stec,pseudo_stec, cycle_slips)
        ipp_sat_stat.append(get_stat_sat_ipp(transmission_time))
        stec_values.append(phase_stec)
    correction = get_gim_correction(ipp_sat_stat, stec_values , gnss_data.times)
    return STEC_GNSS(ipp=ipp_sat_stat,stec_values=stec_values,bias=correction)
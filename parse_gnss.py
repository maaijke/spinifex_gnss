import gzip
from pathlib import Path
import numpy as np
from astropy.constants import c as speed_light
from astropy.time import Time
import astropy.units as u
from typing import NamedTuple, Any
import georinex
import xarray
import concurrent.futures
from datetime import datetime

FREQL1 = 1.57542
FREQL2 = 1.2276
FREQL3 = 1.176
WL1 = speed_light.value / (FREQL1 * 1e9)
WL2 = speed_light.value / (FREQL2 * 1e9)
WL3 = speed_light.value / (FREQL3 * 1e9)
C12 = 100 / (40.3 * (1.0 / FREQL1**2 - 1.0 / FREQL2**2))
# 100 because conversion to Hz and to TECU (1e18/1e16)

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


class DCBdata(NamedTuple):
    """Object containing differential code biases"""

    dcb: np.ndarray[float]
    """array with dcb values and stddev per station/prn code combination"""
    station_code_combination: list[list[str]]
    """list of station_code1_code2 or prn_code1_code2 to find dcb value indices"""


class GNSSData(NamedTuple):
    """Object containing xarray and some metadata for gnss"""

    gnss: xarray.Dataset | None
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


class GNSSTECData(NamedTuple):
    """Object containing the estimated sTEC values for a single station satellite combination,
    as well as the transmission time correction"""

    tec_pseudorange: np.ndarray[float]
    """stec estimated from pseudorange"""
    tec_phase: np.ndarray[float]
    """stec estimated from phases"""
    times: Time
    """times of gnss station"""
    times_of_transmission: Time
    """corrected transmision times of satellite"""
    station: str
    """station name"""
    prn: str
    """prn code (satellite)"""


dummy_gnss_data = lambda station: GNSSData(
    gnss=None,
    c1_str="",
    c2_str="",
    l1_str="",
    l2_str="",
    has_dcb=False,
    is_valid=False,
    station=station,
)


def parse_dcb_sinex(dcbfile: Path):
    """Parse a SINEX Bias (.BIA) file and return a dict of DCB values in meters."""

    if dcbfile.suffix == ".gz":
        with gzip.open(dcbfile, "rt", encoding="utf-8") as file_buffer:
            return _read_dcb_data(file_buffer)


def _read_dcb_data(file_buffer):
    dcb = []
    station_code_combination = []
    in_data = False
    for line in file_buffer:
        if line.startswith("+BIAS/SOLUTION"):
            in_data = True
            continue
        elif line.startswith("-BIAS/SOLUTION"):
            break
        if line.startswith("*"):
            fields = line.strip().split()
        elif in_data and not line.startswith("*"):
            idx = 0
            data = {}
            for field in fields:
                data[field.strip("_")] = line[idx : idx + len(field)].strip()
                idx += len(field) + 1
            if data["UNIT"] == "ns":
                to_seconds = 1e-9
            else:
                to_seconds = 1
            station = data["STATION"]
            cod1 = data["OBS1"]
            cod2 = data["OBS2"]
            bias = float(data["ESTIMATED_VALUE"])
            stddev = float(data["STD_DEV"])
            prn = data["PRN"]
            if station:
                station_code_combination.append(f"{station}_{cod1}_{cod2}")
                dcb.append(
                    np.array((bias, stddev)) * speed_light.value * to_seconds
                )  # bias in meter
            else:
                station_code_combination.append(f"{prn}_{cod1}_{cod2}")
                dcb.append(
                    np.array((bias, stddev)) * speed_light.value * to_seconds
                )  # bias in meter
    return DCBdata(dcb=dcb, station_code_combination=station_code_combination)


def get_gnss_data(gnss_file: Path, times: Time, dcb: dict[Any], station: str):
    try:
        # first load 1s to see available measurements
        gnss = georinex.load(
            gnss_file, tlim=[times[0].isot, (times[0] + 1 * u.min).isot]
        )
        # get L1, L2 data
        labels = [
            i for i in sorted(gnss.data_vars.keys()) if i[1] == "1" or i[1] == "2"
        ]
        dcb_labels = [
            i.split("_")[1:]
            for i in dcb.station_code_combination
            if i.split("_")[0] == station[:4]
        ]
        c_tracking = [
            (i, j)
            for i, j in dcb_labels
            if i in labels
            and j in labels
            and f"L1{i[2]}" in labels
            and f"L2{i[2]}" in labels
        ]
        has_dcb = True
        if not c_tracking:
            print(
                f"no consistent pseudorange codes in {labels} and {dcb_labels}, continue without dcb"
            )
            has_dcb = False
            c_tracking = [
                (f"C1{i}", f"C2{j}")
                for i in [
                    "C",
                    "W",
                    "P",
                ]  # TODO: Add more possibilities? What is the meaning of these
                for j in ["W", "P", "C"]
                if f"C1{i}" in labels
                and f"C2{j}" in labels
                and f"L1{i}" in labels
                and f"L2{j}" in labels
            ]
            if not c_tracking:
                print(f"no consistent pseudorange codes in {labels}")
                return dummy_gnss_data(station=station)
        c_tracking = c_tracking[0]
        gnss = georinex.load(
            gnss_file,
            meas=[
                f"L1{c_tracking[0][-1]}",
                f"L2{c_tracking[1][-1]}",
                c_tracking[0],
                c_tracking[1],
            ],
            tlim=[times[0].isot, (times[-1] + 1 * u.min).isot],
        )
        return GNSSData(
            gnss=gnss,
            c1_str=c_tracking[0],
            c2_str=c_tracking[1],
            l1_str=f"L1{c_tracking[0][-1]}",
            l2_str=f"L2{c_tracking[1][-1]}",
            has_dcb=has_dcb,
            is_valid=True,
            station=station,
        )
    except:
        print(f"failed for station {station}")
        return dummy_gnss_data(station=station)


def process_all_rinex_parallel(rinex_files, times: Time, dcb: dict[Any], max_workers=8):
    """Run get_gnss_data in parallel and gather results."""

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_gnss_data, rf, times, dcb, rf.stem[:9]): rf
            for rf in rinex_files
        }
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    return results



def get_cycle_slips(
    l1_data: np.ndarray[float], l2_data: np.ndarray[float]
) -> np.ndarray[int]:
    diff = np.abs(np.diff(l1_data,prepend=l1_data[0]))
    diff2 = np.abs(np.diff(l2_data,prepend=l2_data[0]))
    #diff = np.diff(diff, prepend=diff[0])
    #diff2 = np.diff(diff2, prepend=diff[0])

    slips = np.logical_or(diff > 3*np.nanmedian(diff), diff2 > 3*np.nanmedian(diff2))
    gaps = np.logical_or(np.isnan(l1_data), np.isnan(l2_data))
    gaps = np.diff(gaps, prepend=0)>0
    seg_id = np.cumsum(np.logical_or(slips, gaps).astype(int))
    return seg_id


def get_tec_gnss(
    gnssdata: GNSSData, dcbdata: DCBdata, clock_correction: u.Quantity = 0 * u.ns
):
    if not gnssdata.is_valid:
        raise ValueError("gnssdata not valid")
    gnss = gnssdata.gnss
    dcb = dcbdata.dcb
    dcbkeys = dcbdata.station_code_combination
    # get geometry free range
    station = gnssdata.station
    c1_str = gnssdata.c1_str
    c2_str = gnssdata.c2_str
    l1_str = gnssdata.l1_str
    l2_str = gnssdata.l2_str
    times = Time(gnss[c1_str].time) + clock_correction
    if gnssdata.has_dcb:
        dcb_stat = dcb[dcbkeys.index(f"{station[:4]}_{c1_str}_{c2_str}")][
            0
        ]  # first is value second is stddev
    else:
        dcb_stat = 0
    stec_list = []

    for prn in gnss[c1_str].sv.values:
        if f"{str(prn)}_{c1_str}_{c2_str}" in dcbkeys:
            index = dcbkeys.index(f"{str(prn)}_{c1_str}_{c2_str}")
            dcb_sat = dcb[index][0]
            tec_c1c2 = C12 * (
                gnss[c1_str].sel(sv=prn).values
                - gnss[c2_str].sel(sv=prn).values
                - (dcb_sat + dcb_stat)
            )
            tec_l1l2 = -C12 * (
                gnss[l1_str].sel(sv=prn).values * WL1
                - gnss[l2_str].sel(sv=prn).values * WL2
            )
            distance = gnss[c2_str].sel(sv=prn).values
            distance[np.isnan(distance)] = 0
            times_of_transmission = (
                times - (distance - (dcb_sat + dcb_stat)) * u.m / speed_light
            )
            seg_id = get_cycle_slips(gnss[l1_str].sel(sv=prn).values, gnss[l2_str].sel(sv=prn).values)
            phase_bias = np.zeros_like(tec_l1l2)
            for seg in np.unique(seg_id):
                seg_idx = np.where(seg_id==seg)

                phase_bias[seg_idx] = np.nanmean(tec_c1c2[seg_idx] - tec_l1l2[seg_idx])  # TODO: better correction of cycle slips, 
                #if there are too many slips the time is to short for reliable P1-P2 correction 
            tec_l1l2 += phase_bias
            stec_list.append(
                GNSSTECData(
                    tec_phase=tec_l1l2,
                    tec_pseudorange=tec_c1c2,
                    station=station,
                    prn=prn,
                    times=times,
                    times_of_transmission=times_of_transmission,
                )
            )
    return stec_list


def parse_clk_data(clkfile: Path) -> dict[Any]:

    satellites = {}
    stations = {}
    with open(clkfile) as myf:
        for line in myf:
            if line.startswith("AS"):  # Satellite record
                parts = line.split()
                sat_name = parts[1]
                date_str = " ".join(parts[2:8])
                clk_bias = float(parts[9])  # clock bias (seconds)
                epoch = datetime.strptime(date_str, "%Y %m %d %H %M %S.%f")

                satellites.setdefault(sat_name, []).append(
                    {"epoch": epoch, "bias": clk_bias}
                )

            elif line.startswith("AR"):  # Station record
                parts = line.split()
                sta_name = parts[1]
                date_str = " ".join(parts[2:8])
                clk_bias = float(parts[9])
                epoch = datetime.strptime(date_str, "%Y %m %d %H %M %S.%f")

                stations.setdefault(sta_name, []).append(
                    {"epoch": epoch, "bias": clk_bias}
                )
    return satellites, stations

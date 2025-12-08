import gzip
from pathlib import Path
import numpy as np
from astropy.constants import c as speed_light
from astropy.time import Time
import astropy.units as u
from typing import NamedTuple, Any, TextIO
import concurrent.futures
from datetime import datetime
from parse_rinex import get_rinex_data, RinexData


# ------------------------------------------------------------
# Mapping of preferred GNSS observation codes, in priority order
# ------------------------------------------------------------
GNSS_OBS_PRIORITY = {
    "G": {  # GPS
        "C1": ["C1W", "C1P", "C1C", "C1Y"],  # W/P(Y) > C/A
        "C2": ["C2W", "C2P", "C2Y", "C2L", "C2X"],
        "L1": ["L1W", "L1P", "L1Y", "L1C"],
        "L2": ["L2W", "L2P", "L2Y", "L2L", "L2X"],
    },
    "E": {  # Galileo
        "C1": ["C1C", "C1X"],  # E1
        "C2": ["C5Q", "C5X"],  # E5a (1176 MHz)
        "L1": ["L1C", "L1X"],
        "L2": ["L5Q", "L5X"],
    },
    "R": {  # GLONASS
        "C1": ["C1C", "C1P"],
        "C2": ["C2C", "C2P"],
        "L1": ["L1C", "L1P"],
        "L2": ["L2C", "L2P"],
    },
    "C": {  # BeiDou-3 (priority); fallback to BeiDou-2
        "C1": ["C1C", "C1X", "C2I"],  # B1C > B1I
        "C2": ["C5Q", "C5X", "C6I"],  # B2a > B3I
        "L1": ["L1C", "L1X", "L2I"],
        "L2": ["L5Q", "L5X", "L6I"],
    },
    "J": {  # QZSS
        "C1": ["C1C", "C1X"],
        "C2": ["C2L", "C2W", "C2X"],
        "L1": ["L1C", "L1X"],
        "L2": ["L2L", "L2W", "L2X"],
    },
    "I": {  # NavIC / IRNSS
        "C1": ["C5Q"],  # L5
        "C2": ["CSQ"],  # S-band
        "L1": ["L5Q"],
        "L2": ["LSQ"],
    },
}


class DCBdata(NamedTuple):
    """Object containing differential code biases"""

    dcb: np.ndarray[float]
    """array with dcb values and stddev per station/prn code combination"""
    station_code_combination: list[list[str]]
    """list of station_code1_code2 or prn_code1_code2 to find dcb value indices"""


class GNSSData(NamedTuple):
    """Object containing data dictionary and some metadata for gnss, one per constellation"""

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
    """Constellation"""
    constellation: str


dummy_gnss_data = lambda station, constellation: GNSSData(
    gnss=None,
    c1_str="",
    c2_str="",
    l1_str="",
    l2_str="",
    has_dcb=False,
    is_valid=False,
    station=station,
    times=None,
    constellation=constellation,
)


def parse_dcb_sinex(dcbfile: Path):
    """Parse a SINEX Bias (.BIA) file and return a dict of DCB values in meters."""

    if dcbfile.suffix == ".gz":
        with gzip.open(dcbfile, "rt", encoding="utf-8") as file_buffer:
            return _read_dcb_data(file_buffer)
    else:
        with open(dcbfile, "rt") as file_buffer:
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


def get_gnss_data(gnss_file: Path, dcb: dict[Any], station: str):
    try:
        rinex_data = get_rinex_data(gnss_file)
    except:
        print(f"rinex data failed for station {station}")
        return []
    constellations = rinex_data.header.datatypes.keys()
    gnss_data_list = []
    for constellation in constellations:
        try:
            c1c2_labels = GNSS_OBS_PRIORITY[constellation]
            rxlabels = rinex_data.header.datatypes[constellation]
            labels = [i for i in sorted(rxlabels) if i[1] == "1" or i[1] == "2"]
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
                and i[1] == "1"
                and j[1] == "2"
                and f"L1{i[2]}" in labels
                and f"L2{j[2]}" in labels
            ]
            has_dcb = True
            if not c_tracking:
                print(
                    f"no consistent pseudorange codes in {labels} and {dcb_labels}, continue without dcb"
                )
                has_dcb = False
                c_tracking = [
                    (i, j)
                    for i in c1c2_labels["C1"]
                    for j in c1c2_labels["C2"]
                    if i in labels
                    and j in labels
                    and f"L1{i[-1]}" in labels
                    and f"L2{j[-1]}" in labels
                ]

                if not c_tracking:
                    print(f"no consistent pseudorange codes in {labels}")
                    return dummy_gnss_data(station=station)
            c_tracking = c_tracking[0]
            c1_str = c_tracking[0]
            c2_str = c_tracking[1]
            l1_str = f"L1{c_tracking[0][-1]}"
            l2_str = f"L2{c_tracking[1][-1]}"
            idx_c1 = rxlabels.index(c1_str)
            idx_c2 = rxlabels.index(c2_str)
            idx_l1 = rxlabels.index(l1_str)
            idx_l2 = rxlabels.index(l2_str)
            data = {}
            for key, rxdata in rinex_data.data.items():
                if key[0] == constellation:
                    data[key] = rxdata[:, (idx_c1, idx_c2, idx_l1, idx_l2)]
            gnss_data_list.append(
                GNSSData(
                    c1_str=c1_str,
                    c2_str=c2_str,
                    l1_str=l1_str,
                    l2_str=l2_str,
                    gnss=data,
                    station=station,
                    has_dcb=has_dcb,
                    is_valid=True,
                    times=rinex_data.times,
                    constellation=constellation,
                )
            )

        except:
            print(f"failed for station {station} {constellation}")
            gnss_data_list.append(
                dummy_gnss_data(station=station, constellation=constellation)
            )
    return gnss_data_list


def process_all_rinex_parallel(rinex_files, dcb: dict[Any], max_workers=6):
    """Run get_gnss_data in parallel and gather results."""

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_gnss_data, rf, dcb, rf.stem[:9]): rf
            for rf in rinex_files
        }
        for fut in concurrent.futures.as_completed(futures):
            results += fut.result()
    return results


def _parse_clk_file(clkfile: TextIO) -> dict[Any]:
    satstat_clk = {}
    for line in clkfile:
        parts = line.split()
        sat_name = parts[1]
        date_str = " ".join(parts[2:8])
        clk_bias = float(parts[9])  # clock bias (seconds)
        epoch = datetime.strptime(date_str, "%Y %m %d %H %M %S.%f")

        satstat_clk.setdefault(sat_name, []).append({"epoch": epoch, "bias": clk_bias})
    return satstat_clk


def parse_clk_data(clkfile: Path) -> dict[Any]:  # rewrite to

    if clkfile.suffix == ".gz":
        with gzip.open(clkfile) as myf:
            return _parse_clk_file(myf)
    else:
        with open(clkfile) as myf:
            return _parse_clk_file(myf)

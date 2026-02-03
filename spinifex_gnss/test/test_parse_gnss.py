from parse_gnss import parse_dcb_sinex, get_gnss_data, process_all_rinex_parallel
from pathlib import Path
import glob

datapath = Path("./data/")


def test_parse_dcb_sinex():
    dcb_data = datapath / "CAS0MGXRAP_20250020000_01D_01D_DCB.BSX.gz"
    return parse_dcb_sinex(dcb_data)


def test_get_gnss_data():
    gnss_file = datapath / "IJMU00NLD_R_20252900000_01D_30S_MO.crx.gz"
    dcb = test_parse_dcb_sinex()
    return get_gnss_data(gnss_file=gnss_file, dcb=dcb, station="IJMU00NLD")


def test_get_gnss_parallel():
    file_list = glob.glob(datapath.as_posix() + "/I*_R_20252900000_01D_30S_MO.crx.gz")
    gnss_files = [Path(i) for i in file_list]
    dcb = test_parse_dcb_sinex()
    return process_all_rinex_parallel(rinex_files=gnss_files, dcb=dcb)

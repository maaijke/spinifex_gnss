from parse_rinex import get_rinex_data
from pathlib import Path

datapath = Path("./data/")

def test_get_rinex_data():
    rinex_file = datapath / "IJMU00NLD_R_20252900000_01D_30S_MO.crx.gz"
    return get_rinex_data(rinex_file)
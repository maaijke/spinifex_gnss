from parse_rinex import get_rinex_data
from pathlib import Path

def test_get_rinex_data():
    rinex_file = Path("/home/mevius/IONO/GPS/spinifex_gnss/data/IJMU00NLD_R_20252900000_01D_30S_MO.crx.gz")
    return get_rinex_data(rinex_file)
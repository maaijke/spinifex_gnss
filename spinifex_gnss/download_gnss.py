import asyncio
import gps_time
from datetime import datetime, timedelta
from spinifex.download import download_or_copy_url
from spinifex.asyncio_wrapper import sync_wrapper
from pathlib import Path
from bs4 import BeautifulSoup
import requests


def get_gps_week(date: datetime):
    gpstime = gps_time.GPSTime.from_datetime(date)
    return gpstime.week_number, int(gpstime.time_of_week / (24 * 3600))


async def _download_satpos_files_coro(
    date: datetime,
    url: str = "ftp://ftp.gfz-potsdam.de/GNSS/products/mgex/",
    datapath: Path = Path("../../GPS/data/"),
) -> list[Path]:

    sp3_names = []
    yesterday = date - timedelta(days=1)
    tomorrow = date + timedelta(days=1)
    gpsweek, _ = get_gps_week(date)
    yesterweek, _ = get_gps_week(yesterday)
    tomorrowweek, _ = get_gps_week(tomorrow)
    sp3_names.append(
        f"{url}{yesterweek}_IGS20/GBM0MGXRAP_{yesterday.year}{yesterday.timetuple().tm_yday:03d}0000_01D_05M_ORB.SP3.gz"
    )
    sp3_names.append(
        f"{url}{gpsweek}_IGS20/GBM0MGXRAP_{date.year}{date.timetuple().tm_yday:03d}0000_01D_05M_ORB.SP3.gz"
    )
    sp3_names.append(
        f"{url}{tomorrowweek}_IGS20/GBM0MGXRAP_{tomorrow.year}{tomorrow.timetuple().tm_yday:03d}0000_01D_05M_ORB.SP3.gz"
    )
    clk_name = f"{url}{gpsweek}_IGS20/GBM0MGXRAP_{date.year}{date.timetuple().tm_yday:03d}0000_01D_30S_CLK.CLK.gz"
    sp3_names.append(clk_name)
    coros = []
    for url in sp3_names:
        coros.append(download_or_copy_url(url, output_directory=datapath))

    return await asyncio.gather(*coros)




def download_satpos_files(
    date: datetime,
    url: str = "ftp://ftp.gfz-potsdam.de/GNSS/products/mgex/",
    datapath: Path = Path("../../GPS/data/"),
) -> list[Path]:
    """Get the sp3 position files and corresponding clock errors for a specific date, the day before and the day after for interpolation purposes

    Parameters
    ----------
    date : datetime
        requested date
    url : str, optional
        server from where to download the data, by default "https://cddis.nasa.gov/archive/gnss/products/"
    datapath : Path, optional
        output directory, by default Path("../../GPS/data/")

    Returns
    -------
    list[Path] : list with the filenames of respectively three sorted sp3 files and clock errors of date

    """
    return _download_satpos_files(date, url, datapath)


async def download_dcb_coro(
    date: datetime, datapath: Path = Path("../../GPS/data/")
) -> Path:
    """Download differential code biases for a given date

    Parameters
    ----------
    date : datetime
        requested date
    datapath : Path, optional
        output directory, by default Path("../../GPS/data/")

    Returns
    -------
    Path
        location of the data
    """
    # TODO: find optional servers for these dcb files
    server = "https://data.bdsmart.cn/pub/product/bias/"
    doy = date.timetuple().tm_yday
    url = f"{server}{date.year}/CAS0MGXRAP_{date.year}{doy:03d}0000_01D_01D_DCB.BSX.gz"
    dcb = download_or_copy_url(url, output_directory=datapath)
    return await asyncio.gather(dcb)


def check_url(url_list: list[str]):
    """
    Check if a given url exists
    """
    files = set()
    for url in url_list:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML directory listing
            soup = BeautifulSoup(response.text, 'html.parser')

            
            # Find all links in the directory
            for link in soup.find_all('a'):
                href = link.get('href')
                if href and not href.startswith(('?', '/', 'http')):
                    files.add(f"{url}{href}")
        except Exception as e:
            print(f"Error parsing {url}: {e}")
            continue
    return files

async def download_rinex_coro(
    date: datetime,
    stations: list[str],
    datapath: Path = Path("../../GPS/data/"),
):
    # TODO: get naming format for dates and servers (like we do for ionex data)
    urls = []
    year = date.year
    yy = date.year - 2000
    doy = date.timetuple().tm_yday
    server_list = [
        "https://cddis.nasa.gov/archive/gnss/data/daily/",
        "https://www.epncb.oma.be/pub/obs/",
        "https://webring.gm.ingv.it:44324/rinex/RING/",
        "https://ga-gnss-data-rinex-v1.s3.amazonaws.com/index.html#public/daily/",
    ]
    url_list = [
        f"{server_list[0]}/{year}/{doy:03d}/{yy}d/",
        f"{server_list[1]}/{year}/{doy:03d}/",
        f"{server_list[2]}/{year}/{doy:03d}/",
        f"{server_list[3]}/{year}/{doy:03d}/"
    ]
    #get and parse directory listing
    print("checking", url_list)
    files_per_url = check_url(url_list)
    for station in stations:
        fname = f"{station}_R_{year}{doy:03d}0000_01D_30S_MO.crx.gz"
        found = False
        # TODO: the checking below is relatively slow, speed up (e.g. download page with all available stations and parse)
        for url in url_list:
            if f"{url}{fname}" in files_per_url:
                urls.append(f"{url}{fname}")
                found = True
                break
        if not found:
            print(f"{fname} not existing")

    coros = []
    for url in urls:
        coros.append(download_or_copy_url(url, output_directory=datapath))
    return await asyncio.gather(*coros)


download_rinex = sync_wrapper(download_rinex_coro)
download_dcb = sync_wrapper(download_dcb_coro)
_download_satpos_files = sync_wrapper(_download_satpos_files_coro)

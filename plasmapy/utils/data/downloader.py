"""
Contains functionality for downloading files from a URL. Intended for
downloading files from the PlasmaPy data repository.

"""

import os
import requests
import hashlib

from urllib.parse import urljoin

__all__ = ["get_file"]

# Note: GitHub links have a problem where the Content-Encoding is set to
# 'gzip' but the file is not actually compressed. This header is just ignored
# by the get_file function.
_BASE_URL = "https://github.com/PlasmaPy/PlasmaPy-data/raw/main/data/"

# TODO: use a config file variable to allow users to set a location for this folder?
_DOWNLOADS_PATH = os.path.join(os.path.dirname(__file__), "downloads")


def filehash(file):
    """
    Creates a hash for a file

    Parameters
    ----------
    file : str
        Path to the file

    Returns
    -------
    hash : str
        A hexidecimal digest of the hash

    """
    
    # Read+hash in chunks in case files are large
    # https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
    BUF_SIZE = 65536
    
    sha1 = hashlib.sha1()
    with open(file, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)
            
    return sha1.hexdigest()


def get_file(basename, base_url=_BASE_URL, directory=None):
    r"""
    Downloads a file from a URL (if the file does not already exist) and
    returns the full local path to the file.

    Parameters
    ----------
    basename : str
        Name of the file to be downloaded (extension included).

    base_url : str, optional
        The base URL of the file to be downloaded. Defaults to the main
        directory of `_PlasmaPy's data repository`.

    directory : str, optional
        The full path to the desired download location. Defaults to the
        default PlasmaPy data download directory:
        plasmapy/utils/data/downloads/

    Returns
    -------
    path : str
        The full local path to the downloaded file.

    """

    if "." not in str(basename):
        raise ValueError(f"'filename' ({basename}) must include an extension.")
        
    if directory is None:  # coverage: ignore
        directory = _DOWNLOADS_PATH

    path = os.path.join(directory, basename)

    # If file doesn't exist locally, download it
    if not os.path.exists(path):

        url = urljoin(base_url, basename)

        reply = requests.get(url)

        # Missing files on GitHub will resolve to a 404 html page, so we use
        # this as an indicator that the file may not exist.
        if "text/html" in reply.headers["Content-Type"]:
            raise OSError(
                "The requested URL returned an html file, which "
                "likely indicates that the file does not exist at the "
                "URL provided."
            )

        with open(path, "wb") as f:
            f.write(reply.content)

    return path

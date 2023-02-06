"""
I/O module.

Adapted from keras: https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/utils/data_utils.py#L150-L270
"""
import logging
import os
import shutil
import tarfile
import zipfile
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import six

logger = logging.getLogger(__name__)


def _extract_archive(filepath, path=".", archive_format="auto"):
    """
    Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.
    Args:
        filepath (str): path to the archive file.
        path (str): path to extract the archive file.
        archive_format (str): Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.
    Returns:
        bool: True if a match was found and an archive extraction
            was completed, False otherwise.
    """
    if archive_format is None:
        return False
    if archive_format == "auto":
        archive_format = ["tar", "zip"]
    if isinstance(archive_format, six.string_types):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == "tar":
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == "zip":
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(filepath):
            with open_fn(filepath) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                    else:
                        shutil.rmtree(path)
                    raise
            return True
    return False


def get_file(
    filename,
    origin,
    cache_subdir="cache",
    extract=False,
    archive_format="auto",
    cache_dir=None,
):
    """
    Downloads a file from a URL if it not already in the cache.
    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.paccmann`, placed in the cache_subdir `cache`,
    and given the filename `filename`. The final location of a file
    `example.txt` would therefore be `~/.pacmann/cache/example.txt`.
    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.

    Args:
        filename (str): Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location.
        origin (str): Original URL of the file.
        cache_subdir (str): Subdirectory under the Keras cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        extract (bool): True tries extracting the file as an Archive, like tar or zip.
        archive_format (str): Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.
        cache_dir (str): Location to store cached files, when None it
            defaults to `~/.paccmann`.
    Returns:
        Path to the downloaded file.
    """
    if cache_dir is None:
        cache_dir = os.path.join("~", ".paccmann")

    datadir_base = os.path.expanduser(cache_dir)
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        try:
            os.makedirs(datadir)
        except Exception:
            datadir = os.path.join("/tmp", ".paccmann", cache_subdir)
            os.makedirs(datadir)

    filepath = os.path.join(datadir, filename)

    download = not os.path.exists(filepath)

    if download:

        logger.info(f"Downloading data from: {origin}")
        error_msg = "URL fetch failure on {}: {} -- {}"
        try:
            try:
                urlretrieve(origin, filepath)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(filepath):
                os.remove(filepath)
            raise

    if extract:
        _extract_archive(filepath, datadir, archive_format)

    return filepath

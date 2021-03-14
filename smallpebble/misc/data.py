"""Rough and ready... load (/download) MNIST/CIFAR data from openml.org."""
from collections import namedtuple
import hashlib
import pathlib
import requests
from tqdm import tqdm
import numpy as np

DEFAULT_SAVEDIR = pathlib.Path.home() / ".smallpebble"
CHUNK_SIZE = 1024 * 1024


def split_mnist(data):
    "https://www.openml.org/d/554"
    images = data[:, : 28 * 28]
    labels = data[:, -1]
    train_slice = slice(0, 60_000)
    test_slice = slice(60_000, 70_000)
    X_train = images[train_slice, :]
    y_train = labels[train_slice]
    X_test = images[test_slice, :]
    y_test = labels[test_slice]
    return X_train, y_train, X_test, y_test


def split_cifar(data):
    "https://www.openml.org/d/40927"
    images = data[:, 0 : 32 * 32 * 3]
    labels = data[:, -1]
    train_slice = slice(0, 50_000)
    test_slice = slice(50_000, 60_000)
    X_train = images[train_slice, :]
    y_train = labels[train_slice]
    X_test = images[test_slice, :]
    y_test = labels[test_slice]
    return X_train, y_train, X_test, y_test


Metadata = namedtuple("Metadata", "filename npy url sha256 rows cols dtype splitdata")
META = {
    "mnist": Metadata(
        "mnist_784.arff",
        "mnist.npy",
        r"https://www.openml.org/data/download/52667/mnist_784.arff",
        "418c0a60d2b4abc95db2e2bbf676f3af93ddaf18f79ba3f640624ab57007fb4b",
        rows=70_000,
        cols=28 * 28 + 1,
        dtype=np.uint8,
        splitdata=split_mnist,
    ),
    "cifar": Metadata(
        "cifar-10.arff",
        "cifar.npy",
        r"https://www.openml.org/data/download/16797613/cifar-10.arff",
        "d28aa6ec1ac50109b54d58c45a4d31be6f9c406b36af81733e09bf9a55b73961",
        rows=60_000,
        cols=32 * 32 * 3 + 1,
        dtype=np.uint8,
        splitdata=split_cifar,
    ),
}


def load(dataname, savedir=None, delete_intermediate_files=True):
    """Load dataname='mnist' or 'cifar', from openml.org.

    >> from smallpebble.misc.data import load
    >> X_train, y_train, X_test, y_test = load('mnist')

    Notes:
    Caches in `savedir` to avoid redownloading (default is ~/.smallpebble/).
    Converts the data into NumPy's 'npy' format, which is smaller and faster to load than 'arff'.
    
    Data is from: https://www.openml.org
    """
    meta = META[dataname]
    savedir = pathlib.Path(savedir) if savedir else DEFAULT_SAVEDIR

    if (savedir / meta.npy).is_file():
        data = np.load(savedir / meta.npy)
    else:
        savedir.mkdir(exist_ok=True)

        print(f"Downloading {dataname} from openml.org")
        download(savedir, dataname)
        print("File successfully downloaded and validated.")

        print("Converting file...")
        data = arff_to_npy(savedir, dataname)
        print("Successfully converted file.")

        if delete_intermediate_files:
            (savedir / meta.filename).unlink()

    return meta.splitdata(data)


def download(savedir, dataname):
    """Download file and check hash."""

    meta = META[dataname]
    filepath = savedir / meta.filename

    # Download the file.
    with open(filepath, "wb") as file:
        session = requests.Session()
        response = session.get(meta.url, stream=True)
        response.raise_for_status()
        for data in tqdm(response.iter_content(chunk_size=CHUNK_SIZE)):
            file.write(data)

    # Validate via sha256.
    with open(filepath, "rb") as file:
        hashed = hashlib.sha256(file.read()).hexdigest()

    if not meta.sha256 == hashed:
        errorfilepath = filepath.with_suffix("-sha256-error")
        filepath.rename(errorfilepath)
        raise ValueError(f"Unexpected hash value. Saved file as {errorfilepath.resolve()}")


def yield_data(savedir, dataname, datalen):
    "A line at a time, yield data from a comma seperated arff file."
    filepath = savedir / META[dataname].filename
    with open(filepath, "r") as file:
        while True:
            line = file.readline()

            if not line:
                break

            if line.startswith("@"):
                continue

            data = line.split(",")

            if not len(data) == datalen:
                continue

            yield data


def arff_to_npy(savedir, dataname):
    meta = META[dataname]
    result = np.zeros([meta.rows, meta.cols], meta.dtype)
    for i, data in tqdm(
        enumerate(yield_data(savedir, dataname, meta.cols)), total=meta.rows
    ):
        result[i, :] = data
    np.save(savedir / meta.npy, result)
    return result


if __name__ == "__main__":
    result = load("mnist", delete_intermediate_files=False)
    for a in result:
        print(a.shape)

    result = load("cifar", delete_intermediate_files=False)
    for a in result:
        print(a.shape)

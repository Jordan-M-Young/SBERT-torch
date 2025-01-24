"""Repo scripts."""

import os
import zipfile

import requests

from app.main import main


def run_train():
    """Runs main training loop."""
    main()


def pull_data() -> None:
    """Script to pull datasets for training."""
    SNLI_URL = ""
    MLNI_URL = ""

    if not os.path.isdir("./data"):
        os.mkdir("./data")

    resp = requests.get(SNLI_URL)
    with open("./data/snli_1.0.zip", "wb") as zfile:
        zfile.write(resp.content)

    resp = requests.get(MLNI_URL)
    with open("./data/multinli_1.0.zip", "wb") as zfile:
        zfile.write(resp.content)

    with zipfile.ZipFile("./data/multinli_1.0.zip", "r") as zip_ref:
        zip_ref.extractall("./data/snli_1.0_test")

    with zipfile.ZipFile("./data/snli_1.0.zip", "r") as zip_ref:
        zip_ref.extractall("./data/multinli_1.0_test")

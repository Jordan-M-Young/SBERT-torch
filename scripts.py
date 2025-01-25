"""Repo scripts."""

import os

import requests

from app.main import main


def run_train():
    """Runs main training loop."""
    main()


def pull_data() -> None:
    """Script to pull datasets for training."""
    SNLI_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    MLNI_URL = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"

    if not os.path.isdir("./data"):
        os.mkdir("./data")

    resp = requests.get(SNLI_URL, stream=True)
    with open("./data/snli_1.0.zip", "wb") as fd:
        for chunk in resp.iter_content(chunk_size=128):
            fd.write(chunk)

    resp = requests.get(MLNI_URL, stream=True)
    with open("./data/mnli_1.0.zip", "wb") as fd:
        for chunk in resp.iter_content(chunk_size=128):
            fd.write(chunk)

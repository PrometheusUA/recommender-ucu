# Recommender systems repository
**by Andrii Shevtsov, Bohdan Vey, Yehor Hryha**

Recommender systems labs repository for Ukrainian Catholic University masters program course. We use [YELP dataset](https://www.yelp.com/dataset) to conduct these labs.

## Setup instructions

### Environment setup

We are using Python's narive virtual environments along with pip package manager and `requirements.txt` files.

To run the project, create a virtual environment via `python<version> -m venv .venv`.

Then, activate the environment:
- On Linux/Mac: `source .venv/bin/activate`.
- On Windows: `.venv/Scripts/activate.bat`.

Then, upgrade pip and install `requirements.txt`:
```bash
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

### Data setup

There are two ways to obtain the data for this repository:
- Obtain the dataset directly from YELP website. Here you should:
    - Open the [dataset download page](https://www.yelp.com/dataset/download).
    - Choose **Download JSON** option and wait untill the archive is downloaded.
    - Decompress obtained `.tar` file, and obtain one more file without extension specified.
    - Add `.tar` to the end of the file name and decompress it too.
    - Move `yelp_dataset` folder (with all `json`es and a `pdf`) into `data` folder of the repository.
- Obtain the dataset via [Kaggle](https://www.kaggle.com/) public API by:
    - Install kaggle package via `pip install kaggle`.
    - If there is no Kaggle access token in your system:
        - Create access token on [Kaggle account](https://www.kaggle.com/settings/account) > API > Create New Token. Download `kaggle.json` file.
        - Move `kaggle.json` to its place:
            - `~/.kaggle/kaggle.json` on Unix-based systems.
            - `C:\Users\<Windows-username>\.kaggle\kaggle.json` on Windows.
    - Move to the data folder: `cd data`.
    - Download the dataset: `kaggle datasets download -d yelp-dataset/yelp-dataset`.
    - Unzip it: `unzip yelp-dataset.zip -d yelp_dataset` or with some app.

## Repository structure

There are several folders in the dataset:
- **`artifacts`** — a folder with models and other deliverables saved for future usage.
- **`data`** — a folder with the dataset and its transformed versions saved.
- **`experiments`** — a folder with experiments tracking, notebooks and their descriptions in markdown format.
- **`scripts`** — a folder with useful python scripts.
- **`src`** — a folder with all source python files for models and their evaluation.

### Experiments

Experiment name is a folder name in `experiments` folder. Usually it contains experiment start date, author(s) of experiment and its main subject.

The folder with experiment can contain several python notebooks, some supplementary code etc. Main files there are `experiment.ipynb` and `description.md`, or other python notebook and markdown file if those are absent. Python notebook here contains experiment main runs, while markdown file contains description of the experiment. Sometimes, description of the experiment is located in the Jupyter notebook itself.

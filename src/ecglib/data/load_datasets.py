import requests
import zipfile
import os

import wfdb
from tqdm import tqdm
import pandas as pd


__all__ = [
    "load_ptb_xl",
    "load_physionet2020",
]


def load_ptb_xl(
    download: bool = False,
    path_to_zip: str = "./",
    path_to_unzip: str = "./",
    delete_zip: bool = True,
) -> pd.DataFrame:
    """
    Load PTB-XL dataset
    :param download: whether to download PTB-XL from Physionet
    :param path_to_zip: path where to store PTB-XL .zip file
    :param path_to_unzip: path where to unarchive PTB-XL .zip file
    :param delete_zip: whether to delete PTB-XL .zip file after unarchiving

    :return: dataframe with PTB-XL dataset info
    """

    if download:
        url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2.zip"
        ptb_xl_zip = os.path.join(path_to_zip, "ptb_xl.zip")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        print("Loading PTB-XL file...")
        with open(ptb_xl_zip, "wb") as f:
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    progress_bar.update(len(chunk))
                    f.write(chunk)
        progress_bar.close()
        print("Loading completed!")
        f.close()

        print("Unzipping PTB-XL file...")
        with zipfile.ZipFile(ptb_xl_zip, "r") as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc=""):
                try:
                    zip_ref.extract(member, path_to_unzip)
                except zipfile.error as e:
                    pass
        print("Unzipping completed!")

        if delete_zip:
            print(f"Deleting {ptb_xl_zip} file...")
            os.remove(ptb_xl_zip)
            print("Deleting completed!")

    ptb_xl_info = pd.read_csv(
        os.path.join(
            path_to_unzip,
            "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2",
            "ptbxl_database.csv",
        )
    )
    ptb_xl_info["fpath"] = [
        os.path.join(
            path_to_unzip,
            "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2",
            ptb_xl_info.iloc[i]["filename_hr"],
        )
        for i in range(len(ptb_xl_info["filename_hr"]))
    ]

    return ptb_xl_info


def load_physionet2020(
    download: bool = False,
    path_to_zip: str = "./",
    path_to_unzip: str = "./",
    delete_zip: bool = True,
    selected_datasets=None,
) -> pd.DataFrame:
    """
    Load physionet2020 challange datasets
    :param download: whether to download physionet2020
    :param path_to_zip: path where to store archive file
    :param path_to_unzip: path where to unarchive selected datasets
    :param delete_zip: whether to delete archive file after unarchiving selected datasets
    :param selected_datasets: list of the dataset names to extract from ['georgia','st_petersburg_incart','cpsc_2018','ptb-xl','ptb',cpsc_2018_extra']
    """

    if download:
        url = "https://physionet.org/static/published-projects/challenge-2020/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2.zip"
        print(f"Loading started...")
        with open(
            os.path.join(
                path_to_zip,
                "classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2.zip",
            ),
            "wb",
        ) as f:
            response = requests.get(url, stream=True)
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
            for chunk in response.iter_content(chunk_size=512):
                if chunk:
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
        print("Loading completed!")

    if not os.path.exists(path_to_unzip):
        os.makedirs(path_to_unzip)

    if not selected_datasets:
        selected_datasets = [
            "georgia",
            "st_petersburg_incart",
            "cpsc_2018",
            "ptb-xl",
            "ptb",
            "cpsc_2018_extra",
        ]

    selected_datasets_paths = tuple(
        [
            f"classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2/training/{dataset}/"
            for dataset in selected_datasets
        ]
    )
    print(f"Unzipping started...")
    with zipfile.ZipFile(
        os.path.join(
            path_to_zip,
            "classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2.zip",
        ),
        "r",
    ) as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc=""):
            if member.filename.startswith(selected_datasets_paths):
                filename = os.path.basename(member.filename)
                if not filename:
                    continue
                member.filename = member.filename.replace(
                    "classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2/training/",
                    "",
                )
                zip_ref.extract(member, path_to_unzip)
    print(f"Unzipping completed!")

    if delete_zip:
        print(f"Deleting zipfile started...")
        os.remove(
            os.path.join(
                path_to_zip,
                "classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2.zip",
            )
        )
        print("Deleting completed!")

    for dataset in selected_datasets:
        dataset_meta = []
        print(f"Collecting {dataset} information...")
        path_to_dataset = os.path.join(path_to_unzip, dataset)
        folders = os.listdir(path_to_dataset)
        for folder in tqdm(folders):
            for filename in os.listdir(os.path.join(path_to_dataset, folder)):
                if not filename.endswith("hea"):
                    continue

                file_info = {}
                record, metadata = wfdb.rdsamp(
                    os.path.join(path_to_dataset, folder, filename[:-4])
                )
                file_info["file_name"] = filename[:-4]
                file_info["fpath"] = os.path.join(
                    path_to_dataset, folder, filename[:-4]
                )
                file_info["ecg_shape"] = record.shape
                file_info["frequency"] = metadata["fs"]
                file_info["ecg_duration"] = metadata["sig_len"] / metadata["fs"]

                for comment in metadata["comments"]:
                    key, val = tuple(comment.strip().replace(" ", "").split(":"))
                    file_info[key] = val
                dataset_meta.append(file_info)

        dataset_meta = pd.DataFrame(dataset_meta)
        dataset_meta["Dx"] = dataset_meta["Dx"].apply(lambda x: x.split(","))
        dataset_meta.to_csv(
            os.path.join(path_to_unzip, f"{dataset}_dataset.csv"), index=False
        )
        print("Information is collected!")
    print("Loading completed!")

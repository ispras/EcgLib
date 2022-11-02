import requests
from zipfile import ZipFile
import os

from tqdm import tqdm
import pandas as pd


__all__ = [
    "load_ptb_xl",
]


def load_ptb_xl(download: bool = False,
                path_to_zip: str = "./",
                path_to_unzip: str = "./",
                delete_zip: bool = True,
               ) -> pd.DataFrame:
    '''
    Load PTB-XL dataset
    :param download: whether to download PTB-XL from Physionet
    :param path_to_zip: path where to store PTB-XL .zip file
    :param path_to_unzip: path where to unarchive PTB-XL .zip file
    :param delete_zip: whether to delete PTB-XL .zip file after unarchiving

    :return: dataframe with PTB-XL dataset info
    '''
    
    if download:
        
        url = 'https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2.zip'
        ptb_xl_zip = os.path.join(path_to_zip, 'ptb_xl.zip')
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        print('Loading PTB-XL file...')
        with open(ptb_xl_zip, 'wb') as f:
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            for chunk in response.iter_content(chunk_size=512):
                if chunk:
                    progress_bar.update(len(chunk))
                    f.write(chunk)
        progress_bar.close()
        print('Loading completed!')
        f.close()
        
        print('Unzipping PTB-XL file...')
        with ZipFile(ptb_xl_zip, 'r') as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc=''):
                try:
                    zip_ref.extract(member, path_to_unzip)
                except zipfile.error as e:
                    pass
        print('Unzipping completed!')
        
        if delete_zip:
            print(f'Deleting {ptb_xl_zip} file...')
            os.remove(ptb_xl_zip)
            print('Deleting completed!')
            
    ptb_xl_info = pd.read_csv(os.path.join(path_to_unzip, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2', 'ptbxl_database.csv'))
    ptb_xl_info['fpath'] = [os.path.join(path_to_unzip, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2', ptb_xl_info.iloc[i]['filename_hr']) for i in range(len(ptb_xl_info['filename_hr']))]
    
    return ptb_xl_info
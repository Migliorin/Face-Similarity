import os

import pandas as pd


def get_images_labels(path:str) -> pd.DataFrame:
    images_paths    = []
    images_labels   = []
    count           = os.listdir(path)
    total           = len(count)
    for root,folder,file in os.walk(path):
        if(len(file) > 0):
            print(f"Classe atual: {root} -> {len(count)}/{total}")
            aux_      = [f'{root}/{x}' for x in file if x.endswith('.jpeg')]
            label_    = root.split('/')[-1]
            images_paths.extend(aux_)
            images_labels.extend([label_]*len(aux_))
            count.remove(label_)
            
    return pd.DataFrame({
    'images_paths'   : images_paths,
    'images_labels'  : images_labels
    })
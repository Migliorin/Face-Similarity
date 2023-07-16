import os

import pandas as pd


def get_images_labels(path:str, format_img='jpeg',verbose=False) -> pd.DataFrame:
    images_paths    = []
    images_labels   = []
    count           = os.listdir(path)
    total           = len(count)
    for root,folder,file in os.walk(path):
        if(len(file) > 0):
            aux_      = [f'{root}/{x}' for x in file if x.endswith(f'.{format_img}')]
            if(len(aux_) > 0):
                if(verbose):
                    print(f"Classe atual: {root} -> {len(count)}/{total}")
                label_    = root.split('/')[-1]
                images_paths.extend(aux_)
                images_labels.extend([label_]*len(aux_))
                count.remove(label_)
                
    print("Images and labels loaded")
            
    return pd.DataFrame({
    'images_paths'   : images_paths,
    'images_labels'  : images_labels
    })
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
import torch


class DatasetGeneral(Dataset):
    def __init__(self, images_path, labels, transform_data = None, resize = (224,224),pos_neg_examples=1):
        self.images_path      = images_path
        self.labels           = labels
        self.transform_data   = transform_data
        self.resize           = resize
        self.pos_neg_examples = pos_neg_examples
        
    def __getitem__(self,index):
        anchor = self.images_path[index]
        current_class = self.labels[index]
        
        negative = []
        while(True):
            negative_index = np.random.randint(0,self.__len__())
            if(self.labels[negative_index] != current_class) and (self.images_path[negative_index] not in negative):
                negative.append(self.images_path[negative_index])
                if(len(negative) == self.pos_neg_examples):
                    break
        
        positive = []
        while(True):
            positive_index = np.random.randint(0,self.__len__())
            if(self.labels[positive_index] == current_class) and (self.images_path[positive_index] not in positive):
                positive.append(self.images_path[positive_index])
                if(len(positive) == self.pos_neg_examples):
                    break
        
        anchor         = self._read_img_path(anchor)
        positive       = [self._read_img_path(x) for x in positive]
        negative       = [self._read_img_path(x) for x in negative]
        
        return torch.tensor(anchor), torch.stack(positive), torch.stack(negative)
    
    def __len__(self):
        return len(self.images_path)
    
    def _read_img_path(self,path):
        #img = cv2.imread(path)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img,self.resize)
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        #img = Image.open(path)
        
        #img = torch.tensor(img).astype(float)
        #img = torch.tensor(img,dtype=float)
        if(self.transform_data is not None):
            img = self.transform_data(img)

        return img
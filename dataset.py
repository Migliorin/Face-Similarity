import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DatasetAnimal(Dataset):
    def __init__(self, images_path, labels, transform_data = None, resize = (224,224)):
        self.images_path      = images_path
        self.labels           = labels
        self.transform_data   = transform_data
        self.resize           = resize
        
    def __getitem__(self,index):
        anchor = self.images_path[index]
        current_class = self.labels[index]
        
        negative_index = np.random.randint(0,self.__len__())
        while(self.labels[negative_index] == current_class):
            negative_index = np.random.randint(0,self.__len__())
        negative = self.images_path[negative_index]
        
        positive_index = np.random.randint(0,self.__len__())
        while(self.labels[positive_index] != current_class):
            positive_index = np.random.randint(0,self.__len__())
        positive = self.images_path[positive_index]
    
        
        anchor         = self._read_img_path(anchor)
        positive       = self._read_img_path(positive)
        negative       = self._read_img_path(negative)
        
        return anchor, positive, negative
    
    def __len__(self):
        return len(self.images_path)
    
    def _read_img_path(self,path):
        #img = cv2.imread(path)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img,self.resize)
        
        img = Image.open(path)
        #img = torch.tensor(img).astype(float)
        #img = torch.tensor(img,dtype=float)
        if(self.transform_data is not None):
            img = self.transform_data(img)

        return img
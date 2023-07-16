import os
from time import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from tqdm import tqdm

from dataset import DatasetGeneral
from backbone.mobilenet_v2 import MobileNetV2
from parse_args import parse_args
from utils import get_images_labels

if __name__ == '__main__':

    args = parse_args()
    print(args)

    if(not os.path.exists(args.snapshot)):
        os.mkdir(args.snapshot)

    output_string = f"{args.snapshot}/{args.output_string}_{int(time())}"

    if(not os.path.exists(output_string)):
        os.mkdir(output_string)

    df = get_images_labels(args.data_dir)
    map_labels = {j:i for i,j in enumerate(df['images_labels'].unique())}
    df['images_labels'] = df['images_labels'].apply(lambda x: map_labels[x])

    train, aux = train_test_split(df,test_size=0.3,random_state=69,stratify=df['images_labels'])
    val, test = train_test_split(aux,test_size=0.5,random_state=69,stratify=aux['images_labels'])


    transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    print("------- Starting dataloaders -------")

    train_dataset   = DatasetGeneral(train['images_paths'].tolist(), train['images_labels'].tolist(), transform_data=transform)
    val_dataset     = DatasetGeneral(val['images_paths'].tolist(), val['images_labels'].tolist(), transform_data=transform)
    test_dataset    = DatasetGeneral(test['images_paths'].tolist(), test['images_labels'].tolist(), transform_data=transform)

    train_dataloader = DataLoader(train_dataset, num_workers=args.workers, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, num_workers=args.workers, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, num_workers=args.workers, batch_size=args.batch_size)

    print("------- Dataloaders started -------")

    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)

    print("------- Starting model -------")

    model = MobileNetV2()
    model = model.model
    model.train()

    if(torch.cuda.is_available()):
        model = model.cuda(args.gpu_id)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    print("------- Model Started -------")


    best_loss = np.inf
    for epoch in tqdm(range(args.num_epochs)):

        train_loss = 0

        for anchor, positive, negative in tqdm(train_dataloader,total=train_dataloader.__len__()):
            if(torch.cuda.is_available()):
                anchor = anchor.cuda(args.gpu_id)
                positive = positive.cuda(args.gpu_id)
                negative = negative.cuda(args.gpu_id)

            anchor_pred = model(anchor)
            positive_pred = model(positive)
            negative_pred = model(negative)
            
            loss = triplet_loss(anchor_pred,positive_pred,negative_pred)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= train_dataloader.__len__()

        val_loss = 0

        with torch.no_grad():
            for anchor, positive, negative in tqdm(val_dataloader,total=val_dataloader.__len__()):
                if(torch.cuda.is_available()):
                    anchor = anchor.cuda(args.gpu_id)
                    positive = positive.cuda(args.gpu_id)
                    negative = negative.cuda(args.gpu_id)

                anchor_pred = model(anchor)
                positive_pred = model(positive)
                negative_pred = model(negative)
                
                loss = triplet_loss(anchor_pred,positive_pred,negative_pred)
                val_loss += loss.item()
            val_loss /= val_dataloader.__len__()

            if(best_loss > val_loss):
                torch.save(model.state_dict(), f"{output_string}/best_model.pt")
                best_loss = val_loss

        print(f"Epoch [{epoch+1}/{args.num_epochs}] - Loss: {train_loss} - Val_Loss: {val_loss}")
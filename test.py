import os,argparse
import numpy as np
import cv2
from models import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import torchvision.utils as vutils
from osgeo import gdal
from PIL import Image

testPath=r'/home/server4/lkq/RS_Data/test231114/in'
jpgPath=r'/home/server4/lkq/RS_Data/test231114/out'
modelPath=r'/home/server4/lmk/decloudnet/trained_models/decloudnet_1219/Net_54000.pth'
if not os.path.exists(jpgPath):
    os.makedirs(jpgPath)
    
device='cuda' if torch.cuda.is_available() else 'cpu'
net=torch.load(modelPath)

tmp=[os.path.join(testPath,x) for x in os.listdir(testPath)]
for i in tmp:
    ids = i.split('/')[-1]
    #print(ids)
    im_data = Image.open(i)
    im_data = im_data.convert('RGB')
    im_data = tfs.ToTensor()(im_data)
    img = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(im_data)

    img = img.to('cuda')
    img = img.unsqueeze(0)
    with torch.no_grad():
        pred = net(img)
    pred = pred.squeeze(0)
    pred = pred.cpu().numpy().transpose(1, 2, 0)
    pred[:, :, 0]=pred[:, :, 0]*0.14+0.64
    pred[:, :, 1]=pred[:, :, 1]*0.15+0.6
    pred[:, :, 2]=pred[:, :, 2]*0.152+0.58
    pred = np.clip(pred*255, 0, 255)
    pred = pred.astype(np.uint8)

    savePath = os.path.join(jpgPath, ids)
    cv2.imwrite(savePath, pred)

    print(savePath, 'completed')

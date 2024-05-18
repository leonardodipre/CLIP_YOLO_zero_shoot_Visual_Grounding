import json
import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image, to_tensor
import random
import pandas as pd

import pickle

class SplitData():
    def __init__(self, root_dir, umd_file,):

        self.root_dir = root_dir
       

        self.annotation_path = os.path.join(root_dir, "annotations", "instances.json")
        self.images_dir = os.path.join(root_dir, "images")
        self.umd_file = umd_file
        
        # Load annotations and image metadata
        with open(self.annotation_path, 'r') as file:
            self.data = json.load(file)
        
        # Build an index of image IDs to image file info
        self.img_id_to_file = {img['id']: img for img in self.data['images']}
        
        # Load sentence references
        self.refs = pickle.load( open(os.path.join(root_dir, "annotations", self.umd_file), 'rb' )  )
        self.refs = pd.DataFrame.from_dict(self.refs)

        
        
        self.imm_list = {}

        for img in self.data['images']:
            
            self.imm_list[img['id']] = {}
            self.imm_list[img['id']]['file_name'] = img['file_name']
            self.imm_list[img['id']]['width'] = img['width']
            self.imm_list[img['id']]['height'] = img['height']

        self.annotation_list = {}

        for ann in self.data['annotations']:
            
            self.annotation_list[ann['id']] = {}
            self.annotation_list[ann['id']]['image_id']  = ann['image_id']
            self.annotation_list[ann['id']]['bbox']  =  ann['bbox']

            self.annotation_list[ann['id']] ['immage_name'] = self.imm_list[ann['image_id']]['file_name']
            self.annotation_list[ann['id']] ['width'] = self.imm_list[ann['image_id']]['width']
            self.annotation_list[ann['id']] ['height'] = self.imm_list[ann['image_id']]['height']

        #Pickel
        self.data_obj = {"immage_name": [] , "caption": [] , "split":[] , "bbox":[]}

        for index, annotation in self.refs.iterrows():

            annId = annotation.ann_id

            for i in range(len(annotation.sentences)):

                self.data_obj["immage_name"].append( self.annotation_list[annId]["immage_name"] ) 
                self.data_obj["caption"].append(annotation.sentences[i]["sent"])
                self.data_obj["split"].append(annotation.split)
                self.data_obj["bbox"].append(self.annotation_list[annId]["bbox"])

        
        self.data_obj = pd.DataFrame.from_dict(self.data_obj)

        self.train_set  = self.data_obj.loc[self.data_obj["split"]== "train"]
        self.test_set = self.data_obj.loc[self.data_obj["split"]== "test"]
        self.val_set = self.data_obj.loc[self.data_obj["split"]== "val"]

    def return_split(self):

        return self.train_set , self.test_set , self.val_set


class RefCOCODataset(Dataset):
    def __init__(self, data_split , root_dir , transform):
        
        self.transform = transform
        self.data = data_split
        self.img_path = []
        self.caption = []
        self.bbox = []

        for index , item in self.data.iterrows():
            
            self.img_path.append( (os.path.join(root_dir, "images/") + item["immage_name"] ) )
            
            self.caption.append(item["caption"])
            self.bbox.append(item["bbox"])

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        image_path= self.img_path[idx]
        image = Image.open(image_path).convert("RGB")

        width , height = image.size

        image = self.transform(image)

        caption = self.caption[idx]
      
        bbox_norm = self.normalize( self.bbox[idx] , width, height)
  

        return image , caption  , bbox_norm

    def normalize(self, bbox, width , height):
        
        scalex = 256 /width
        scaley = 256 /height

        bbox[0] *= scalex
        bbox[1] *= scaley

        bbox[2] = (bbox[2] * scalex) + bbox[0]
        bbox[3] = (bbox[3] * scaley ) +  bbox[1]

        return torch.Tensor(bbox) 

    




"""
path = "/home/leo/Documents/Dataset/RefCocog/refcocog/"


spit_data = SplitData(root_dir=path, umd_file='refs(umd).p')

transform = transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train , test , val = spit_data.return_split()

dataset = RefCOCODataset(data_split = train , root_dir=path , transform =transform)




image , caption , bbox  = dataset[17]

print(caption)

immagine = to_pil_image(image)

x1 , x2 , y1, y2 = bbox



draw = ImageDraw.Draw(immagine)
draw.rectangle( [x1 , x2 , y1, y2 ], outline="green", width=3)
immagine.save("output.jpg")

"""
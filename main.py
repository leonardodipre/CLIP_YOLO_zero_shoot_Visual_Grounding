import numpy as np
from Dataloader import *
import open_clip
from torch.utils.data import DataLoader
from open_clip import tokenizer
from torchvision.transforms.functional import to_pil_image, to_tensor
from utils_ import *
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  


path = "/home/leo/Documents/Dataset/RefCocog/refcocog/"

spit_data = SplitData(root_dir=path, umd_file='refs(umd).p')

transform = transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


train , test , val = spit_data.return_split()

dataset_train = RefCOCODataset(data_split = train , root_dir=path , transform =transform)
dataset_test = RefCOCODataset(data_split = test , root_dir=path , transform =transform)
dataset_val = RefCOCODataset(data_split = val , root_dir=path , transform =transform)

print("Dimentions of sets")
print("tarin" , len(dataset_train))
print("Test" , len(dataset_test))
print("Val" , len(dataset_val))

batch_size = 64  # You can adjust this according to your system's capability


# Setting up DataLoaders
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)



#CLIP pretrain model
model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')

#Model Yolo
model_YOLO = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

loss_array = []
#Enumeriamo sul dataloader
for sample in tqdm(val_loader, desc="Processing Validation Data"):

    image_tensors, sentences , target_box = sample

    images = [to_pil_image(img) for img in image_tensors]
    results = model_YOLO(images)
    
    #Per ogni immagine passata con yolo andiamo a predere le BBOX 
    for idx, image in enumerate(images):
        #print("IDX" , idx , "Image" , image)
        sent = sentences[idx]
        
       
        target = target_box[idx]
        
        xyxy_bboxes = results.xyxy[idx]
      

        croped = cropp_immage(image , xyxy_bboxes , preprocess)

        image_input_clip = torch.tensor(np.stack(croped))

        with torch.no_grad():

            image_features = model.encode_image(image_input_clip).float()
           
            text_tokens = tokenizer.tokenize(sent)

            text_features = model.encode_text(text_tokens).float()
            

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T 
            
            max_index = np.argmax(similarity[0])  
            max_value = similarity[0][max_index]   

           
            if ( len(xyxy_bboxes)  == 0):
               
                loss = torch.tensor(0.0)

            else:
                pred_bbox = xyxy_bboxes[max_index].cpu()

                loss = bbox_iou(pred_bbox , target)

              
            loss_array.append(loss)

            
print("Total loss IOU accuracy is: " , np.asarray(loss_array).mean())            
           
            
            
          

    
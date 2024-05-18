import torch
from PIL import Image, ImageDraw
import numpy as np


def cropp_immage(image  , xyxy_bboxes, preprocess):
    
    cropped_images = []

    #se immagine non ha YOLO bbox, aggiunge l'immagine
    if(len(xyxy_bboxes ) == 0):
        cropped_images.append( preprocess(image))
    else:

        for idx_box , bbox in enumerate(xyxy_bboxes):

                xmin  ,  ymin  ,  xmax ,  ymax , _  ,  _ = bbox.tolist()
                
                cropped_image = image.crop((xmin, ymin, xmax, ymax))
                
                cropped_images.append( preprocess(cropped_image))

      
    
    return cropped_images



def bbox_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    bbox1, bbox2 (torch.Tensor): bounding boxes in the format [x1, y1, x2, y2]

    Returns:
    float: the IoU ratio.
    """
    # Convert tensors to items if they are 0-dimensional
    if bbox1.dim() == 0:
        bbox1 = bbox1.item()
    if bbox2.dim() == 0:
        bbox2 = bbox2.item()

    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    box1Area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    box2Area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou




"""
            for drawing bbox
            x1 , x2 , y1, y2 , _ , _= pred_bbox
            x1_t , x2__t , y1_t, y2_t = target_box


            draw = ImageDraw.Draw(image)

            draw.rectangle( [x1 , x2 , y1, y2 ], outline="red", width=3)
            draw.rectangle( [x1_t , x2__t , y1_t, y2_t ], outline="black", width=3)
            image.save("output.jpg")
"""
# ------------------------------------------------------------------------------
# Modified based on https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------

import os
import copy
import random

import cv2
import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
import torch
from .base_dataset import BaseDataset
import matplotlib.pyplot as plt

def show_mask(mask, ax=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return mask_image

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

class SkyGround(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 sam_checkpoint,
                 sam_model_type,
                 sam_device,
                 use_teacher_model,
                 num_classes=2,
                 multi_scale=False,
                 flip=True, 
                 ignore_label=255, 
                 base_size=1280, 
                 crop_size=(480, 640),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4
                 ):

        super(SkyGround, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path

        ### teacher model ###
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.sam_device = sam_device
        #####################

        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip
        self.img_list = [line.strip().split() for line in open(root+list_path)]
        self.files = self.read_files()

        self.label_mapping = {0: 0, 1: 1}
        self.class_weights = torch.FloatTensor([0.900, 1.100]).cuda()
        
        self.bd_dilate_size = bd_dilate_size
        # Initialize the teacher model (SAM)
        self.use_teacher_model = use_teacher_model
        teacher_model = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
        teacher_model.to(device=self.sam_device)
        self.teacher_model = teacher_model
        predictor = SamPredictor(teacher_model)
        self.predictor = predictor


    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                new_files = [{
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "aug": []
                }]

                files.extend(new_files)
        return files
        
    def convert_label(self, label, inverse=False):
        temp = np.full_like(label, self.ignore_label)
        if inverse:
            for v, k in self.label_mapping.items():
                temp[label == k] = v
        else:
            for k, v in self.label_mapping.items():
                temp[label == k] = v
        return temp

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root, item["img"]))
        # change image to grayscale and resize it to 640x480
        # Resize the image to 640x480
        resized_image = cv2.resize(image, (640, 480))

        # Convert the resized image to grayscale
        #image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        if self.use_teacher_model:
            ### instead of reading segmentaion mask from dataset -> use SAM model
            # Generate segment masks using the teacher mode
            with torch.no_grad():
                self.predictor.set_image(image)

                # if one wants to change the star position which SAM generates the sky mask relay on.
                input_point = np.array([[30, 30], [image.shape[1] / 2, 30], [image.shape[1] - 30, 30]])
                input_label = np.array([1, 1, 1])

                masks, scores, logits = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False,
                )

                best_output = np.max(scores)
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    if (best_output == score):
                        label = mask.astype(np.uint8)

        else:
            label = cv2.imread(os.path.join(self.root, item["label"]),
                               cv2.IMREAD_GRAYSCALE)
            # Resize the label to 640x480
            label = cv2.resize(label, (640, 480))
            # Convert the label image to a binary mask (sky-0,ground-1)
            label = np.where(label == 0, 0, 1).astype(np.uint8)

        label = self.convert_label(label)
        image, label, edge = self.gen_sample(image, label,
                                self.multi_scale, self.flip, edge_size=self.bd_dilate_size,city=False)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred


    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.tif'))

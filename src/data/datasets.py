# import cv2
# import torch
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset
# import utils


# class AVLip(Dataset):
#     def __init__(self, opt):
#         assert opt.data_label in ["train", "val"]
#         self.data_label = opt.data_label
#         self.real_list = utils.get_list(opt.real_list_path)
#         self.fake_list = utils.get_list(opt.fake_list_path)
#         self.label_dict = dict()
#         for i in self.real_list:
#             self.label_dict[i] = 0
#         for i in self.fake_list:
#             self.label_dict[i] = 1
#         self.total_list = self.real_list + self.fake_list

#     def __len__(self):
#         return len(self.total_list)

#     def __getitem__(self, idx):
#         img_path = self.total_list[idx]
#         label = self.label_dict[img_path]
#         img = torch.tensor(cv2.imread(img_path), dtype=torch.float32)
#         img = img.permute(2, 0, 1)
#         crops = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
#                                      std=[0.26862954, 0.26130258, 0.27577711])(img)
#         # crop images
#         # crops[0]: 1.0x, crops[1]: 0.65x, crops[2]: 0.45x
#         crops = [[transforms.Resize((224, 224))(img[:, 500:, i:i + 500]) for i in range(5)], [], []]
#         crop_idx = [(28, 196), (61, 163)]
#         for i in range(len(crops[0])):
#             crops[1].append(transforms.Resize((224, 224))
#                             (crops[0][i][:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]))
#             crops[2].append(transforms.Resize((224, 224))
#                             (crops[0][i][:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]))
#         img = transforms.Resize((1120, 1120))(img)

#         return img, crops, label

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from retinaface import RetinaFace
from typing import Tuple, List, Optional
from PIL import Image
import os
import utils


# class AVLip(Dataset):
#     def __init__(self, opt):
#         assert opt.data_label in ["train", "val"]
#         self.data_label = opt.data_label
#         self.real_list = utils.get_list(opt.real_list_path)
#         self.fake_list = utils.get_list(opt.fake_list_path)
#         self.label_dict = dict()
#         for i in self.real_list:
#             self.label_dict[i] = 0
#         for i in self.fake_list:
#             self.label_dict[i] = 1
#         self.total_list = self.real_list + self.fake_list

#     def __len__(self):
#         return len(self.total_list)

#     def __getitem__(self, idx):
#         img_path = self.total_list[idx]
#         label = self.label_dict[img_path]
#         img = torch.tensor(cv2.imread(img_path), dtype=torch.float32)
#         img = img.permute(2, 0, 1)
#         crops = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
#                                      std=[0.26862954, 0.26130258, 0.27577711])(img)
#         # crop images
#         # crops[0]: 1.0x, crops[1]: 0.65x, crops[2]: 0.45x
#         crops = [[transforms.Resize((224, 224))(img[:, 500:, i:i + 500]) for i in range(5)], [], []]
#         crop_idx = [(28, 196), (61, 163)]
#         for i in range(len(crops[0])):
#             crops[1].append(transforms.Resize((224, 224))
#                             (crops[0][i][:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]))
#             crops[2].append(transforms.Resize((224, 224))
#                             (crops[0][i][:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]))
#         img = transforms.Resize((1120, 1120))(img)

#         return img, crops, label


class AVLip(Dataset):
    def __init__(self, opt):
        assert opt.data_label in ["train", "val"]
        self.data_label = opt.data_label
        
        # Load dataset lists and create label dictionary
        self.real_list = utils.get_list(opt.real_list_path)
        self.fake_list = utils.get_list(opt.fake_list_path)
        self.label_dict = {path: 0 for path in self.real_list}
        self.label_dict.update({path: 1 for path in self.fake_list})
        self.total_list = self.real_list + self.fake_list
        
        # Initialize transforms
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.resize = transforms.Resize((224, 224))

        self.total_images_attempted = 0
        self.valid_images = 0

    def __len__(self):
        return len(self.total_list)

    def detect_and_crop(self, img):
        try:
            resp = RetinaFace.detect_faces(img)
            if resp is None or not resp:
                return None, None

            face_data = resp['face_1']
            x1, y1, x2, y2 = face_data['facial_area']
            landmarks = face_data['landmarks']
            mouth_right = landmarks['mouth_right']
            mouth_left = landmarks['mouth_left']

            face_height = y2 - y1
            face_center_x = (x1 + x2) // 2
            face_x1 = max(0, face_center_x - face_height//2)
            face_x2 = min(img.shape[1], face_center_x + face_height//2)
            face_crop = img[y1:y2, face_x1:face_x2]

            lip_y_center = int((mouth_right[1] + mouth_left[1]) / 2)
            lip_x_center = int((mouth_right[0] + mouth_left[0]) / 2)
            lip_height = int((y2 - y1) * 0.45)
            lip_y1 = max(0, lip_y_center - lip_height//2)
            lip_y2 = min(img.shape[0], lip_y_center + lip_height//2)
            lip_x1 = max(0, lip_x_center - lip_height//2)
            lip_x2 = min(img.shape[1], lip_x_center + lip_height//2)
            lip_crop = img[lip_y1:lip_y2, lip_x1:lip_x2]

            return face_crop, lip_crop
        except Exception as e:
            return None, None

    def __getitem__(self, idx):
        while idx < len(self.total_list):
            try:
                self.total_images_attempted += 1
                img_path = self.total_list[idx]
                label = self.label_dict[img_path]
               
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    idx += 1
                    continue
                
                img = torch.tensor(img_bgr, dtype=torch.float32)
                img = img.permute(2, 0, 1)
                crops = self.normalize(img)
                
                initial_crops = []
                face_crops = []
                lip_crops = []
                
                all_crops_valid = True
                for i in range(5):
                    crop_region = img[:, 500:, i:i + 500].permute(1, 2, 0).numpy().astype(np.uint8)
                    face_crop, lip_crop = self.detect_and_crop(crop_region)
                    
                    if face_crop is None or lip_crop is None:
                        all_crops_valid = False
                        break
                    
                    initial_tensor = torch.tensor(crop_region, dtype=torch.float32).permute(2, 0, 1)
                    face_tensor = torch.tensor(face_crop, dtype=torch.float32).permute(2, 0, 1)
                    lip_tensor = torch.tensor(lip_crop, dtype=torch.float32).permute(2, 0, 1)
                    
                    initial_tensor = self.resize(initial_tensor)
                    face_tensor = self.resize(face_tensor)
                    lip_tensor = self.resize(lip_tensor)
                    
                    initial_crops.append(initial_tensor)
                    face_crops.append(face_tensor)
                    lip_crops.append(lip_tensor)
                
                if not all_crops_valid:
                    idx += 1
                    continue
                
                crops = [initial_crops, face_crops, lip_crops]
                
                img = transforms.Resize((1120, 1120))(img)
                
                self.valid_images += 1

                return img, crops, label
                
            except Exception as e:
                idx += 1
                continue
            
        raise RuntimeError("No valid images found in dataset")

    def get_stats(self):
        return {
            'total_images': self.total_images_attempted,
            'valid_images': self.valid_images,
            'invalid_rate': (self.total_images_attempted - self.valid_images) / self.total_images_attempted if self.total_images_attempted > 0 else 0
        }
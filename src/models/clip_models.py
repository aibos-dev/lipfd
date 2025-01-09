# import torch
# import torch.nn as nn
# from .clip import clip

# CHANNELS = {
#     "RN50" : 1024,
#     "ViT-L/14" : 768
# }

# class CLIPModel(nn.Module):
#     def __init__(self, name, num_classes=1):
#         super(CLIPModel, self).__init__()

#         # Load the CLIP model
#         self.model, self.preprocess = clip.load(name, device="cpu")  # Initially load on CPU
        
#         # Move model to multiple GPUs using DataParallel
#         if torch.cuda.is_available():
#             device_ids = [0, 1, 2, 3]  # List of GPU IDs
#             self.model = nn.DataParallel(self.model, device_ids=device_ids)
#             self.model = self.model.cuda()  # Move the model to GPUs
        
#         # Add the final linear layer
#         self.fc = nn.Linear(CHANNELS[name], num_classes)

#     def forward(self, x, return_feature=False):
#         # Move input tensor to the same device as the model
#         x = x.to(self.model.device)  # Ensure input is on the same device as the model
        
#         # Forward pass through the model
#         features = self.model.module.encode_image(x)  # Use .module to access the model wrapped by DataParallel
        
#         if return_feature:
#             return features
#         return self.fc(features)

from .clip import clip 
from PIL import Image
import torch.nn as nn


CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear( CHANNELS[name], num_classes )
 

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x) 
        if return_feature:
            return features
        return self.fc(features)


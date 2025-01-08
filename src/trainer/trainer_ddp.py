# import os
# import torch
# import torch.nn as nn
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel
# import torch.nn.functional as F
# from models import build_model, get_loss


# class DistributedTrainer(nn.Module):
#     """Distributed version of the Trainer class"""
#     def __init__(self, opt, local_rank, find_unused_parameters=False):
#         super().__init__()
#         self.opt = opt
#         self.total_steps = 0
#         self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
#         self.device = torch.device(f'cuda:{local_rank}')
        
#         # Create directories only on main process
#         if dist.get_rank() == 0:
#             os.makedirs(self.save_dir, exist_ok=True)
            
#         # Build and wrap model with DDP
#         self.model = build_model(opt.arch)
#         self.model.to(self.device)
        
#         # Add support for find_unused_parameters in DDP
#         self.model = DistributedDataParallel(
#             self.model,
#             device_ids=[local_rank],
#             find_unused_parameters=find_unused_parameters  # Added this line
#         )
        
#         # Load pretrained if fine-tuning
#         self.step_bias = 0
#         if opt.fine_tune:
#             self.step_bias = int(opt.pretrained_model.split("_")[-1].split(".")[0]) + 1
#             state_dict = torch.load(opt.pretrained_model, map_location=self.device)
#             # Load into DDP's module
#             self.model.module.load_state_dict(state_dict["model"])
#             self.total_steps = state_dict["total_steps"]
#             if dist.get_rank() == 0:
#                 print(f"Model loaded @ {opt.pretrained_model.split('/')[-1]}")

#         # Setup optimizer
#         if opt.fix_encoder:
#             params = []
#             for name, p in self.model.named_parameters():
#                 if name.split(".")[0] in ["encoder"]:
#                     p.requires_grad = False
#                 else:
#                     p.requires_grad = True
#                     params.append(p)
#         else:
#             params = self.model.parameters()

#         if opt.optim == "adam":
#             self.optimizer = torch.optim.AdamW(
#                 params,
#                 lr=opt.lr,
#                 betas=(opt.beta1, 0.999),
#                 weight_decay=opt.weight_decay,
#             )
#         elif opt.optim == "sgd":
#             self.optimizer = torch.optim.SGD(
#                 params, 
#                 lr=opt.lr, 
#                 momentum=0.0, 
#                 weight_decay=opt.weight_decay
#             )
        
#         self.criterion = get_loss().to(self.device)
#         self.criterion1 = nn.CrossEntropyLoss()
#         self.loss = None

#     def set_input(self, input):
#         """Move input data to device"""
#         self.input = input[0].to(self.device)
#         self.crops = [[t.to(self.device) for t in sublist] for sublist in input[1]]
#         self.label = input[2].to(self.device).float()

#     def forward(self):
#         """Forward pass using DDP model"""
#         self.get_features()
#         self.output, self.weights_max, self.weights_org = self.model(
#             self.crops, self.features
#         )
#         self.output = self.output.view(-1)
#         self.loss = self.criterion(
#             self.weights_max, self.weights_org
#         ) + self.criterion1(self.output, self.label)

#     def get_features(self):
#         """Get features using DDP model's module"""
#         self.features = self.model.module.get_features(self.input).to(self.device)

#     def optimize_parameters(self):
#         """Optimize with gradient synchronization handled by DDP"""
#         if self.loss is None:
#             if dist.get_rank() == 0:
#                 print("Warning: Loss is None!")
#             return None
#         if not self.loss.requires_grad:
#             if dist.get_rank() == 0:
#                 print("Warning: Loss doesn't require gradient!")
#             return None

#         self.optimizer.zero_grad()
#         loss_value = self.loss.item()
#         self.loss.backward()
        
#         # Gradient monitoring
#         total_grad_norm = 0
#         num_params_with_grad = 0
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 if param.grad is None:
#                     if dist.get_rank() == 0:
#                         print(f"Warning: Parameter {name} has no gradient!")
#                 else:
#                     grad_norm = param.grad.norm().item()
#                     total_grad_norm += grad_norm
#                     num_params_with_grad += 1
                    
#                     if torch.isnan(param.grad).any():
#                         if dist.get_rank() == 0:
#                             print(f"Warning: NaN gradients detected in {name}")
#                     if torch.isinf(param.grad).any():
#                         if dist.get_rank() == 0:
#                             print(f"Warning: Inf gradients detected in {name}")
        
#         # Gradient clipping (optional)
#         if hasattr(self.opt, 'grad_clip') and self.opt.grad_clip > 0:
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.grad_clip)
        
#         self.optimizer.step()
        
#         return {
#             'loss': loss_value,
#             'avg_grad_norm': total_grad_norm / num_params_with_grad if num_params_with_grad > 0 else 0
#         }

#     def eval(self):
#         """Set model to evaluation mode"""
#         self.model.eval()

#     def train(self):
#         """Set model to training mode"""
#         self.model.train()

#     def save_networks(self, save_filename):
#         """Save model (only on main process)"""
#         if dist.get_rank() != 0:
#             return
#         save_path = os.path.join(self.save_dir, save_filename)
#         state_dict = {
#             "model": self.model.module.state_dict(),
#             "optimizer": self.optimizer.state_dict(),
#             "total_steps": self.total_steps,
#         }
#         torch.save(state_dict, save_path)

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from models import build_model, get_loss

class DistributedTrainer(nn.Module):
    """Distributed version of the Trainer class"""
    def __init__(self, opt, local_rank, find_unused_parameters=False):
        super().__init__()
        self.opt = opt
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device(f'cuda:{local_rank}')
        
        # Create directories only on main process
        if dist.get_rank() == 0:
            os.makedirs(self.save_dir, exist_ok=True)
            
        # Set same seed for model initialization across all ranks
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # Build model
        self.model = build_model(opt.arch)
        self.model.to(self.device)
        
        # Synchronize model parameters across ranks before DDP wrapping
        with torch.no_grad():
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)
        
        # Wait for all processes to finish parameter sync
        dist.barrier()
        
        # Wrap with DDP after synchronization
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[local_rank],
            find_unused_parameters=find_unused_parameters,
            broadcast_buffers=True  # Ensure buffers are synchronized
        )
        
        # Load pretrained if fine-tuning
        self.step_bias = 0
        if opt.fine_tune:
            self.step_bias = int(opt.pretrained_model.split("_")[-1].split(".")[0]) + 1
            # Load state dict on rank 0 and broadcast to others
            if dist.get_rank() == 0:
                state_dict = torch.load(opt.pretrained_model, map_location=self.device)
                self.model.module.load_state_dict(state_dict["model"])
                self.total_steps = state_dict["total_steps"]
                print(f"Model loaded @ {opt.pretrained_model.split('/')[-1]}")
            
            # Broadcast loaded parameters to all ranks
            with torch.no_grad():
                for param in self.model.parameters():
                    dist.broadcast(param.data, src=0)
            dist.barrier()
            
            # Broadcast total_steps to all ranks
            total_steps_tensor = torch.tensor([self.total_steps], device=self.device)
            dist.broadcast(total_steps_tensor, src=0)
            self.total_steps = total_steps_tensor.item()

        # Setup optimizer
        if opt.fix_encoder:
            params = []
            for name, p in self.model.named_parameters():
                if name.split(".")[0] in ["encoder"]:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                    params.append(p)
        else:
            params = self.model.parameters()

        if opt.optim == "adam":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                params, 
                lr=opt.lr, 
                momentum=0.0, 
                weight_decay=opt.weight_decay
            )
        
        self.criterion = get_loss().to(self.device)
        self.criterion1 = nn.CrossEntropyLoss()
        self.loss = None
        
        # Final barrier to ensure all processes are ready
        dist.barrier()

        # Initialize gradient accumulation parameters
        self.accumulation_steps = opt.accumulation_steps if hasattr(opt, 'accumulation_steps') else 1
        self.accumulated_steps = 0

    def set_input(self, input):
        """Move input data to device"""
        self.input = input[0].to(self.device)
        self.crops = [[t.to(self.device) for t in sublist] for sublist in input[1]]
        self.label = input[2].to(self.device).float()

    def forward(self):
        """Forward pass using DDP model"""
        self.get_features()
        self.output, self.weights_max, self.weights_org = self.model(
            self.crops, self.features
        )
        self.output = self.output.view(-1)
        self.loss = self.criterion(
            self.weights_max, self.weights_org
        ) + self.criterion1(self.output, self.label)

    def get_features(self):
        """Get features using DDP model's module"""
        self.features = self.model.module.get_features(self.input).to(self.device)

    def optimize_parameters(self):
        """Optimize with gradient synchronization handled by DDP and gradient accumulation"""
        if self.loss is None:
            if dist.get_rank() == 0:
                print("Warning: Loss is None!")
            return None
        if not self.loss.requires_grad:
            if dist.get_rank() == 0:
                print("Warning: Loss doesn't require gradient!")
            return None

        self.optimizer.zero_grad()
        loss_value = self.loss.item()
        self.loss.backward()

        # Accumulate gradients
        self.accumulated_steps += 1
        if self.accumulated_steps % self.accumulation_steps == 0:
            self.optimizer.step()
            self.accumulated_steps = 0  # Reset the step counter after an update

        total_grad_norm = 0
        num_params_with_grad = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    if dist.get_rank() == 0:
                        print(f"Warning: Parameter {name} has no gradient!")
                else:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    num_params_with_grad += 1
                    
                    if torch.isnan(param.grad).any():
                        if dist.get_rank() == 0:
                            print(f"Warning: NaN gradients detected in {name}")
                    if torch.isinf(param.grad).any():
                        if dist.get_rank() == 0:
                            print(f"Warning: Inf gradients detected in {name}")
        
        if hasattr(self.opt, 'grad_clip') and self.opt.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.grad_clip)
        
        return {
            'loss': loss_value,
            'avg_grad_norm': total_grad_norm / num_params_with_grad if num_params_with_grad > 0 else 0
        }

    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()

    def train(self):
        """Set model to training mode"""
        self.model.train()

    def save_networks(self, save_filename):
        """Save model (only on main process)"""
        if dist.get_rank() != 0:
            return
        save_path = os.path.join(self.save_dir, save_filename)
        state_dict = {
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }
        torch.save(state_dict, save_path)

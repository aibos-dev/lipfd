# import os
# import torch
# import torch.distributed as dist
# from torch.cuda.amp import GradScaler, autocast
# from validate import validate
# from data import create_distributed_dataloader
# from trainer.trainer_ddp import DistributedTrainer
# from options.train_options import TrainOptions
# from tqdm import tqdm
# import random
# import warnings

# warnings.filterwarnings("ignore", category=UserWarning)

# # Define the Genetic Algorithm Class
# class GeneticAlgorithm:
#     def __init__(self, population_size, mutation_rate, crossover_rate, max_generations):
#         self.population_size = population_size
#         self.mutation_rate = mutation_rate
#         self.crossover_rate = crossover_rate
#         self.max_generations = max_generations

#     def initialize_population(self, model):
#         population = []
#         for _ in range(self.population_size):
#             individual = self.create_individual(model)
#             population.append(individual)
#         return population

#     def create_individual(self, model):
#         individual = {}
#         for name, param in model.named_parameters():
#             individual[name] = torch.randn_like(param)
#         return individual

#     def crossover(self, parent1, parent2):
#         child = {}
#         for name in parent1:
#             if random.random() < self.crossover_rate:
#                 child[name] = parent1[name]  # Copy from parent1
#             else:
#                 child[name] = parent2[name]  # Copy from parent2
#         return child

#     def mutate(self, individual):
#         for name in individual:
#             if random.random() < self.mutation_rate:
#                 individual[name] += torch.randn_like(individual[name]) * 0.1
#         return individual

#     def evaluate(self, individual, model, trainer):
#         with torch.no_grad():
#             for name, param in model.named_parameters():
#                 param.copy_(individual[name])
#         trainer.forward()  # Calculate loss using the current individual weights
#         return trainer.loss.item()

#     def select_parents(self, population, model, trainer):
#         fitness_scores = [self.evaluate(individual, model, trainer) for individual in population]
#         sorted_population = [individual for _, individual in sorted(zip(fitness_scores, population))]
#         return sorted_population[:2]  # Select two best parents

#     def evolve(self, model, trainer):
#         population = self.initialize_population(model)
#         best_individual = None
#         best_fitness = float('inf')

#         for generation in range(self.max_generations):
#             print(f"Generation {generation + 1}/{self.max_generations}")
#             new_population = []

#             while len(new_population) < self.population_size:
#                 parent1, parent2 = self.select_parents(population, model, trainer)

#                 child = self.crossover(parent1, parent2)
#                 child = self.mutate(child)

#                 new_population.append(child)

#             for individual in new_population:
#                 fitness = self.evaluate(individual, model, trainer)
#                 if fitness < best_fitness:
#                     best_fitness = fitness
#                     best_individual = individual

#             population = new_population

#         return best_individual, best_fitness


# def get_val_opt():
#     val_opt = TrainOptions().parse(print_options=False)
#     val_opt.isTrain = False
#     val_opt.data_label = "val"
#     val_opt.real_list_path = "/workspace/datasets/AVlips_dataset/preprocessed_exp/val/0_real"
#     val_opt.fake_list_path = "/workspace/datasets/AVlips_dataset/preprocessed_exp/val/1_fake"
#     return val_opt

# def main():
#     try:
#         # Specify GPUs to use (0, 1, 2, 3)
#         os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

#         # Initialize TrainOptions here
#         opt = TrainOptions().parse()

#         # Check if the environment variables are set for distributed training
#         local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Default to 0 if not set
#         world_size = int(os.environ.get("WORLD_SIZE", 4))  # Default to 4 if not set (4 GPUs)

#         print(f"Starting process with LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}")

#         # Initialize distributed environment
#         dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
#         torch.cuda.set_device(local_rank)  # Ensure each process uses the right GPU

#         print(f"Distributed environment initialized on rank {local_rank} using GPU {local_rank}")

#         # Add local_rank to opt
#         opt.local_rank = local_rank

#         # Add this check and update if world_size is not set in opt
#         if not hasattr(opt, 'world_size'):
#             opt.world_size = world_size  # Set world_size manually from environment variable

#         opt.real_list_path = "/workspace/datasets/AVlips_dataset/cropped/train/0_real"
#         opt.fake_list_path = "/workspace/datasets/AVlips_dataset/cropped/train/1_fake"

#         val_opt = get_val_opt()
#         val_opt.real_list_path = "/workspace/datasets/AVlips_dataset/cropped/val/0_real"
#         val_opt.fake_list_path = "/workspace/datasets/AVlips_dataset/cropped/val/1_fake"

#         # Ensure that batch size is not zero after division
#         if opt.batch_size % world_size != 0:
#             print(f"Warning: Batch size {opt.batch_size} is not divisible by world size {world_size}.")
#             opt.batch_size = opt.batch_size // world_size + (1 if opt.batch_size % world_size > 0 else 0)

#         if val_opt.batch_size % world_size != 0:
#             print(f"Warning: Validation batch size {val_opt.batch_size} is not divisible by world size {world_size}.")
#             val_opt.batch_size = val_opt.batch_size // world_size + (1 if val_opt.batch_size % world_size > 0 else 0)

#         print(f"Training batch size: {opt.batch_size}")
#         print(f"Validation batch size: {val_opt.batch_size}")

#         model = DistributedTrainer(opt, local_rank)
#         data_loader, train_sampler = create_distributed_dataloader(opt, num_workers=4, pin_memory=True, rank=opt.local_rank)
#         val_loader, _ = create_distributed_dataloader(val_opt, num_workers=4, pin_memory=True, rank=opt.local_rank)

#         if local_rank == 0:
#             print(f"Length of data loader: {len(data_loader)}")
#             print(f"Length of val loader: {len(val_loader)}")

#         # Initialize GA
#         ga = GeneticAlgorithm(
#             population_size=10,
#             mutation_rate=0.1,
#             crossover_rate=0.7,
#             max_generations=5
#         )

#         scaler = GradScaler()
#         best_val_metrics = {'ap': 0.0, 'acc': 0.0, 'epoch': -1}

#         for epoch in range(opt.epoch):
#             model.train()
#             train_sampler.set_epoch(epoch)

#             if local_rank == 0:
#                 print(f"Epoch: {epoch + model.step_bias}")

#             total_loss = 0
#             num_batches = 0

#             for i, (img, crops, label) in enumerate(tqdm(
#                 data_loader,
#                 desc=f"Training Epoch {epoch + 1}",
#                 disable=local_rank != 0
#             )):

#                 model.total_steps += 1
#                 model.set_input((img, crops, label))

#                 with autocast():
#                     model.forward()
#                     opt_info = model.optimize_parameters()

#                 if opt_info is not None:
#                     total_loss += opt_info['loss']
#                     num_batches += 1

#                     if local_rank == 0 and i % 100 == 0:
#                         print(f"Step {i}: Loss={opt_info['loss']:.4f}, "
#                               f"Avg Grad Norm={opt_info['avg_grad_norm']:.4f}")

#                     if local_rank == 0 and model.total_steps % opt.loss_freq == 0:
#                         tqdm.write(
#                             f"Train loss: {opt_info['loss']:.4f}\tstep: {model.total_steps}"
#                         )

#             avg_train_loss = total_loss / num_batches if num_batches > 0 else float('inf')

#             dist.barrier()

#             # Validation
#             if local_rank == 0:
#                 model.eval()
#                 with torch.no_grad():
#                     ap, fpr, fnr, acc = validate(model.model.module, val_loader, [local_rank])
#                     print(
#                         f"(Val @ epoch {epoch + model.step_bias}) "
#                         f"acc: {acc:.4f} ap: {ap:.4f} fpr: {fpr:.4f} fnr: {fnr:.4f}"
#                     )

#                     if acc > best_val_metrics['acc']:
#                         best_val_metrics = {'ap': ap, 'acc': acc, 'epoch': epoch}
#                         model.save_networks("/workspace/checkpoints/best_model.pth")  # Saved as "best_model.pth"
#                         print(f"New best model saved! Accuracy: {acc:.4f}")

#                     if epoch % opt.save_epoch_freq == 0:
#                         model.save_networks(f"/workspace/checkpoints/model_epoch_{epoch + model.step_bias}.pth")  # Saved as "model_epoch_{epoch}.pth"

#             # Use GA to evolve the model parameters
#             if local_rank == 0:
#                 best_individual, _ = ga.evolve(model.model, model)
#                 with torch.no_grad():
#                     for name, param in model.model.named_parameters():
#                         param.copy_(best_individual[name])

#         if local_rank == 0:
#             print("\nTraining completed!")
#             print(f"Best model performance (epoch {best_val_metrics['epoch']}):")
#             print(f"Accuracy: {best_val_metrics['acc']:.4f}")
#             print(f"AP: {best_val_metrics['ap']:.4f}")

#     except KeyboardInterrupt:
#         print("\nTraining interrupted by user")
#     except Exception as e:
#         print(f"Error on rank {local_rank}: {str(e)}")
#         raise e
#     finally:
#         dist.destroy_process_group()
#         torch.cuda.empty_cache()

# if __name__ == "__main__":
#     main()

import os
import torch
import torch.distributed as dist
from validate import validate
from data import create_distributed_dataloader
from trainer.trainer_ddp import DistributedTrainer
from options.train_options import TrainOptions
from tqdm import tqdm
import warnings
import numpy as np
from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore", category=UserWarning)

def synchronize_model(model, local_rank):
    """Ensure model parameters are synchronized across all ranks"""
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    return model

def evaluate_hyperparameters(individual, local_rank):
    """Modified to ensure model parameters are synchronized across ranks"""
    lr, batch_size = individual
    
    # Create a new TrainOptions instance with updated hyperparameters
    opt = TrainOptions().parse(print_options=False)
    opt.lr = lr
    opt.batch_size = int(batch_size)

    try:
        # Ensure each process uses its assigned GPU
        torch.cuda.set_device(local_rank)
        
        # Initialize the model and synchronize parameters
        model = DistributedTrainer(opt, local_rank)
        model = synchronize_model(model.model, local_rank)
        
        # Wait for all processes to finish model initialization
        dist.barrier()
        
        data_loader, _ = create_distributed_dataloader(opt)

        # Train for a single epoch to evaluate hyperparameters
        total_loss = 0
        num_batches = 0
        
        model.train()
        for img, crops, label in data_loader:
            model.set_input((img, crops, label))
            model.forward()
            opt_info = model.optimize_parameters()

            if opt_info is not None:
                total_loss += opt_info['loss']
                num_batches += 1

        # Gather losses from all processes
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        gathered_losses = [torch.tensor(avg_loss).cuda(local_rank)]
        dist.all_gather(gathered_losses, torch.tensor(avg_loss).cuda(local_rank))
        
        # Use the mean loss across all processes
        mean_loss = torch.mean(torch.stack(gathered_losses)).item()
        return mean_loss,

    except Exception as e:
        print(f"Error during evaluation on rank {local_rank}: {e}")
        return float('inf'),

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.data_label = "val"
    val_opt.real_list_path = "/workspace/datasets/AVlips_dataset/preprocessed_exp/val/0_real"
    val_opt.fake_list_path = "/workspace/datasets/AVlips_dataset/preprocessed_exp/val/1_fake"
    return val_opt

def main():
    # Initialize distributed environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    
    # Initialize the distributed process group before anything else
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    print(f"Starting process with LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}")

    try:
        # Set same random seed for all processes
        torch.manual_seed(42)
        np.random.seed(42)
        
        # GA setup - only on rank 0
        if local_rank == 0:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()
            toolbox.register("attr_lr", np.random.uniform, 1e-5, 1e-2)
            toolbox.register("attr_batch_size", np.random.randint, 16, 128)
            toolbox.register("individual", tools.initCycle, creator.Individual,
                           (toolbox.attr_lr, toolbox.attr_batch_size), n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", lambda ind: evaluate_hyperparameters(ind, local_rank))
            toolbox.register("mate", tools.cxBlend, alpha=0.5)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)

            population = toolbox.population(n=10)
            ngen = 5
            cxpb = 0.5
            mutpb = 0.2

            print("Running Genetic Algorithm for hyperparameter optimization...")
            population, logbook = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, 
                                                    verbose=True)

            # Retrieve the best hyperparameters
            best_individual = tools.selBest(population, k=1)[0]
            best_lr, best_batch_size = best_individual
            
            # Broadcast best hyperparameters to all processes
            best_params = torch.tensor([best_lr, float(best_batch_size)], device=f'cuda:{local_rank}')
        else:
            best_params = torch.zeros(2, device=f'cuda:{local_rank}')

        # Broadcast best parameters from rank 0 to all processes
        dist.broadcast(best_params, 0)
        best_lr, best_batch_size = best_params.tolist()

        if local_rank == 0:
            print(f"Best hyperparameters: Learning Rate={best_lr}, Batch Size={best_batch_size}")

        # Update training options with the best hyperparameters
        opt = TrainOptions().parse()
        val_opt = get_val_opt()
        opt.lr = best_lr
        opt.batch_size = int(best_batch_size)
        val_opt.batch_size = int(best_batch_size // world_size)

        # Initialize model and synchronize parameters
        model = DistributedTrainer(opt, local_rank)
        model.model = synchronize_model(model.model, local_rank)
        
        # Wait for all processes to finish model initialization
        dist.barrier()
        
        # Initialize data loaders
        data_loader, train_sampler = create_distributed_dataloader(opt)
        val_loader, _ = create_distributed_dataloader(val_opt)

        if local_rank == 0:
            print(f"Length of data loader: {len(data_loader)}")
            print(f"Length of val loader: {len(val_loader)}")

        best_val_metrics = {
            'ap': 0.0,
            'acc': 0.0,
            'epoch': -1
        }

        for epoch in range(opt.epoch):
            model.train()
            train_sampler.set_epoch(epoch)

            if local_rank == 0:
                print(f"Epoch: {epoch + model.step_bias}")

            total_loss = 0
            num_batches = 0

            for i, (img, crops, label) in enumerate(tqdm(
                data_loader, 
                desc=f"Training Epoch {epoch+1}",
                disable=local_rank != 0
            )):
                model.total_steps += 1
                model.set_input((img, crops, label))
                model.forward()
                opt_info = model.optimize_parameters()

                if opt_info is not None:
                    total_loss += opt_info['loss']
                    num_batches += 1

                    if local_rank == 0 and i % 100 == 0:
                        print(f"Step {i}: Loss={opt_info['loss']:.4f}, "
                              f"Avg Grad Norm={opt_info['avg_grad_norm']:.4f}")

            dist.barrier()

            if local_rank == 0:
                model.eval()
                with torch.no_grad():
                    ap, fpr, fnr, acc = validate(model.model.module, val_loader, [local_rank])
                    print(
                        f"(Val @ epoch {epoch + model.step_bias}) "
                        f"acc: {acc:.4f} ap: {ap:.4f} fpr: {fpr:.4f} fnr: {fnr:.4f}"
                    )

                    if acc > best_val_metrics['acc']:
                        best_val_metrics = {
                            'ap': ap,
                            'acc': acc,
                            'epoch': epoch
                        }
                        model.save_networks("best_model.pth")
                        print(f"New best model saved! Accuracy: {acc:.4f}")

                    if epoch % opt.save_epoch_freq == 0:
                        model.save_networks(f"model_epoch_{epoch + model.step_bias}.pth")

        if local_rank == 0:
            print("\nTraining completed!")
            print(f"Best model performance (epoch {best_val_metrics['epoch']}):")
            print(f"Accuracy: {best_val_metrics['acc']:.4f}")
            print(f"AP: {best_val_metrics['ap']:.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error on rank {local_rank}: {str(e)}")
        raise e
    finally:
        dist.destroy_process_group()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
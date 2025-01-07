# # from validate import validate
# # from data import create_dataloader
# # from trainer.trainer import Trainer
# # from options.train_options import TrainOptions


# # def get_val_opt():
# #     val_opt = TrainOptions().parse(print_options=False)
# #     val_opt.isTrain = False
# #     val_opt.data_label = "val"
# #     val_opt.real_list_path = "./datasets/val/0_real"
# #     val_opt.fake_list_path = "./datasets/val/1_fake"
# #     return val_opt


# # if __name__ == "__main__":
# #     opt = TrainOptions().parse()
# #     val_opt = get_val_opt()
# #     model = Trainer(opt)

# #     data_loader = create_dataloader(opt)
# #     val_loader = create_dataloader(val_opt)

# #     print("Length of data loader: %d" % (len(data_loader)))
# #     print("Length of val  loader: %d" % (len(val_loader)))

# #     for epoch in range(opt.epoch):
# #         model.train()
# #         print("epoch: ", epoch + model.step_bias)
# #         for i, (img, crops, label) in enumerate(data_loader):
# #             model.total_steps += 1

# #             model.set_input((img, crops, label))
# #             model.forward()
# #             loss = model.get_loss()

# #             model.optimize_parameters()

# #             if model.total_steps % opt.loss_freq == 0:
# #                 print(
# #                     "Train loss: {}\tstep: {}".format(
# #                         model.get_loss(), model.total_steps
# #                     )
# #                 )

# #         if epoch % opt.save_epoch_freq == 0:
# #             print("saving the model at the end of epoch %d" % (epoch + model.step_bias))
# #             model.save_trainer("model_epoch_%s.pth" % (epoch + model.step_bias))

# #         model.eval()
# #         ap, fpr, fnr, acc = validate(model.model, val_loader, opt.gpu_ids)
# #         print(
# #             "(Val @ epoch {}) acc: {} ap: {} fpr: {} fnr: {}".format(
# #                 epoch + model.step_bias, acc, ap, fpr, fnr
# #             )
# #         )

# from validate import validate
# from data import create_dataloader
# from trainer.trainer import Trainer
# from options.train_options import TrainOptions
# from tqdm import tqdm

# import wandb

# def get_val_opt():
#     val_opt = TrainOptions().parse(print_options=False)
#     val_opt.isTrain = False
#     val_opt.data_label = "val"
#     val_opt.real_list_path = "/workspace/datasets/AVlips_dataset/preprocessed_20241130/val/0_real"
#     val_opt.fake_list_path = "/workspace/datasets/AVlips_dataset/preprocessed_20241130/val/1_fake"
#     return val_opt


# if __name__ == "__main__":
#     opt = TrainOptions().parse()
#     val_opt = get_val_opt()
#     model = Trainer(opt)

#     data_loader = create_dataloader(opt)
#     val_loader = create_dataloader(val_opt)

#     print("Length of data loader: %d" % (len(data_loader)))
#     print("Length of val  loader: %d" % (len(val_loader)))

#     wandb.init(project="Lipfd", config={
#         "epochs": opt.epoch,
#         "batch_size": opt.batch_size,
#         "learning_rate": opt.lr,
#         "save_epoch_freq": opt.save_epoch_freq
#     })

#     for epoch in range(opt.epoch):
#         model.train()
#         print("epoch: ", epoch + model.step_bias)

#         for i, (img, crops, label) in enumerate(tqdm(data_loader, desc=f"Training Epoch {epoch+1}")):
#             model.total_steps += 1

#             model.set_input((img, crops, label))
#             model.forward()
#             # loss = model.get_loss()

#             opt_info = model.optimize_parameters() # model.optimize_parameters()

#             if i % 100 == 0:
#                 print(f"Step {i}: Loss={opt_info['loss']:.4f}, "
#                     f"Avg Grad Norm={opt_info['avg_grad_norm']:.4f}")
            
#             # wandb
#             if model.total_steps % opt.loss_freq == 0:
#                 wandb.log({
#                     "Train Loss": opt_info['loss'], 
#                     "Average Gradient Norm": opt_info['avg_grad_norm'],
#                     "Step": model.total_steps
#                     })
#                 tqdm.write(
#                     "Train loss: {}\tstep: {}".format(opt_info['loss'], model.total_steps)
#                 )

#         if epoch % opt.save_epoch_freq == 0:
#             print("saving the model at the end of epoch %d" % (epoch + model.step_bias))
#             model.save_networks("model_epoch_%s.pth" % (epoch + model.step_bias))

#         model.eval()
#         ap, fpr, fnr, acc = validate(model.model, val_loader, opt.gpu_ids)

#         wandb.log({
#             "Validation Accuracy": acc,
#             "Validation AP": ap,
#             "Validation FPR": fpr,
#             "Validation FNR": fnr,
#             "Epoch": epoch + model.step_bias
#         })

#         print(
#             "(Val @ epoch {}) acc: {} ap: {} fpr: {} fnr: {}".format(
#                 epoch + model.step_bias, acc, ap, fpr, fnr
#             )
#         )

# from validate import validate
# from data import create_dataloader
# from trainer.trainer import Trainer
# from options.train_options import TrainOptions


# def get_val_opt():
#     val_opt = TrainOptions().parse(print_options=False)
#     val_opt.isTrain = False
#     val_opt.data_label = "val"
#     val_opt.real_list_path = "./datasets/val/0_real"
#     val_opt.fake_list_path = "./datasets/val/1_fake"
#     return val_opt


# if __name__ == "__main__":
#     opt = TrainOptions().parse()
#     val_opt = get_val_opt()
#     model = Trainer(opt)

#     data_loader = create_dataloader(opt)
#     val_loader = create_dataloader(val_opt)

#     print("Length of data loader: %d" % (len(data_loader)))
#     print("Length of val  loader: %d" % (len(val_loader)))

#     for epoch in range(opt.epoch):
#         model.train()
#         print("epoch: ", epoch + model.step_bias)
#         for i, (img, crops, label) in enumerate(data_loader):
#             model.total_steps += 1

#             model.set_input((img, crops, label))
#             model.forward()
#             loss = model.get_loss()

#             model.optimize_parameters()

#             if model.total_steps % opt.loss_freq == 0:
#                 print(
#                     "Train loss: {}\tstep: {}".format(
#                         model.get_loss(), model.total_steps
#                     )
#                 )

#         if epoch % opt.save_epoch_freq == 0:
#             print("saving the model at the end of epoch %d" % (epoch + model.step_bias))
#             model.save_trainer("model_epoch_%s.pth" % (epoch + model.step_bias))

#         model.eval()
#         ap, fpr, fnr, acc = validate(model.model, val_loader, opt.gpu_ids)
#         print(
#             "(Val @ epoch {}) acc: {} ap: {} fpr: {} fnr: {}".format(
#                 epoch + model.step_bias, acc, ap, fpr, fnr
#             )
#         )

from validate import validate
from data import create_dataloader
from trainer.trainer import Trainer
from options.train_options import TrainOptions
from tqdm import tqdm

import wandb

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.data_label = "val"
    val_opt.real_list_path = "/workspace/datasets/AVlips_dataset/preprocessed_20241130/val/0_real"
    val_opt.fake_list_path = "/workspace/datasets/AVlips_dataset/preprocessed_20241130/val/1_fake"
    return val_opt


if __name__ == "__main__":
    opt = TrainOptions().parse()
    val_opt = get_val_opt()
    model = Trainer(opt)

    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    print("Length of data loader: %d" % (len(data_loader)))
    print("Length of val  loader: %d" % (len(val_loader)))

    # wandb.init(project="LipFD", config={
    #     "epochs": opt.epoch,
    #     "batch_size": opt.batch_size,
    #     "learning_rate": opt.lr,
    #     "save_epoch_freq": opt.save_epoch_freq
    # })

    for epoch in range(opt.epoch):
        model.train()
        print("epoch: ", epoch + model.step_bias)

        for i, (img, crops, label) in enumerate(tqdm(data_loader, desc=f"Training Epoch {epoch+1}")):
            model.total_steps += 1

            model.set_input((img, crops, label))
            model.forward()
            # loss = model.get_loss()

            opt_info = model.optimize_parameters() # model.optimize_parameters()

            if i % 100 == 0:
                print(f"Step {i}: Loss={opt_info['loss']:.4f}, "
                    f"Avg Grad Norm={opt_info['avg_grad_norm']:.4f}")
            
            # wandb
            if model.total_steps % opt.loss_freq == 0:
                # wandb.log({
                #     "Train Loss": opt_info['loss'], 
                #     "Average Gradient Norm": opt_info['avg_grad_norm'],
                #     "Step": model.total_steps
                #     })
                tqdm.write(
                    "Train loss: {}\tstep: {}".format(opt_info['loss'], model.total_steps)
                )

        if epoch % opt.save_epoch_freq == 0:
            print("saving the model at the end of epoch %d" % (epoch + model.step_bias))
            model.save_networks("model_epoch_%s.pth" % (epoch + model.step_bias))

        model.eval()
        ap, fpr, fnr, acc = validate(model.model, val_loader, opt.gpu_ids)

        # wandb.log({
        #     "Validation Accuracy": acc,
        #     "Validation AP": ap,
        #     "Validation FPR": fpr,
        #     "Validation FNR": fnr,
        #     "Epoch": epoch + model.step_bias
        # })

        print(
            "(Val @ epoch {}) acc: {} ap: {} fpr: {} fnr: {}".format(
                epoch + model.step_bias, acc, ap, fpr, fnr
            )
        )


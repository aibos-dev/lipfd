"""
# Milestone 1: Ensuring Proper Training of LipFD

## Summary of Changes

### `./train.py`

1. Added settings for `tqdm` and `wandb`.
2. Set `val_opt.isTrain` to training mode.

### Added `./run.sh`

- Configuration for training jobs:
  - Folder to save log files.
  - Environment setup and execution.

### Added `./requirements_milestone_1.txt`

- Python 3.10.
- Environment configuration.

### `./trainer/trainer.py`

1. Modified the initialization method:
    1. Added `super(Trainer, self).__init__()`.
    2. For `if opt.fix_encoder:`:
        1. Adjusted gradient tracking.
        2. Changed the method for adding to `params`.
2. Modified the `get_loss` method.
3. Updated the `optimize_parameters` method:
    1. Recorded loss values.
    2. Verified accurate gradient updates.

### `./options/train_options.py`

1. Adjusted `'--save_epoch_freq'`, `'--epoch'`, and `'--fine-tune'`.
2. Added `'--weight_decay'` and `'--class_bal'`.

### `./options/base_options.py`

- Commented out the following sections:
    
    ```python
            # # process opt.suffix
            # if opt.suffix:
            #     suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
            #     opt.name = opt.name + suffix
    ```
    
    ```python
            # opt.rz_interp = opt.rz_interp.split(",")
            # opt.blur_sig = [float(s) for s in opt.blur_sig.split(",")]
            # opt.jpg_method = opt.jpg_method.split(",")
            # opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(",")]
            # if len(opt.jpg_qual) == 2:
            #     opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
            # elif len(opt.jpg_qual) > 2:
            #     raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")
    ```
    
- Adjusted `"--batch_size"` and `"--gpu_ids"`.

### `./models/LipFD.py`

- Modified the `forward` method in `class RALoss` as follows:
    
    ```python
                    # loss_wt += torch.Tensor([10]).to(alphas_max[i][j].device) / np.exp(
                    #     alphas_max[i][j] - alphas_org[i][j]
                    # )
                    loss_wt += torch.tensor([10.0], device=alphas_max[i][j].device) / torch.exp(
                        alphas_max[i][j] - alphas_org[i][j]
                    )
    ```

## Verification Using Metrics

- Confirmed that the model begins learning based on the following metrics:

    ![index.png](./index.png)
## Requirements

- ascend 910B
- MindSpore==1.8.1

## Quick Start

To modify the data path in the `./config/AFNO.yaml` file, please update the following parameters:
```yaml
exp_dir: './W-MAE/'
train_data_path: './train'
valid_data_path: './train'
inf_data_path:   './out_of_sample'
time_means_path:   './additional/time_means.npy'
global_means_path: './additional/global_means.npy'
global_stds_path:  './additional/global_stds.npy'
```

After updating the path, run the following commands in the terminal. If you're running distributed learning, add the `--distributed` flag. 

```bash
python3 main_new.py --config mae_backbone
python3 main_new.py --config afno_one_step --resume_checkpoint_path ./mae/mae_backbone/best_ckpt_rank11.ckpt
python3 main_new.py --config precip --model_wind_path ./mea/afno_one_step/0_202303232223/training_checkpoints/best_ckpt_rank0.ckpt
```

## Acknowledgements
Special thanks to the developers of [FourCastNet](https://github.com/NVlabs/FourCastNet) for providing the reference code used in this project.
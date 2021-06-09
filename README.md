# styleNeRF

Style transfer for neural radiance fields, based on a [PyTorch Lightning implementation](https://github.com/kwea123/nerf_pl) and [the official PyTorch tutorial for neural style transfer](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html). 

### Download the blender dataset

Download `nerf_synthetic.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and extract the content under `./datasets`

### Train the model

From the command line:

```
python train.py --dataset_name blender --root_dir $BLENDER_DIR --N_importance 64 --img_wh 400 400 --noise_std 0 --num_epochs 20 --batch_size 1024 --optimizer adam --lr 5e-4 --lr_scheduler cosine --exp_name exp
```

You can monitor the training process by `tensorboard --logdir runs` and go to `localhost:6006` in your browser.

### Pretrained models and logs
You can download the pretrained models and training logs in [release](https://github.com/kwea123/nerf_pl/releases).

### Testing

Use [eval.py](eval.py) to create the whole sequence of moving views.
E.g.
```
python eval.py --root_dir $BLENDER --dataset_name blender --scene_name lego --img_wh 400 400 --N_importance 64 --ckpt_path $CKPT_PATH
```

It will create folder `results/{dataset_name}/{scene_name}` and run inference on all test data, finally create a gif out of them.

![nerf-u](https://user-images.githubusercontent.com/11364490/105578186-a9933400-5dc1-11eb-8865-e276b581d8fd.gif)

### Notes on differences with the paper

*  Current base MLP uses 8 layers of 256 units as the original NeRF, while NeRF-W uses **512** units each.
*  Current static head uses 1 layer as the original NeRF, while NeRF-W uses **4** layers.
*  **Softplus** activation for sigma (reason explained [here](https://github.com/bmild/nerf/issues/29#issuecomment-765335765)) while NeRF-W uses **relu**.
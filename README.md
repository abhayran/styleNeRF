# styleNeRF

Style transfer for neural radiance fields in PyTorch. 

This project is built upon a [PyTorch Lightning implementation](https://github.com/kwea123/nerf_pl) and [the official PyTorch tutorial for neural style transfer](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html). 

## Download the blender dataset

Download `nerf_synthetic.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and extract the content under `~/datasets`.

## Train the model

After downloading the dataset, run this command to train the NeRF model:

```
python train.py --dataset_name blender --root_dir datasets/nerf_synthetic/lego --img_wh 400 400 --num_epochs_density 1 --num_epochs_style 1 --batch_size 1024 --lr 1e-3
```

You can monitor the training process by `tensorboard --logdir runs` and go to `localhost:6006` in your browser.

## Testing

Use [eval.py](eval.py) to create the whole sequence of moving views.
E.g.
```
python eval.py --dataset_name blender --root_dir datasets/nerf_synthetic/lego --scene_name lego --img_wh 400 400 --ckpt_path $CKPT_PATH
```

It will create folder `results/{dataset_name}/{scene_name}` and run inference on all test data, finally create a gif out of them.

![ezgif com-gif-maker](https://user-images.githubusercontent.com/40629249/124356665-6ca45680-dc17-11eb-8830-45399841ecdd.gif)

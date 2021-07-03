# styleNeRF

Style transfer for neural radiance fields in PyTorch. 

This project is built upon a [PyTorch Lightning implementation](https://github.com/kwea123/nerf_pl) and [the official PyTorch tutorial for neural style transfer](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html). 

## Model architecture

We employ a two stage training. First stage is the standard NeRF training procedure. Our primary aim is to train the "geometry MLP" which is responsible for generating the density, while "style MLP" acts as an auxilary network and will be modified later on.

In the second stage, we freeze the geometry MLP and train style MLP by the style loss between rendered images and the style image. By doing so, we disentangle the geometry and appearance of the scene, hence manage to transfer the style while making sure the geometry is fixed.

![image](https://user-images.githubusercontent.com/40629249/124357777-91033180-dc1d-11eb-9293-1f8dde0609c5.png)

## Download the blender dataset

Download `nerf_synthetic.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and extract the content under `./datasets`.

## Stage 1: Train the density model

After downloading the dataset, run the following command for the Stage 1 where the scene geometry is learned:

```
python pipeline.py --is_learning_density True --prefix $PREFIX --suffix $SUFFIX --dataset_name blender --root_dir datasets/nerf_synthetic/lego --img_wh 400 400 --num_epochs_density 1 --batch_size 1024 --lr 5e-4
```

The models will be logged under `./ckpts/$PREFIX_nerf_coarse_$SUFFIX.pt` and `./ckpts/$PREFIX_nerf_coarse_$SUFFIX.pt`

You can monitor the training process by `tensorboard --logdir runs` and go to `localhost:6006` in your browser.

## Stage 2: Transfer the style

Having trained the density model, we now transfer the style while freezing the geometry MLP with the following command:

```
python pipeline.py --is_learning_density False --prefix $PREFIX --suffix $SUFFIX --style_dir $STYLE_DIR --style_mode $STYLE_MODE --coarse_path $COARSE_PATH --fine_path $FINE_PATH --dataset_name blender --root_dir datasets/nerf_synthetic/lego --img_wh 400 400 --num_epochs_style 1 --lr 1e-3
```

where `$STYLE_DIR` is the path to style image, `$COARSE_PATH` is the path to previously logged coarse NeRF model and `$FINE_PATH` is the path to previously logged fine NeRF model.

Style transfer stage is very heavy in terms of memory consumption, since we can't process the pixels individually due to the nature of style loss. We propose three different ways to alleviate that problem which you can specify by `$STYLE_MODE`:
- `small`: This is the most straightforward approach, where you render smaller images. Limited to around a size of 50 x 50 for a GPU with 8 GB of memory, hence the fine detail is lost.
- `patch`: Render patches and transfer the style to each patch individually. Better performance, but lacks global consistency and fails to capture higher level features since we have to process patches individually. 
- `memory_saving`: This is a practical workaround which enables us to process pixels individually, yielding superior results. This approach consists of two substages:
    - Substage 1: Render image in inference mode, calculate and store gradients per pixel by the style loss
    - Substage 2: Render pixels while freezing the geometry MLP, then backpropagate the corresponding stored pixel gradients. Wait for every pixel to be processed before making an update.

![image](https://user-images.githubusercontent.com/40629249/124358547-1d632380-dc21-11eb-8695-2d05484ae04e.png)

## Testing

Use [eval.py](eval.py) to create the whole sequence of moving views.
E.g.
```
python eval.py --dataset_name blender --root_dir datasets/nerf_synthetic/lego --scene_name lego --img_wh 400 400 --coarse_path $COARSE_PATH --fine_path $FINE_PATH
```

It will create folder `results/{dataset_name}/{scene_name}` and run inference on all test data, finally create a gif out of them and save to `./nerf.gif`.

![ezgif com-gif-maker](https://user-images.githubusercontent.com/40629249/124356665-6ca45680-dc17-11eb-8830-45399841ecdd.gif)

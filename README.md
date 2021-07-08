# styleNeRF

Memory efficient style transfer for neural radiance fields [1] in PyTorch. 

This project is built upon a [PyTorch Lightning implementation](https://github.com/kwea123/nerf_pl) [2] and [the official PyTorch tutorial for neural style transfer](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html). 

## Download the blender dataset

Download `nerf_synthetic.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and extract the content under `./datasets`.

## Quick demonstration

If you want to give this repo a quick try, you can run `demo.ipynb` which will first train a NeRF model on the synthetic lego truck scene, then transfer the style from Van Gogh's starry night image, and finally render a gif. Whole process takes around 3 hours on a single RTX 2070 GPU with 8GB of memory.

You can monitor the training process via the command `tensorboard --logdir runs` and going to `localhost:6006` in your browser.

## Model architecture

We employ a two stage training. First stage is the standard NeRF training procedure. Our primary aim is to train the "geometry MLP" which is responsible for generating the density, while "style MLP" acts as an auxilary network and will be modified later on.

In the second stage, we freeze the geometry MLP and train style MLP by the style loss between rendered images and the style image. By doing so, we disentangle the geometry and appearance of the scene, hence manage to transfer the style while making sure the geometry is fixed.

![image](https://user-images.githubusercontent.com/40629249/124367192-5d43fe00-dc55-11eb-9408-6e99529007e2.png)

## Stage 1: Train the density model

After downloading the dataset, run the following command for the Stage 1 where the scene geometry is learned:

```
python pipeline.py --stage geometry --prefix $PREFIX --suffix $SUFFIX --dataset_name blender --root_dir datasets/nerf_synthetic/lego --img_wh 400 400 --num_epochs_density 1 --batch_size 1024 --lr 5e-4
```

The models will be logged under `./ckpts/$PREFIX_nerf_coarse_$SUFFIX.pt` and `./ckpts/$PREFIX_nerf_fine_$SUFFIX.pt`

## Stage 2: Transfer the style

Having trained the density model, we now transfer the style while freezing the geometry MLP with the following command:

```
python pipeline.py --stage style --prefix $PREFIX --suffix $SUFFIX --style_dir $STYLE_DIR --style_mode $STYLE_MODE --coarse_path $COARSE_PATH --fine_path $FINE_PATH --dataset_name blender --root_dir datasets/nerf_synthetic/lego --img_wh 400 400 --num_epochs_style 1 --lr 1e-3
```

where `$STYLE_DIR` is the path to style image, `$COARSE_PATH` is the path to previously logged coarse NeRF model and `$FINE_PATH` is the path to previously logged fine NeRF model.

Style transfer stage is very heavy in terms of memory consumption, since we can't process the pixels individually due to the nature of style loss. We propose three different ways to alleviate that problem which you can specify by `$STYLE_MODE`:
- `small`: This is the most straightforward approach, where you render smaller images. Limited to around a size of 50 x 50 for a GPU with 8 GB of memory, hence the fine detail is lost. Significant size mismatch between rendered images and the style image is a problem.
- `patch`: Render patches and transfer the style to each patch individually. Better performance, but lacks global consistency and fails to capture higher level features since we have to process patches individually. Size mismatch problem persists. 
- `memory_saving`: A practical workaround for working with arbitrarily big images without using patching, which enables us to process pixels even individually. This way we mitigate both the memory limitations and the size mismatch issue. This approach consists of two substages:
    - Substage 1: Render image in inference mode, calculate and store gradients per pixel by the style loss
    - Substage 2: Render pixels while freezing the geometry MLP, then backpropagate the corresponding stored pixel gradients. Wait for every pixel to be processed before stepping the optimizer.

![image](https://user-images.githubusercontent.com/40629249/124367388-03443800-dc57-11eb-9e93-5c80d1a724f4.png)

## Testing

Use [eval.py](eval.py) to create the whole sequence of moving views.
E.g.
```
python eval.py --dataset_name blender --root_dir datasets/nerf_synthetic/lego --scene_name lego --img_wh 400 400 --coarse_path $COARSE_PATH --fine_path $FINE_PATH --gif_name $GIF_NAME
```

It will create folder `results/{dataset_name}/{scene_name}` and run inference on all test data, finally create a gif out of them and save to `./$GIF_NAME.gif`.

- Example: A NeRF model trained first on the synthetic lego truck data with 400 x 400 images, then learned the style of the Van Gogh's starry night image using the `memory_saving` mode, using a GPU with 8GB of memory. Memory saving mode enables working with GPUs having quite low memory.   

![gif](https://user-images.githubusercontent.com/40629249/124567099-f3d61200-de43-11eb-86dc-035567213182.gif)

## References
<a id="1">[1]</a> 
Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. *Nerf: Representing scenes as neural radiance fields for view synthesis.* In European Conference on Computer Vision, pages 405–421. Springer, 2020.

<a id="2">[2]</a> 
Chen Quei-An.  *Nerf_pl: A pytorch-lightning implementation of neural radiance fields*, 2020.

# Dynamic Novel View Synthesis

## Dependencies

Besides python package, you will need

- `ImageMagick`
  - Use `Mogrify` to minify the images
- `colmap`
  - Extract camera parameters and scene bounds.

to run the code.

## Datasets Preprocessing
```
├──data/
  ├──<scene_name>/
    ├──images/
      ├──(Your images with frame index#) #.jpg/.png
```
The Preprocessing is refered to LLFF: https://github.com/Fyusion/LLFF#1-recover-camera-poses \
In the root folder, execute `sh gen_pose.sh <dataset_path>` 

e.g.
```
sh gen_pose.sh ~/IBRNet/data/<scene_name>
```

## Training
To train a model, first edit the config `configs/finetune_llff.txt`, and execute
```
sh train.sh
```

You can monitor the training status by using tensorboard
```
sh tb.sh
```

## Inference
To render the video, first edit the config `configs/eval_llff.txt`, and execute
```
sh render.sh
```

You may need to edit `eval/render_llff_video.py #62` to change setting of `pose_path`
```
# "fixed_time": Time is set to 0 
# "fixed_pose": Pose is set to poses_avg(poses)
# others (None): Both time and pose are moved
_, poses, bds, render_poses, i_test, rgb_files, time_indices, render_time, time_max = load_llff_data(
                scene_path, load_imgs=False, factor=8, pose_path=None, N_views=120)
```


## Citation
```
@inproceedings{wang2021ibrnet,
  author    = {Wang, Qianqian and Wang, Zhicheng and Genova, Kyle and Srinivasan, Pratul and Zhou, Howard  and Barron, Jonathan T. and Martin-Brualla, Ricardo and Snavely, Noah and Funkhouser, Thomas},
  title     = {IBRNet: Learning Multi-View Image-Based Rendering},
  booktitle = {CVPR},
  year      = {2021}
}

```

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import imageio
from numpy.random import triangular
import torch
import sys
sys.path.append('../')
from torch.utils.data import Dataset
from .data_utils import random_crop, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses
from itertools import combinations
from tqdm import tqdm
from RAFT import raft_helper


class LLFFTestDataset(Dataset):
    def __init__(self, args, mode, scenes=(), random_crop=False, **kwargs):
        # self.folder_path = os.path.join(args.rootdir, 'data/nerf_llff_data/')
        self.folder_path = os.path.join(args.rootdir, 'data/')
        self.args = args
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.random_crop = random_crop
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []
        self.render_time_indices = []
        self.time_max = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []
        self.train_time_indices = []


        all_scenes = os.listdir(self.folder_path)
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes

        print("loading {} for {}".format(scenes, mode))
        for i, scene in enumerate(scenes):
            scene_path = os.path.join(self.folder_path, scene)
            # FIXME: factor=4
            _, poses, bds, render_poses, i_test, rgb_files, time_indices, time_max = load_llff_data(scene_path, load_imgs=False, factor=8)
            
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)

            i_test = np.arange(poses.shape[0])[::self.args.llffhold]
            i_train = np.array([j for j in np.arange(int(poses.shape[0])) if
                                (j not in i_test and j not in i_test)])

            if mode == 'train':
                i_render = i_train
            else:
                i_render = i_test

            self.time_max.append(time_max)
            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            self.train_time_indices.append(np.array(time_indices)[i_train].tolist())

            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_time_indices.extend(np.array(time_indices)[i_render].tolist())

            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)
        self.train_time_indices = np.array(self.train_time_indices)
        self.render_time_indices = np.array(self.render_time_indices)
        self.time_indices = np.array(time_indices)
        self.rgb_files = rgb_files
        # WARNING: only one scene considered
        self.img_path = rgb_files[0].rsplit('/', maxsplit=1)[0]
        self.optical_flows = self.read_optical_flow(args.raft_path)
        self.optical_flows = self.optical_flows.permute([0, 1, 3, 2, 4]) # (tar, src, W, H, (W_flow, H_flow))

    def __len__(self):
        return len(self.render_rgb_files) * 100000 if self.mode == 'train' else len(self.render_rgb_files)

    def make_optical_flow(self, files, times, raft_path):
        print("Make Optical Flows:")
        raft = raft_helper.load(raft_path)
        imgs = [None]*len(times)

        def load_img(idx):
            t = times[idx]
            img = imageio.imread(files[idx]).astype(np.float32)
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            return img[None].cuda()

        flows = torch.zeros((len(times), len(times))).tolist()
        for i, j in tqdm(combinations(range(len(times)), 2)):
            if imgs[i] is None: imgs[i] = load_img(i)
            if imgs[j] is None: imgs[j] = load_img(j)
            flows[i][j] = raft_helper.get_flow(raft, imgs[i], imgs[j])
            flows[j][i] = raft_helper.get_flow(raft, imgs[j], imgs[i])
        
        for i in range(len(times)):
            flows[i][i] = torch.zeros_like(flows[0][1])

        flows = torch.stack([torch.stack(row) for row in flows])
        print(flows.min(), flows.max())
        print(f"Optical Flows done: {flows.shape}")
        torch.save(flows, self.img_path + f"/optical_flows.pt")
        return flows
        

    def read_optical_flow(self, raft_path):
        '''
        tar: target(ibrnet) image
        src: source(ibrnet) image
        '''
        file_name = self.img_path + f"/optical_flows.pt"
        if os.path.isfile(file_name):
            flows = torch.load(file_name)
            print(f"Read Optical Flows: {flows.shape}")
        else:
            flows = self.make_optical_flow(self.rgb_files, self.time_indices, raft_path)
        return flows
        

    def __getitem__(self, idx):
        idx = idx % len(self.render_rgb_files)
        rgb_file = self.render_rgb_files[idx]
        time_index = self.render_time_indices[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        time_max = self.time_max[train_set_id]
        train_time_indices = self.train_time_indices[train_set_id]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        if self.mode == 'train':
            if rgb_file in train_rgb_files:
                id_render = train_rgb_files.index(rgb_file)
            else:
                id_render = -1
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=2)
        else:
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views
        
        nearest_pose_ids = get_nearest_pose_ids(time_index, train_time_indices,
                                                render_pose, train_poses,
                                                min(self.num_source_views*subsample_factor, 28),
                                                tar_id=id_render,
                                                angular_dist_method='dist')
        nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == 'train':
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        src_time_indices = train_time_indices[nearest_pose_ids.tolist()]
        optical_flows = self.optical_flows[time_index][src_time_indices]
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32)
            
            src_rgb /= 255.
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                         train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        if self.mode == 'train' and self.random_crop:
            crop_h = np.random.randint(low=250, high=750)
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(400 * 600 / crop_h)
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            rgb, camera, src_rgbs, src_cameras = random_crop(rgb, camera, src_rgbs, src_cameras,
                                                             (crop_h, crop_w))

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])

        return {'rgb': torch.from_numpy(rgb[..., :3]),
                'time_index': time_index/time_max,
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'src_time_indices': torch.from_numpy(src_time_indices)/time_max,
                'depth_range': depth_range,
                'optical_flows': optical_flows
                # TODO: src_flows
                }


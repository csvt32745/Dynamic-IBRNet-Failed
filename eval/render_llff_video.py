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


from torch.utils.data import Dataset
import sys
sys.path.append('../')
from torch.utils.data import DataLoader
import imageio
from config import config_parser
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.render_image import render_single_image
from ibrnet.model import IBRNetModel
from utils import *
from ibrnet.projection import Projector
from ibrnet.data_loaders import get_nearest_pose_ids, llff_test
from ibrnet.data_loaders.llff_data_utils import load_llff_data, batch_parse_llff_poses, poses_avg
import time


class LLFFRenderDataset(Dataset):
    def __init__(self, args,
                 scenes='fern',  # 'fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex'
                 **kwargs):

        self.folder_path = os.path.join(args.rootdir, 'data/')
        self.num_source_views = args.num_source_views

        print("loading {} for rendering".format(scenes))
        self.time_max = []
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []
        self.render_time_indices = []
        self.h = []
        self.w = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []
        self.train_time_indices = []

        for i, scene in enumerate(scenes):
            scene_path = os.path.join(self.folder_path, scene)
            # "fixed_time": Time is set to 0 
            # "fixed_pose": Pose is set to poses_avg(poses)
            # others (None): Both time and pose are moved
            _, poses, bds, render_poses, i_test, rgb_files, time_indices, render_time, time_max = load_llff_data(
                scene_path, load_imgs=False, factor=8, pose_path=None, N_views=120)

            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)
            h, w = poses[0][:2, -1]
            # FIXME
            render_intrinsics, render_c2w_mats = batch_parse_llff_poses(render_poses)
            # render_intrinsics, render_c2w_mats = batch_parse_llff_poses(poses)
            i_test = [i_test]
            i_val = i_test
            # i_train = np.array([i for i in np.arange(len(rgb_files)) if
            #                     (i not in i_test and i not in i_val)])
            i_train = np.arange(len(rgb_files))

            self.time_max.append(time_max)
            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            self.train_time_indices.append(np.array(time_indices)[i_train].tolist())
            
            num_render = len(render_intrinsics)
            # num_render = len(time_indices)*2


            # self.render_time_indices.extend(np.array(time_indices).repeat(num_render//len(time_indices))[:num_render].tolist())
            # self.render_time_indices.extend([9.]*num_render)
            self.render_time_indices.extend(render_time)
            print(self.render_time_indices)
            # FIXME
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in render_intrinsics[:num_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in render_c2w_mats[:num_render]])

            # self.render_intrinsics.extend([intrinsics[0]]*num_render)
            # self.render_poses.extend([c2w_mats[0]]*num_render)
            # self.render_poses.extend([batch_parse_llff_poses(np.expand_dims(poses_avg(poses), 0))[1]]*num_render)

            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)
            self.h.extend([int(h)]*num_render)
            self.w.extend([int(w)]*num_render)
        self.train_time_indices = np.array(self.train_time_indices)
        self.render_time_indices = np.array(self.render_time_indices)

    def __len__(self):
        return len(self.render_poses)

    def __getitem__(self, idx):
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]
        time_index = self.render_time_indices[idx]

        train_set_id = self.render_train_set_ids[idx]
        time_max = self.time_max[train_set_id]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]
        train_time_indices = self.train_time_indices[train_set_id]

        h, w = self.h[idx], self.w[idx]
        camera = np.concatenate(([h, w], intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        id_render = time_index
        nearest_pose_ids = get_nearest_pose_ids(time_index, train_time_indices,
                                                render_pose, train_poses,
                                                self.num_source_views,
                                                tar_id=id_render,
                                                angular_dist_method='dist',
                                                is_preserve=True)

        src_rgbs = []
        src_cameras = []
        src_time_indices = train_time_indices[nearest_pose_ids.tolist()]
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                         train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])
        norm = lambda x: (x*2.)-1.

        return {'camera': torch.from_numpy(camera),
                'time_index': norm(time_index/time_max),
                'rgb_path': '',
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                'src_time_indices': norm(torch.from_numpy(src_time_indices)/time_max),
                }


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    args.distributed = False

    # print(args.num_source_views)
    # assert False

    # Create ibrnet model
    model = IBRNetModel(args, load_scheduler=False, load_opt=False, load_deform=True)
    eval_dataset_name = args.eval_dataset
    extra_out_dir = '{}/{}'.format(eval_dataset_name, args.expname)
    print('saving results to {}...'.format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)

    projector = Projector(device='cuda:0')

    assert len(args.eval_scenes) == 1, "only accept single scene"
    scene_name = args.eval_scenes[0]
    out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}'.format(scene_name, model.start_step), 'videos')
    print('saving results to {}'.format(out_scene_dir))

    os.makedirs(out_scene_dir, exist_ok=True)

    test_dataset = LLFFRenderDataset(args, scenes=args.eval_scenes)
    # test_dataset = llff_test.LLFFTestDataset(args, 'validation',
    #                                               scenes=args.eval_scenes)
    save_prefix = scene_name
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)
    out_frames = []
    crop_ratio = 0.075

    for i, data in enumerate(test_loader):
        start = time.time()
        src_rgbs = data['src_rgbs'][0].cpu().numpy()
        averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_average.png'.format(i)), averaged_img)

        model.switch_to_eval()
        with torch.no_grad():
            ray_sampler = RaySamplerSingleImage(data, device='cuda:0')
            ray_batch = ray_sampler.get_all()
            ray_batch['src_time_indices'] = data['src_time_indices']
            ray_batch['time_index'] = data['time_index']
            # ray_batch['time_index'] = torch.Tensor([10.])

            featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
            ret = render_single_image(ray_sampler=ray_sampler,
                                        ray_batch=ray_batch,
                                        model=model,
                                        projector=projector,
                                        chunk_size=args.chunk_size,
                                        det=True,
                                        N_samples=args.N_samples,
                                        inv_uniform=args.inv_uniform,
                                        N_importance=args.N_importance,
                                        white_bkgd=args.white_bkgd,
                                        featmaps=featmaps)
            
            torch.cuda.empty_cache()
        # 

        
        coarse_pred_rgb = ret['outputs_coarse']['rgb'].detach().cpu()
        coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_coarse.png'.format(i)), coarse_pred_rgb)

        coarse_pred_depth = ret['outputs_coarse']['depth'].detach().cpu()
        imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_coarse.png'.format(i)),
                        (coarse_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
        coarse_pred_depth_colored = colorize_np(coarse_pred_depth,
                                                range=tuple(data['depth_range'].squeeze().numpy()))
        imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_vis_coarse.png'.format(i)),
                        (255 * coarse_pred_depth_colored).astype(np.uint8))

        coarse_acc_map = torch.sum(ret['outputs_coarse']['weights'].detach().cpu(), dim=-1)
        coarse_acc_map_colored = (colorize_np(coarse_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_acc_map_coarse.png'.format(i)),
                        coarse_acc_map_colored)

        if ret['outputs_fine'] is not None:
            fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
            fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_fine.png'.format(i)), fine_pred_rgb)
            fine_pred_depth = ret['outputs_fine']['depth'].detach().cpu()
            imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_fine.png'.format(i)),
                            (fine_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
            fine_pred_depth_colored = colorize_np(fine_pred_depth,
                                                  range=tuple(data['depth_range'].squeeze().cpu().numpy()))
            imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_vis_fine.png'.format(i)),
                            (255 * fine_pred_depth_colored).astype(np.uint8))
            fine_acc_map = torch.sum(ret['outputs_fine']['weights'].detach().cpu(), dim=-1)
            fine_acc_map_colored = (colorize_np(fine_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_acc_map_fine.png'.format(i)),
                            fine_acc_map_colored)
        else:
            fine_pred_rgb = None

        out_frame = fine_pred_rgb if fine_pred_rgb is not None else coarse_pred_rgb
        h, w = averaged_img.shape[:2]
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        # crop out image boundaries
        out_frame = out_frame[crop_h:h - crop_h, crop_w:w - crop_w, :]
        out_frames.append(out_frame)

        print('frame {} completed, {}'.format(i, time.time() - start))

    imageio.mimwrite(os.path.join(extra_out_dir, '{}.mp4'.format(scene_name)), out_frames, fps=30, quality=8)

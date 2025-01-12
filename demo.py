import argparse
import random

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image

from dataset.config import set_resolution, get_camera_intrinsic
from dataset.evaluation import (anchor_output_process, collision_detect,
                                detect_2d_grasp, detect_6d_grasp_multi)
from dataset.pc_dataset_tools import data_process, feature_fusion
from models.anchornet import AnchorGraspNet, Backbone
from models.localgraspnet import PointMultiGraspNet
from models.pointnet import PointNetfeat

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', default='realsense_checkpoint')
parser.add_argument('--random-seed', type=int, default=1)

# image input
parser.add_argument('--rgb-path', default='images/demo_rgb.png')
parser.add_argument('--depth-path', default='images/demo_depth.png')
parser.add_argument('--input-h', type=int, default=int(720/2.25)) # height of input image
parser.add_argument('--input-w', type=int, default=int(1280/2)) # width of input image
parser.add_argument('--export-onnx', type=bool, default=False) # export .onnx files

# 2d grasping
parser.add_argument('--sigma', type=int, default=10)
parser.add_argument('--ratio', type=int, default=8) # grasp attributes prediction downsample ratio, must be 2^N
parser.add_argument('--anchor-k', type=int, default=6) # in-plane rotation anchor number
parser.add_argument('--anchor-w', type=float, default=50.0) # grasp width anchor size
parser.add_argument('--anchor-z', type=float, default=20.0) # grasp depth anchor size
parser.add_argument('--grid-size', type=int, default=8) # grid size for grid-based center sampling
parser.add_argument('--feature-dim', type=int, default=128) # feature dimension for anchor net

# 6d grasping
parser.add_argument('--heatmap-thres', type=float, default=0.01) # heatmap threshold
parser.add_argument('--local-k', type=int, default=10) # grasp detection number in each local region (localnet)
parser.add_argument('--depth-thres', type=float, default=0.01) # depth threshold for collision detection
parser.add_argument('--max-points', type=int, default=25600) # downsampled max number of points in point cloud
parser.add_argument('--anchor-num', type=int, default=7) # spatial rotation anchor number
parser.add_argument('--center-num', type=int, default=64) # sampled local center/region number
parser.add_argument('--group-num', type=int, default=512) # local region fps number

args = parser.parse_args()

# --------------------------------------------------------------------------- #

class PointCloudHelper:

    def __init__(self) -> None:
        # precalculate x,y map
        self.all_points_num = args.max_points
        self.output_shape = (args.input_w//16, args.input_h//16) # downsampled aspect ratio of input image

        # get intrinsics
        intrinsics = get_camera_intrinsic()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # calculate x, y regions
        ymap, xmap = np.meshgrid(np.arange(args.input_h), np.arange(args.input_w))
        points_x = (xmap - cx) / fx
        points_y = (ymap - cy) / fy
        self.points_x = torch.from_numpy(points_x).float()
        self.points_y = torch.from_numpy(points_y).float()

        # for to_xyz_maps()
        ymap, xmap = np.meshgrid(np.arange(self.output_shape[1]), np.arange(self.output_shape[0]))
        factor = args.input_w / self.output_shape[0]
        points_x = (xmap - cx / factor) / (fx / factor)
        points_y = (ymap - cy / factor) / (fy / factor)
        self.points_x_downscale = torch.from_numpy(points_x).float()
        self.points_y_downscale = torch.from_numpy(points_y).float()

    # turn rgb + depth into point cloud
    def to_scene_points(self, rgbs: torch.Tensor, depths: torch.Tensor, include_rgb=True):
        batch_size = rgbs.shape[0]
        feature_len = 3 + 3 * include_rgb
        points_all = -torch.ones((batch_size, self.all_points_num, feature_len), dtype=torch.float32)
        if gpu:
            points_all = points_all.cuda()

        # calculate z
        idxs = []
        masks = (depths > 0)
        cur_zs = depths / 1000.0
        if gpu:
            cur_xs = self.points_x.cuda() * cur_zs
            cur_ys = self.points_y.cuda() * cur_zs
        else:
            cur_xs = self.points_x * cur_zs
            cur_ys = self.points_y * cur_zs
        for i in range(batch_size):
            # convert point cloud to xyz maps
            points = torch.stack([cur_xs[i], cur_ys[i], cur_zs[i]], axis=-1)
            # remove zero depth
            mask = masks[i]
            points = points[mask]
            colors = rgbs[i][:, mask].T

            # random sample if too many points
            if len(points) >= self.all_points_num:
                cur_idxs = random.sample(range(len(points)), self.all_points_num)
                points = points[cur_idxs]
                colors = colors[cur_idxs]
                # save idxs for concat fusion
                idxs.append(cur_idxs)

            # concat rgb and features after translation
            if include_rgb:
                points_all[i] = torch.concat([points, colors], axis=1)
            else:
                points_all[i] = points
        return points_all, idxs, masks

    # get a downsampled xyz map
    def to_xyz_maps(self, depths):
        # downsample
        downsample_depths = F.interpolate(depths[:, None], size=self.output_shape).squeeze(1)
        if gpu:
            downsample_depths = downsample_depths.cuda()
        # convert xyzs
        cur_zs = downsample_depths / 1000.0
        if gpu:
            cur_xs = self.points_x_downscale.cuda() * cur_zs
            cur_ys = self.points_y_downscale.cuda() * cur_zs
        else:
            cur_xs = self.points_x_downscale * cur_zs
            cur_ys = self.points_y_downscale * cur_zs
        xyzs = torch.stack([cur_xs, cur_ys, cur_zs], axis=-1)
        return xyzs.permute(0, 3, 1, 2)

if __name__ == '__main__':
    # set torch and gpu setting
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    gpu = torch.cuda.is_available()
    if gpu:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

    # random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # init the model
    resnet = Backbone(in_dim=4, planes=args.feature_dim//16, mode='18')
    anchornet = AnchorGraspNet(feature_dim=args.feature_dim, ratio=args.ratio, anchor_k=args.anchor_k)
    pointnet = PointNetfeat(feature_len=3, use_bn=(not args.export_onnx)) # don't include bn in onnx
    localnet = PointMultiGraspNet(info_size=3, k_cls=args.anchor_num**2)
    if gpu:
        resnet = resnet.cuda()
        anchornet = anchornet.cuda()
        pointnet = pointnet.cuda()
        localnet = localnet.cuda()

    # load checkpoint
    check_point = torch.load(args.checkpoint_path, weights_only=True, map_location=torch.device('cpu'))
    # anchornet.load_state_dict(check_point['anchor'])
    # localnet.load_state_dict(check_point['local'])

    # load anchors
    basic_ranges = torch.linspace(-1, 1, args.anchor_num + 1)
    if gpu:
        basic_ranges = basic_ranges.cuda()
    basic_anchors = (basic_ranges[1:] + basic_ranges[:-1]) / 2
    anchors = {'gamma': basic_anchors, 'beta': basic_anchors}
    anchors['gamma'] = check_point['gamma']
    anchors['beta'] = check_point['beta']
    print('-> loaded checkpoint %s ' % (args.checkpoint_path))
    print("gamma anchors", anchors['gamma'])
    print("beta anchors", anchors['beta'])

    # network eval mode
    anchornet.eval()
    localnet.eval()

    # read image and convert to tensor
    ori_depth = np.array(Image.open(args.depth_path).resize((args.input_w, args.input_h), Image.Resampling.NEAREST))
    ori_rgb = np.array(Image.open(args.rgb_path).resize((args.input_w, args.input_h), Image.Resampling.LANCZOS)) / 255.0
    ori_depth = np.clip(ori_depth, 0, 1000)
    ori_rgb = torch.from_numpy(ori_rgb).permute(2, 1, 0)[None]
    ori_rgb = ori_rgb.to(device='cuda' if gpu else 'cpu', dtype=torch.float32)
    ori_depth = torch.from_numpy(ori_depth).T[None]
    ori_depth = ori_depth.to(device='cuda' if gpu else 'cpu', dtype=torch.float32)
    print("ori_rgb:", ori_rgb.shape, "ori_depth:", ori_depth.shape)
    set_resolution(args.input_w, args.input_h)
    print("camera_intrinsics:", get_camera_intrinsic()[0], get_camera_intrinsic()[1])

    # get scene points and xyz maps
    pc_helper = PointCloudHelper()
    view_points, _, _ = pc_helper.to_scene_points(ori_rgb, ori_depth)
    xyzs = pc_helper.to_xyz_maps(ori_depth)
    print("view_points:", view_points.shape) # 3rd dimension: [x, y, z, r, g, b]
    print("xyz:", xyzs.shape)

    # downscale image and normalize depth
    rgb = F.interpolate(ori_rgb, (args.input_w//2, args.input_h//2))
    depth = F.interpolate(ori_depth[None], (args.input_w//2, args.input_h//2))[0]
    depth = depth / 1000.0
    depth = torch.clip((depth - depth.mean()), -1, 1)

    # generate 2d input
    x = torch.concat([depth[None], rgb], 1)
    x = x.to(device='cuda' if gpu else 'cpu', dtype=torch.float32)
    print("x:", x.shape)

    # inference
    with torch.no_grad():
        # use ResNet to get downscaled features
        xs = resnet(x)

        # 2d prediction (GHM)
        pred_2d, perpoint_features = anchornet(xs)
        loc_map, cls_mask, theta_offset, height_offset, width_offset = anchor_output_process(*pred_2d, sigma=args.sigma)

        # detect 2d grasps
        rect_gg = detect_2d_grasp(loc_map,
                                  cls_mask,
                                  theta_offset,
                                  height_offset,
                                  width_offset,
                                  ratio=args.ratio,
                                  anchor_k=args.anchor_k,
                                  anchor_w=args.anchor_w,
                                  anchor_z=args.anchor_z,
                                  mask_thre=args.heatmap_thres,
                                  center_num=args.center_num,
                                  grid_size=args.grid_size,
                                  grasp_nms=args.grid_size)
        print("# 2d grasps found:", rect_gg.size)

        # check if a grasp is found
        assert rect_gg.size != 0, "No 2d grasp found"

        # show heatmap
        rgb_t = x[0, 1:].cpu().numpy().squeeze().transpose(2, 1, 0)
        depth_t = ori_depth.cpu().numpy().squeeze().T / 1000.0
        resized_rgb = Image.fromarray((rgb_t * 255.0).astype(np.uint8))
        resized_rgb = np.array(resized_rgb.resize((args.input_w//2, args.input_h//2))) / 255.0
        rect_rgb = rect_gg.plot_rect_grasp_group(resized_rgb, 0).clip(0, 1)
        plt.subplot(221)
        plt.imshow(rgb_t) # original image
        plt.subplot(222)
        plt.imshow(depth_t) # depth image
        plt.subplot(223)
        plt.imshow(loc_map.squeeze().T, cmap='jet') # heatmap
        plt.subplot(224)
        plt.imshow(rect_rgb) # grasps
        plt.tight_layout()
        plt.show()

        # -------------------------------- #
        # wait for user to close window... #
        # -------------------------------- #

        if args.export_onnx:
            torch.onnx.export(
                resnet,
                x,                          # model input
                "resnet.onnx",              # file to save
                export_params=True,         # store the trained parameter weights inside the onnx file
                opset_version=11,           # ONNX opset version
                do_constant_folding=True,   # fold constant ops for optimization
                input_names=["rgbd_image"],
                output_names=[
                    "layer_1_features",
                    "layer_2_features",
                    "layer_3_features",
                    "layer_4_features",
                    "layer_5_features"
                ]
            )
            print("resnet.onnx saved")

            torch.onnx.export(
                anchornet,
                xs,                         # model input
                "anchornet.onnx",           # file to save
                export_params=True,         # store the trained parameter weights inside the onnx file
                opset_version=11,           # ONNX opset version
                do_constant_folding=True,   # fold constant ops for optimization
                input_names=[
                    "layer_1_features",
                    "layer_2_features",
                    "layer_3_features",
                    "layer_4_features",
                    "layer_5_features"
                ],
                output_names=[
                    "loc_map",
                    "cls_mask",
                    "theta_offset",
                    "depth_offset",
                    "width_offset",
                    "perpoint_features"
                ]
            )
            print("anchornet.onnx saved")

        # feature fusion
        points_all = feature_fusion(view_points[..., :3], perpoint_features, xyzs)
        pc_group, valid_local_centers, new_rect_ggs = data_process(
            points_all,
            ori_depth,
            [rect_gg],
            args.center_num,
            args.group_num,
            (args.input_w, args.input_h),
            min_points=32,
            is_training=False)
        rect_gg = new_rect_ggs[0] # overwrite rect_gg to update valid mask
        points_all = points_all.squeeze()
        pc_group = pc_group.transpose(1, 2) # change structure for localnet
        print("points_all:", points_all.shape)
        print("pc_group:", pc_group.shape)

        # get 2d grasp info (not grasp itself) for trainning
        grasp_info = np.zeros((0, 3), dtype=np.float32)
        g_thetas = rect_gg.thetas[None]
        g_ws = rect_gg.widths[None]
        g_ds = rect_gg.depths[None]
        cur_info = np.vstack([g_thetas, g_ws, g_ds])
        grasp_info = np.vstack([grasp_info, cur_info.T])
        grasp_info = torch.from_numpy(grasp_info).to(dtype=torch.float32, device='cuda' if gpu else 'cpu')
        print("grasp_info:", grasp_info.shape)

        # pointnet for feature extraction
        features = pointnet(pc_group)

        # localnet (NMG)
        pred, offset = localnet(features, grasp_info)

        # detect 6d grasp from 2d output and 6d output
        _, pred_rect_gg = detect_6d_grasp_multi(rect_gg,
                                                pred,
                                                offset,
                                                valid_local_centers,
                                                (args.input_w//2, args.input_h//2),
                                                anchors,
                                                k=args.local_k)

        # collision detect
        pred_grasp_from_rect = pred_rect_gg.to_6d_grasp_group(depth=args.depth_thres)
        pred_gg, _ = collision_detect(points_all, pred_grasp_from_rect, mode='graspnet')
        pred_gg = pred_gg.nms() # remove redundant grasps
        print("# 6d grasps found:", pred_gg.size)

        # show grasps in 3d
        grasp_geo = pred_gg.to_open3d_geometry_list()
        points = view_points[..., :3].cpu().numpy().squeeze()
        colors = view_points[..., 3:6].cpu().numpy().squeeze()
        vispc = o3d.geometry.PointCloud()
        vispc.points = o3d.utility.Vector3dVector(points)
        vispc.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([vispc] + grasp_geo)

        if args.export_onnx:
            torch.onnx.export(
                pointnet,
                pc_group,     # model input
                "pointnet.onnx",            # file to save
                export_params=True,         # store the trained parameter weights inside the onnx file
                opset_version=11,           # ONNX opset version
                do_constant_folding=True,   # fold constant ops for optimization
                input_names=["pointcloud"],
                output_names=[
                    "pointnet_features"
                ]
            )
            print("pointnet.onnx saved")

            torch.onnx.export(
                localnet,
                (features, grasp_info),     # model input
                "localnet.onnx",            # file to save
                export_params=True,         # store the trained parameter weights inside the onnx file
                opset_version=11,           # ONNX opset version
                do_constant_folding=True,   # fold constant ops for optimization
                input_names=["pointnet_features", "grasp_info"],
                output_names=[
                    "anchor_pred",
                    "offset_pred"
                ]
            )
            print("localnet.onnx saved")

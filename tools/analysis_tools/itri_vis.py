# Copyright (c) Phigent Robotics. All rights reserved.
import argparse
import json
import os
import pickle

import cv2
import numpy as np
from pyquaternion.quaternion import Quaternion

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB


def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(
        valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    return valid


def depth2color(depth):
    gray = max(0, min((depth + 2.5) / 3.0, 1.0))
    max_lumi = 200
    colors = np.array(
        [[max_lumi, 0, max_lumi], [max_lumi, 0, 0], [max_lumi, max_lumi, 0],
         [0, max_lumi, 0], [0, max_lumi, max_lumi], [0, 0, max_lumi]],
        dtype=np.float32)
    if gray == 1:
        return tuple(colors[-1].tolist())
    num_rank = len(colors) - 1
    rank = np.floor(gray * num_rank).astype(np.int)
    diff = (gray - rank / num_rank) * num_rank
    return tuple(
        (colors[rank] + (colors[rank + 1] - colors[rank]) * diff).tolist())


def lidar2img(points_lidar, camrera_info):
    points_lidar_homogeneous = \
        np.concatenate([points_lidar,
                        np.ones((points_lidar.shape[0], 1),
                                dtype=points_lidar.dtype)], axis=1)
    camera2lidar = np.eye(4, dtype=np.float32)
    camera2lidar[:3, :3] = camrera_info['sensor2lidar_rotation']
    camera2lidar[:3, 3] = camrera_info['sensor2lidar_translation']
    lidar2camera = np.linalg.inv(camera2lidar)
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = np.array(camrera_info['cam_intrinsic']).reshape((3,3))
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid


def get_lidar2global(infos):
    lidar2ego = np.eye(4, dtype=np.float32)
    # lidar2ego[:3, :3] = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
    # lidar2ego[:3, 3] = infos['lidar2ego_translation']
    ego2global = np.eye(4, dtype=np.float32)
    # ego2global[:3, :3] = Quaternion(
    #     infos['ego2global_rotation']).rotation_matrix
    # ego2global[:3, 3] = infos['ego2global_translation']
    return ego2global @ lidar2ego

def align_coordinate(arr):
    arr = np.array(arr)
    arr = arr[[1, 0, 2]]
    arr[0] = -arr[0]
    # arr[1] = -arr[1]
    return list(arr)

def align_coordinate_2(arr):
    arr = np.array(arr)
    arr = arr[[1, 0, 2]]
    return list(arr)

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the predicted '
                                     'result of nuScenes')
    parser.add_argument(
        'res', help='Path to the predicted result in json format')
    parser.add_argument(
        '--show-range',
        type=int,
        default=50,
        help='Range of visualization in BEV')
    parser.add_argument(
        '--canva-size', type=int, default=1000, help='Size of canva in pixel')
    parser.add_argument(
        '--vis-frames',
        type=int,
        default=500,
        help='Number of frames for visualization')
    parser.add_argument(
        '--scale-factor',
        type=int,
        default=4,
        help='Trade-off between image-view and bev in size of '
        'the visualized canvas')
    parser.add_argument(
        '--vis-thred',
        type=float,
        default=0.3,
        help='Threshold the predicted results')
    parser.add_argument('--draw-gt', action='store_true')
    parser.add_argument('--save-result', action='store_true')
    parser.add_argument(
        '--version',
        type=str,
        default='val',
        help='Version of nuScenes dataset')
    parser.add_argument(
        '--root_path',
        type=str,
        default='./data/nuscenes',
        help='Path to nuScenes dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./vis',
        help='Path to save visualization results')
    parser.add_argument(
        '--format',
        type=str,
        default='video',
        choices=['video', 'image'],
        help='The desired format of the visualization result')
    parser.add_argument(
        '--fps', type=int, default=20, help='Frame rate of video')
    parser.add_argument(
        '--video-prefix', type=str, default='vis', help='name of video')
    parser.add_argument(
        '--skip-frames', type=int, default=150, help='Frames to skip at the beginning')
    args = parser.parse_args()
    return args


color_map = {0: (255, 255, 0), 1: (0, 255, 255)}


def main():
    args = parse_args()
    # load predicted results
    if ".json" in args.res:
        res = json.load(open(args.res, 'r'))
    elif ".pkl" in args.res:
        with open(args.res, "rb") as f:
            res = pickle.load(f)
    # load dataset information
    info_path = \
        args.root_path + '/itri_infos_%s.pkl' % args.version
    dataset = pickle.load(open(info_path, 'rb'))
    # prepare save path and medium
    vis_dir = args.save_path
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    print('saving visualized result to %s' % vis_dir)
    scale_factor = args.scale_factor
    canva_size = args.canva_size
    show_range = args.show_range
    if args.format == 'video':
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        vout = cv2.VideoWriter(
            os.path.join(vis_dir, '%s.mp4' % args.video_prefix), fourcc,
            args.fps, (int(608 / scale_factor * 4),
                       int(380 / scale_factor + canva_size)))

    draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]
    draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                                   (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
                                   (2, 6), (3, 7)]
    views = [
        # 'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        # 'CAM_BACK', 'CAM_BACK_RIGHT'
        "image_front_bottom_60", "image_front_top_far_30", "image_left_back_60", "image_right_back_60"
    ]
    print('start visualizing results')
    dataset['infos'] = sorted(dataset['infos'], key = lambda ele: int(ele['timestamp']))
    for cnt, infos in enumerate(
            dataset['infos'][:min(args.vis_frames, len(dataset['infos']))], args.skip_frames):
        if cnt % 10 == 0:
            print('%d/%d' % (cnt, min(args.vis_frames, len(dataset['infos']))))
        # collect instances
        pred_res = res['results'][infos['token']]
        pred_boxes = [
            align_coordinate(pred_res[rid]['translation']) + align_coordinate_2(pred_res[rid]['size']) + [
                (Quaternion(pred_res[rid]['rotation']).yaw_pitch_roll[0])
            ] for rid in range(len(pred_res))
        ]

        """ only visualize ground-truth
        pred_res = []
        pred_boxes = []
        """
        if len(pred_boxes) == 0:
            corners_lidar = np.zeros((0, 3), dtype=np.float32)
        else:
            pred_boxes = np.array(pred_boxes, dtype=np.float32)
            boxes = LB(pred_boxes, origin=(0.0, 0.0, 0.0))
            corners_lidar = boxes.corners.numpy().reshape(-1, 3)
            # corners_global = boxes.corners.numpy().reshape(-1, 3)
            # corners_global = np.concatenate(
            #     [corners_global,
            #      np.ones([corners_global.shape[0], 1])],
            #     axis=1)
            # l2g = get_lidar2global(infos)
            # corners_lidar = corners_global @ np.linalg.inv(l2g).T
            # corners_lidar = corners_lidar[:, :3]
        pred_flag = np.ones((corners_lidar.shape[0] // 8, ), dtype=np.bool)
        scores = [
            pred_res[rid]['detection_score'] for rid in range(len(pred_res))
        ]
        if args.draw_gt or args.save_result:
            gt_boxes = np.array(infos['gt_boxes'])
            if not gt_boxes.shape[0] == 0:
                gt_boxes = gt_boxes[:, :7]
                # gt_boxes[:, -1] = gt_boxes[:, -1] + np.pi / 2
                x = gt_boxes[:, 0].copy()
                gt_boxes[:, 0] = gt_boxes[:, 1]
                gt_boxes[:, 1] = x
                gt_boxes[:, 0] = -gt_boxes[:, 0]
                # gt_boxes[:, 1] = -gt_boxes[:, 1]
                # width = gt_boxes[:, 4].copy()
                # gt_boxes[:, 4] = gt_boxes[:, 3]
                # gt_boxes[:, 3] = width
                corners_lidar_gt = \
                    LB(gt_boxes,
                    origin=(0.0, 0.0, 0.0)).corners.numpy().reshape(-1, 3)
                corners_lidar = np.concatenate([corners_lidar, corners_lidar_gt],
                                            axis=0)
                gt_flag = np.ones((corners_lidar_gt.shape[0] // 8), dtype=np.bool)
                pred_flag = np.concatenate(
                    [pred_flag, np.logical_not(gt_flag)], axis=0)
                scores = scores + [0 for _ in range(gt_boxes.shape[0])]
        scores = np.array(scores, dtype=np.float32)
        sort_ids = np.argsort(scores)

        if args.save_result: # save per frame result as txt file for evaluation
            if not os.path.exists(f'{args.save_path}/pred_labels/'):
                os.mkdir(f'{args.save_path}/pred_labels/')
            if not os.path.exists(f'{args.save_path}/gt_labels/'):
                os.mkdir(f'{args.save_path}/gt_labels/')
            with open(f'{args.save_path}/pred_labels/{infos["token"]}.txt', 'w') as f:
                for c, i, s in zip((pred_res[rid]['detection_name'] for rid in range(len(pred_res))), pred_boxes, (pred_res[rid]['detection_score'] for rid in range(len(pred_res)))):
                    a = ','.join(str(x) for x in ([c] + [str(y) for y in i] + [str(s)]))
                    print(a, file=f)
            with open(f'{args.save_path}/gt_labels/{infos["token"]}.txt', 'w') as f:
                for c, i in zip(np.array(infos['gt_names']), gt_boxes):
                    a = ','.join(str(x) for x in ([c] + [str(y) for y in i]))
                    print(a, file=f)

        # image view
        imgs = []
        for view in views:
            img = cv2.imread(infos['cams'][view]['data_path'])
            # draw instances
            # corners_img, valid = lidar2img(corners_lidar, infos['cams'][view])
            # valid = np.logical_and(
            #     valid,
            #     check_point_in_img(corners_img, img.shape[0], img.shape[1]))
            # valid = valid.reshape(-1, 8)
            # corners_img = corners_img.reshape(-1, 8, 2).astype(np.int)
            # for aid in range(valid.shape[0]):
            #     for index in draw_boxes_indexes_img_view:
            #         if valid[aid, index[0]] and valid[aid, index[1]] and scores[aid] > args.vis_thred:
            #             cv2.line(
            #                 img,
            #                 corners_img[aid, index[0]],
            #                 corners_img[aid, index[1]],
            #                 color=color_map[int(pred_flag[aid])],
            #                 thickness=scale_factor)
            imgs.append(img)

        # bird-eye-view
        canvas = np.zeros((int(canva_size), int(canva_size), 3),
                          dtype=np.uint8)
        # draw lidar points
        if args.draw_gt:
            with open(infos['lidar_path'], 'rb') as f:
                header = f.read(4096)
                num_points_line = [line for line in header.split(b'\n') if line.startswith(b'POINTS')][0]
                num_points = int(num_points_line.split(b' ')[-1])

                lidar_points = np.fromfile(f, dtype=np.float32, count=num_points*3)
                lidar_points = lidar_points.reshape(-1, 3)[:, :3]
                lidar_points[:, [0, 1]] = lidar_points[:, [1, 0]]
                lidar_points[:, 0] = -lidar_points[:, 0]
                lidar_points[:, 1] = -lidar_points[:, 1]
                lidar_points[:, :2] = \
                    (lidar_points[:, :2] + show_range) / show_range / 2.0 * canva_size
                for p in lidar_points:
                    if check_point_in_img(
                            p.reshape(1, 3), canvas.shape[1], canvas.shape[0])[0]:
                        color = depth2color(p[2])
                        cv2.circle(
                            canvas, (int(p[0]), int(p[1])),
                            radius=0,
                            color=color,
                            thickness=1)

        # draw instances
        corners_lidar = corners_lidar.reshape(-1, 8, 3)
        corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
        bottom_corners_bev = corners_lidar[:, [0, 3, 7, 4], :2]
        bottom_corners_bev = \
            (bottom_corners_bev + show_range) / show_range / 2.0 * canva_size
        bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
        center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)
        head_bev = corners_lidar[:, [0, 4], :2].mean(axis=1)
        canter_canvas = \
            (center_bev + show_range) / show_range / 2.0 * canva_size
        center_canvas = canter_canvas.astype(np.int32)
        head_canvas = (head_bev + show_range) / show_range / 2.0 * canva_size
        head_canvas = head_canvas.astype(np.int32)

        for rid in sort_ids:
            score = scores[rid]
            if score < args.vis_thred and pred_flag[rid]:
                continue
            score = min(score * 2.0, 1.0) if pred_flag[rid] else 1.0
            color = color_map[int(pred_flag[rid])]
            for index in draw_boxes_indexes_bev:
                cv2.line(
                    canvas,
                    bottom_corners_bev[rid, index[0]],
                    bottom_corners_bev[rid, index[1]],
                    [color[0] * score, color[1] * score, color[2] * score],
                    thickness=1)
            cv2.line(
                canvas,
                center_canvas[rid],
                head_canvas[rid],
                [color[0] * score, color[1] * score, color[2] * score],
                1,
                lineType=8)

        # fuse image-view and bev
        
        img = np.zeros((380 + canva_size, 608*4, 3),
                       dtype=np.uint8)
        # img[:380, 608:608*2, :] = np.concatenate(imgs[:], axis=1)
        img[:380, :, :] = np.concatenate(imgs[:], axis=1)
        # img_back = np.concatenate(
        #     [imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]],
        #     axis=1)
        # img[900 + canva_size * scale_factor:, :, :] = img_back
        w_begin = int((608 * 4 - canva_size) // 2)
        
        def draw_grid(img, grid_shape, color=(0, 255, 0), thickness=1):
            h, w, _ = img.shape
            rows, cols = grid_shape
            dy, dx = h / rows, w / cols

            # draw vertical lines
            for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
                x = int(round(x))
                cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

            # draw horizontal lines
            for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
                y = int(round(y))
                cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

            return img
        canvas = cv2.addWeighted(canvas, 0.9, draw_grid(canvas, (6, 6), (220, 220, 220), 1), 0.1, 0)
        img[int(380):int(380) + canva_size,
            w_begin:w_begin + canva_size, :] = canvas

        img = cv2.resize(img, (int(608 * 4),
                               int(380 + canva_size)))

        if args.format == 'image':
            cv2.imwrite(os.path.join(vis_dir, '%s.jpg' % infos['token']), img)
        elif args.format == 'video':
            vout.write(img)
    if args.format == 'video':
        vout.release()


if __name__ == '__main__':
    main()

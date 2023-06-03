import os
import random
import torch
import trimesh
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R

from torch.utils.data import Dataset, DataLoader
from functools import partial

from .kpconv_utils import collate_fn_kpconv, calibrate_neighbors


class PairwisePieceDataset(Dataset):
    """Geometry part assembly dataset, with fracture surface information

    We follow the data prepared by Breaking Bad dataset:
        https://breaking-bad-dataset.github.io/
    """

    def __init__(
            self,
            data_dir,
            data_fn,
            data_keys,
            category='',
            num_points=1000,
            min_num_part=2,
            max_num_part=20,
            shuffle_parts=False,
            rot_range=-1,
            overfit=-1,

            resample_ratio=-1,
            length=-1,

            require_fracture_points=-1,
            label_threshold=10,

            sample_by='point'
    ):
        # store parameters
        self.category = category if category.lower() != 'all' else ''
        self.data_dir = data_dir
        self.num_points = num_points
        self.min_num_part = min_num_part
        self.max_num_part = max_num_part  # ignore shapes with more parts
        self.shuffle_parts = shuffle_parts  # shuffle part orders
        self.rot_range = rot_range  # rotation range in degree
        self.require_fracture_points = require_fracture_points  # -1 for no labeled fracture points, 1 for gt labeled

        self.sample_by = sample_by  # ['point', 'area'] sample by fixed point number or mesh area

        # list of fracture folder path
        self.data_list, self.meta_list, self.meta_index, self.labels = self._read_data(data_fn)
        self.labels_01 = [1 if label >= label_threshold else 0 for label in self.labels]

        # TRAIN
        # dataset length:  7828990
        # positive:  828078
        # dataset length:  7837198
        # positive:  1016555
        # VAL
        # dataset length:  1800229
        # positive:  188211

        # surface
        # TRAIN
        # dataset length:  7837198
        # positive:  877221
        # VAL
        # dataset length:  1800229
        # positive:  198549

        print("dataset length: ", len(self.data_list))
        print("positive: ", sum(self.labels_01))

        if overfit > 0:
            # brute force cut the data length, not recommended
            self.data_list = self.data_list[:overfit]
            self.meta_list = self.meta_list[:overfit]
            self.meta_index = self.meta_index[:overfit]
            self.labels = self.labels[:overfit]
            self.labels_01 = self.labels_01[:overfit]

        if resample_ratio > 0:
            # Reweight the data distribution, resample_ratio = positive / (positive + negative)
            positive = sum(self.labels_01)
            total = int(positive / resample_ratio)
            if total < len(self.data_list):
                id_0 = [t for (t, itm) in enumerate(self.labels) if itm < label_threshold]
                pos_pos = [t for (t, itm) in enumerate(self.labels) if itm >= label_threshold]
                neg_pos = random.sample(id_0, total - positive)
                pos = pos_pos + neg_pos
                pos.sort()
                self.data_list = [self.data_list[i] for i in pos]
                self.meta_list = [self.meta_list[i] for i in pos]
                self.meta_index = [self.meta_index[i] for i in pos]
                self.labels = [self.labels[i] for i in pos]
                self.labels_01 = [self.labels_01[i] for i in pos]
            else:
                print("resample ratio is too low")
        print("resample length: ", len(self.data_list))

        # additional data to load, e.g. ('part_ids', 'instance_label')
        self.data_keys = data_keys

        if 0 < length < len(self.data_list):
            # If you only need a portion of data, we will resample it
            self.length = length
            pos = list(range(len(self.data_list)))
            random.shuffle(pos)
            self.data_list = [self.data_list[i] for i in pos]
            self.meta_list = [self.meta_list[i] for i in pos]
            self.meta_index = [self.meta_index[i] for i in pos]
            self.labels = [self.labels[i] for i in pos]
            self.labels_01 = [self.labels_01[i] for i in pos]
        else:
            self.length = len(self.data_list)

    def __len__(self):
        return self.length

    def _read_data(self, data_fn):
        """Filter out invalid number of parts."""
        if os.path.exists(os.path.join(self.data_dir, 'meta_for_piece_matching_' + data_fn)):
            with open(os.path.join(self.data_dir, 'meta_for_piece_matching_' + data_fn), 'rb') as meta_table:
                meta_dict = pickle.load(meta_table)
                data_list = meta_dict['data_list']
                meta_list = meta_dict['meta_list']
                meta_index = meta_dict['meta_index']
                labels = meta_dict['labels']
                print('load from existing files')
            return data_list, meta_list, meta_index, labels

        with open(os.path.join(self.data_dir, data_fn), 'r') as f:
            mesh_list = [line.strip() for line in f.readlines()]
            if self.category:
                mesh_list = [
                    line for line in mesh_list
                    if self.category in line.split('/')
                ]
        data_list = []
        meta_list = []
        meta_index = []
        labels = []
        print('start generating files')
        for mesh in mesh_list:
            mesh_dir = os.path.join(self.data_dir, mesh)
            mesh_path_split = mesh.partition('/')
            meta = os.path.join(mesh_path_split[0] + '_matching', mesh_path_split[2])
            meta_dir = os.path.join(self.data_dir, meta)
            if not os.path.isdir(mesh_dir):
                print(f'{mesh} does not exist')
                continue
            for frac in os.listdir(mesh_dir):
                # print(frac)
                # we take both fractures and modes for training
                if 'fractured' not in frac and 'mode' not in frac:
                    continue
                meta_frac = os.path.join(meta, frac)
                frac = os.path.join(mesh, frac)

                pieces = os.listdir(os.path.join(self.data_dir, frac))
                pieces.sort()
                meta_content = torch.load(os.path.join(self.data_dir, meta_frac, 'matching.pt'))
                matching_graph = meta_content['matching_graph']
                idx = 0
                for piece in pieces:
                    idx2 = 0
                    for piece2 in pieces:
                        if idx2 >= idx:
                            break
                        data_list.append([os.path.join(frac, piece), os.path.join(frac, piece2)])
                        meta_list.append([os.path.join(meta_frac, piece + '.matching.pt'),
                                          os.path.join(meta_frac, piece2 + '.matching.pt')])
                        meta_index.append([idx, idx2])
                        label = matching_graph[idx, idx2].item()
                        label = int(label)
                        labels.append(label)
                        idx2 += 1
                    idx += 1
        print("finish generation, start saving")
        meta_dict = {'data_list': data_list,
                     'meta_list': meta_list,
                     'meta_index': meta_index,
                     'labels': labels}
        with open(os.path.join(self.data_dir, 'meta_for_piece_matching_' + data_fn), 'wb') as meta_table:
            pickle.dump(meta_dict, meta_table, protocol=pickle.HIGHEST_PROTOCOL)
        return data_list, meta_list, meta_index, labels

    @staticmethod
    def _recenter_pc(pc):
        """pc: [N, 3]"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid[None]
        return pc, centroid

    def _rotate_pc(self, pc):
        """pc: [N, 3]"""
        if self.rot_range > 0.:
            rot_euler = (np.random.rand(3) - 0.5) * 2. * self.rot_range
            rot_mat = R.from_euler('xyz', rot_euler, degrees=True).as_matrix()
        else:
            rot_mat = R.random().as_matrix()
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        return pc, quat_gt

    @staticmethod
    def _shuffle_pc(pc, pc_gt, is_critical):
        """pc: [N, 3]"""
        order = np.arange(pc.shape[0])
        random.shuffle(order)
        pc = pc[order]
        pc_gt = pc_gt[order]
        is_critical = is_critical[order]
        return pc, pc_gt, is_critical

    def _pad_data(self, data, pad_size=None):
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...]."""
        if pad_size is None:
            pad_size = self.max_num_part
        data = np.array(data)
        pad_shape = (pad_size,) + tuple(data.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[:data.shape[0]] = data
        return pad_data

    def _get_pcs(self, pieces, metas, meta_idx):
        """Read mesh and sample point cloud from a folder."""
        # `piece`: xxx/plate/1d4093ad2dfad9df24be2e4f911ee4af/fractured_0/piece_0.obj
        piece_file = os.path.join(self.data_dir, pieces[0])
        piece2_file = os.path.join(self.data_dir, pieces[1])
        mesh = trimesh.load(piece_file, force='mesh')
        mesh2 = trimesh.load(piece2_file, force='mesh')
        meta = torch.load(os.path.join(self.data_dir, metas[0]))
        meta = meta['vertex_matching_piecewise'].to_dense()
        meta2 = torch.load(os.path.join(self.data_dir, metas[1]))
        meta2 = meta2['vertex_matching_piecewise'].to_dense()

        # read mesh and sample points
        pcs = []
        nps = []
        # meta = torch.load(os.path.join(self.data_dir, meta_folder, 'matching.pt'))
        if self.sample_by == 'area':
            area, area2 = mesh.area, mesh2.area
            total_area = sum([area, area2])
            num_point = int(area * self.num_points / total_area)
            num_point2 = self.num_points - num_point
            samples, fid = mesh.sample(num_point, return_index=True)
            samples2, fid2 = mesh2.sample(num_point2, return_index=True)
            pcs.extend([samples, samples2])
            nps.extend([num_point, num_point2])
        else:  # default sample by num_points
            samples, fid = mesh.sample(self.num_points, return_index=True)
            pcs.append(samples)
            samples2, fid2 = mesh2.sample(self.num_points, return_index=True)
            pcs.append(samples2)
            nps.extend([self.num_points, self.num_points])
        bary = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[fid], points=samples)
        bary2 = trimesh.triangles.points_to_barycentric(triangles=mesh2.triangles[fid2], points=samples2)
        bary = torch.Tensor(bary).to(torch.float32)
        bary2 = torch.Tensor(bary2).to(torch.float32)
        close_vertices = torch.LongTensor(mesh.faces[fid])
        close_vertices2 = torch.LongTensor(mesh2.faces[fid2])
        closest_vertex = close_vertices[torch.arange(self.num_points), torch.argmax(bary, dim=1)]
        closest_vertex2 = close_vertices2[torch.arange(self.num_points), torch.argmax(bary2, dim=1)]
        is_critical = torch.logical_and(meta[closest_vertex, 0] == 1, meta[closest_vertex, 1] == meta_idx[1]).to(
            torch.int)
        is_critical2 = torch.logical_and(meta2[closest_vertex2, 0] == 1, meta2[closest_vertex2, 1] == meta_idx[0]).to(
            torch.int)
        if self.sample_by == 'area':
            return pcs, [is_critical, is_critical2], nps
        else:
            return np.stack(pcs, axis=0), np.stack([is_critical, is_critical2], axis=0), nps

    def __getitem__(self, index):
        piece0_dir, piece1_dir = self.data_list[index]
        piece0_idx, piece1_idx = self.meta_index[index]
        pcs, is_criticals, nps = self._get_pcs(self.data_list[index], self.meta_list[index], self.meta_index[index])
        label = int(self.labels_01[index])
        cur_pts, cur_quat, cur_trans, cur_pts_gt, cur_is_critical, n_critical_pcs = [], [], [], [], [], []
        for i in range(2):
            pc = pcs[i]
            pc_gt = pcs[i].copy()
            is_critical = is_criticals[i]
            pc, gt_trans = self._recenter_pc(pc)
            pc, gt_quat = self._rotate_pc(pc)
            pc_shuffle, pc_gt_shuffle, is_critical = self._shuffle_pc(pc, pc_gt, is_critical)
            # print(is_critical.nonzero())
            is_critical = is_critical.nonzero()
            if type(is_critical) is tuple:
                is_critical = is_critical[0]
            n_critical_pc = is_critical.shape[0]
            is_critical = np.concatenate([is_critical, np.zeros(nps[i] - n_critical_pc)]).astype(int)
            n_critical_pcs.append(n_critical_pc)
            cur_pts.append(pc_shuffle)
            cur_quat.append(gt_quat)
            cur_trans.append(gt_trans)
            cur_pts_gt.append(pc_gt_shuffle)

            cur_is_critical.append(is_critical)
        if self.sample_by == 'area':
            cur_pts = np.concatenate(cur_pts).astype(np.float32)  # [N_sum, 3]
            cur_pts_gt = np.concatenate(cur_pts_gt).astype(np.float32)  # [N_sum, 3]
            cur_is_critical = np.concatenate(cur_is_critical).astype(np.int64)  # [N_sum]
            cur_quat = np.stack(cur_quat, axis=0).astype(np.float32)  # [2, 4]
            cur_trans = np.stack(cur_trans, axis=0).astype(np.float32)  # [2, 3]
            n_critical_pcs = np.array(n_critical_pcs).astype(np.int64)  # [2]
            n_pcs = np.array(nps).astype(np.int64)  # [2]
        else:
            cur_pts = np.stack(cur_pts, axis=0).astype(np.float32)  # [2, N, 3]
            cur_quat = np.stack(cur_quat, axis=0).astype(np.float32)  # [2, 4]
            cur_trans = np.stack(cur_trans, axis=0).astype(np.float32)  # [2, 3]
            cur_pts_gt = np.stack(cur_pts_gt, axis=0).astype(np.float32)  # [2, N, 3]
            cur_is_critical = np.stack(cur_is_critical, axis=0).astype(np.int64)  # [2, N]
            n_critical_pcs = np.array(n_critical_pcs).astype(np.int64)  # [2]
            n_pcs = np.array(nps).astype(np.int64)  # [2]
        # print("cur_pts", cur_pts.shape)
        """
        data_dict = {
            'part_pcs': [P, N, 3] The points sampled from each part.
            'gt_pcs': [P, N, 3] The ground truth points for each part.
            'n_pcs': [P] The number of point for each part.

            'critical_pcs_idx': [P, N] The index of critical points.
            'n_critical_pcs': [P] number of critical points for each part.

            'label': [1] 1 for matchable part, 0 for not matchable part.

            'part_trans': [P, 3] Translation vector
            'part_quat': [P, 4] Rotation as quaternion.
            'part_valids': [P] 1 for shape parts, 0 for padded zeros.
            'data_id': int ID of the data.
        }
        """
        data_dict = {'part_pcs': cur_pts,
                     'part_quat': cur_quat,
                     'part_trans': cur_trans,
                     'gt_pcs': cur_pts_gt,
                     'n_pcs': n_pcs,
                     'label': label,
                     'data_id': index,
                     'piece0_dir': piece0_dir,
                     'piece1_dir': piece1_dir,
                     'piece0_idx': piece0_idx,
                     'piece1_idx': piece1_idx, }
        if self.require_fracture_points > 0:
            data_dict.update({
                'critical_pcs_idx': cur_is_critical,
                'n_critical_pcs': n_critical_pcs
            })

        # label: 0 dimensional tensor
        return data_dict


def build_pairwise_piece_dataloader(cfg):
    data_dict = dict(
        data_dir=cfg.DATA.DATA_DIR,
        data_fn=cfg.DATA.DATA_FN.format('train'),
        data_keys=cfg.DATA.DATA_KEYS,
        category=cfg.DATA.CATEGORY,
        num_points=cfg.DATA.NUM_PC_POINTS,
        min_num_part=cfg.DATA.MIN_NUM_PART,
        max_num_part=cfg.DATA.MAX_NUM_PART,
        shuffle_parts=cfg.DATA.SHUFFLE_PARTS,
        rot_range=cfg.DATA.ROT_RANGE,
        overfit=cfg.DATA.OVERFIT,

        resample_ratio=cfg.DATA.RESAMPLE_RATIO,
        length=cfg.DATA.LENGTH * cfg.BATCH_SIZE,
        label_threshold=cfg.DATA.LABEL_THRESHOLD,

        require_fracture_points=cfg.DATA.REQUIRE_FRACTURE_POINTS,
        sample_by=cfg.DATA.SAMPLE_BY,
    )
    train_set = PairwisePieceDataset(**data_dict)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    data_dict['data_fn'] = cfg.DATA.DATA_FN.format('val')
    data_dict['shuffle_parts'] = False
    if data_dict['resample_ratio'] < 0.99:
        data_dict['resample_ratio'] = -1
    data_dict['length'] = cfg.DATA.TEST_LENGTH
    val_set = PairwisePieceDataset(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    return train_loader, val_loader


def build_pairwise_piece_kpconv_dataloader(cfg):
    data_dict = dict(
        data_dir=cfg.DATA.DATA_DIR,
        data_fn=cfg.DATA.DATA_FN.format('train'),
        data_keys=cfg.DATA.DATA_KEYS,
        category=cfg.DATA.CATEGORY,
        num_points=cfg.DATA.NUM_PC_POINTS,
        min_num_part=cfg.DATA.MIN_NUM_PART,
        max_num_part=cfg.DATA.MAX_NUM_PART,
        shuffle_parts=cfg.DATA.SHUFFLE_PARTS,
        rot_range=cfg.DATA.ROT_RANGE,
        overfit=cfg.DATA.OVERFIT,

        resample_ratio=cfg.DATA.RESAMPLE_RATIO,
        length=cfg.DATA.LENGTH * cfg.BATCH_SIZE,
        label_threshold=cfg.DATA.LABEL_THRESHOLD,

        require_fracture_points=cfg.DATA.REQUIRE_FRACTURE_POINTS,
    )
    train_set = PairwisePieceDataset(**data_dict)

    data_dict['data_fn'] = cfg.DATA.DATA_FN.format('val')
    data_dict['shuffle_parts'] = False
    if data_dict['resample_ratio'] < 0.99:
        data_dict['resample_ratio'] = -1
    data_dict['length'] = cfg.DATA.TEST_LENGTH
    val_set = PairwisePieceDataset(**data_dict)

    neighborhood_limits = calibrate_neighbors(train_set, cfg.MODEL, collate_fn=collate_fn_kpconv)
    print("neighborhood:", neighborhood_limits)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_fn_kpconv, config=cfg.MODEL, neighborhood_limits=neighborhood_limits),
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn_kpconv, config=cfg.MODEL, neighborhood_limits=neighborhood_limits),
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    return train_loader, val_loader

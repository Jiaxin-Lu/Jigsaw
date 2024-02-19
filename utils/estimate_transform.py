import numpy as np
import open3d as o3d
import torch

from scipy.spatial.transform import Rotation as R
import numpy as np
from .utils import lexico_iter
from .global_alignment import global_alignment


def estimate_global_transform(perm_mat, part_pcs, n_valid, n_pcs, n_critical_pcs, critical_pcs_idx, part_quat, part_trans, align_pivot=True):
    """
    Estimate global transformation based on the matching and points.
    Align to the largest piece if `align_pivot` is True.
    """
    B, P = n_critical_pcs.shape
    n_critical_pcs_cumsum = np.cumsum(n_critical_pcs, axis=-1)
    n_pcs_cumsum = np.cumsum(n_pcs, axis=-1)
    pred_dict = dict()
    pred_dict['rot'] = np.zeros((B, P, 3, 3))
    pred_dict['trans'] = np.zeros((B, P, 3))
    for b in range(B):
        piece_connections = np.zeros(n_valid[b])
        sum_full_matched = np.sum(perm_mat[b])
        edges, transformations, uncertainty = [], [], []

        # compute pairwise relative pose
        for idx1, idx2 in lexico_iter(np.arange(n_valid[b])):
            cri_st1 = 0 if idx1 == 0 else n_critical_pcs_cumsum[b, idx1 - 1]
            cri_ed1 = n_critical_pcs_cumsum[b, idx1]
            cri_st2 = 0 if idx2 == 0 else n_critical_pcs_cumsum[b, idx2 - 1]
            cri_ed2 = n_critical_pcs_cumsum[b, idx2]

            pc_st1 = 0 if idx1 == 0 else n_pcs_cumsum[b, idx1 - 1]
            pc_ed1 = n_pcs_cumsum[b, idx1]
            pc_st2 = 0 if idx2 == 0 else n_pcs_cumsum[b, idx2 - 1]
            pc_ed2 = n_pcs_cumsum[b, idx2]

            n1 = n_critical_pcs[b, idx1]
            n2 = n_critical_pcs[b, idx2]
            if n1 == 0 or n2 == 0:
                continue
            mat = perm_mat[b, cri_st1:cri_ed1, cri_st2:cri_ed2]  # [N1, N2]
            mat_s = np.sum(mat).astype(np.int32)
            mat2 = perm_mat[b, cri_st2:cri_ed2, cri_st1:cri_ed1]
            mat_s2 = np.sum(mat2).astype(np.int32)
            if mat_s < mat_s2:
                mat = mat2.transpose(1, 0)
                mat_s = mat_s2
            if n_valid[b] > 2 and mat_s == 0 and sum_full_matched > 0:
                continue
            if mat_s < 3:
                continue
            pc1 = part_pcs[b, pc_st1:pc_ed1]  # N, 3
            pc2 = part_pcs[b, pc_st2:pc_ed2]  # N, 3
            if critical_pcs_idx is not None:
                critical_pcs_src = pc1[critical_pcs_idx[b, pc_st1: pc_st1 + n1]]
                critical_pcs_tgt = pc2[critical_pcs_idx[b, pc_st2: pc_st2 + n2]]
                trans_mat = get_trans_from_mat(critical_pcs_src, critical_pcs_tgt, mat)
                edges.append(np.array([idx2, idx1]))
                transformations.append(trans_mat)
                uncertainty.append(1.0 / (mat_s))
                piece_connections[idx1] = piece_connections[idx1] + 1
                piece_connections[idx2] = piece_connections[idx2] + 1

        # connect small pieces with less than 3 correspondence
        for idx1, idx2 in lexico_iter(np.arange(n_valid[b])):
            if piece_connections[idx1] > 0 and piece_connections[idx2] > 0:
                continue
            if piece_connections[idx1] == 0 and piece_connections[idx2] == 0:
                continue
            cri_st1 = 0 if idx1 == 0 else n_critical_pcs_cumsum[b, idx1 - 1]
            cri_ed1 = n_critical_pcs_cumsum[b, idx1]
            cri_st2 = 0 if idx2 == 0 else n_critical_pcs_cumsum[b, idx2 - 1]
            cri_ed2 = n_critical_pcs_cumsum[b, idx2]

            pc_st1 = 0 if idx1 == 0 else n_pcs_cumsum[b, idx1 - 1]
            pc_ed1 = n_pcs_cumsum[b, idx1]
            pc_st2 = 0 if idx2 == 0 else n_pcs_cumsum[b, idx2 - 1]
            pc_ed2 = n_pcs_cumsum[b, idx2]

            n1 = n_critical_pcs[b, idx1]
            n2 = n_critical_pcs[b, idx2]
            if n1 == 0 or n2 == 0:
                edges.append(np.array([idx2, idx1]))
                trans_mat = np.eye(4)
                pc1 = part_pcs[b, pc_st1:pc_ed1]
                pc2 = part_pcs[b, pc_st2:pc_ed2]
                if n2 > 0:
                    trans_mat[:3, 3] = pc2[critical_pcs_idx[b, pc_st2]] - np.sum(pc1, axis=0)
                elif n1 > 0:
                    trans_mat[:3, 3] = np.sum(pc2, axis=0) - pc1[critical_pcs_idx[b, pc_st1]]
                else:
                    trans_mat[:3, 3] = np.sum(pc2, axis=0) - np.sum(pc1, axis=0)
                transformations.append(trans_mat)
                uncertainty.append(1)
                piece_connections[idx1] = piece_connections[idx1] + 1
                piece_connections[idx2] = piece_connections[idx2] + 1
                continue

            mat = perm_mat[b, cri_st1:cri_ed1, cri_st2:cri_ed2]  # [N1, N2]
            mat_s = np.sum(mat).astype(np.int32)
            mat2 = perm_mat[b, cri_st2:cri_ed2, cri_st1:cri_ed1]
            mat_s2 = np.sum(mat2).astype(np.int32)
            if mat_s < mat_s2:
                mat = mat2.transpose(1, 0)
                mat_s = mat_s2
            pc1 = part_pcs[b, pc_st1:pc_ed1]  # N, 3
            pc2 = part_pcs[b, pc_st2:pc_ed2]  # N, 3
            if critical_pcs_idx is not None:
                critical_pcs_src = pc1[critical_pcs_idx[b, pc_st1: pc_st1 + n1]]
                critical_pcs_tgt = pc2[critical_pcs_idx[b, pc_st2: pc_st2 + n2]]
                trans_mat = np.eye(4)
                matching1, matching2 = np.nonzero(mat)
                trans_mat[:3, 3] = np.sum(critical_pcs_tgt[matching2], axis=0) - \
                                    np.sum(critical_pcs_src[matching1], axis=0)
                edges.append(np.array([idx2, idx1]))
                transformations.append(trans_mat)
                uncertainty.append(1)
                piece_connections[idx1] = piece_connections[idx1] + 1
                piece_connections[idx2] = piece_connections[idx2] + 1
                
        if len(edges) > 0:
            edges = np.stack(edges)
            transformations = np.stack(transformations)
            uncertainty = np.array(uncertainty)
            global_transformations = global_alignment(n_valid[b], edges, transformations, uncertainty)
            pivot = 1
            for idx in range(n_valid[b]):
                num_points = n_pcs[b, idx]
                if num_points > n_pcs[b, pivot]:
                    pivot = idx
        else:
            global_transformations = np.repeat(np.eye(4).reshape((1, 4, 4)), n_valid[b], axis=0)
            pivot = 0
        
        if align_pivot:
            global_transformations = align_to_pivot(global_transformations, part_quat[b], part_trans[b], n_valid[b], pivot=pivot)

        pred_dict['rot'][b, :n_valid[b], :, :] = global_transformations[:, :3, :3]
        pred_dict['trans'][b, :n_valid[b], :] = global_transformations[:, :3, 3]
    return pred_dict


def align_to_pivot(global_transformations, part_quat, part_trans, P, pivot=0):
    to_pivot_trans_mat = np.eye(4)
    quat = part_quat[pivot]
    to_pivot_trans_mat[:3, :3] = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    to_pivot_trans_mat[:3, 3] = part_trans[pivot]

    offset = to_pivot_trans_mat @ np.linalg.inv(global_transformations[pivot, :, :])
    for idx in range(P):
        global_transformations[idx, :, :] = offset @ global_transformations[idx, :, :]
    return global_transformations


def run_ransac(xyz_i, xyz_j, corr_idx):
    """
    Ransac based estimation of the transformation paramaters of the congurency transformation. Estimates the
    transformation parameters thtat map xyz0 to xyz1. Implementation is based on the open3d library
    (http://www.open3d.org/docs/release/python_api/open3d.registration.registration_ransac_based_on_correspondence.html)

    Args:
    xyz_i (numpy array): coordinates of the correspondences from the first point cloud [n,3]
    xyz_j (numpy array): coordinates of the correspondences from the second point cloud [n,3]
    Returns:
    trans_param (float): mean registration precision over the scene (not so important because it can be increased see papers)
    recall (float): mean registration recall over the scene (deciding parameter for the performance of the algorithm)
    """

    # Distance threshold as specificed by 3DMatch dataset
    distance_threshold = 0.05

    # Convert the point to an open3d PointCloud object
    xyz0 = o3d.geometry.PointCloud()
    xyz1 = o3d.geometry.PointCloud()

    xyz0.points = o3d.utility.Vector3dVector(xyz_i)
    xyz1.points = o3d.utility.Vector3dVector(xyz_j)

    # Correspondences are already sorted
    # corr_idx = np.tile(np.expand_dims(np.arange(len(xyz0.points)), 1), (1, 2))
    corrs = o3d.utility.Vector2iVector(corr_idx)

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source=xyz0,
        target=xyz1,
        corres=corrs,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            50000, 2500
        ),
    )

    trans_mat = result_ransac.transformation
    return trans_mat


def get_corr_from_mat(mat):
    if isinstance(mat, np.ndarray):
        corr = np.vstack(mat.nonzero()).transpose(1, 0)
    else:
        assert isinstance(mat, torch.Tensor)
        corr = np.vstack(mat.nonzero())
    return corr


def get_trans_from_mat(pc_src, pc_tgt, mat):
    corr = get_corr_from_mat(mat)
    trans_mat = run_ransac(pc_src, pc_tgt, corr)
    return trans_mat


def get_trans_from_corr(pc_src, pc_tgt, corr):
    """
    pc_src: [N1, 3], positions of source point cloud
    pc_tgt: [N2, 3], positions of target point cloud
    corr: [N', 2], each row [idx1, idx2] matches point idx1 in source to point idx2 in target
    """
    trans_mat = run_ransac(pc_src, pc_tgt, corr)
    return trans_mat


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as R

    n = 400
    pc1 = np.random.random((n, 3))
    pc2 = pc1.copy()
    trans_mat = np.zeros((4, 4))
    trans_mat[:3, :3] = R.random().as_matrix()
    trans_mat[:3, 3] = np.random.random(3)
    trans_mat[3, 3] = 1
    pc1 = np.concatenate([pc1, np.ones((pc1.shape[0], 1))], axis=1)
    pc1 = trans_mat @ pc1.transpose(1, 0)
    pc1 = pc1[:3, :].transpose(1, 0)
    corr = np.vstack([np.arange(n), np.arange(n)]).transpose(1, 0)
    trans_mat2 = get_trans_from_corr(pc1, pc2, corr)
    print(trans_mat)
    print(trans_mat2)
    print(np.linalg.inv(trans_mat2))

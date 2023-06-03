import numpy as np
import torch
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
from utils import Timer


def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0,
                                  random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                                batches_len,
                                                                                features=features,
                                                                                classes=labels,
                                                                                sampleDl=sampleDl,
                                                                                max_p=max_p,
                                                                                verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(
            s_labels)


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)


def collate_fn_kpconv(list_data, config, neighborhood_limits):
    batched_lengths_list = []

    batched_part_pcs_list = []
    # batched_gt_pcs_list = []

    if len(list_data[0]['part_pcs'].shape) == 3:
        P, N, n_dim = list_data[0]['part_pcs'].shape
    else:
        n_dim = list_data[0]['part_pcs'].shape[-1]
        N = list_data[0]['part_pcs'].shape[0]
        P = 1
    B = len(list_data)

    for ind, (data_dict) in enumerate(list_data):
        batched_part_pcs_list.append(data_dict['part_pcs'].reshape(-1, n_dim))
        # batched_gt_pcs_list.append(data_dict['gt_pcs'].reshape(P*N, -1))
        if 'n_pcs' not in data_dict:
            if 'part_valids' in data_dict:
                p_valid = np.sum(data_dict['part_valids'])
                batched_lengths_list.extend([N for _ in range(p_valid)])
            else:
                batched_lengths_list.extend([N for _ in range(P)])
        else:
            batched_lengths_list.extend([n for n in data_dict['n_pcs'] if n > 0])

    batched_part_pcs = torch.from_numpy(np.concatenate(batched_part_pcs_list, axis=0))
    # batched_gt_pcs = torch.from_numpy(np.concatenate(batched_gt_pcs_list, axis=0))
    batched_lengths = torch.from_numpy(np.array(batched_lengths_list)).int()

    part_feats = torch.ones_like(batched_part_pcs[:, :1], dtype=torch.float32)
    if 'pos' in config.kp_feats:
        part_feats = torch.cat([part_feats, batched_part_pcs], dim=-1)

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []

    for block_i, block in enumerate(config.architecture):

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************

        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            conv_i = batch_neighbors_kpconv(batched_part_pcs, batched_part_pcs, batched_lengths, batched_lengths, r,
                                            neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv(batched_part_pcs, batched_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(pool_p, batched_part_pcs, pool_b, batched_lengths, r,
                                            neighborhood_limits[layer])

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(batched_part_pcs, pool_p, batched_lengths, pool_b, 2 * r,
                                          neighborhood_limits[layer])

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_part_pcs.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_part_pcs = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer += 1
        layer_blocks = []

    ###############
    # Return inputs
    ###############
    dict_inputs = {
        'points': input_points,  # list [N_sum, input_feat_dim]
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        # to align with the KPConv architecture, not needed specifically
        'features': part_feats,  # list [N_sum_last, dim]
        'stack_lengths': input_batches_len,  # list [tensor shape (B)]
    }

    # stack or cat the original data dict
    for k in list_data[0].keys():
        if isinstance(list_data[0][k], int):
            dict_inputs[k] = torch.from_numpy(np.array([data_dict[k] for data_dict in list_data]).astype(np.int64))
        elif isinstance(list_data[0][k], float):
            dict_inputs[k] = torch.from_numpy(np.array([data_dict[k] for data_dict in list_data]))
        elif isinstance(list_data[0][k], np.ndarray):
            dict_inputs[k] = torch.from_numpy(np.stack([data_dict[k] for data_dict in list_data]))
        elif isinstance(list_data[0][k], torch.Tensor):
            dict_inputs[k] = torch.stack([data_dict[k] for data_dict in list_data])
        elif isinstance(list_data[0][k], str):
            dict_inputs[k] = [data_dict[k] for data_dict in list_data]
        elif isinstance(list_data[0][k], np.number):
            dict_inputs[k] = torch.from_numpy(np.array([data_dict[k]
                                                        for data_dict in list_data]).astype(list_data[0][k].dtype))
        else:
            raise NotImplementedError(f"data_dict[{k}] should be in [float, int, str, np.ndarray, torch.Tensor], "
                                      f"but got {type(list_data[0][k])}")

    return dict_inputs


def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=2000):
    timer = Timer()
    last_display = timer.total_time

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        timer.tic()
        batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * 5)

        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in
                  batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)
        timer.toc()

        if timer.total_time - last_display > 0.1:
            last_display = timer.total_time
            print(f"Calib Neighbors {i:08d}: timings {timer.total_time:4.2f}s")

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles
    print('\n')

    return neighborhood_limits

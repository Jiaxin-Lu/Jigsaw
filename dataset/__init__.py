from .all_piece_matching_dataset import build_all_piece_matching_dataloader, build_all_piece_matching_kpconv_dataloader
from .geometry_dataset import build_geometry_dataloader
from .pairwise_piece_dataset import build_pairwise_piece_dataloader, \
    build_pairwise_piece_kpconv_dataloader
from .dataset_config import dataset_cfg


def build_dataloader(cfg):
    dataset = cfg.DATASET.lower().split(".")
    if dataset[0] == "breaking_bad":
        if dataset[1] == "assembly":
            return build_geometry_dataloader(cfg)
        elif dataset[1] == "all_piece_matching":
            return build_all_piece_matching_dataloader(cfg)
        elif dataset[1] == "all_piece_matching_kpconv":
            return build_all_piece_matching_kpconv_dataloader(cfg)
        elif dataset[1] == "pairwise_piece":
            return build_pairwise_piece_dataloader(cfg)
        elif dataset[1] == "pairwise_piece":
            return build_pairwise_piece_kpconv_dataloader(cfg)
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented")
    elif dataset[0] == "part_net":
        raise NotImplementedError(f"Dataset {dataset} not implemented")
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")

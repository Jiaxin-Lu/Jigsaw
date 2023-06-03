from easydict import EasyDict as edict

__C = edict()

dataset_cfg = __C

# Breaking Bad geometry assembly dataset

__C.BREAKING_BAD = edict()
__C.BREAKING_BAD.DATA_DIR = 'data/breaking_bad'
__C.BREAKING_BAD.DATA_FN = 'artifact.{}.txt'  # this would vary due to different subset, use artifact as default
__C.BREAKING_BAD.DATA_KEYS = ('part_ids', )

__C.BREAKING_BAD.SUBSET = ''  # must in ['artifact', 'everyday', 'other']
__C.BREAKING_BAD.CATEGORY = ''  # empty means all categories
__C.BREAKING_BAD.ALL_CATEGORY = [
    'BeerBottle', 'Bowl', 'Cup', 'DrinkingUtensil', 'Mug', 'Plate', 'Spoon',
    'Teacup', 'ToyFigure', 'WineBottle', 'Bottle', 'Cookie', 'DrinkBottle',
    'Mirror', 'PillBottle', 'Ring', 'Statue', 'Teapot', 'Vase', 'WineGlass'
]  # Only used for everyday

__C.BREAKING_BAD.ROT_RANGE = - 1.0  # rotation range for curriculum learning
__C.BREAKING_BAD.NUM_PC_POINTS = 1000  # points per part
__C.BREAKING_BAD.MIN_PART_POINT = 30  # if sampled by area, want to make sure all piece have >30 points
__C.BREAKING_BAD.MIN_NUM_PART = 2
__C.BREAKING_BAD.MAX_NUM_PART = 20
__C.BREAKING_BAD.SHUFFLE_PARTS = False
__C.BREAKING_BAD.SAMPLE_BY = 'num_points'

__C.BREAKING_BAD.NUM_SPARSE_POINTS = 100  # points per part after sampling
__C.BREAKING_BAD.POINT_LIMIT = 30000
__C.BREAKING_BAD.USE_AUGMENTATION = 1
__C.BREAKING_BAD.AUGMENTATION_NOISE = 0.005
__C.BREAKING_BAD.AUGMENTATION_ROTATION = 1.0

__C.BREAKING_BAD.RESAMPLE_RATIO = -1.0
__C.BREAKING_BAD.LENGTH = -1
__C.BREAKING_BAD.TEST_LENGTH = -1
__C.BREAKING_BAD.OVERFIT = -1
__C.BREAKING_BAD.LABEL_THRESHOLD = 10
__C.BREAKING_BAD.P2P_THRESHOLD = 0.02
__C.BREAKING_BAD.REQUIRE_FRACTURE_POINTS = -1

__C.BREAKING_BAD.COLORS = [
    [0, 204, 0],
    [204, 0, 0],
    [0, 0, 204],
    [127, 127, 0],
    [127, 0, 127],
    [0, 127, 127],
    [76, 153, 0],
    [153, 0, 76],
    [76, 0, 153],
    [153, 76, 0],
    [76, 0, 153],
    [153, 0, 76],
    [204, 51, 127],
    [204, 51, 127],
    [51, 204, 127],
    [51, 127, 204],
    [127, 51, 204],
    [127, 204, 51],
    [76, 76, 178],
    [76, 178, 76],
    [178, 76, 76],
]


# PartNet semantic assembly dataset

__C.PART_NET = edict()
__C.PART_NET.DATA_DIR = 'data/partnet'
__C.PART_NET.DATA_FN = 'Chair.{}.npy'  # this would vary due to different categories, use chair as default
__C.PART_NET.DATA_KEYS = ('part_ids', 'match_ids', 'contact_points')

__C.PART_NET.CATEGORY = ''  # must in ['chair', 'lamp', 'table']

__C.PART_NET.NUM_PC_POINTS = 1000
__C.PART_NET.NUM_PART_CATEGORY = 57  # Chair: 57, Lamp: 83, Table: 82
__C.PART_NET.MIN_NUM_PART = 2
__C.PART_NET.MAX_NUM_PART = 20
__C.PART_NET.SHUFFLE_PARTS = False

__C.PART_NET.OVERFIT = -1
__C.PART_NET.COLORS = [
    [0, 204, 0],
    [204, 0, 0],
    [0, 0, 204],
    [127, 127, 0],
    [127, 0, 127],
    [0, 127, 127],
    [76, 153, 0],
    [153, 0, 76],
    [76, 0, 153],
    [153, 76, 0],
    [76, 0, 153],
    [153, 0, 76],
    [204, 51, 127],
    [204, 51, 127],
    [51, 204, 127],
    [51, 127, 204],
    [127, 51, 204],
    [127, 204, 51],
    [76, 76, 178],
    [76, 178, 76],
    [178, 76, 76],
]

import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
cfg = __C


#########################
#   Directory options   #
#########################
__C.data_dir = "bbdd"
__C.model_dir = 'models'
__C.result_dir = 'results'

#########################
#      BBDD options     #
#########################
# Classes in the dataset for object detection (kidney)
__C.num_classes1 = 2  
__C.class_names1 = ['background','kidney']

# Classes in the dataset for classification
__C.num_classes2 = 7 
__C.class_names2 = ['sane','cyst', 'pyramid', 'hydronephrosis','others','bad_diff','hyperecogenic']
__C.num_global_classes = 2

#########################
#     Model options     #
#########################
# Type of aggregation of local lesion scores for global diagnosis
__C.agg_method='max' # max, avg, lme or area

#########################
#  Experiment options   #
#########################

__C.folds=[1,2,3,4,5]
__C.batch_size = 1

#########################
#      Test options     #
#########################
# Mask R-CNN options (1, for kidney) and Faster R-CNN options (2, for local pathologies)
cfg.th_score1=0.5
cfg.th_score2=0.5
cfg.th_iou1=0.5
cfg.th_iou2=0.5
cfg.th_mask=0.5
cfg.GLOBAL_PAT=True
cfg.VERBOSE=True

def get_output_dir():
    """Return the directory where experimental artifacts are placed.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    path = osp.abspath(osp.join(__C.OUT_DIR, 'output', __C.EXP_DIR))
    return path
    
def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            v=np.array(v)
            if type(b[k]) is not type(v):
                raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f,Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value

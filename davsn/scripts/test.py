import argparse
import os
import os.path as osp
import pprint
import warnings

from torch.utils import data

from davsn.model.accel_deeplabv2 import get_accel_deeplab_v2
from davsn.dataset.CityscapesSeq import CityscapesSeqDataSet
from davsn.domain_adaptation.eval_video_UDA import evaluate_domain_adaptation

from davsn.domain_adaptation.config import cfg, cfg_from_file

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    return parser.parse_args()


def main(config_file, exp_suffix):
    # LOAD ARGS
    assert config_file is not None, 'Missing cfg file'
    cfg_from_file(config_file)
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'
    if exp_suffix:
        cfg.EXP_NAME += f'_{exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TEST.SNAPSHOT_DIR[0] == '':
        cfg.TEST.SNAPSHOT_DIR[0] = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TEST.SNAPSHOT_DIR[0], exist_ok=True)

    print('Using config:')
    pprint.pprint(cfg)
    # load models
    models = []
    n_models = len(cfg.TEST.MODEL)
    if cfg.TEST.MODE == 'best':
        assert n_models == 1, 'Not yet supported'
    for i in range(n_models):
        if cfg.TEST.MODEL[i] == 'ACCEL_DeepLabv2':
            model = get_accel_deeplab_v2(num_classes=cfg.NUM_CLASSES,
                                   multi_level=cfg.TEST.MULTI_LEVEL[i])
        else:
            raise NotImplementedError(f"Not yet supported {cfg.TEST.MODEL[i]}")
        models.append(model)

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    # dataloaders
    test_dataset = CityscapesSeqDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                     list_path=cfg.DATA_LIST_TARGET,
                                     set=cfg.TEST.SET_TARGET,
                                     info_path=cfg.TEST.INFO_TARGET,
                                     crop_size=cfg.TEST.INPUT_SIZE_TARGET,
                                     mean=cfg.TEST.IMG_MEAN,
                                     labels_size=cfg.TEST.OUTPUT_SIZE_TARGET)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=cfg.TEST.BATCH_SIZE_TARGET,
                                  num_workers=cfg.NUM_WORKERS,
                                  shuffle=False,
                                  pin_memory=True)
    # eval
    evaluate_domain_adaptation(models, test_loader, cfg)


if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    print(args)
    main(args.cfg, args.exp_suffix)

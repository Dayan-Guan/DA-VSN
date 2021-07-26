import os
import os.path as osp
import time
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from advent.utils.func import per_class_iu, fast_hist
from advent.utils.serialization import pickle_dump, pickle_load
from PIL import Image

def evaluate_domain_adaptation( models, test_loader, cfg,
                                fixed_test_size=True,
                                verbose=True):
    device = cfg.GPU_ID
    interp = None
    if fixed_test_size:
        interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear', align_corners=True)
    # eval
    if cfg.TEST.MODE == 'video_single':
        eval_video_single(cfg, models,
                    device, test_loader, interp, fixed_test_size,
                    verbose)
    elif cfg.TEST.MODE == 'video_best':
        eval_video_best(cfg, models,
                  device, test_loader, interp, fixed_test_size,
                  verbose)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")

### Viper
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 190, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 0, 230, 119, 11, 32]
### SynthiaSeq
# palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 190, 153, 153, 153, 153, 153, 250, 170, 30,
#            220, 220, 0, 107, 142, 35, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def eval_video_single(cfg, models,
                device, test_loader, interp,
                fixed_test_size, verbose):
    num_classes = cfg.NUM_CLASSES
    assert len(cfg.TEST.RESTORE_FROM) == len(models), 'Number of models are not matched'
    for checkpoint, model in zip(cfg.TEST.RESTORE_FROM, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    for index, batch in tqdm(enumerate(test_loader)):
        # image, label, _, name = batch
        image, label, image2, _, name = batch
        file_name = name[0].split('/')[-1]
        frame = int(file_name.replace('_leftImg8bit.png', '')[-6:])
        frame1 = frame - 1
        flow_int16_x10_name = file_name.replace('leftImg8bit.png', str(frame1).zfill(6) + '_int16_x10')
        flow_int16_x10 = np.load(os.path.join(cfg.TEST.flow_path, flow_int16_x10_name + '.npy'))
        flow = torch.from_numpy(flow_int16_x10 / 10.0).permute(2, 0, 1).unsqueeze(0)
        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            output = None
            for model, model_weight in zip(models, cfg.TEST.MODEL_WEIGHT):
                # pred_main = model(image.cuda(device))[1]
                pred_main = models[0](image.cuda(device), image2.cuda(device), flow, device)[1]
                output_ = interp(pred_main).cpu().data[0].numpy()
                if output is None:
                    output = model_weight * output_
                else:
                    output += model_weight * output_

            assert output is not None, 'Output is None'
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)

            amax_output_col = colorize_mask(np.asarray(output, dtype=np.uint8))
            name = name[0].split('/')[-1]
            image_name = name.split('.')[0]
            # vis seg maps
            os.makedirs(cfg.TEST.SNAPSHOT_DIR[0] + '/best_results', exist_ok=True)
            amax_output_col.save('%s/%s_color.png' % (cfg.TEST.SNAPSHOT_DIR[0] + '/best_results', image_name))

        label = label.numpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
    inters_over_union_classes = per_class_iu(hist)
    print(f'mIoU = \t{round(np.nanmean(inters_over_union_classes) * 100, 2)}')
    if verbose:
        display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)
    print([np.round(iou*100, 1) for iou in inters_over_union_classes.tolist()])

def eval_video_best(cfg, models,
              device, test_loader, interp,
              fixed_test_size, verbose):
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    # start_iter = cfg.TEST.SNAPSHOT_STEP
    start_iter = cfg.TEST.SNAPSHOT_START_ITER
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER
    cache_path = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'all_res.pkl')
    if osp.exists(cache_path):
        all_res = pickle_load(cache_path)
    else:
        all_res = {}
    cur_best_miou = -1
    cur_best_model = ''
    for i_iter in range(start_iter, max_iter + 1, step):
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{i_iter}.pth')
        if not osp.exists(restore_from):
            # continue
            if cfg.TEST.WAIT_MODEL:
                print('Waiting for model..!')
                while not osp.exists(restore_from):
                    time.sleep(5)
        print("Evaluating model", restore_from)
        if i_iter not in all_res.keys():
            load_checkpoint_for_evaluation(models[0], restore_from, device)
            hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            test_iter = iter(test_loader)
            for index in tqdm(range(len(test_loader))):
                # image, label, _, name = next(test_iter)
                image, label, image2, _, name = next(test_iter)
                if not fixed_test_size:
                    interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
                with torch.no_grad():
                    file_name = name[0].split('/')[-1]
                    frame = int(file_name.replace('_leftImg8bit.png', '')[-6:])
                    frame1 = frame - 1
                    flow_int16_x10_name = file_name.replace('leftImg8bit.png', str(frame1).zfill(6) + '_int16_x10')
                    flow_int16_x10 = np.load(os.path.join(cfg.TEST.flow_path, flow_int16_x10_name + '.npy'))
                    flow = torch.from_numpy(flow_int16_x10 / 10.0).permute(2, 0, 1).unsqueeze(0)
                    # pred_main = models[0](image.cuda(device))[1]
                    pred_main= models[0](image.cuda(device), image2.cuda(device), flow, device)[1]
                    pred_argmax = torch.argmax(interp(pred_main),dim=1)
                    output = pred_argmax.cpu().data[0].numpy()
                label = label.numpy()[0]
                hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
                if verbose and index > 0 and index % 100 == 0:
                    print('{:d} / {:d}: {:0.2f}'.format(
                        index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
            inters_over_union_classes = per_class_iu(hist)
            all_res[i_iter] = inters_over_union_classes
            pickle_dump(all_res, cache_path)
        else:
            inters_over_union_classes = all_res[i_iter]
        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        if cur_best_miou < computed_miou:
            cur_best_miou = computed_miou
            cur_best_iou = inters_over_union_classes * 100
            cur_best_model = restore_from
        print('\tCurrent mIoU:', computed_miou)
        print('\tCurrent best model:', cur_best_model)
        print('\tCurrent best mIoU:', cur_best_miou)
        # print(cur_best_iou)
        print([np.round(iou,1) for iou in cur_best_iou])
        if verbose:
            display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)

def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))
import os
import sys
from pathlib import Path
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm
from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc, bce_loss
from advent.utils.loss import entropy_loss
from advent.utils.func import prob_2_entropy
from advent.utils.viz_segmask import colorize_mask

def train_domain_adaptation(model, trainloader, targetloader, cfg):
    if cfg.TRAIN.DA_METHOD == 'DAVSN':
        train_DAVSN(model, trainloader, targetloader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")

def train_DAVSN(model, trainloader, targetloader, cfg):
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)
    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True
    # DISCRIMINATOR NETWORK
    d_temporal_aux = get_fc_discriminator(num_classes=num_classes * 2)
    d_temporal_aux.train()
    d_temporal_aux.to(device)
    d_temporal_main = get_fc_discriminator(num_classes=num_classes * 2)
    d_temporal_main.train()
    d_temporal_main.to(device)
    d_spatial_aux = get_fc_discriminator(num_classes=num_classes*2)
    d_spatial_aux.train()
    d_spatial_aux.to(device)
    d_spatial_main = get_fc_discriminator(num_classes=num_classes*2)
    d_spatial_main.train()
    d_spatial_main.to(device)

    # OPTIMIZERS
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    # discriminators' optimizers
    optimizer_d_temporal_aux = optim.Adam(d_temporal_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                          betas=(0.9, 0.99))
    optimizer_d_temporal_main = optim.Adam(d_temporal_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                           betas=(0.9, 0.99))
    optimizer_d_spatial_aux = optim.Adam(d_spatial_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_spatial_main = optim.Adam(d_spatial_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))
    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)
    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    aux_factor = cfg.TRAIN.LAMBDA_ADV_AUX / cfg.TRAIN.LAMBDA_ADV_MAIN #as in Advent
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):
        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_temporal_aux.zero_grad()
        optimizer_d_temporal_main.zero_grad()
        optimizer_d_spatial_aux.zero_grad()
        optimizer_d_spatial_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_temporal_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_temporal_main, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_spatial_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_spatial_main, i_iter, cfg)
        # Source-domain supervised training
        for param in d_temporal_aux.parameters():
            param.requires_grad = False
        for param in d_temporal_main.parameters():
            param.requires_grad = False
        for param in d_spatial_aux.parameters():
            param.requires_grad = False
        for param in d_spatial_main.parameters():
            param.requires_grad = False
        _, batch = trainloader_iter.__next__()
        image, label, image2, _, name = batch
        if label.dim() == 4:
            label = label.squeeze(-1)
        file_name = name[0].split('/')[-1]
        if cfg.SOURCE == 'Viper':
            frame = int(file_name.replace('.jpg', '')[-5:])
            frame1 = frame - 1
            flow_int16_x10_name = file_name.replace('.jpg', str(frame1).zfill(5) + '_int16_x10')
        elif cfg.SOURCE == 'SynthiaSeq':
            flow_int16_x10_name = file_name.replace('.png', '_int16_x10')
        flow_int16_x10 = np.load(os.path.join(cfg.TRAIN.flow_path_src, flow_int16_x10_name + '.npy'))
        flow = torch.from_numpy(flow_int16_x10 / 10.0).permute(2, 0, 1).unsqueeze(0)
        pred_src_aux, pred_src, pred_src_aux1, pred_src1, pred_src_aux2, pred_src2 = model(image.cuda(device), image2.cuda(device), flow, device)
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, label, device)
        else:
            loss_seg_src_aux = 0
        pred_src = interp(pred_src)
        loss_seg_src_main = loss_calc(pred_src, label, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # Usupervised domain adaptation
        _, batch_trg = targetloader_iter.__next__()
        image_trg, _, image_trg2, _, name = batch_trg
        file_name = name[0].split('/')[-1]
        frame = int(file_name.replace('_leftImg8bit.png', '')[-6:])
        frame1 = frame - 1
        flow_int16_x10_name_trg = file_name.replace('leftImg8bit.png', str(frame1).zfill(6) + '_int16_x10')
        flow_int16_x10_trg = np.load(os.path.join(cfg.TRAIN.flow_path, flow_int16_x10_name_trg + '.npy'))
        flow = torch.from_numpy(flow_int16_x10_trg / 10.0).permute(2, 0, 1).unsqueeze(0)
        pred_trg_aux, pred_trg, pred_trg_aux1, pred_trg1, pred_trg_aux2, pred_trg2 = model(image.cuda(device), image2.cuda(device), flow, device)

        ### Intra-domain TCR
        prob_trg1 = F.softmax(pred_trg1)
        prob_trg_aux1 = F.softmax(pred_trg_aux1)
        ent_trg1 = torch.mean(prob_2_entropy(prob_trg1), dim=1).detach().cpu()
        ent_trg_aux1 = torch.mean(prob_2_entropy(prob_trg_aux1), dim=1).detach().cpu()
        prob_trg2 = F.softmax(pred_trg2).cpu().numpy()
        prob_trg_aux2 = F.softmax(pred_trg_aux2).cpu().numpy()
        ent_trg2 = torch.mean(prob_2_entropy(F.softmax(pred_trg2)), dim=1).cpu().numpy()
        ent_trg_aux2 = torch.mean(prob_2_entropy(F.softmax(pred_trg_aux2)), dim=1).cpu().numpy()
        interp_flow2trg = nn.Upsample(size=(prob_trg1.shape[-2], prob_trg1.shape[-1]), mode='bilinear', align_corners=True)
        interp_flow2trg_ratio = prob_trg1.shape[-2] / flow.shape[-2]
        flow_trg = interp_flow2trg(flow) * interp_flow2trg_ratio
        flow_trg = flow_trg.cpu().numpy()
        prob_trg_rec = np.zeros(prob_trg1.shape)
        prob_trg_aux_rec = np.zeros(prob_trg1.shape)
        ent_trg_rec = np.zeros(ent_trg1.shape)
        ent_trg_aux_rec = np.zeros(ent_trg_aux1.shape)
        for x in range(prob_trg2.shape[-1]):
            for y in range(prob_trg2.shape[-2]):
                x_flow = int(round(x + flow_trg[:, 0, y, x][0]))
                y_flow = int(round(y + flow_trg[:, 1, y, x][0]))
                if x_flow >= 0 and x_flow < prob_trg2.shape[-1] and y_flow >= 0 and y_flow < prob_trg2.shape[-2]:
                    prob_trg_rec[:, :, y_flow, x_flow] = prob_trg2[:, :, y_flow, x_flow]
                    prob_trg_aux_rec[:, :, y_flow, x_flow] = prob_trg_aux2[:, :, y_flow, x_flow]
                    ent_trg_rec[:,y_flow,x_flow] = ent_trg2[:,y_flow,x_flow]
                    ent_trg_aux_rec[:,y_flow,x_flow] = ent_trg_aux2[:,y_flow,x_flow]
        prob_trg_rec = torch.from_numpy(prob_trg_rec)
        prob_trg_aux_rec = torch.from_numpy(prob_trg_aux_rec)
        prob_trg_rec_positions = torch.sum(prob_trg_rec,1)
        ent_trg_rec = torch.from_numpy(ent_trg_rec)
        ent_trg_aux_rec = torch.from_numpy(ent_trg_aux_rec)
        loss_tim_weights = prob_trg_rec_positions.float()*(ent_trg_rec.float() < ent_trg1).float()
        loss_tim_aux_weights = prob_trg_rec_positions.float()*(ent_trg_aux_rec.float() < ent_trg_aux1).float()
        print('\n### rec_detach_ratio: %.2f rec_aux_detach_ratio:  %.2f' %(loss_tim_weights.sum()/(loss_tim_weights.shape[-2]*loss_tim_weights.shape[-1]),
                                                                        loss_tim_aux_weights.sum()/(loss_tim_aux_weights.shape[-2]*loss_tim_aux_weights.shape[-1])))
        loss_tim = weighted_l1_loss(prob_trg1, prob_trg_rec.float().cuda(device), loss_tim_weights.float().cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            loss_tim_aux = weighted_l1_loss(prob_trg_aux1, prob_trg_aux_rec.float().cuda(device), loss_tim_aux_weights.float().cuda(device))
        else:
            loss_tim_aux = 0
        loss = (cfg.TRAIN.lamda_u * loss_tim + cfg.TRAIN.lamda_u * loss_tim_aux)

        ### Cross-domain TCR
        # adversarial training to fool the discriminator
        pred_src_concat = torch.cat((pred_src1, pred_src2), dim=1)
        pred_trg_concat = torch.cat((pred_trg1, pred_trg2), dim=1)
        pred_src_concat = interp(pred_src_concat)
        pred_trg_concat = interp_target(pred_trg_concat)
        d_out = d_temporal_main(F.softmax(pred_trg_concat))
        loss_adv = bce_loss(d_out, source_label)
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux_concat = torch.cat((pred_src_aux1, pred_src_aux2), dim=1)
            pred_trg_aux_concat = torch.cat((pred_trg_aux1, pred_trg_aux2), dim=1)
            pred_src_aux_concat = interp(pred_src_aux_concat)
            pred_trg_aux_concat = interp_target(pred_trg_aux_concat)
            d_out_aux = d_temporal_aux(F.softmax(pred_trg_aux_concat))
            loss_adv_aux = bce_loss(d_out_aux, source_label)
        else:
            loss_adv_aux = 0
        loss = loss + (cfg.TRAIN.lamda_u * loss_adv
                + cfg.TRAIN.lamda_u * aux_factor * loss_adv_aux)
        pred_src_concat_spatial = torch.cat((pred_src1, pred_src1), dim=1)
        pred_trg_concat_spatial = torch.cat((pred_trg1, pred_trg1), dim=1)
        pred_src_concat_spatial = interp(pred_src_concat_spatial)
        pred_trg_concat_spatial = interp_target(pred_trg_concat_spatial)
        d_out2 = d_spatial_main(F.softmax(pred_trg_concat_spatial))
        loss_adv2 = bce_loss(d_out2, source_label)
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux_concat_spatial = torch.cat((pred_src_aux1, pred_src_aux1), dim=1)
            pred_trg_aux_concat_spatial = torch.cat((pred_trg_aux1, pred_trg_aux1), dim=1)
            pred_src_aux_concat_spatial = interp(pred_src_aux_concat_spatial)
            pred_trg_aux_concat_spatial = interp_target(pred_trg_aux_concat_spatial)
            d_out_aux2 = d_spatial_aux(F.softmax(pred_trg_aux_concat_spatial))
            loss_adv_aux2 = bce_loss(d_out_aux2, source_label)
        else:
            loss_adv_aux2 = 0
        loss = loss + (cfg.TRAIN.lamda_u * cfg.TRAIN.lamda_sa * loss_adv2
                + cfg.TRAIN.lamda_u * cfg.TRAIN.lamda_sa * aux_factor * loss_adv_aux2)
        loss.backward()
        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_temporal_aux.parameters():
            param.requires_grad = True
        for param in d_temporal_main.parameters():
            param.requires_grad = True
        for param in d_spatial_aux.parameters():
            param.requires_grad = True
        for param in d_spatial_main.parameters():
            param.requires_grad = True
        # train with source
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux_concat = pred_src_aux_concat.detach()
            d_out_aux = d_temporal_aux(F.softmax(pred_src_aux_concat))
            loss_d_aux_src = bce_loss(d_out_aux, source_label) / 2
            loss_d_aux_src.backward()
        pred_src_concat = pred_src_concat.detach()
        d_out_main = d_temporal_main(F.softmax(pred_src_concat))
        loss_d_src = bce_loss(d_out_main, source_label) / 2
        loss_d_src.backward()
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux_concat_spatial = pred_src_aux_concat_spatial.detach()
            d_out_aux2 = d_spatial_aux(F.softmax(pred_src_aux_concat_spatial))
            loss_d2_aux_src = bce_loss(d_out_aux2, source_label) / 2
            loss_d2_aux_src.backward()
        pred_src_concat_spatial = pred_src_concat_spatial.detach()
        d_out_main2 = d_spatial_main(F.softmax(pred_src_concat_spatial))
        loss_d2_src = bce_loss(d_out_main2, source_label) / 2
        loss_d2_src.backward()
        # train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux_concat = pred_trg_aux_concat.detach()
            d_out_aux = d_temporal_aux(F.softmax(pred_trg_aux_concat))
            loss_d_aux = bce_loss(d_out_aux, target_label) / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_concat = pred_trg_concat.detach()
        d_out_main = d_temporal_main(F.softmax(pred_trg_concat))
        loss_d = bce_loss(d_out_main, target_label) / 2
        loss_d.backward()
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux_concat_spatial = pred_trg_aux_concat_spatial.detach()
            d_out_aux2 = d_spatial_aux(F.softmax(pred_trg_aux_concat_spatial))
            loss_d2_aux = bce_loss(d_out_aux2, target_label) / 2
            loss_d2_aux.backward()
        else:
            loss_d2_aux = 0
        pred_trg_concat_spatial = pred_trg_concat_spatial.detach()
        d_out_main2 = d_spatial_main(F.softmax(pred_trg_concat_spatial))
        loss_d2 = bce_loss(d_out_main2, target_label) / 2
        loss_d2.backward()
        # Discriminators Discrepancy Loss
        k = 0
        loss_dis = 0
        for (W1, W2) in zip(d_temporal_main.parameters(), d_spatial_main.parameters()):
            W1 = W1.view(-1)
            W2 = W2.view(-1)
            loss_dis = loss_dis + (torch.matmul(W1, W2) / (torch.norm(W1) * torch.norm(W2)) + 1) # +1 is for a positive loss
            k += 1
        loss_dis = loss_dis / k
        k = 0
        loss_dis_aux = 0
        if cfg.TRAIN.MULTI_LEVEL:
            for (W_aux1, W_aux2) in zip(d_temporal_aux.parameters(), d_spatial_aux.parameters()):
                W_aux1 = W_aux1.view(-1)
                W_aux2 = W_aux2.view(-1)
                loss_dis_aux = loss_dis_aux + (torch.matmul(W_aux1, W_aux2) / (torch.norm(W_aux1) * torch.norm(W_aux2)) + 1)
                k += 1
            loss_dis_aux = loss_dis_aux / k
        else:
            loss_dis_aux = 0
        loss = (cfg.TRAIN.lamda_u * cfg.TRAIN.lamda_wd * loss_dis + cfg.TRAIN.lamda_u * cfg.TRAIN.lamda_wd * loss_dis_aux)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(d_temporal_aux.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(d_temporal_main.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(d_spatial_aux.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(d_spatial_main.parameters(), 1)
        optimizer.step()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_temporal_aux.step()
            optimizer_d_spatial_aux.step()
        optimizer_d_temporal_main.step()
        optimizer_d_spatial_main.step()
        current_losses = {'loss_src_aux': loss_seg_src_aux,
                          'loss_src': loss_seg_src_main,
                          'loss_itcr_aux': loss_tim_aux,
                          'loss_itcr': loss_tim,
                          'loss_adv_aux': loss_adv_aux,
                          'loss_adv_aux2': loss_adv_aux2,
                          'loss_adv': loss_adv,
                          'loss_adv2': loss_adv2,
                          'loss_d_aux': loss_d_aux,
                          'loss_d2_aux': loss_d2_aux,
                          'loss_d': loss_d,
                          'loss_d2': loss_d2,
                          'loss_dis_aux': loss_dis_aux,
                          'loss_dis': loss_dis}
        print_losses(current_losses, i_iter)
        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

def weighted_l1_loss(input, target, weights):
    loss = weights * torch.abs(input - target)
    loss = torch.mean(loss)
    return loss

def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')

def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

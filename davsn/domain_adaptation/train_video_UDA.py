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

def train_domain_adaptation(model, source_loader, target_loader, cfg):
    if cfg.TRAIN.DA_METHOD == 'DAVSN':
        train_DAVSN(model, source_loader, target_loader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")

def train_DAVSN(model, source_loader, target_loader, cfg):
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
    d_sta_aux = get_fc_discriminator(num_classes=num_classes*2)
    d_sta_aux.train()
    d_sta_aux.to(device)
    d_sta_main = get_fc_discriminator(num_classes=num_classes*2)
    d_sta_main.train()
    d_sta_main.to(device)
    d_sa_aux = get_fc_discriminator(num_classes=num_classes*2)
    d_sa_aux.train()
    d_sa_aux.to(device)
    d_sa_main = get_fc_discriminator(num_classes=num_classes*2)
    d_sa_main.train()
    d_sa_main.to(device)

    # OPTIMIZERS
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    # discriminators' optimizers
    optimizer_d_sta_aux = optim.Adam(d_sta_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                          betas=(0.9, 0.99))
    optimizer_d_sta_main = optim.Adam(d_sta_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                           betas=(0.9, 0.99))
    optimizer_d_sa_aux = optim.Adam(d_sa_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_sa_main = optim.Adam(d_sa_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))
    # interpolate output segmaps
    interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)
    # labels for adversarial training
    source_label = 0
    target_label = 1
    source_loader_iter = enumerate(source_loader)
    target_loader_iter = enumerate(target_loader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):
        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_sta_aux.zero_grad()
        optimizer_d_sta_main.zero_grad()
        optimizer_d_sa_aux.zero_grad()
        optimizer_d_sa_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_sta_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_sta_main, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_sa_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_sa_main, i_iter, cfg)

        ######### Source-domain supervised training
        for param in d_sta_aux.parameters():
            param.requires_grad = False
        for param in d_sta_main.parameters():
            param.requires_grad = False
        for param in d_sa_aux.parameters():
            param.requires_grad = False
        for param in d_sa_main.parameters():
            param.requires_grad = False
        _, source_batch = source_loader_iter.__next__()
        src_img_cf, src_label, src_img_kf, _, src_img_name = source_batch
        if src_label.dim() == 4:
            src_label = src_label.squeeze(-1)
        file_name = src_img_name[0].split('/')[-1]
        if cfg.SOURCE == 'Viper':
            frame = int(file_name.replace('.jpg', '')[-5:])
            frame1 = frame - 1
            flow_int16_x10_name = file_name.replace('.jpg', str(frame1).zfill(5) + '_int16_x10')
        elif cfg.SOURCE == 'SynthiaSeq':
            flow_int16_x10_name = file_name.replace('.png', '_int16_x10')
        flow_int16_x10 = np.load(os.path.join(cfg.TRAIN.flow_path_src, flow_int16_x10_name + '.npy'))
        src_flow = torch.from_numpy(flow_int16_x10 / 10.0).permute(2, 0, 1).unsqueeze(0)
        src_pred_aux, src_pred, src_pred_cf_aux, src_pred_cf, src_pred_kf_aux, src_pred_kf = model(src_img_cf.cuda(device), src_img_kf.cuda(device), src_flow, device)
        src_pred = interp_source(src_pred)
        loss_seg_src_main = loss_calc(src_pred, src_label, device)
        if cfg.TRAIN.MULTI_LEVEL:
            src_pred_aux = interp_source(src_pred_aux)
            loss_seg_src_aux = loss_calc(src_pred_aux, src_label, device)
        else:
            loss_seg_src_aux = 0
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        ######### Usupervised domain adaptation
        _, target_batch = target_loader_iter.__next__()
        trg_img_cf, _, image_trg_kf, _, name = target_batch
        file_name = name[0].split('/')[-1]
        frame = int(file_name.replace('_leftImg8bit.png', '')[-6:])
        frame1 = frame - 1
        flow_int16_x10_name_trg = file_name.replace('leftImg8bit.png', str(frame1).zfill(6) + '_int16_x10')
        flow_int16_x10_trg = np.load(os.path.join(cfg.TRAIN.flow_path, flow_int16_x10_name_trg + '.npy'))
        trg_flow = torch.from_numpy(flow_int16_x10_trg / 10.0).permute(2, 0, 1).unsqueeze(0)
        trg_pred_aux, trg_pred, trg_pred_cf_aux, trg_pred_cf, trg_pred_kf_aux, trg_pred_kf = model(trg_img_cf.cuda(device), image_trg_kf.cuda(device), trg_flow, device)

        ###### Intra-domain TCR
        adversarial_factor_aux = cfg.TRAIN.LAMBDA_ADV_AUX / cfg.TRAIN.LAMBDA_ADV_MAIN  # as in Advent
        # for current frame (cf)
        trg_prob_cf = F.softmax(trg_pred_cf)
        trg_prob_cf_aux = F.softmax(trg_pred_cf_aux)
        trg_ent_cf = torch.mean(prob_2_entropy(trg_prob_cf), dim=1).detach().cpu()
        trg_ent_cf_aux = torch.mean(prob_2_entropy(trg_prob_cf_aux), dim=1).detach().cpu()
        # for key frame (kf)
        trg_prob_kf = F.softmax(trg_pred_kf).cpu().numpy()
        trg_prob_aux_kf = F.softmax(trg_pred_kf_aux).cpu().numpy()
        trg_ent_kf = torch.mean(prob_2_entropy(F.softmax(trg_pred_kf)), dim=1).cpu().numpy()
        trg_ent_kf_aux = torch.mean(prob_2_entropy(F.softmax(trg_pred_kf_aux)), dim=1).cpu().numpy()
        # generate propogated prediction via optical flow
        interp_flow2trg = nn.Upsample(size=(trg_prob_cf.shape[-2], trg_prob_cf.shape[-1]), mode='bilinear', align_corners=True)
        interp_flow2trg_ratio = trg_prob_cf.shape[-2] / trg_flow.shape[-2]
        trg_flow_interp = interp_flow2trg(trg_flow) * interp_flow2trg_ratio
        trg_flow_interp = trg_flow_interp.cpu().numpy()
        trg_prob_propagated = np.zeros(trg_prob_cf.shape)
        trg_prob_propagated_aux = np.zeros(trg_prob_cf_aux.shape)
        trg_ent_propagated = np.zeros(trg_ent_cf.shape)
        trg_ent_propagated_aux = np.zeros(trg_ent_cf_aux.shape)
        for x in range(trg_prob_kf.shape[-1]):
            for y in range(trg_prob_kf.shape[-2]):
                x_flow = int(round(x + trg_flow_interp[:, 0, y, x][0]))
                y_flow = int(round(y + trg_flow_interp[:, 1, y, x][0]))
                if x_flow >= 0 and x_flow < trg_prob_kf.shape[-1] and y_flow >= 0 and y_flow < trg_prob_kf.shape[-2]:
                    trg_prob_propagated[:, :, y_flow, x_flow] = trg_prob_kf[:, :, y, x]
                    trg_prob_propagated_aux[:, :, y_flow, x_flow] = trg_prob_aux_kf[:, :, y, x]
                    trg_ent_propagated[:,y_flow,x_flow] = trg_ent_kf[:,y,x]
                    trg_ent_propagated_aux[:,y_flow,x_flow] = trg_ent_kf_aux[:,y,x]
        trg_prob_propagated = torch.from_numpy(trg_prob_propagated)
        trg_prob_propagated_aux = torch.from_numpy(trg_prob_propagated_aux)
        trg_propagated_positions = torch.sum(trg_prob_propagated,1)
        trg_ent_propagated = torch.from_numpy(trg_ent_propagated)
        trg_ent_propagated_aux = torch.from_numpy(trg_ent_propagated_aux)
        # force unconfident predictions in the current frame to be consistent with confident predictions propagated from the previous frames
        loss_itcr_weights = trg_propagated_positions.float()*(trg_ent_propagated.float() < trg_ent_cf).float()
        loss_itcr = weighted_l1_loss(trg_prob_cf, trg_prob_propagated.float().cuda(device), loss_itcr_weights.float().cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            loss_itcr_aux_weights = trg_propagated_positions.float() * (trg_ent_propagated_aux.float() < trg_ent_cf_aux).float()
            loss_itcr_aux = weighted_l1_loss(trg_prob_cf_aux, trg_prob_propagated_aux.float().cuda(device), loss_itcr_aux_weights.float().cuda(device))
        else:
            loss_itcr_aux = 0
        loss = (cfg.TRAIN.lamda_u * loss_itcr + cfg.TRAIN.lamda_u * loss_itcr_aux)

        ###### Cross-domain TCR
        ### adversarial training ot fool the discriminator
        # spatial-temporal alignment (sta)
        src_sta_pred = torch.cat((src_pred_cf, src_pred_kf), dim=1)
        trg_sta_pred = torch.cat((trg_pred_cf, trg_pred_kf), dim=1)
        src_sta_pred = interp_source(src_sta_pred)
        trg_sta_pred = interp_target(trg_sta_pred)
        d_out_sta = d_sta_main(F.softmax(trg_sta_pred))
        loss_sta = bce_loss(d_out_sta, source_label)
        if cfg.TRAIN.MULTI_LEVEL:
            src_sta_pred_aux = torch.cat((src_pred_cf_aux, src_pred_kf_aux), dim=1)
            trg_sta_pred_aux = torch.cat((trg_pred_cf_aux, trg_pred_kf_aux), dim=1)
            src_sta_pred_aux = interp_source(src_sta_pred_aux)
            trg_sta_pred_aux = interp_target(trg_sta_pred_aux)
            d_out_sta_aux = d_sta_aux(F.softmax(trg_sta_pred_aux))
            loss_sta_aux = bce_loss(d_out_sta_aux, source_label)
        else:
            loss_sta_aux = 0
        loss = loss + (cfg.TRAIN.lamda_u * loss_sta
                + cfg.TRAIN.lamda_u * adversarial_factor_aux * loss_sta_aux)
        # spatial alignment (sa)
        src_sa_pred = torch.cat((src_pred_cf, src_pred_cf), dim=1)
        trg_sa_pred = torch.cat((trg_pred_cf, trg_pred_cf), dim=1)
        src_sa_pred = interp_source(src_sa_pred)
        trg_sa_pred = interp_target(trg_sa_pred)
        d_out_sa = d_sa_main(F.softmax(trg_sa_pred))
        loss_sa = bce_loss(d_out_sa, source_label)
        if cfg.TRAIN.MULTI_LEVEL:
            src_sa_pred_aux = torch.cat((src_pred_cf_aux, src_pred_cf_aux), dim=1)
            trg_sa_pred_aux = torch.cat((trg_pred_cf_aux, trg_pred_cf_aux), dim=1)
            src_sa_pred_aux = interp_source(src_sa_pred_aux)
            trg_sa_pred_aux = interp_target(trg_sa_pred_aux)
            d_out_sa_aux = d_sa_aux(F.softmax(trg_sa_pred_aux))
            loss_sa_aux = bce_loss(d_out_sa_aux, source_label)
        else:
            loss_sa_aux = 0
        loss = loss + (cfg.TRAIN.lamda_u * cfg.TRAIN.lamda_sa * loss_sa
                + cfg.TRAIN.lamda_u * cfg.TRAIN.lamda_sa * adversarial_factor_aux * loss_sa_aux)
        loss.backward()
        ### Train discriminator networks (Enable training mode on discriminator networks)
        for param in d_sta_aux.parameters():
            param.requires_grad = True
        for param in d_sta_main.parameters():
            param.requires_grad = True
        for param in d_sa_aux.parameters():
            param.requires_grad = True
        for param in d_sa_main.parameters():
            param.requires_grad = True
        ## Train with source
        # spatial-temporal alignment (sta)
        src_sta_pred = src_sta_pred.detach()
        d_out_sta = d_sta_main(F.softmax(src_sta_pred))
        loss_d_sta = bce_loss(d_out_sta, source_label) / 2
        loss_d_sta.backward()
        if cfg.TRAIN.MULTI_LEVEL:
            src_sta_pred_aux = src_sta_pred_aux.detach()
            d_out_sta_aux = d_sta_aux(F.softmax(src_sta_pred_aux))
            loss_d_sta_aux = bce_loss(d_out_sta_aux, source_label) / 2
            loss_d_sta_aux.backward()
        # spatial alignment (sa)
        src_sa_pred = src_sa_pred.detach()
        d_out_sa = d_sa_main(F.softmax(src_sa_pred))
        loss_d_sa = bce_loss(d_out_sa, source_label) / 2
        loss_d_sa.backward()
        if cfg.TRAIN.MULTI_LEVEL:
            src_sa_pred_aux = src_sa_pred_aux.detach()
            d_out_sa_aux = d_sa_aux(F.softmax(src_sa_pred_aux))
            loss_d_sa_aux = bce_loss(d_out_sa_aux, source_label) / 2
            loss_d_sa_aux.backward()
        ## Train with target
        # spatial-temporal alignment (sta)
        trg_sta_pred = trg_sta_pred.detach()
        d_out_sta = d_sta_main(F.softmax(trg_sta_pred))
        loss_d_sta = bce_loss(d_out_sta, target_label) / 2
        loss_d_sta.backward()
        if cfg.TRAIN.MULTI_LEVEL:
            trg_sta_pred_aux = trg_sta_pred_aux.detach()
            d_out_sta_aux = d_sta_aux(F.softmax(trg_sta_pred_aux))
            loss_d_sta_aux = bce_loss(d_out_sta_aux, target_label) / 2
            loss_d_sta_aux.backward()
        else:
            loss_d_sta_aux = 0
        # spatial alignment (sa)
        trg_sa_pred = trg_sa_pred.detach()
        d_out_sa = d_sa_main(F.softmax(trg_sa_pred))
        loss_d_sa = bce_loss(d_out_sa, target_label) / 2
        loss_d_sa.backward()
        if cfg.TRAIN.MULTI_LEVEL:
            trg_sa_pred_aux = trg_sa_pred_aux.detach()
            d_out_sa_aux = d_sa_aux(F.softmax(trg_sa_pred_aux))
            loss_d_sa_aux = bce_loss(d_out_sa_aux, target_label) / 2
            loss_d_sa_aux.backward()
        else:
            loss_d_sa_aux = 0

        # Discriminators' weights discrepancy (wd)
        k = 0
        loss_wd = 0
        for (W1, W2) in zip(d_sta_main.parameters(), d_sa_main.parameters()):
            W1 = W1.view(-1)
            W2 = W2.view(-1)
            loss_wd = loss_wd + (torch.matmul(W1, W2) / (torch.norm(W1) * torch.norm(W2)) + 1)
            k += 1
        loss_wd = loss_wd / k
        if cfg.TRAIN.MULTI_LEVEL:
            k = 0
            loss_wd_aux = 0
            for (W1_aux, W2_aux) in zip(d_sta_aux.parameters(), d_sa_aux.parameters()):
                W1_aux = W1_aux.view(-1)
                W2_aux = W2_aux.view(-1)
                loss_wd_aux = loss_wd_aux + (torch.matmul(W1_aux, W2_aux) / (torch.norm(W1_aux) * torch.norm(W2_aux)) + 1)
                k += 1
            loss_wd_aux = loss_wd_aux / k
        else:
            loss_wd_aux = 0
        loss = (cfg.TRAIN.lamda_u * cfg.TRAIN.lamda_wd * loss_wd + cfg.TRAIN.lamda_u * cfg.TRAIN.lamda_wd * loss_wd_aux)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(d_sta_aux.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(d_sta_main.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(d_sa_aux.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(d_sa_main.parameters(), 1)
        optimizer.step()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_sta_aux.step()
            optimizer_d_sa_aux.step()
        optimizer_d_sta_main.step()
        optimizer_d_sa_main.step()
        current_losses = {'loss_src_aux': loss_seg_src_aux,
                          'loss_src': loss_seg_src_main,
                          'loss_itcr_aux': loss_itcr_aux,
                          'loss_itcr': loss_itcr,
                          'loss_sta_aux': loss_sta_aux,
                          'loss_sa_aux': loss_sa_aux,
                          'loss_sta': loss_sta,
                          'loss_sa': loss_sa,
                          'loss_d_sta_aux': loss_d_sta_aux,
                          'loss_d_sa_aux': loss_d_sa_aux,
                          'loss_d_sta': loss_d_sta,
                          'loss_d_sa': loss_d_sa,
                          'loss_wd_aux': loss_wd_aux,
                          'loss_wd': loss_wd}
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

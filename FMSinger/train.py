import os
import sys
import argparse
import math
import time
import logging

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from dataset_korean import DistributedBucketSampler, TextAudioCollate, TextAudioLoader

sys.path.append('../..')
import modules.commons as commons
import utils.utils as utils

from models import (
  SynthesizerTrn,
  Discriminator
)

from modules.losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss,
)
from preprocess.mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch

torch.backends.cudnn.benchmark = True
global_step = 0
use_cuda = torch.cuda.is_available()


numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

def main():
  """Assume Single Node Multi GPUs Training Only"""

  hps = utils.get_hparams()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(hps.train.port)

  if(torch.cuda.is_available()):
    n_gpus = torch.cuda.device_count()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
  else:
    cpurun(0, 1, hps)
    
def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.train.save_dir)
    logger.info(hps.train)
    logger.info(hps.data)
    logger.info(hps.model)
    utils.check_git_hash(hps.train.save_dir)
    writer = SummaryWriter(log_dir=hps.train.save_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.train.save_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)


  train_dataset = TextAudioLoader(
    hps.data.fpath_train,
    sampling_rate=hps.data.sample_rate,
    max_wav_value=hps.data.max_wav_value,
    filter_length=hps.data.win_size,
    win_length=hps.data.win_size,
    hop_length=hps.data.hop_size,
    num_mels=hps.data.acoustic_dim,
    fmin=hps.data.fmin,
    fmax=hps.data.fmax,
    min_text_len=hps.data.min_text_len,
    max_text_len=hps.data.max_text_len,
    spk_dict = hps.speaker_elf,
    )

  train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [300, 400, 500, 600, 700, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
      )
  collate_fn = TextAudioCollate()

  train_loader = DataLoader(
        train_dataset,
        num_workers=8,          
        prefetch_factor=2,      
        persistent_workers=True,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )
  
  eval_dataset = None
  valid_loader  = None
  if rank == 0:
      eval_dataset = TextAudioLoader(
            hps.data.fpath_eval,
            sampling_rate=hps.data.sample_rate,
            max_wav_value=hps.data.max_wav_value,
            filter_length=hps.data.win_size,
            win_length=hps.data.win_size,
            hop_length=hps.data.hop_size,
            num_mels=hps.data.acoustic_dim,
            fmin=hps.data.fmin,
            fmax=hps.data.fmax,
            min_text_len=hps.data.min_text_len,
            max_text_len=hps.data.max_text_len,
            spk_dict = hps.speaker_elf,
      )
      valid_loader = DataLoader(
            eval_dataset,
            num_workers=8,
            shuffle=False,
            batch_size=hps.train.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
      )

  net_g = SynthesizerTrn(hps).cuda(rank)
  net_d = Discriminator(hps, hps.model.use_spectral_norm).cuda(rank)
  
  optim_g = torch.optim.AdamW(
      net_g.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
  net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
  
  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.train.save_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.train.save_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], [train_loader, valid_loader], logger, [writer, writer_eval],train_sampler)
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], [train_loader, None], None, None,train_sampler)
    scheduler_g.step()
    scheduler_d.step()

def cpurun(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.train.save_dir)
    logger.info(hps.train)
    logger.info(hps.data)
    logger.info(hps.model)
    utils.check_git_hash(hps.train.save_dir)
    writer = SummaryWriter(log_dir=hps.train.save_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.train.save_dir, "eval"))
  torch.manual_seed(hps.train.seed)
  train_dataset = TextAudioLoader(
    hps.data.fpath_train,
    sampling_rate=hps.data.sample_rate,
    max_wav_value=hps.data.max_wav_value,
    filter_length=hps.data.win_size,
    win_length=hps.data.win_size,
    hop_length=hps.data.hop_size,
    num_mels=hps.data.acoustic_dim,
    fmin=hps.data.fmin,
    fmax=hps.data.fmax,
    min_text_len=hps.data.min_text_len,
    max_text_len=hps.data.max_text_len,
    spk_dict = hps.speaker_elf,
    )

  train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [300, 400, 500, 600, 700, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
      )
  collate_fn = TextAudioCollate()

  train_loader = DataLoader(
        train_dataset,
        num_workers=8,          
        prefetch_factor=2,      
        persistent_workers=True,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )
  
  eval_dataset = None
  valid_loader  = None
  if rank == 0:
      eval_dataset = TextAudioLoader(
            hps.data.fpath_eval,
            sampling_rate=hps.data.sample_rate,
            max_wav_value=hps.data.max_wav_value,
            filter_length=hps.data.win_size,
            win_length=hps.data.win_size,
            hop_length=hps.data.hop_size,
            num_mels=hps.data.acoustic_dim,
            fmin=hps.data.fmin,
            fmax=hps.data.fmax,
            min_text_len=hps.data.min_text_len,
            max_text_len=hps.data.max_text_len,
            spk_dict = hps.speaker_elf,
      )
      valid_loader = DataLoader(
            eval_dataset,
            num_workers=8,
            shuffle=False,
            batch_size=hps.train.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
      )


  net_g = SynthesizerTrn(hps)
  net_d = Discriminator(hps, hps.model.use_spectral_norm)
  
  optim_g = torch.optim.AdamW(
      net_g.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.train.save_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.train.save_dir, "D_*.pth"), net_g, optim_g)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], [train_loader, valid_loader], logger, [writer, writer_eval],train_sampler)
    
    scheduler_g.step()
    scheduler_d.step()

def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, loaders, logger, writers, sampler):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers
  sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()
  scaler_g = GradScaler() if hps.train.fp16_run else None
  scaler_d = GradScaler() if hps.train.fp16_run else None
  
  accumulation_steps = hps.train.accumulation_steps
  
  for batch_idx, data_dict in enumerate(train_loader):
    x, x_lengths, spec, spec_lengths,\
    wav, wav_lengths, gt_f0, _, smoothed_f0, f0_lengths,\
    note, note_lengths, note_dur, note_dur_lengths,\
    note_boundary_start, note_boundary_end, note_boundary_flag, note_boundary_cnt, note_boundary_lengths, \
    note_dur_input, note_dur_input_lengths, vq, vq_lengths, genre_id = data_dict 
    
    if(use_cuda):
        x                = x.cuda(rank, non_blocking=True)
        x_lengths        = x_lengths.cuda(rank, non_blocking=True)
        spec             = spec.cuda(rank, non_blocking=True)
        spec_lengths     = spec_lengths.cuda(rank, non_blocking=True)
        wav                = wav.cuda(rank, non_blocking=True)
        wav_lengths        = wav_lengths.cuda(rank, non_blocking=True)
        gt_f0            = gt_f0.cuda(rank, non_blocking=True)
        smoothed_f0      = smoothed_f0.cuda(rank, non_blocking=True)
        f0_lengths       = f0_lengths.cuda(rank, non_blocking=True)
        note             = note.cuda(rank, non_blocking=True)
        note_lengths     = note_lengths.cuda(rank, non_blocking=True)
        note_dur         = note_dur.cuda(rank, non_blocking=True)
        note_dur_lengths = note_dur_lengths.cuda(rank, non_blocking=True)
        note_boundary_start = note_boundary_start.cuda(rank, non_blocking=True)
        note_boundary_end = note_boundary_end.cuda(rank, non_blocking=True)
        note_boundary_flag = note_boundary_flag.cuda(rank, non_blocking=True)
        note_boundary_cnt =  note_boundary_cnt.cuda(rank, non_blocking=True)
        note_boundary_lengths = note_boundary_lengths.cuda(rank, non_blocking=True)
        note_dur_input = note_dur_input.cuda(rank, non_blocking=True)
        note_dur_input_lengths = note_dur_input_lengths.cuda(rank, non_blocking=True)
        vq               = vq.cuda(rank, non_blocking=True)
        vq_lengths       = vq_lengths.cuda(rank, non_blocking=True)
        genre_id           = genre_id.cuda(rank, non_blocking=True)



    has_note_boundary = (note_boundary_start is not None and note_boundary_end is not None)
    
    if hps.train.fp16_run and scaler_g is not None:
        with autocast():
            if has_note_boundary:
                y_hat, ids_slice, gt_dur, predict_lf0, LF0, y_ddsp, kl_div, predict_mel, mask, phoneme_duration_pred, phoneme_duration_target, note_duration_target, loss_cnf, x_mask, kl_div_mas, loss_fm_mas = net_g(
                    x, x_lengths, note, note_dur_input, note_dur, gt_f0, spec, spec_lengths, vq,
                    note_boundary_start, note_boundary_end, note_boundary_flag, note_boundary_cnt, note_boundary_lengths
                )
            else:
                y_hat, ids_slice, gt_dur, predict_lf0, LF0, y_ddsp, kl_div, predict_mel, mask, phoneme_duration_pred, phoneme_duration_target, note_duration_target, loss_cnf, x_mask = net_g(
                    x, x_lengths, note, note_dur_input, note_dur, gt_f0, spec, spec_lengths, vq,
                    note_boundary_start, note_boundary_end, note_boundary_flag, note_boundary_cnt, note_boundary_lengths
                )
                kl_div_mas = 0.0  
                loss_fm_mas = 0.0 
    else:
        if has_note_boundary:
            y_hat, ids_slice, gt_dur, predict_lf0, LF0, y_ddsp, kl_div, predict_mel, mask, phoneme_duration_pred, phoneme_duration_target, note_duration_target, loss_cnf, x_mask, kl_div_mas, loss_fm_mas = net_g(
                x, x_lengths, note, note_dur_input, note_dur, gt_f0, spec, spec_lengths, vq,
                note_boundary_start, note_boundary_end, note_boundary_flag, note_boundary_cnt, note_boundary_lengths
            )
        else:
            y_hat, ids_slice, gt_dur, predict_lf0, LF0, y_ddsp, kl_div, predict_mel, mask, phoneme_duration_pred, phoneme_duration_target, note_duration_target, loss_cnf, x_mask = net_g(
                x, x_lengths, note, note_dur_input, note_dur, gt_f0, spec, spec_lengths, vq,
                note_boundary_start, note_boundary_end, note_boundary_flag, note_boundary_cnt, note_boundary_lengths
            )
            kl_div_mas = 0.0  
            loss_fm_mas = 0.0  
    
    y_ddsp = y_ddsp.unsqueeze(1)


    # Discriminator
    y = commons.slice_segments(wav, ids_slice * hps.data.hop_size, hps.train.segment_size) # slice 
    y_ddsp_mel = mel_spectrogram_torch(
          y_ddsp.squeeze(1), 
          hps.data.n_fft, 
          hps.data.acoustic_dim, 
          hps.data.sample_rate, 
          hps.data.hop_size, 
          hps.data.win_size, 
          hps.data.fmin, 
          hps.data.fmax
      )

    y_logspec = torch.log(spectrogram_torch(
          y.squeeze(1),
          hps.data.n_fft,
          hps.data.sample_rate,
          hps.data.hop_size,
          hps.data.win_size
    ) + 1e-7)

    y_ddsp_logspec = torch.log(spectrogram_torch(
          y_ddsp.squeeze(1),
          hps.data.n_fft,
          hps.data.sample_rate,
          hps.data.hop_size,
          hps.data.win_size
    ) + 1e-7)

    y_mel = mel_spectrogram_torch(
          y.squeeze(1), 
          hps.data.n_fft, 
          hps.data.acoustic_dim, 
          hps.data.sample_rate, 
          hps.data.hop_size, 
          hps.data.win_size, 
          hps.data.fmin, 
          hps.data.fmax
      )
    y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.n_fft, 
          hps.data.acoustic_dim, 
          hps.data.sample_rate, 
          hps.data.hop_size, 
          hps.data.win_size, 
          hps.data.fmin, 
          hps.data.fmax
      )

    # Discriminator forward
    if hps.train.fp16_run and scaler_d is not None:
        with autocast():
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc
    else:
        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc

    optim_d.zero_grad()
    if hps.train.fp16_run and scaler_d is not None:
        scaler_d.scale(loss_disc_all).backward()
        scaler_d.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler_d.step(optim_d)
        scaler_d.update()
    else:
        loss_disc_all.backward()
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        optim_d.step()

    # Generator loss
    if hps.train.fp16_run and scaler_d is not None:
        with autocast():
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
    else:
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
    
    loss_f0 = F.mse_loss(predict_lf0 * mask, LF0 * mask) * 10
    loss_mel = F.l1_loss(y_mel, y_hat_mel) * 45
    loss_mel_dsp = F.l1_loss(y_mel, y_ddsp_mel) * 45
    loss_spec_dsp = F.l1_loss(y_logspec, y_ddsp_logspec) * 45
 
   
    # Duration Loss 
    if phoneme_duration_target is not None and note_duration_target is not None and phoneme_duration_pred is not None:

        text_mask = x_mask.squeeze(1)  
        
        loss_dur = F.mse_loss(phoneme_duration_pred[:, 0, :], phoneme_duration_target) * 0.1
        
        loss_note_dur = F.mse_loss(phoneme_duration_pred[:, 1, :], note_dur_input) * 0.1
    else:
        loss_dur = torch.tensor(0.0, device=y_hat.device)
        loss_note_dur = torch.tensor(0.0, device=y_hat.device)

    loss_mel_am = F.mse_loss(spec * mask, predict_mel * mask) #* 10

    loss_fm = feature_loss(fmap_r, fmap_g)
    loss_gen, losses_gen = generator_loss(y_d_hat_g)

    loss_fm = loss_fm / 9 * 6
    loss_gen = loss_gen / 9 * 6
    
    cnf_weight = 1.0  
    loss_cnf_weighted = loss_cnf * cnf_weight
    
    # Main Flow + CFM Loss = Total Generator Loss
    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_f0 + loss_mel_dsp + loss_dur + kl_div + kl_div_mas + loss_mel_am + loss_spec_dsp + loss_cnf_weighted

    loss_gen_all = loss_gen_all / hps.train.accumulation_steps
    
    # Generator Backward with mixed precision
    if hps.train.fp16_run and scaler_g is not None:
        scaler_g.scale(loss_gen_all).backward()
        if((global_step+1) % hps.train.accumulation_steps == 0):
            scaler_g.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler_g.step(optim_g)
            scaler_g.update()
            optim_g.zero_grad()
    else:
        loss_gen_all.backward()
        if((global_step+1) % hps.train.accumulation_steps == 0):
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            optim_g.step()
            optim_g.zero_grad()
    
    if rank==0:
      if (global_step+1) % (hps.train.accumulation_steps * 10) == 0:
        logger.info(["step&time", global_step, time.asctime( time.localtime(time.time()) )])
        logger.info(["mel&mel_dsp&spec_dsp: " ,loss_mel, loss_mel_dsp, loss_spec_dsp])
        logger.info(["f0: " ,loss_f0])
        logger.info(["adv&fm: " ,loss_gen, loss_fm])
        logger.info(["kl: " ,kl_div])
        logger.info(["kl_mas: " ,kl_div_mas])
        logger.info(["cnf_raw: " , loss_cnf])      
        logger.info(["cnf_weighted: " , loss_cnf_weighted])  
        logger.info(["cnf_weight: " , cnf_weight])  
        logger.info(["am&dur: " , loss_mel_am, loss_dur, loss_note_dur])

        
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_gen_all, loss_mel, loss_f0]
        
        if phoneme_duration_pred is not None:
            dur_pred_max = phoneme_duration_pred.abs().max().item()
            if dur_pred_max > 10:  # 임계값 설정
                logger.warning(f"[DURATION WARNING FM] Step {global_step}: Duration predictor 출력이 큼! Max: {dur_pred_max:.6f}")
        
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        
        scalar_dict = {"loss/total": loss_gen_all, 
                       "loss/mel": loss_mel,
                       "loss/adv": loss_gen,
                       "loss/fm": loss_fm,
                       "loss/mel_ddsp": loss_mel_dsp,
                       "loss/spec_ddsp": loss_spec_dsp,
                       "loss/dur": loss_dur,
                       "loss/note_dur": loss_note_dur,
                       "loss/mel_am": loss_mel_am,
                       "loss/kl_div": kl_div,
                       "loss/kl_div_mas": kl_div_mas,
                       "loss/f0": loss_f0,
                       "learning_rate": lr}
        
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        logger.info(['All training params(G): ', utils.count_parameters(net_g), ' M'])
        
        evaluate(hps, net_g, eval_loader, writer_eval)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.train.save_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.train.save_dir, "D_{}.pth".format(global_step)))
        net_g.train()
    global_step += 1
    
    if global_step % 100 == 0:
        torch.cuda.empty_cache()
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
      for batch_idx, data_dict in enumerate(eval_loader):
        x, x_lengths, spec, spec_lengths,\
        wav, wav_lengths, gt_f0, _, smoothed_f0, f0_lengths,\
        note, note_lengths, note_dur, note_dur_lengths,\
        note_boundary_start, note_boundary_end, note_boundary_flag, note_boundary_cnt, note_boundary_lengths, \
        note_dur_input, note_dur_input_lengths, vq, vq_lengths, genre_id = data_dict

        if(use_cuda):
            x                = x.cuda(0, non_blocking=True)
            x_lengths        = x_lengths.cuda(0, non_blocking=True)
            wav                = wav.cuda(0, non_blocking=True)
            note             = note.cuda(0, non_blocking=True)
            note_dur_input   = note_dur_input.cuda(0, non_blocking=True)
            vq               = vq.cuda(0, non_blocking=True)
            vq_lengths       = vq_lengths.cuda(0, non_blocking=True)
            note_boundary_start = note_boundary_start.cuda(0, non_blocking=True)
            note_boundary_end = note_boundary_end.cuda(0, non_blocking=True)
            note_boundary_flag = note_boundary_flag.cuda(0, non_blocking=True)
            note_boundary_cnt = note_boundary_cnt.cuda(0, non_blocking=True)
            note_boundary_lengths = note_boundary_lengths.cuda(0, non_blocking=True)
            # genre_id           = genre_id.cuda(0, non_blocking=True)

        # remove else
        x                       = x[:1]
        x_lengths               = x_lengths[:1]
        spec                    = spec[:1]
        spec_lengths            = spec_lengths[:1]                                        
        wav                       = wav[:1]
        note                    = note[:1]
        note_dur_input          = note_dur_input[:1]
        vq                      = vq[:1]
        note_boundary_start     = note_boundary_start[:1]
        note_boundary_end       = note_boundary_end[:1]
        note_boundary_flag      = note_boundary_flag[:1]
        note_boundary_cnt       = note_boundary_cnt[:1]
        note_boundary_lengths   = note_boundary_lengths[:1]
        # genre_id                = genre_id[:1]
        break
      
      # Inference
      model = generator.module if hasattr(generator, 'module') else generator
      y_hat, y_harm, y_noise, flow_z_l,_ = model.infer(x, x_lengths, note, note_dur_input, vq)
      spec = spectrogram_torch(
            wav.squeeze(1), 
            hps.data.n_fft, 
            hps.data.sample_rate, 
            hps.data.hop_size, 
            hps.data.win_size
        )

      y_mel = mel_spectrogram_torch(
          wav.squeeze(1), 
          hps.data.n_fft, 
          hps.data.acoustic_dim, 
          hps.data.sample_rate, 
          hps.data.hop_size, 
          hps.data.win_size, 
          hps.data.fmin, 
          hps.data.fmax
      )
    y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.n_fft, 
          hps.data.acoustic_dim, 
          hps.data.sample_rate, 
          hps.data.hop_size, 
          hps.data.win_size, 
          hps.data.fmin, 
          hps.data.fmax
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:],
      "gen/harm": y_harm[0,:,:],
      "gen/noise": y_noise[0,:,:]
    }
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(spec[0].cpu().numpy())})
      audio_dict.update({"gt/audio": wav[0,:,:wav_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sample_rate
    )
    generator.train()

if __name__ == "__main__":
  main()

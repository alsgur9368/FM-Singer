#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FM-Singer Inference Script
Core code for generating waveforms from checkpoints
"""

import sys
import os
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
import soundfile as sf
from torch.utils.data import DataLoader

sys.path.append('../..')
import modules.commons as commons
import utils.utils as utils
from models import SynthesizerTrn
from dataset_korean import TextAudioLoader, TextAudioCollate


class HParams:
    """HParams class for configuration"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, HParams(**v))
            else:
                setattr(self, k, v)


def load_config(config_path):
    """Load config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def load_model(checkpoint_path, config_path):
    """Load model from checkpoint."""
    print(f"⚙️  Loading config: {config_path}")
    config = load_config(config_path)
    hps = HParams(**config)
    
    print(f"🏗️  Creating model...")
    model = SynthesizerTrn(hps)
    
    if os.path.isdir(checkpoint_path):
        checkpoint_path = utils.latest_checkpoint_path(checkpoint_path)
        if checkpoint_path is None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"🔍 Latest checkpoint: {checkpoint_path}")
    
    print(f"📥 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    return model, hps


def inference_from_dataset(model, hps, dataset_path, output_dir, device='cpu', max_samples=10):
    """Perform inference from dataset."""
    print(f"📁 Loading dataset: {dataset_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = TextAudioLoader(
        dataset_path,
        max_wav_value=hps.data.max_wav_value,
        sampling_rate=hps.data.sample_rate,
        filter_length=hps.data.n_fft,
        win_length=hps.data.win_size,
        hop_length=hps.data.hop_size,
        num_mels=hps.data.acoustic_dim,
        fmin=hps.data.fmin,
        fmax=hps.data.fmax if hps.data.fmax is not None else 8000,
        min_text_len=hps.data.min_text_len,
        max_text_len=hps.data.max_text_len,
        spk_dict=dict(getattr(hps, 'speaker_elf', {}).__dict__) if hasattr(getattr(hps, 'speaker_elf', {}), '__dict__') else {}
    )
    
    print(f"📊 Dataset size: {len(dataset)}")
    print(f"🎯 Max samples: {max_samples}")
    
    collate_fn = TextAudioCollate()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    model = model.to(device)
    
    for idx, batch in enumerate(tqdm(dataloader, desc="🎵 Inference")):
        if idx >= max_samples:
            break
        
        (x, x_lengths, spec, spec_lengths,
         wav, wav_lengths, gt_f0, f0_lengths, smoothed_f0, f0_lengths_2,
         note, note_lengths, note_dur, note_dur_lengths,
         note_boundary_start, note_boundary_end, note_boundary_flag, note_boundary_cnt, note_boundary_lengths,
         note_dur_input, note_dur_input_lengths, vq, vq_lengths, genre_id) = batch
        
        x = x.to(device)
        x_lengths = x_lengths.to(device)
        note = note.to(device)
        note_dur_input = note_dur_input.to(device)
        vq = vq.to(device)
        
        with torch.no_grad():
            audio, _, _, _, _ = model.infer(x, x_lengths, note, note_dur_input, vq)
        
        audio_np = audio[0, 0].cpu().numpy()
        audio_np = audio_np * hps.data.max_wav_value
        audio_np = audio_np.astype(np.int16)
        
        output_path = os.path.join(output_dir, f"sample_{idx+1:03d}.wav")
        sf.write(output_path, audio_np, hps.data.sample_rate)
    
    print(f"✅ Inference completed! Generated {min(idx+1, max_samples)} files")
    print(f"📁 Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='FM-Singer Inference')
    parser.add_argument('--checkpoint', type=str, 
                       default='/data/SVS/FM-Singer/chkp_FMSinger',
                       help='Checkpoint file or directory path')
    parser.add_argument('--config', type=str, 
                       default='/data/SVS/FM-Singer/egs/FMSinger/config.json',
                       help='Config file path')
    parser.add_argument('--output_dir', type=str, 
                       default='./outputs/1.A.fm_singer',
                       help='Output directory')
    parser.add_argument('--dataset_path', type=str,
                       #default='/data/SVS/KDG_Code/egs/train_period_singer/filelists/test_eval.txt',
                       default='/data/SVS/FM-Singer/filelist/test_eval.txt',
                       help='Dataset file path')
    parser.add_argument('--max_samples', type=int, default=1, help='Maximum number of samples')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    model, hps = load_model(args.checkpoint, args.config)
    print(f"✅ Model loaded successfully")
    
    print(f"📁 Dataset: {args.dataset_path}")
    inference_from_dataset(model, hps, args.dataset_path, args.output_dir, device, args.max_samples)


if __name__ == '__main__':
    main()


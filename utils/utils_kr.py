#! -*- coding:utf-8 -*-
#
import numpy as np
from scipy.io.wavfile import read
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import torch.nn.functional as F
import hgtk

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_state_dict(model, model_name, ckpt, load_only_params=None):
    saved_state_dict = ckpt[model_name]
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
  
    new_state_dict= {}
    if load_only_params is None:
        for k, v in state_dict.items():
            if k in saved_state_dict:
                new_state_dict[k] = saved_state_dict[k]
            else:
                new_state_dict[k] = v
    else:
        target_params = None
        if model_name in load_only_params:
            target_params = load_only_params[model_name]
        if target_params is None:
            new_state_dict = state_dict
        else:
            for k, v in state_dict.items():
                if k in target_params:
                    new_state_dict[k] = saved_state_dict[k]
                else:
                    new_state_dict[k] = v

    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)


def load_filepaths_and_text(fpath, spk_dict, delim="|"):
    dataset = []
    
    with open(fpath, 'r') as fp:
        for line in fp:
            line = line.strip()
            if line == '':
                continue
            if line.startswith('#'):
                continue
                        
            # wav_fname(str)|symbol ids(ints)|notes(ints)|notes durations(ints)|speaker_vq_fname(str)|genre_ids(int)
            token = line.split(delim)
            
            spk_id = token[4]
            spk_id = spk_dict[spk_id]
            
            if len(token) > 0:
                dataset.append([token[0], token[1], token[2], token[3], spk_id, token[5]])
                
    return dataset



def load_checkpoint(checkpoint_fpath, net_g, net_d, net_dur_d, optim_g=None, optim_d=None, optim_dur_d=None, scheduler_g=None, scheduler_d=None, scheduler_dur_d=None, warm_start=False, load_only_params=None):
    ckpt = torch.load(checkpoint_fpath, map_location="cpu")
    
    iteration = 0
    epoch     = 0
    
    if not warm_start:
        iteration = ckpt['iteration']
        epoch = ckpt['epoch']
        if optim_g is not None: optim_g.load_state_dict(ckpt['optim_g'])
        if optim_d is not None: optim_d.load_state_dict(ckpt['optim_d'])
        if optim_dur_d is not None: optim_dur_d.load_state_dict(ckpt['optim_dur_d'])
        if scheduler_g is not None: scheduler_g.load_state_dict(ckpt['scheduler_g'])
        if scheduler_d is not None: scheduler_d.load_state_dict(ckpt['scheduler_d'])
        if scheduler_dur_d is not None: scheduler_dur_d.load_state_dict(ckpt['scheduler_dur_d'])

    load_state_dict(net_g,     'net_g',     ckpt, load_only_params=load_only_params)
    load_state_dict(net_d,     'net_d',     ckpt, load_only_params=load_only_params)
    load_state_dict(net_dur_d, 'net_dur_d', ckpt, load_only_params=load_only_params)

    return epoch, iteration


def save_checkpoint(net_g, net_d, net_dur_d, optim_g, optim_d, optim_dur_d, scheduler_g, scheduler_d, scheduler_dur_d, epoch, iteration, checkpoint_path):    
    ckpt = {}
    
    ckpt['net_g'] = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
    ckpt['net_d'] = net_d.module.state_dict() if hasattr(net_d, "module") else net_d.state_dict()
    ckpt['net_dur_d'] = net_dur_d.module.state_dict() if hasattr(net_dur_d, "module") else net_dur_d.state_dict()
    
    ckpt['optim_g'] = optim_g.state_dict()
    ckpt['optim_d'] = optim_d.state_dict()
    ckpt['optim_dur_d'] = optim_dur_d.state_dict()
    
    ckpt['scheduler_g'] = scheduler_g.state_dict()
    ckpt['scheduler_d'] = scheduler_d.state_dict()
    ckpt['scheduler_dur_d'] = scheduler_dur_d.state_dict()
    
    ckpt['epoch'] = epoch
    ckpt['iteration'] = iteration
    
    torch.save(ckpt, checkpoint_path)


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def adjust_note_durations(note_durations):
    adjusted_durations = []   
    odd_cnt = 0 
    
    for i in range(0, len(note_durations), 3):
        inital, mid, final = note_durations[i:i+3]
        if mid % 2 == 1:
            odd_cnt += 1
            if odd_cnt == 2:
                inital, final = 2, 2
                mid = mid // 2
                odd_cnt = 0
            else:
                inital, final = 2, 1
                mid = mid // 2
        else:
            inital, final = 2, 1
            mid = mid // 2
        adjusted_durations.extend([inital, mid, final])       
    return adjusted_durations

def create_note_boundaries(note, note_dur):    
    note_boundaries = []
    start = 0
    end = 0
    note_cnt = 0

    for i in range(0, len(note_dur), 3):
        initial, middle, final = note_dur[i], note_dur[i + 1], note_dur[i + 2]
        duration = initial + middle + final
        note_cnt += 3  # 3개 단위로 note_cnt 증가
        
        # 다음 음절이 현재 음절과 같다면 end만 갱신
        if i + 3 != len(note_dur) and note[i + 2] == note[i + 3]:
            end += duration
        else:
            # note_cnt가 3을 넘으면 단일 음소가 아님
            single_phoneme_flags = (note_cnt == 3)
            end += duration

            # 현재 음절에 대한 시작, 종료, 플래그 값, 요소 개수 추가
            note_boundaries.append((start, end, single_phoneme_flags, note_cnt))

            # 초기화
            start = end
            note_cnt = 0

    return note_boundaries


def create_alignment_matrix(durations, frame_len):
    # durations를 수정하지 않고, 복사본을 생성하여 사용
    durations = durations.clone()

    num_phonemes = len(durations)  # 열의 길이 (음소 개수)
    
    if frame_len != sum(durations):
        diff = frame_len - sum(durations)
        if len(durations) > 1:
            durations[-2] = durations[-2] + diff
        else:
            durations[0] = durations[0] + diff
    
    # 행렬 초기화 (0으로 채움) - [frame_len, num_phonemes] -> [1, frame_len, num_phonemes]
    alignment_matrix = torch.zeros(1, frame_len, num_phonemes, dtype=torch.int)

    # 각 음소에 해당하는 영역을 1로 채우기
    start = 0
    for idx, duration in enumerate(durations):
        # duration을 정수로 변환
        duration_int = int(duration.item())
        end = start + duration_int
        alignment_matrix[0, start:end, idx] = 1  # 해당 음소의 범위를 1로 채움
        start = end  # 다음 음소의 시작 위치로 업데이트

    return alignment_matrix


def log_scale_f0(f0, C=1, clip_val=1e-5):
    """
    f0 값을 log 스케일로 변환하여 log mel spectrogram과 유사한 스케일로 맞춥니다.

    Parameters:
    - f0: (Tensor) 변환할 f0 텐서
    - C: (float) 스케일링 상수
    - clip_val: (float) 매우 작은 값으로, 0 이하의 값이 없도록 보장

    Returns:
    - log_f0: (Tensor) log 스케일로 변환된 f0 텐서
    """
    # f0의 최소값을 clip_val로 클램프하여 로그 변환에 문제가 없도록 보장
    log_f0 = torch.log(torch.clamp(f0, min=clip_val) * C + 1e-8)
    return log_f0



def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)


def pad_v2(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded




def replace_zeros_with_neighbors_mean(f0):
    # f0 배열을 복사하여 작업 (원본 수정 방지)
    f0_replaced = f0.copy()
    
    # f0 배열의 길이
    length = len(f0)
    
    for i in range(length):
        if f0_replaced[i] < 100:
            # 앞뒤 값을 사용할 수 있는지 확인
            left = f0_replaced[i - 1] if i > 0 else None
            right = f0_replaced[i + 1] if i < length - 1 else None
            
            # 앞과 뒤 값이 둘 다 존재할 때 평균을 계산하여 대체
            if left is not None and right is not None:
                f0_replaced[i] = (left + right) / 2
            # 왼쪽 값만 존재할 때 왼쪽 값으로 대체
            elif left is not None:
                f0_replaced[i] = left
            # 오른쪽 값만 존재할 때 오른쪽 값으로 대체
            elif right is not None:
                f0_replaced[i] = right

    return f0_replaced
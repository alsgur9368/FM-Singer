#! -*- coding:utf-8 -*-
import os, sys
import random
import time
import pyworld as pw
import numpy as np
import torch
import torch.utils.data
import modules.commons_kr as commons

from scipy.signal import medfilt
from utils.utils_kr import load_filepaths_and_text, load_wav_to_torch, adjust_note_durations, create_note_boundaries, replace_zeros_with_neighbors_mean
from preprocess.mel_processing_kr import mel_spectrogram_torch
from text_korean.korean import text_to_sequence

class TextAudioLoader(torch.utils.data.Dataset):
    """
    data format should be:
        wav_fname(str)|symbol ids(ints)|notes(ints)|notes durations(ints)|speaker_vq_fname(str)|genre_ids(int)
        
    """
    def __init__(
        self, 
        dataset_fpath,
        max_wav_value=32768.0,
        sampling_rate=44100,
        filter_length=2048,
        win_length=2048,
        hop_length=512,
        num_mels=80,
        fmin=0,
        fmax=8000,
        min_text_len=1,
        max_text_len=190,
        delim='|',
        spk_dict = {},
        medfilt_kernel_size = 13,
    ):
        self.audiopaths_and_text = load_filepaths_and_text(dataset_fpath, spk_dict)
        self.max_wav_value = max_wav_value
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.win_length = win_length
        self.hop_length = hop_length
        self.delim = delim
        self.n_mel_channels =num_mels
        self.fmin = fmin
        self.fmax = fmax
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.frame_period = round(hop_length/sampling_rate*1000, 2)
        self.kernel_size = medfilt_kernel_size
                    
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()

    def _filter(self):
        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text, note, noteduration, spk_id, genre in self.audiopaths_and_text: 
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text, note, noteduration, spk_id, genre])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths
        
    def get_audio_text_pair(self, data):
        audiopath, text, note, noteduration, spk_id, genre = data
        wav, mels, gt_f0, smoothed_f0   = self.get_audio(audiopath) 
        text            = self.get_text(text)
        note, note_dur, note_boundary  = self.get_notes(note, noteduration, mels)

        note_dur_input = note_dur.float() * (self.hop_length / self.sampling_rate)
        note_dur_input = torch.round(note_dur_input * 10000) / 10000  

        vq              = spk_id
        genre_id        = self.get_genre_id(genre)
                
        assert mels.size(1) == sum(note_dur), f'Error: mels frame : {mels.size(1)} and note_dur frame : {sum(note_dur)} must have the same length.'          
        assert len(text) == len(note) == len(note_dur), 'Error: text, note, and note_dur must have the same length.'  
                      
        return (text, mels, wav, gt_f0, smoothed_f0, note, note_dur, note_boundary, note_dur_input, vq, genre_id)

    def get_audio(self, filename):
        audio, sr = load_wav_to_torch(filename)
        if sr != self.sampling_rate:
            raise ValueError(f'[ERR] SR mismatch: fname={filename} sr_this={sr} sr_need={self.sampling_rate}')

        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
                    
        mels = mel_spectrogram_torch(
            audio_norm,
            self.filter_length,
            self.n_mel_channels,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            self.fmin,
            self.fmax,
            center=False,
        )
        mels = torch.squeeze(mels, 0)
        
        # extract f0
        wav = audio.numpy().astype(np.float64)
        f0 = np.load(filename.replace('.wav', '_f0.npy'))
        f0_gt = replace_zeros_with_neighbors_mean(f0)
        f0 = medfilt(f0_gt, kernel_size=self.kernel_size)

        f0_smoothed = medfilt(f0_gt, kernel_size=self.kernel_size)
        
        # f0 길이와 mel-spectrogram 길이 맞춤
        if mels.size(1) < len(f0_smoothed):
            f0_smoothed = f0_smoothed[:-1]
            f0 = f0[:-1]
        elif mels.size()[1] > len(f0_smoothed):
            raise ValueError(
                f"Mel-spectrogram의 프레임 수({mels.size(1)})가 F0 시퀀스의 길이({len(f0_smoothed)})보다 큽니다. "
                "입력 오디오와 설정을 확인하세요."
            )
        
        f0_tensor = torch.from_numpy(f0).unsqueeze(0) 
        f0_smoothed_tensor = torch.from_numpy(f0_smoothed).unsqueeze(0)  
        
        return audio_norm, mels, f0_tensor, f0_smoothed_tensor
    
    def get_text(self, text, cleaned_text=False):
        print    
        text_norm = text_to_sequence(text)
        text_norm = torch.LongTensor(text_norm)
        return text_norm
    
    def get_notes(self, note, noteduration, mels):
        note = eval(note)
        note_dur = eval(noteduration)
        
        note_dur = adjust_note_durations(note_dur)
        note_boundaries = create_note_boundaries(note, note_dur)
                   
        note = torch.LongTensor(note)
        noteduration = torch.LongTensor(note_dur)
        
        if mels.size(1) < sum(noteduration):
            if noteduration[-2] - (sum(noteduration) - mels.size(1)) >= 1:
                noteduration[-2] -= (sum(noteduration) - mels.size(1))
            else:
                noteduration[-5] -= (sum(noteduration) - mels.size(1))
        
        return note, noteduration, note_boundaries
    
    def get_vq(self, fname):
        vq = np.load(fname)
        vq_tensor = torch.from_numpy(vq)

        return vq_tensor
    
    def get_genre_id(self, genre_id):
        if genre_id == 'Ballade':
            genre_id = 1
        elif genre_id == 'Rock':
            genre_id = 2
        elif genre_id == 'Trot':
            genre_id = 3
        elif genre_id == 'Unknown':
            genre_id = 4
        
        return torch.LongTensor([genre_id])
    
    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])
    
    def __len__(self):
        return len(self.audiopaths_and_text)

class TextAudioCollate:
    def __init__(self, return_ids=False):
        self.return_ids = return_ids
    
    def __call__(self, batch):
        """
        batch: (text, spec, wav, gt_f0, smoothed_f0, note, note_dur, note_boundary, vq, genre_id)
        """
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )
                              
        max_text_len = max([x[0].size(0) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len  = max([x[2].size(1) for x in batch])
        max_gt_f0_len   = max([x[3].size(1) for x in batch])
        max_smootehd_f0_len   = max([x[4].size(1) for x in batch])
        max_note_len = max([x[5].size(0) for x in batch])
        max_note_dur_len = max([x[6].size(0) for x in batch])
        
        max_note_boundary_len = max([len(x[7]) for x in batch])

        max_note_dur_input_len = max([x[8].size(0) for x in batch])

        text_padded  = torch.LongTensor(len(batch), max_text_len).zero_()
        spec_padded  = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len).zero_()
        wav_padded   = torch.FloatTensor(len(batch), 1, max_wav_len).zero_()
        gt_f0_padded = torch.FloatTensor(len(batch), 1, max_gt_f0_len).zero_()
        smoothed_f0_padded = torch.FloatTensor(len(batch), 1, max_smootehd_f0_len).zero_()
        note_padded  = torch.LongTensor(len(batch), max_note_len).zero_()
        note_dur_padded = torch.LongTensor(len(batch), max_note_dur_len).zero_()
        note_dur_input_padded = torch.FloatTensor(len(batch), max_note_dur_input_len).zero_()
        
        note_boundary_start = torch.LongTensor(len(batch), max_note_boundary_len).zero_()
        note_boundary_end   = torch.LongTensor(len(batch), max_note_boundary_len).zero_()
        note_boundary_flag  = torch.BoolTensor(len(batch), max_note_boundary_len).zero_()
        note_boundary_cnt   = torch.LongTensor(len(batch), max_note_boundary_len).zero_()
                
        # vq_padded    = torch.FloatTensor(len(batch), batch[0][9].size(0), batch[0][9].size(1)).zero_()
        spk_ids        = torch.LongTensor(len(batch))
        genre_id       = torch.LongTensor(len(batch))

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths  = torch.LongTensor(len(batch))
        f0_lengths   = torch.LongTensor(len(batch))
        note_lengths = torch.LongTensor(len(batch))
        note_dur_lengths = torch.LongTensor(len(batch))
        spk_ids_length   = torch.LongTensor(len(batch))
        note_boundary_lengths = torch.LongTensor(len(batch))
        note_dur_input_lengths = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            # Lyrics
            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            # Mel-spectrogram
            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            # Waveform
            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            # gt f0
            gt_f0 = row[3]
            gt_f0_padded[i, :, : gt_f0.size(1)] = gt_f0

            # smoothed f0 
            smoothed_f0 = row[4]
            smoothed_f0_padded[i, :, : smoothed_f0.size(1)] = smoothed_f0

            # Note
            note = row[5]
            note_padded[i, : note.size(0)] = note
            note_lengths[i] = note.size(0)

            # Note duration
            note_dur = row[6]
            note_dur_padded[i, : note_dur.size(0)] = note_dur
            note_dur_lengths[i] = note_dur.size(0)

            # Note boundary (start, end, flag, cnt)
            note_boundary = row[7]
            for j, (start, end, flag, cnt) in enumerate(note_boundary):
                note_boundary_start[i, j] = start
                note_boundary_end[i, j] = end
                note_boundary_flag[i, j] = flag
                note_boundary_cnt[i, j] = cnt
            note_boundary_lengths[i] = len(note_boundary)
            
            note_dur_input = row[8]
            note_dur_input_padded[i, : note_dur_input.size(0)] = note_dur_input  # 추가

            # vq = row[9]
            # vq_padded[i, :, :] = vq
            # vq_lengths[i] = vq.size(1)
            spk_id = row[9]
            spk_ids[i] = spk_id

            # Genre ID
            genre_id[i] = row[10]

        return (
            text_padded, text_lengths,
            spec_padded, spec_lengths,
            wav_padded, wav_lengths,
            gt_f0_padded, f0_lengths,
            smoothed_f0_padded, f0_lengths,
            note_padded, note_lengths,
            note_dur_padded, note_dur_lengths,
            note_boundary_start, note_boundary_end, note_boundary_flag, note_boundary_cnt, note_boundary_lengths,
            note_dur_input_padded, note_dur_input_lengths,
            # vq_padded, vq_lengths,
            spk_ids,spk_ids_length,
            genre_id
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
       
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket            
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size

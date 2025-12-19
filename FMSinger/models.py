import sys
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

sys.path.append('../..')
import modules.commons as commons
import modules.modules as modules
import modules.attentions as attentions
from modules.flow_matching import TorchCFMWrapper
from modules.commons import init_weights, get_padding
from modules.ddsp import scale_function, remove_above_nyquist, upsample
from modules.stft import TorchSTFT

import torch.distributions as D

from modules.losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)

def create_alignment_matrix(durations, frame_len):
    durations = durations.clone()
    num_phonemes = len(durations)
    
    if frame_len != sum(durations):
        diff = frame_len - sum(durations)
        durations[-2] = durations[-2] + diff
    
    alignment_matrix = torch.zeros(frame_len, num_phonemes, dtype=torch.float32, device=durations.device)
    
    start = 0
    for idx, duration in enumerate(durations):
        end = start + int(duration.item())
        alignment_matrix[start:end, idx] = 1.0
        start = end
    
    return alignment_matrix

import monotonic_align

LRELU_SLOPE=0.1

class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_speakers=0, spk_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.spk_channels = spk_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    self.conv_3 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_3 = modules.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 2, 1)

    if n_speakers != 0:
      self.cond = nn.Conv1d(spk_channels, in_channels, 1)

  def forward(self, x, x_mask, spk_emb=None):
    x = torch.detach(x)  
    if spk_emb is not None:
      spk_emb = torch.detach(spk_emb)
      x = x + self.cond(spk_emb)

    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)

    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)

    x = self.conv_3(x * x_mask)
    x = torch.relu(x)
    x = self.norm_3(x)
    x = self.drop(x)

    x = self.proj(x * x_mask)
    return x * x_mask


class TextEncoder(nn.Module):
  def __init__(self,
      n_symbols,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout):
    super().__init__()
    self.n_symbols = n_symbols
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.emb_phone = nn.Embedding(self.n_symbols, 256)
    nn.init.normal_(self.emb_phone.weight, 0.0, 256**-0.5)

    self.emb_pitch = nn.Embedding(128, 128)
    nn.init.normal_(self.emb_pitch.weight, 0.0, 128**-0.5)

    
    self.emb_dur = torch.nn.Linear(1, 128)

    self.pre_net = torch.nn.Linear(512, hidden_channels)
    self.pre_dur_net = torch.nn.Linear(512, hidden_channels)

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.proj= nn.Conv1d(hidden_channels, out_channels, 1)
    self.proj_pitch = nn.Conv1d(128, out_channels, 1)

  def forward(self, phone, phone_lengths, pitchid, dur):

    phone_end = self.emb_phone(phone) * math.sqrt(256)
    pitch_end = self.emb_pitch(pitchid) * math.sqrt(128)
    dur = dur.float()
    dur_end = self.emb_dur(dur.unsqueeze(-1))
    x = torch.cat([phone_end, pitch_end, dur_end], dim=-1)

    dur_input = self.pre_dur_net(x)
    dur_input = torch.transpose(dur_input, 1, -1)
    
    x = self.pre_net(x)
    x = torch.transpose(x, 1, -1) # [b, h, t]
    
    x_mask = torch.unsqueeze(commons.sequence_mask(phone_lengths, x.size(2)), 1).to(x.dtype)

    x = self.encoder(x * x_mask, x_mask)
    x = self.proj(x) * x_mask

    pitch_info = self.proj_pitch(pitch_end.transpose(1, 2))

    return x, x_mask, dur_input, pitch_info


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


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
      super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
      x = torch.transpose(x, 1, 2)
      output = list()
      mel_len = list()
      for batch, expand_target in zip(x, duration):
        expanded = self.expand(batch, expand_target)
        output.append(expanded)
        mel_len.append(expanded.shape[0])

      if max_len is not None:
        output = pad_v2(output, max_len)
      else:
        output = pad_v2(output)
      output = torch.transpose(output, 1, 2)
      return output, torch.LongTensor(mel_len)

    def expand(self, batch, predicted):
      predicted = torch.squeeze(predicted)
      out = list()

      for i, vec in enumerate(batch):
        expand_size_raw = predicted[i].item()
        expand_size = max(1, min(int(abs(expand_size_raw)), 10000))  # 1~10000 범위로 제한
        
        state_info_index = torch.unsqueeze(torch.arange(0, expand_size), 1).float()
        state_info_length = torch.unsqueeze(torch.Tensor([expand_size] * expand_size), 1).float()
        state_info = torch.cat([state_info_index, state_info_length], 1).to(vec.device) 
        new_vec = vec.expand(max(int(expand_size), 0), -1)
        new_vec = torch.cat([new_vec, state_info], 1)
        out.append(new_vec)
      out = torch.cat(out, 0)
      return out

    def forward(self, x, duration, max_len):
      output, mel_len = self.LR(x, duration, max_len)
      return output, mel_len

class PriorDecoder(nn.Module):
  def __init__(self,
      out_bn_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout,
      n_speakers=0,
      spk_channels=0):
    super().__init__()
    self.out_bn_channels = out_bn_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.spk_channels = spk_channels

    self.prenet = nn.Conv1d(hidden_channels + 2, hidden_channels, 3, padding=1)
    self.decoder = attentions.FFT(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.proj = nn.Conv1d(hidden_channels, out_bn_channels, 1)

    if n_speakers != 0:
      self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

  def forward(self, x, x_lengths, spk_emb=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    x = self.prenet(x) * x_mask

    if(spk_emb is not None):
      x = x + self.cond(spk_emb)

    x = self.decoder(x * x_mask, x_mask)

    bn = self.proj(x) * x_mask

    return bn, x_mask

class Decoder(nn.Module):
  def __init__(self,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout,
      n_speakers=0,
      spk_channels=0):
    super().__init__()
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.spk_channels = spk_channels

    self.prenet = nn.Conv1d(hidden_channels + 2, hidden_channels, 3, padding=1)
    self.decoder = attentions.FFT(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    if n_speakers != 0:
      self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

  def forward(self, x, x_lengths, spk_emb=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    x = self.prenet(x) * x_mask

    if(spk_emb is not None):
      x = x + self.cond(spk_emb)

    x = self.decoder(x * x_mask, x_mask)

    x = self.proj(x) * x_mask

    return x, x_mask

class ConvReluNorm(nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
    super().__init__()
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.p_dropout = p_dropout
    assert n_layers > 1, "Number of layers should be larger than 0."

    self.conv_layers = nn.ModuleList()
    self.norm_layers = nn.ModuleList()
    self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2))
    self.norm_layers.append(LayerNorm(hidden_channels))
    self.relu_drop = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(p_dropout))
    for _ in range(n_layers-1):
      self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))
      self.norm_layers.append(LayerNorm(hidden_channels))
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    self.proj.weight.data.zero_()
    self.proj.bias.data.zero_()

  def forward(self, x):
    x = self.conv_layers[0](x)
    x = self.norm_layers[0](x)
    x = self.relu_drop(x)

    for i in range(1, self.n_layers):
      x_ = self.conv_layers[i](x)
      x_ = self.norm_layers[i](x_)
      x_ = self.relu_drop(x_)
      x = (x + x_) / 2
    x = self.proj(x)
    return x


class PosteriorEncoder(nn.Module):
  def __init__(self,
      hps,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, n_speakers=hps.data.n_speakers, spk_channels=hps.model.spk_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x,x_mask, g=g)
    stats = self.proj(x) * x_mask
    return stats, x_mask

class ResBlock3(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock3, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

class Generator_Harm(torch.nn.Module):
    def __init__(self, hps):
        super(Generator_Harm, self).__init__()
        self.hps = hps
        
        self.prenet = Conv1d(hps.model.hidden_channels, hps.model.hidden_channels, 3, padding=1)
                
        self.net = ConvReluNorm(hps.model.hidden_channels,
                                    hps.model.hidden_channels,
                                    hps.model.hidden_channels,
                                    hps.model.kernel_size,
                                    8,
                                    hps.model.p_dropout)

        #self.rnn = nn.LSTM(input_size=hps.model.hidden_channels, 
        #    hidden_size=hps.model.hidden_channels,
        #    num_layers=1, 
        #    bias=True, 
        #    batch_first=True, 
        #    dropout=0.5,
        #    bidirectional=True)
        self.postnet = Conv1d(hps.model.hidden_channels, hps.model.n_harmonic+1, 3, padding=1)
    
    def forward(self, f0, harm, mask):
        pitch = f0.transpose(1, 2)
        harm = self.prenet(harm)
        harm = self.net(harm) * mask
        harm = self.postnet(harm)
        harm = harm.transpose(1, 2)
        param = harm
        
        param = scale_function(param)
        total_amp = param[..., :1]
        amplitudes = param[..., 1:]
        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.hps.data.sample_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.hps.data.hop_size)
        pitch = upsample(pitch, self.hps.data.hop_size)
        
        n_harmonic = amplitudes.shape[-1]
        omega = torch.cumsum(2 * math.pi * pitch / self.hps.data.sample_rate, 1)
        omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
        signal_harmonics = (torch.sin(omegas) * amplitudes)
        signal_harmonics = signal_harmonics.transpose(1, 2)
        return signal_harmonics

class Generator(torch.nn.Module):
    def __init__(self, hps, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, n_speakers=0, spk_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.upsample_rates = upsample_rates
        self.n_speakers = n_speakers

        resblock = modules.ResBlock1 if resblock == '1' else modules.R
        
        self.downs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            i = len(upsample_rates) - 1 - i
            u = upsample_rates[i]
            k = upsample_kernel_sizes[i]

            self.downs.append(weight_norm(
                Conv1d(hps.model.n_harmonic + 2, hps.model.n_harmonic + 2,
                       k, u, padding=k//2)))
        
        self.resblocks_downs = nn.ModuleList()
        for i in range(len(self.downs)):
            j = len(upsample_rates) - 1 - i
            self.resblocks_downs.append(ResBlock3(hps.model.n_harmonic + 2, 3, (1, 3)))

        
        self.concat_pre = Conv1d(upsample_initial_channel + hps.model.n_harmonic + 2, upsample_initial_channel, 3, 1, padding=1)
        self.concat_conv = nn.ModuleList()
        for i in range(len(upsample_rates)):
          ch = upsample_initial_channel//(2**(i+1))
          self.concat_conv.append(Conv1d(ch + hps.model.n_harmonic + 2, ch, 3, 1, padding=1, bias=False))


        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if self.n_speakers != 0:
            self.cond = nn.Conv1d(spk_channels, upsample_initial_channel, 1)
            
    def forward(self, x, ddsp, g=None):

        x = self.conv_pre(x)

        if g is not None:
          x = x + self.cond(g)

        se = ddsp
        res_features = [se]
        for i in range(self.num_upsamples):
            in_size = se.size(2)
            se = self.downs[i](se)
            se = self.resblocks_downs[i](se)
            up_rate = self.upsample_rates[self.num_upsamples - 1 - i]
            se = se[:, :, : in_size // up_rate]
            res_features.append(se)

        x = torch.cat([x, se], 1)
        x = self.concat_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            in_size = x.size(2)
            x = self.ups[i](x)
            x = x[:, :, : in_size * self.upsample_rates[i]]

            x = torch.cat([x, res_features[self.num_upsamples - 1 - i]], 1)
            x = self.concat_conv[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
            
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x

    def remove_weight_norm(self):

        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

class Generator_Noise(nn.Module):
    def __init__(self, hps):
        super(Generator_Noise, self).__init__()
        self.hps = hps
        self.win_size = hps.data.win_size
        self.hop_size = hps.data.hop_size
        self.fft_size = hps.data.n_fft
        
        self.istft_pre = Conv1d(
            hps.model.hidden_channels, 
            hps.model.hidden_channels, 
            3, 
            padding=1
        )
        
        self.net = ConvReluNorm(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            hps.model.kernel_size,
            8,
            hps.model.p_dropout
        )

        self.istft_amplitude = nn.Conv1d(
            hps.model.hidden_channels, 
            self.fft_size//2 + 1, 
            1, 
            1
        )

        self.window = torch.hann_window(self.win_size)

    def forward(self, x, mask):
        istft_x = x
        istft_x = self.istft_pre(istft_x)

        istft_x = self.net(istft_x) * mask

        # shape: (B, fft_size//2+1, time) -> unsqueeze(-1): (B, F, T, 1)
        amp = self.istft_amplitude(istft_x).unsqueeze(-1)

        phase = (torch.rand(amp.shape) * 2 * 3.14 - 3.14).to(amp)

        real = amp * torch.cos(phase)  # (B, F, T, 1)
        imag = amp * torch.sin(phase)  # (B, F, T, 1)

        spec = torch.cat([real, imag], dim=-1)

        spec_complex = torch.view_as_complex(spec)  # (B, F, T), complex64

        istft_x = torch.istft(
            spec_complex,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window.to(x),  
            center=True,
            length=x.shape[2] * self.hop_size,  
            return_complex=False
        )

        # (B, T) -> (B, 1, T) 
        return istft_x.unsqueeze(1)

class LayerNorm(nn.Module):
  def __init__(self, channels, eps=1e-5):
    super().__init__()
    self.channels = channels
    self.eps = eps

    self.gamma = nn.Parameter(torch.ones(channels))
    self.beta = nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    x = x.transpose(1, -1)
    x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
    return x.transpose(1, -1)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiFrequencyDiscriminator(nn.Module):
    def __init__(self,
                 hop_lengths=[128, 256, 512],
                 hidden_channels=[256, 512, 512],
                 domain='double', mel_scale=True):
        super(MultiFrequencyDiscriminator, self).__init__()

        self.stfts = nn.ModuleList([
            TorchSTFT(fft_size=x * 4, hop_size=x, win_size=x * 4,
                      normalized=True, domain=domain, mel_scale=mel_scale)
            for x in hop_lengths])

        self.domain = domain
        if domain == 'double':
            self.discriminators = nn.ModuleList([
                BaseFrequenceDiscriminator(2, c)
                for x, c in zip(hop_lengths, hidden_channels)])
        else:
            self.discriminators = nn.ModuleList([
                BaseFrequenceDiscriminator(1, c)
                for x, c in zip(hop_lengths, hidden_channels)])

    def forward(self, x):
        scores, feats = list(), list()
        for stft, layer in zip(self.stfts, self.discriminators):

            mag, phase = stft.transform(x.squeeze())
            if self.domain == 'double':
                mag = torch.stack(torch.chunk(mag, 2, dim=1), dim=1)
            else:
                mag = mag.unsqueeze(1)

            score, feat = layer(mag)
            scores.append(score)
            feats.append(feat)
        return scores, feats

class BaseFrequenceDiscriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels=512):
        super(BaseFrequenceDiscriminator, self).__init__()

        self.discriminator = nn.ModuleList()
        self.discriminator += [
            nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    in_channels, hidden_channels // 32,
                    kernel_size=(3, 3), stride=(1, 1)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels // 32, hidden_channels // 16,
                    kernel_size=(3, 3), stride=(2, 2)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels // 16, hidden_channels // 8,
                    kernel_size=(3, 3), stride=(1, 1)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels // 8, hidden_channels // 4,
                    kernel_size=(3, 3), stride=(2, 2)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels // 4, hidden_channels // 2,
                    kernel_size=(3, 3), stride=(1, 1)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels // 2, hidden_channels,
                    kernel_size=(3, 3), stride=(2, 2)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels, 1,
                    kernel_size=(3, 3), stride=(1, 1)))
            )
        ]

    def forward(self, x):
        hiddens = []
        for layer in self.discriminator:
            x = layer(x)
            hiddens.append(x)
        return x, hiddens[-1]

class Discriminator(torch.nn.Module):
    def __init__(self, hps, use_spectral_norm=False):
        super(Discriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)
        self.disc_multfrequency = MultiFrequencyDiscriminator(hop_lengths=[ 
                                                                           int(hps.data.sample_rate * 5 / 1000),
                                                                           int(hps.data.sample_rate * 10 / 1000),
                                                                           int(hps.data.sample_rate * 12.5 / 1000)],
                                                              hidden_channels=[256, 256, 256])
        
    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        scores_r, fmaps_r = self.disc_multfrequency(y)
        scores_g, fmaps_g = self.disc_multfrequency(y_hat)
        for i in range(len(scores_r)):
            y_d_rs.append(scores_r[i])
            y_d_gs.append(scores_g[i])
            fmap_rs.append(fmaps_r[i])
            fmap_gs.append(fmaps_g[i])
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrn(nn.Module):
  """
  Model
  """

  def __init__(self, hps):
    super().__init__()
    self.hps = hps

    self.text_encoder = TextEncoder(
        224,
        hps.model.prior_hidden_channels,
        hps.model.prior_hidden_channels,
        hps.model.prior_filter_channels,
        hps.model.prior_n_heads,
        hps.model.prior_n_layers,
        hps.model.prior_kernel_size,
        hps.model.prior_p_dropout)

    self.decoder = PriorDecoder(
        hps.model.hidden_channels * 2,
        hps.model.prior_hidden_channels,
        hps.model.prior_filter_channels,
        hps.model.prior_n_heads,
        hps.model.prior_n_layers,
        hps.model.prior_kernel_size,
        hps.model.prior_p_dropout,
        n_speakers=hps.data.n_speakers,
        spk_channels=hps.model.spk_channels
        )

    self.f0_decoder = Decoder(
        1,
        hps.model.prior_hidden_channels,
        hps.model.prior_filter_channels,
        hps.model.prior_n_heads,
        hps.model.prior_n_layers,
        hps.model.prior_kernel_size,
        hps.model.prior_p_dropout,
        n_speakers=hps.data.n_speakers,
        spk_channels=hps.model.spk_channels
        )

    self.mel_decoder = Decoder(
        hps.data.acoustic_dim,
        hps.model.prior_hidden_channels,
        hps.model.prior_filter_channels,
        hps.model.prior_n_heads,
        hps.model.prior_n_layers,
        hps.model.prior_kernel_size,
        hps.model.prior_p_dropout,
        n_speakers=hps.data.n_speakers,
        spk_channels=hps.model.spk_channels
        )
   
    self.posterior_encoder = PosteriorEncoder(
        hps,
        hps.data.acoustic_dim,
        hps.model.hidden_channels, 
        hps.model.hidden_channels, 5, 1, 16)
    
    self.dropout = nn.Dropout(0.2)
    
    self.duration_predictor = DurationPredictor(
        hps.model.prior_hidden_channels, 
        hps.model.prior_hidden_channels, 
        3, 
        0.5, 
        n_speakers=hps.data.n_speakers, 
        spk_channels=hps.model.spk_channels)
    self.LR = LengthRegulator()

    torchcfm_config = getattr(hps.model, 'torchcfm', None)
    if torchcfm_config and getattr(torchcfm_config, 'enabled', False):
        self.flow_main = TorchCFMWrapper(
            hidden_channels=hps.model.hidden_channels,
            filter_channels=hps.model.hidden_channels,
            kernel_size=3,
            n_layers=4,
            p_dropout=0.1,
            n_speakers=hps.data.n_speakers,
            spk_channels=hps.model.spk_channels,
            solver='dopri5',
            atol=1e-5,
            rtol=1e-5,
            use_adjoint=True,
            sigma=getattr(torchcfm_config, 'sigma', 0.0)
        )
        self.use_main_flow = True
    else:
        self.flow_main = None
        self.use_main_flow = False
    
    self.flow_mas = modules.ResidualCouplingBlock(
        channels=hps.model.hidden_channels,
        hidden_channels=hps.model.filter_channels,
        kernel_size=5,
        dilation_rate=1,
        n_layers=4,
        n_flows=4,
        n_speakers=hps.data.n_speakers,
        gin_channels=hps.model.spk_channels
    )
    self.mas_flow_type = 'residual_coupling'

    self.dec = Generator(hps,
                         hps.model.hidden_channels,
                         hps.model.resblock, 
                         hps.model.resblock_kernel_sizes, 
                         hps.model.resblock_dilation_sizes, 
                         hps.model.upsample_rates, 
                         hps.model.upsample_initial_channel, 
                         hps.model.upsample_kernel_sizes, 
                         n_speakers=hps.data.n_speakers,
                         spk_channels=hps.model.spk_channels)
    
    self.dec_harm = Generator_Harm(hps)
    
    self.dec_noise = Generator_Noise(hps)
    
    self.f0_prenet = nn.Conv1d(1, hps.model.prior_hidden_channels + 2, 3, padding=1)
    self.energy_prenet = nn.Conv1d(1, hps.model.prior_hidden_channels + 2, 3, padding=1)
    self.mel_prenet = nn.Conv1d(hps.data.acoustic_dim, hps.model.prior_hidden_channels + 2, 3, padding=1)    
    self.sin_prenet = nn.Conv1d(1, hps.model.n_harmonic + 2, 3, padding=1)

    if hps.data.n_speakers > 0:
        self.emb_spk = nn.Embedding( hps.data.n_speakers, hps.model.spk_channels)
    
    self.proj_prior = nn.Conv1d(hps.model.prior_hidden_channels, hps.model.hidden_channels * 2, 1)
  
  def perform_note_boundary_mas(self, neg_cent, base_attn_mask, x_mask, y_mask,
                               note_boundary_start, note_boundary_end, 
                               note_boundary_flag, note_boundary_cnt, 
                               note_boundary_lengths, bn_lengths):

    batch_size, mel_frames, text_length = neg_cent.shape
    attn = torch.zeros(batch_size, 1, mel_frames, text_length, device=neg_cent.device)
    
    for batch_idx in range(batch_size):
        actual_mel_len = bn_lengths[batch_idx].item()
        actual_text_len = int(x_mask[batch_idx].sum().item())
        
        #  note boundary mas
        phn_start = 0
        single_phoneme_count = 0
        multiple_phoneme_count = 0
        total_note_frames = 0
        
        for boundary_idx in range(note_boundary_lengths[batch_idx]):
            note_start = note_boundary_start[batch_idx, boundary_idx].item()
            note_end = note_boundary_end[batch_idx, boundary_idx].item()
            phn_count = note_boundary_cnt[batch_idx, boundary_idx].item()
            phn_end = phn_start + phn_count
            
            single_phoneme_per_note = (phn_count == 1)
            
            note_start = max(0, min(note_start, actual_mel_len))
            note_end = max(note_start + 1, min(note_end, actual_mel_len))
            phn_end = min(phn_end, actual_text_len)
            
            note_frame_len = note_end - note_start
            total_note_frames += note_frame_len
            
            if single_phoneme_per_note:
                phn_idx = phn_start
                if phn_idx < actual_text_len:
                    attn[batch_idx, 0, note_start:note_end, phn_idx] = 1.0
                single_phoneme_count += 1
            else:
                note_neg_cent = neg_cent[batch_idx:batch_idx+1, note_start:note_end, phn_start:phn_end]
                note_attn_mask = base_attn_mask[batch_idx:batch_idx+1, :, note_start:note_end, phn_start:phn_end].squeeze(1)
                
                mas_result = monotonic_align.maximum_path(
                    note_neg_cent,  # [1, note_frame_len, phn_count]
                    note_attn_mask  # [1, note_frame_len, phn_count]
                ).unsqueeze(1).detach()  # [1, 1, note_frame_len, phn_count]
                
                attn[batch_idx, :, note_start:note_end, phn_start:phn_end] = mas_result.squeeze(1)
                multiple_phoneme_count += 1
            
            phn_start = phn_end
        
        frame_assignments = attn[batch_idx, 0, :actual_mel_len, :actual_text_len].sum(dim=1)
        
        if not torch.allclose(frame_assignments, torch.ones_like(frame_assignments)):
            unassigned_frames = (frame_assignments == 0).nonzero(as_tuple=True)[0]
            for frame_idx in unassigned_frames:
                active_phonemes = (attn[batch_idx, 0, :actual_mel_len, :actual_text_len].sum(dim=0) > 0).nonzero(as_tuple=True)[0]
                if len(active_phonemes) > 0:
                    closest_phn = active_phonemes[0]  
                    attn[batch_idx, 0, frame_idx, closest_phn] = 1.0
        
    
    return attn

  def forward(self, phone, phone_lengths, pitchid, dur, gtdur, F0, mel, bn_lengths, spk_id,
              note_boundary_start=None, note_boundary_end=None, note_boundary_flag=None, 
              note_boundary_cnt=None, note_boundary_lengths=None):
    if self.hps.data.n_speakers > 0:
      g = self.emb_spk(spk_id).unsqueeze(-1) # [b, h, 1]
    else:
      g = None
    
    # Encoder
    x, x_mask, dur_input, x_pitch = self.text_encoder(phone, phone_lengths, pitchid, dur)
    
    if note_boundary_start is not None and note_boundary_end is not None:
        posterior, y_mask = self.posterior_encoder(mel, bn_lengths, g)
        posterior_mean = posterior[:,:self.hps.model.hidden_channels,:]
        posterior_logstd = posterior[:,self.hps.model.hidden_channels:,:]
        posterior_norm = D.Normal(posterior_mean, torch.exp(posterior_logstd))
        posterior_z = posterior_norm.rsample()
        
        loss_fm_mas = torch.tensor(0.0, device=posterior_z.device)
        
        z_p_mas = self.flow_mas(posterior_z, y_mask, g=g)
        
        prior_stats = self.proj_prior(x) * x_mask
        m_p, logs_p = torch.split(prior_stats, self.hps.model.hidden_channels, dim=1)
        
        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t_s]
            
            batch_size, mel_frames, text_length = z_p_mas.shape[0], z_p_mas.shape[2], m_p.shape[2]
            
            # neg_cent1: 상수항
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t_s]
            
            neg_cent = torch.zeros(batch_size, mel_frames, text_length, device=z_p_mas.device)
            for b in range(batch_size):
                z_p_b = z_p_mas[b:b+1]  # [1, d, t_t]
                m_p_b = m_p[b:b+1]  # [1, d, t_s]
                s_p_sq_r_b = s_p_sq_r[b:b+1]  # [1, d, t_s]
                
                neg_cent2_b = torch.matmul(-0.5 * (z_p_b**2).transpose(1, 2), s_p_sq_r_b)  # [1, t_t, t_s]
                neg_cent3_b = torch.matmul(z_p_b.transpose(1, 2), (m_p_b * s_p_sq_r_b))  # [1, t_t, t_s]
                neg_cent4_b = torch.sum(-0.5 * (m_p_b**2) * s_p_sq_r_b, [1], keepdim=True)  # [1, 1, t_s]
                
                neg_cent[b] = neg_cent1[b] + neg_cent2_b.squeeze(0) + neg_cent3_b.squeeze(0) + neg_cent4_b.squeeze(0)
            
            base_attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        
        attn = self.perform_note_boundary_mas(
            neg_cent, base_attn_mask, x_mask, y_mask,
            note_boundary_start, note_boundary_end, 
            note_boundary_flag, note_boundary_cnt, 
            note_boundary_lengths, bn_lengths
        )
        
        w_mas = attn.sum(2)  # [b, 1, t_s] 
        phoneme_duration_gt = w_mas.squeeze(1)  
        
        m_p_expanded = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p_expanded = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        
        loss_kl_mas = kl_loss(z_p_mas, posterior_logstd, m_p_expanded, logs_p_expanded, y_mask)
        
        logw_pred = self.duration_predictor(dur_input, x_mask, spk_emb=g)
        
        gt_dur = torch.clamp(torch.round(phoneme_duration_gt), min=1).long()
        
        logw_pred_scaled = (torch.exp(logw_pred) - 1) * x_mask
        logw_pred_scaled = logw_pred_scaled * self.hps.data.sample_rate / self.hps.data.hop_size
        
        phoneme_duration_pred = logw_pred_scaled  
        phoneme_duration_target = phoneme_duration_gt  
        note_duration_target = gtdur  
        
    else:
        logw_pred = self.duration_predictor(dur_input, x_mask, spk_emb=g)
        predict_dur = (torch.exp(logw_pred) - 1) * x_mask
        predict_dur = predict_dur * self.hps.data.sample_rate / self.hps.data.hop_size
        predict_dur = torch.max(predict_dur, torch.ones_like(predict_dur).to(x))
        predict_dur = torch.ceil(predict_dur).long()
        gt_dur = gtdur
        attn = None
        
        phoneme_duration_pred = logw_pred  
        phoneme_duration_target = None  
        note_duration_target = gtdur 

    # Length Regulator 
    decoder_input, mel_len = self.LR(x, gt_dur, None)
    decoder_input_pitch, mel_len = self.LR(x_pitch, gt_dur, None)
    
    max_mel_len = max(bn_lengths).item()
    if decoder_input.size(2) != max_mel_len:
        if decoder_input.size(2) < max_mel_len:
            pad_len = max_mel_len - decoder_input.size(2)
            decoder_input = F.pad(decoder_input, (0, pad_len))
            decoder_input_pitch = F.pad(decoder_input_pitch, (0, pad_len))
        else:
            decoder_input = decoder_input[:, :, :max_mel_len]
            decoder_input_pitch = decoder_input_pitch[:, :, :max_mel_len]
 
    LF0 = 2595. * torch.log10(1. + F0 / 700.)
    LF0 = LF0 / 500
   
    # aam
    predict_lf0, predict_bn_mask = self.f0_decoder(decoder_input + decoder_input_pitch, bn_lengths, spk_emb=g)
    predict_mel, predict_bn_mask = self.mel_decoder(decoder_input + self.f0_prenet(LF0), bn_lengths, spk_emb=g)

    predict_energy = predict_mel.detach().sum(1).unsqueeze(1) / self.hps.data.acoustic_dim

    decoder_input = decoder_input + \
                        self.f0_prenet(LF0) + \
                        self.energy_prenet(predict_energy) + \
                        self.mel_prenet(predict_mel.detach())
    decoder_output, predict_bn_mask = self.decoder(decoder_input, bn_lengths, spk_emb=g)
    

    prior_info = decoder_output
    prior_mean = prior_info[:, :self.hps.model.hidden_channels, :]
    prior_logstd = prior_info[:, self.hps.model.hidden_channels:, :]   
    
    
    prior_norm = D.Normal(prior_mean, torch.exp(prior_logstd))
    prior_z = prior_norm.rsample()

    if note_boundary_start is not None and note_boundary_end is not None:
        posterior_mean = posterior[:,:self.hps.model.hidden_channels,:]
        posterior_logstd = posterior[:,self.hps.model.hidden_channels:,:]
    else:
        posterior, y_mask = self.posterior_encoder(mel, bn_lengths, g)
        posterior_mean = posterior[:,:self.hps.model.hidden_channels,:]
        posterior_logstd = posterior[:,self.hps.model.hidden_channels:,:]
    
    posterior_norm = D.Normal(posterior_mean, torch.exp(posterior_logstd))
    posterior_z = posterior_norm.rsample()
    
    loss_kl = kl_loss(posterior_z, posterior_logstd, prior_mean, prior_logstd, y_mask)

    if self.use_main_flow and self.flow_main is not None:
        z_enhanced = self.flow_main(prior_z, y_mask, g=g, reverse=False)
        
        loss_cnf = self.flow_main.compute_flow_matching_loss(
            x0=prior_z,      
            x1=posterior_z,  
            x_mask=y_mask,
            g=g
        )
    else:
        loss_kl = kl_loss(posterior_z, posterior_logstd, prior_mean, prior_logstd, y_mask)
        loss_cnf = torch.tensor(0.0, device=posterior_z.device)

    p_z = posterior_z
    p_z = self.dropout(p_z)
 
    pitch = upsample(F0.transpose(1, 2), self.hps.data.hop_size)
    omega = torch.cumsum(2 * math.pi * pitch / self.hps.data.sample_rate, 1)
    sin = torch.sin(omega).transpose(1, 2)
    
    # dsp synthesize
    noise_x = self.dec_noise(p_z, y_mask)
    harm_x = self.dec_harm(F0, p_z, y_mask)

    # dsp waveform
    dsp_o = torch.cat([harm_x, noise_x], axis=1)

    #decoder_condition = torch.cat([harm_x, noise_x, sin], axis=1)    
    decoder_condition = self.sin_prenet(sin)

    # dsp based HiFiGAN vocoder
    x_slice, ids_slice = commons.rand_slice_segments(p_z, bn_lengths, self.hps.train.segment_size // self.hps.data.hop_size)
    F0_slice = commons.slice_segments(F0, ids_slice, self.hps.train.segment_size // self.hps.data.hop_size)
    dsp_slice = commons.slice_segments(dsp_o, ids_slice * self.hps.data.hop_size, self.hps.train.segment_size)
    condition_slice = commons.slice_segments(decoder_condition, ids_slice * self.hps.data.hop_size, self.hps.train.segment_size)
    o = self.dec(x_slice, condition_slice, g=g)

    if note_boundary_start is not None and note_boundary_end is not None:
        return o, ids_slice, gt_dur, predict_lf0, LF0 * predict_bn_mask, dsp_slice.sum(1), loss_kl, predict_mel, predict_bn_mask, phoneme_duration_pred, phoneme_duration_target, note_duration_target, loss_cnf, x_mask, loss_kl_mas, loss_fm_mas
    else:
        return o, ids_slice, gt_dur, predict_lf0, LF0 * predict_bn_mask, dsp_slice.sum(1), loss_kl, predict_mel, predict_bn_mask, phoneme_duration_pred, phoneme_duration_target, note_duration_target, loss_cnf, x_mask

  def infer(self, phone, phone_lengths, pitchid, dur, spk_id, spec=None, spec_lengths=None, length_scale=1.):
    if self.hps.data.n_speakers > 0:
      g = self.emb_spk(spk_id).unsqueeze(-1) # [b, h, 1]
    else:
      g = None
    
    # Encoder
    x, x_mask, dur_input, x_pitch = self.text_encoder(phone, phone_lengths, pitchid, dur)
    
    logw_pred = self.duration_predictor(dur_input, x_mask, spk_emb=g)
    
    predict_dur = (torch.exp(logw_pred) - 1) * x_mask
    predict_dur = predict_dur * self.hps.data.sample_rate / self.hps.data.hop_size
    predict_dur = torch.max(predict_dur, torch.ones_like(predict_dur).to(x))
    
    predict_dur = torch.clamp(predict_dur, min=1, max=2000) 
    predict_dur = torch.ceil(predict_dur).long()
    predict_dur = predict_dur[:, 0, :]
    
    y_lengths = torch.clamp_min(torch.sum(predict_dur, [1]), 1).long()

    # Length Regulator 
    decoder_input, mel_len = self.LR(x, predict_dur, None)
    decoder_input_pitch, mel_len = self.LR(x_pitch, predict_dur, None)
    
    # aam
    predict_lf0, predict_bn_mask = self.f0_decoder(decoder_input + decoder_input_pitch, y_lengths, spk_emb=g)
    predict_mel, predict_bn_mask = self.mel_decoder(decoder_input + self.f0_prenet(predict_lf0), y_lengths, spk_emb=g)

    predict_lf0 = torch.max(predict_lf0, torch.zeros_like(predict_lf0).to(predict_lf0))
    predict_energy = predict_mel.sum(1).unsqueeze(1) / self.hps.data.acoustic_dim
    
    decoder_input = decoder_input + \
                        self.f0_prenet(predict_lf0) + \
                        self.energy_prenet(predict_energy) + \
                        self.mel_prenet(predict_mel)
    decoder_output, y_mask = self.decoder(decoder_input, y_lengths, spk_emb=g)
 
    prior_info = decoder_output
    prior_mean = prior_info[:, :self.hps.model.hidden_channels, :]
    prior_std = prior_info[:, self.hps.model.hidden_channels:, :]   
    
    if self.use_main_flow and self.flow_main is not None:
        prior_z_sample = prior_mean + torch.randn_like(prior_std) * torch.exp(prior_std) * 0.8
        prior_z = self.flow_main(prior_z_sample, y_mask, g=g, reverse=False)
    else:
        prior_z = prior_mean + torch.randn_like(prior_std) * torch.exp(prior_std) * 0.8

    noise_x = self.dec_noise(prior_z, y_mask)

    F0_std = 500
    F0 = predict_lf0 * F0_std
    F0 = F0 / 2595
    F0 = torch.pow(10, F0)
    F0 = (F0 - 1) * 700.

    harm_x = self.dec_harm(F0, prior_z, y_mask)
    
    pitch = upsample(F0.transpose(1, 2), self.hps.data.hop_size)
    omega = torch.cumsum(2 * math.pi * pitch / self.hps.data.sample_rate, 1)
    sin = torch.sin(omega).transpose(1, 2)

    #decoder_condition = torch.cat([harm_x, noise_x, sin], axis=1)
    decoder_condition = self.sin_prenet(sin)

    # dsp based HiFiGAN vocoder
    o = self.dec(prior_z, decoder_condition, g=g)
 
    return o, harm_x.sum(1).unsqueeze(1), noise_x, prior_z, predict_dur

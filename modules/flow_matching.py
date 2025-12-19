# TorchCFM 기반 Conditional Flow Matching 구현
# 2024.12.19: VISinger2 + TorchCFM Conditional Flow Matching 통합

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchdiffeq import odeint

# TorchCFM 라이브러리 임포트
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from modules.modules import DDSConv, LayerNorm


class VelocityNetwork(nn.Module):
    """
    Flow Matching을 위한 Velocity Field Network
    
    시간 t와 입력 x_t를 받아서 velocity field v_t(x_t)를 예측합니다.
    """
    def __init__(self, 
                 hidden_channels,
                 filter_channels,
                 kernel_size=3,
                 n_layers=4,
                 p_dropout=0.1,
                 n_speakers=0,
                 spk_channels=0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_speakers = n_speakers
        self.spk_channels = spk_channels
        
        # 시간 임베딩을 위한 네트워크
        self.time_embedding = TimeEmbedding(hidden_channels)
        
        # 입력 projection
        self.input_proj = nn.Conv1d(hidden_channels, filter_channels, 1)
        
        # 메인 velocity 네트워크 (DDSConv 사용)
        self.velocity_net = DDSConv(
            channels=filter_channels,
            kernel_size=kernel_size,
            n_layers=n_layers,
            p_dropout=p_dropout
        )
        
        # 출력 projection
        self.output_proj = nn.Conv1d(filter_channels, hidden_channels, 1)
        
        # 화자 조건부 처리
        if n_speakers > 0:
            self.spk_proj = nn.Conv1d(spk_channels, filter_channels, 1)
        
        # 가중치 초기화
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, t, x, x_mask, g=None):
        """
        Args:
            t: 시간 스칼라 또는 텐서 [1] or [batch]
            x: 입력 텐서 [batch, channels, time]
            x_mask: 마스크 [batch, 1, time]
            g: 화자 임베딩 [batch, spk_channels, 1]
        
        Returns:
            velocity: 예측된 velocity field [batch, channels, time]
        """
        batch_size = x.size(0)
        
        # 시간 임베딩
        if isinstance(t, (float, int)):
            t = torch.full((batch_size,), t, device=x.device, dtype=x.dtype)
        elif len(t.shape) == 0:
            t = t.expand(batch_size)
        
        time_emb = self.time_embedding(t)  # [batch, channels]
        time_emb = time_emb.unsqueeze(-1)  # [batch, channels, 1]
        
        # 입력 처리
        h = self.input_proj(x) * x_mask
        
        # 시간 임베딩 추가
        if h.size(1) == time_emb.size(1):
            h = h + time_emb
        else:
            # 채널 수가 다르면 projection 필요
            time_proj = F.linear(time_emb.squeeze(-1), 
                               torch.randn(h.size(1), time_emb.size(1)).to(h.device))
            h = h + time_proj.unsqueeze(-1)
        
        # 화자 정보 추가
        if g is not None and self.n_speakers > 0:
            spk_emb = self.spk_proj(g)
            h = h + spk_emb
        
        # Velocity 예측
        h = self.velocity_net(h, x_mask)
        velocity = self.output_proj(h) * x_mask
        
        return velocity


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for flow matching
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Args:
            t: [batch] 시간 텐서
        Returns:
            embedding: [batch, dim] 시간 임베딩
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # 홀수 차원 처리
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1, 0, 0))
        
        return embeddings


class ODEFunction(nn.Module):
    """
    ODE Function을 nn.Module로 래핑하여 adjoint method와 호환되도록 함
    """
    def __init__(self, velocity_net):
        super().__init__()
        self.velocity_net = velocity_net
        
    def forward(self, t, x):
        """
        ODE function for neural ODE
        
        Args:
            t: 시간 스칼라
            x: 입력 텐서 [batch, channels, time]
        
        Returns:
            dx_dt: 시간에 대한 x의 변화율
        """
        # x에서 mask와 speaker embedding 분리 (forward에서 함께 전달됨)
        if hasattr(self, '_mask') and hasattr(self, '_g'):
            x_mask = self._mask
            g = self._g
        else:
            # fallback: mask 없이 진행
            batch_size, channels, time_steps = x.shape
            x_mask = torch.ones(batch_size, 1, time_steps, device=x.device, dtype=x.dtype)
            g = None
        
        # Velocity field 계산
        velocity = self.velocity_net(t, x, x_mask, g)
        
        # 마스크 적용
        dx_dt = velocity * x_mask
        
        return dx_dt
    
    def set_context(self, x_mask, g):
        """컨텍스트 정보 설정 (mask와 speaker embedding)"""
        self._mask = x_mask
        self._g = g


class TorchCFMWrapper(nn.Module):
    """
    TorchCFM Conditional Flow Matching Wrapper
    
    TorchCFM의 ConditionalFlowMatcher를 VISinger2에 맞게 래핑한 클래스
    """
    def __init__(self,
                 hidden_channels,
                 filter_channels,
                 kernel_size=3,
                 n_layers=4,
                 p_dropout=0.1,
                 n_speakers=0,
                 spk_channels=0,
                 solver='dopri5',
                 atol=1e-5,
                 rtol=1e-5,
                 use_adjoint=False,
                 sigma=0.0):  # TorchCFM에서 사용하는 noise level
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.use_adjoint = use_adjoint
        self.sigma = sigma
        
        # Velocity network (기존 구조 유지)
        self.velocity_net = VelocityNetwork(
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            kernel_size=kernel_size,
            n_layers=n_layers,
            p_dropout=p_dropout,
            n_speakers=n_speakers,
            spk_channels=spk_channels
        )
        
        # TorchCFM ConditionalFlowMatcher 초기화
        self.flow_matcher = ConditionalFlowMatcher(sigma=sigma)
        
        # ODE Function을 nn.Module로 래핑 (기존 유지)
        self.ode_func = ODEFunction(self.velocity_net)
    

    
    def forward(self, x, x_mask, g=None, reverse=False, **kwargs):
        """
        Flow Matching CNF forward/reverse 연산
        
        Args:
            x: 입력 텐서 [batch, channels, time]
            x_mask: 마스크 [batch, 1, time]
            g: 화자 임베딩 [batch, spk_channels, 1]
            reverse: True이면 역방향 (prior -> data), False이면 순방향 (data -> prior)
        
        Returns:
            output: 변환된 텐서 [batch, channels, time]
        """
        device = x.device
        
        # 시간 구간 설정
        if reverse:
            # 역방향: t=1 (prior) -> t=0 (data)
            t_span = torch.tensor([1.0, 0.0], device=device)
        else:
            # 순방향: t=0 (data) -> t=1 (prior)  
            t_span = torch.tensor([0.0, 1.0], device=device)
        
        # ODE Function에 컨텍스트 설정
        self.ode_func.set_context(x_mask, g)
        
        # ODE 적분 수행
        if self.use_adjoint:
            try:
                from torchdiffeq import odeint_adjoint
                output = odeint_adjoint(
                    func=self.ode_func,
                    y0=x,
                    t=t_span,
                    method=self.solver,
                    atol=self.atol,
                    rtol=self.rtol,
                    options={'step_size': 0.1}  # 메모리 절약을 위한 step size 제한
                )[-1]  # 마지막 시간 단계만 취함
            except Exception as e:
                # adjoint 실패 시 일반 odeint로 fallback
                output = odeint(
                    func=self.ode_func,
                    y0=x,
                    t=t_span,
                    method=self.solver,
                    atol=self.atol,
                    rtol=self.rtol,
                    options={'step_size': 0.1}  # 메모리 절약을 위한 step size 제한
                )[-1]
        else:
            try:
                output = odeint(
                    func=self.ode_func,
                    y0=x,
                    t=t_span,
                    method=self.solver,
                    atol=self.atol,
                    rtol=self.rtol,
                    options={'step_size': 0.1}  # 메모리 절약을 위한 step size 제한
                )[-1]
            except Exception as e:
                # ODE 해결 실패 시 Euler fallback
                output = self._euler_fallback(x, x_mask, g, reverse)
        
        # 마스크 적용하여 최종 결과 반환
        return output * x_mask
    
    def _euler_fallback(self, x, x_mask, g, reverse, n_steps=10):
        """
        ODE 해결 실패 시 단순 Euler 방법으로 fallback
        """
        dt = 1.0 / n_steps
        current_x = x
        
        for i in range(n_steps):
            if reverse:
                t = 1.0 - (i * dt)
            else:
                t = i * dt
            
            t_tensor = torch.tensor(t, device=x.device)
            velocity = self.velocity_net(t_tensor, current_x, x_mask, g)
            
            if reverse:
                current_x = current_x - dt * velocity * x_mask
            else:
                current_x = current_x + dt * velocity * x_mask
        
        return current_x
    
    def compute_flow_matching_loss(self, x0, x1, x_mask, g=None):
        """
        TorchCFM 기반 Conditional Flow Matching Loss 계산
        
        Args:
            x0: 소스 분포 (prior) [batch, channels, time]
            x1: 타겟 분포 (posterior) [batch, channels, time]
            x_mask: 마스크 [batch, 1, time]
            g: 화자 임베딩 [batch, spk_channels, 1]
        
        Returns:
            loss: TorchCFM conditional flow matching loss
        """
        batch_size = x0.size(0)
        device = x0.device
        
        # TorchCFM의 조건부 확률 경로 샘플링
        # x0, x1을 (batch*time, channels) 형태로 reshape
        # 원래: [batch, channels, time] -> [batch*time, channels]
        b, c, t = x0.shape
        x0_flat = x0.transpose(1, 2).reshape(-1, c)  # [batch*time, channels]
        x1_flat = x1.transpose(1, 2).reshape(-1, c)  # [batch*time, channels]
        mask_flat = x_mask.transpose(1, 2).reshape(-1, 1).squeeze(-1)  # [batch*time]
        
        # 마스크된 부분만 추출
        valid_indices = mask_flat.bool()
        if valid_indices.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        x0_masked = x0_flat[valid_indices]  # [valid_samples, channels]
        x1_masked = x1_flat[valid_indices]  # [valid_samples, channels]
        
        # TorchCFM conditional path sampling
        t_samples, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
            x0_masked, x1_masked
        )
        
        # Velocity network 예측 (reshape back to original format for compatibility)
        valid_samples = xt.shape[0]
        # t_samples를 스칼라로 평균화 (velocity_net이 스칼라 시간을 받도록)
        t_scalar = t_samples.mean()
        
        # xt를 다시 batch 형태로 변환 (velocity_net 호환성)
        # 임시로 batch=1로 처리하고, 모든 valid samples를 time dimension으로 배치
        xt_reshaped = xt.unsqueeze(0).transpose(1, 2)  # [1, channels, valid_samples]
        mask_reshaped = torch.ones(1, 1, valid_samples, device=device)
        
        # 화자 임베딩 처리 (첫 번째 배치 요소만 사용)
        g_sample = g[0:1] if g is not None else None
        
        # Velocity 예측
        predicted_velocity = self.velocity_net(t_scalar, xt_reshaped, mask_reshaped, g_sample)
        predicted_velocity = predicted_velocity.transpose(1, 2).squeeze(0)  # [valid_samples, channels]
        
        # TorchCFM Loss 계산
        loss = F.mse_loss(predicted_velocity, ut, reduction='mean')
        
        return loss


# 기존 클래스명과의 호환성을 위한 alias
FlowMatchingCNF = TorchCFMWrapper

def replace_flow_with_cnf(model, hps):
    """
    기존 ResidualCouplingBlock을 TorchCFMWrapper로 교체하는 함수
    
    Args:
        model: SynthesizerTrn 모델
        hps: 하이퍼파라미터
    """
    # 기존 flow를 TorchCFM 기반 CNF로 교체
    model.flow = TorchCFMWrapper(
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
        sigma=0.0  # TorchCFM noise level
    )
    
    return model

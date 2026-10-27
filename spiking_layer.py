import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SpikeModule(nn.Module):

    def __init__(self):
        super().__init__()
        self._spiking = True

    def set_spike_state(self, use_spike=True):
        self._spiking = use_spike

    def forward(self, x):
        # shape correction
        if self._spiking is not True and len(x.shape) == 5:
            x = x.mean([0])
        return x

def spike_activation(x, binary=False, temp=1.0):
    

    if binary:
        out_s = torch.gt(x, 0.5)
        out_bp = torch.clamp(x, 0, 1)
        return (out_s.float() - out_bp).detach() + out_bp
    else:
        #out_s = torch.sign(x)
        #out_s[torch.abs(x)<0.5] = torch.tensor(0.)
        #out_bp = torch.clamp(x, -1, 1)
        #return (out_s.float() - out_bp).detach() + out_bp
        #out_s = torch.where(x > 0.5, torch.tensor(1.0, device=x.device),
        #                   torch.where(x < -0.5, torch.tensor(-1.0, device=x.device), torch.tensor(0.0, device=x.device)))
        # 反向传播路径：在[-1,1]区间内保持梯度为1
        #out_bp = torch.clamp(x, -1, 1)
        # 直通估计器技巧：前向传播使用离散值，反向传播使用连续梯度
        #return (out_s.float() - out_bp).detach() + out_bp
        x = x.contiguous()
        out_s = torch.clamp(torch.round(x), 0, 2)
        out_bp = torch.clamp(x, 0, 2)
        #print("Ternary")
        return (out_s.float() - out_bp).detach() + out_bp


def gradient_scale(x, scale):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    return y


def mem_update(x_in, mem, V_th, decay, fire_ratio,grad_scale=1., temp=1.0):
    
    x_in = x_in.contiguous()
    mem = mem.contiguous()
    
    mem = mem * decay + x_in
    #if mem.shape[1]==256:
    #    embed()
    if isinstance(V_th, torch.Tensor):
        V_th_scaled = gradient_scale(V_th, grad_scale)
    else:
        # 如果是标量值，直接使用
        V_th_scaled = V_th
    #V_th = gradient_scale(V_th, grad_scale)
    #mem2 = MPR(mem, 0.5)
    normalized_mem = mem / V_th_scaled
    normalized_mem = normalized_mem.contiguous()

    spike = spike_activation(normalized_mem, binary=False, temp=temp)
    #mem = mem * (1 - torch.abs(spike))
    #mem = mem - spike
    mem = mem * (1 - spike>0.1)
    #spike = spike 
    return mem, spike


class LIFAct(SpikeModule):
    """ Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self, T):
        super(LIFAct, self).__init__()
        self.step = T
        #self.V_th = nn.Parameter(torch.tensor(1.0))
        self.V_th = 1.0
        # self.tau = nn.Parameter(torch.tensor(-1.1))
        self.temp = 3.0
        #self.temp = nn.Parameter(torch.tensor(1.))
        self.grad_scale = None
        
        #self.fire_ratio = nn.Parameter(torch.ones(1,512,1,1), requires_grad=True)
        #self.fire_ratio = nn.Parameter(torch.tensor(1.))
        self.fire_ratio = 1
        self.learnable_threshold = nn.Parameter(torch.tensor(1.0))
    
    def get_threshold(self):
        """获取当前阈值，可结合固定值和可学习值"""
        return self.V_th * self.learnable_threshold.item()

    def forward(self, x):
        
        if self._spiking is not True:
            return F.relu(x)
        
        # adjust the size of x (B*T,C,H,W) to (B,T,C,H,W)
        x = x.view(self.step, -1, *x.shape[1:])
        
        if self.grad_scale is None:
            self.grad_scale = 1 / math.sqrt(x[1].numel()*self.step)
        mem = 0
        u = torch.zeros_like(x[0])
        out = []
        T, B, C, H, W = x.shape
        current_threshold = self.get_threshold()
        for i in range(self.step):
        
            u, out_i = mem_update(x_in=x[i], mem=u, V_th=current_threshold,fire_ratio=self.fire_ratio,
                                  grad_scale=self.grad_scale, decay=1.0, temp=self.temp)
            out += [out_i]
        out = torch.stack(out)
        return out.view(-1, *out.shape[2:])


class LIFAct_nolearned_th(SpikeModule):
    """ Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self, T):
        super(LIFAct_nolearned_th, self).__init__()
        self.step = T
        # self.V_th = nn.Parameter(torch.tensor(1.0))
        self.V_th = 1.0
        # self.tau = nn.Parameter(torch.tensor(-1.1))
        self.temp = 3.0
        # self.temp = nn.Parameter(torch.tensor(1.))
        self.grad_scale = None

        # self.fire_ratio = nn.Parameter(torch.ones(1,512,1,1), requires_grad=True)
        # self.fire_ratio = nn.Parameter(torch.tensor(1.))
        self.fire_ratio = 1
        # self.V_th = 1.0

    # def get_threshold(self):
    #     """获取当前阈值，可结合固定值和可学习值"""
    #     return self.V_th * self.learnable_threshold.item()

    def forward(self, x):

        if self._spiking is not True:
            return F.relu(x)

        # adjust the size of x (B*T,C,H,W) to (B,T,C,H,W)
        x = x.view(self.step, -1, *x.shape[1:])

        if self.grad_scale is None:
            self.grad_scale = 1 / math.sqrt(x[1].numel() * self.step)
        mem = 0
        u = torch.zeros_like(x[0])
        out = []
        T, B, C, H, W = x.shape
        # current_threshold = self.get_threshold()
        for i in range(self.step):
            u, out_i = mem_update(x_in=x[i], mem=u, V_th=self.V_th, fire_ratio=self.fire_ratio,
                                  grad_scale=self.grad_scale, decay=1.0, temp=self.temp)
            out += [out_i]
        out = torch.stack(out)
        return out.view(-1, *out.shape[2:])



class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, others) = ctx.saved_tensors
        gama = others[0].item()

        grad_input = grad_output.clone()
        tmp = ((1 / gama) * (1 / gama) * ((gama - input.abs())).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class DSPIKE(nn.Module):
    def __init__(self, region=1.0):
        super(DSPIKE, self).__init__()
        self.region = region

    def forward(self, x, temp):
        out_bpn = torch.clamp(x, -self.region, self.region)
        out_bp = (torch.tanh(temp * out_bpn)) / \
                 (2 * np.tanh(self.region * temp)) + 0.5
        out_s = (x >= 0).float()
        return (out_s.float() - out_bp).detach() + out_bp

class BDspike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        out = (input >= 0).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, others) = ctx.saved_tensors
        gama = others[0].item()
        gamma1 = 1.0
        gamma2 = 2.5
        r = 2.0
        grad_input = grad_output.clone()
        # tmp1 = ((1 / gamma1) * (1 / gamma1) * ((gamma1 - input.abs())).clamp(min=0))
        # tmp2 = ((1 / gamma2) * (1 / gamma2) * ((gamma2 - input.abs())).clamp(min=0))
        # sigma = 1/4*(tmp1**2 - tmp2**2)
        # tmp = (tmp1 * tmp2 - sigma)**0.5
        out_bpn = torch.clamp(input, -r, r)
        k1 = 1/(2 * np.tanh(r * gamma1))
        k2 = 1/(2 * np.tanh(r * gamma2))
        tmp1 = gamma1 * k1 * (1 - torch.tanh(gamma1 * out_bpn)**2)
        tmp2 = gamma2 * k2 * (1 - torch.tanh(gamma2 * out_bpn)**2)
        # sigma = 0.25 * ((tmp1+tmp2) ** 2 - (tmp1-tmp2) ** 2)+1e-8
        tmp = tmp2 * 1.2 - tmp1 * 0.2
        # tmp = (tmp1 + tmp2) / 2
        # tmp = tmp2
        grad_input = grad_input * tmp
        return grad_input, None

class ZDSPIKE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, others) = ctx.saved_tensors
        gama = others[0].item()
        region = 1.0
        k = 1 / (2 * np.tanh(gama * region))
        grad_input = grad_output.clone()
        m = torch.tanh(gama * input)
        tmp = k * (1 - m * m) * gama
        tmp[input > region] = 0
        tmp[input < -region] = 0
        grad_input = grad_input * tmp
        return grad_input, None

class LIFSpike(nn.Module):
    def __init__(self, T=1, thresh=1.0, tau=0.5, gamma=2.5, use_ann=False):
        super(LIFSpike, self).__init__()
        self.use_ann = use_ann
        self.act_ann = nn.ReLU()
        # self.snn_act = DSPIKE(region=1.0)
        self.snn_act = ZDSPIKE.apply
        self.T = T # time steps
        self.thresh = thresh
        self.tau = tau
        self.gamma = gamma
        self.mem_detach = False

    def forward(self, x):
        if self.use_ann:
            return self.act_ann(x)
        else:
            if self.T == 1:
                spike =  self.snn_act(x - self.thresh, self.gamma)
                # TODO: maybe compute the KL divergence between the two distributions
                return spike
            else:
                # adjust the size of x (B*T,C,H,W) to (B,T,C,H,W)
                x = x.view(self.T, -1, *x.shape[1:])
                mem = 0
                spikes = torch.zeros_like(x)
                for t in range(self.T):
                    # calculate the membrane potential
                    mem = mem * self.tau + x[t,...]
                    spike = self.snn_act(mem - self.thresh, self.gamma)
                    spikes[t,...] = spike
                    if self.mem_detach:
                        mem = mem * (1 - spike.detach())
                    else:
                        mem = mem * (1 - spike)
                # adjust the size of spikes (B,T,C,H,W) to (B*T,C,H,W)
                return spikes.view(-1, *spikes.shape[2:])

    def set_ANN(self, act: bool):
        self.use_ann = act

    def set_mem_detach(self, mem_detach):
        self.mem_detach = mem_detach

class singleLIF(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gamma=2.5):
        super(singleLIF, self).__init__()
        self.act = DSPIKE(region=1.0)
        self.thresh = thresh
        self.tau = tau
        self.gamma = gamma
        self.mem = 0

    def forward(self, x):
        self.mem = self.mem * self.tau + x
        spike = self.act(self.mem - self.thresh, self.gamma)
        self.mem = self.mem * (1 - spike)
        return spike

    def reset(self):
        self.mem = 0


class LIAFSpike(nn.Module):
    def __init__(self, T=1, thresh=1.0, tau=0.5):
        super(LIAFSpike, self).__init__()
        self.use_ann = False
        self.act_ann = nn.ReLU()
        # self.snn_act = BDspike.apply
        self.T = T # time steps
        self.thresh = thresh
        self.tau = tau
        self.mem_detach = False

    def forward(self, x):
        if self.T == 1:
            return self.act_ann(x - self.thresh)
        else:
            # adjust the size of x (B*T,C,H,W) to (B,T,C,H,W)
            x = x.view(self.T, -1, *x.shape[1:])
            mem = 0
            outs = torch.zeros_like(x)
            for t in range(self.T):
                # calculate the membrane potential
                mem = mem * self.tau + x[t,...]
                out = self.act_ann(mem - self.thresh)
                spike = (mem >= self.thresh).float().detach()
                outs[t,...] = out
                mem = mem * (1 - spike)
            # adjust the size of spikes (B,T,C,H,W) to (B*T,C,H,W)
            return outs.view(-1, *outs.shape[2:])

    def set_ANN(self, act: bool):
        self.use_ann = act

    def set_mem_detach(self, mem_detach):
        self.mem_detach = mem_detach

class ExpandTime(nn.Module):
    def __init__(self, T=1):
        super(ExpandTime, self).__init__()
        self.T = T

    def forward(self, x):
        x_seq = x[None,:,:,:,:]
        x_seq = x_seq.repeat(self.T, 1 , 1, 1, 1)
        # adjust the size of spikes (B,T,C,H,W) to (B*T,C,H,W)
        return x_seq.view(-1, *x_seq.shape[2:])

class RateEncoding(nn.Module):
    def __init__(self, T=1):
        super(RateEncoding, self).__init__()
        self.T = T

    def forward(self, x):
        x_seq = x[None, :, :, :, :]
        x_seq = x_seq.repeat(self.T, 1, 1, 1, 1)
        x_seq = x_seq.view(-1, *x_seq.shape[2:])
        # poision noise
        noise = torch.randn_like(x_seq)
        x_seq = (x_seq>=noise).float().detach()
        return x_seq

if __name__ == '__main__':
    x = torch.randn(2,3,8,8)
    expt = ExpandTime(T=2)
    y = expt(x)
    print(y.shape)



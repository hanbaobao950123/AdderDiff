import copy
import json
import os
import warnings

import torch
from absl import app, flags
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet
from score.both import get_inception_and_fid_score, get_inception_and_fid_score_fromdataset
import torchvision
import torch.nn as nn
from spiking_layer import LIFSpike, ExpandTime, RateEncoding, singleLIF, LIFAct, LIFAct_nolearned_th
from model import Swish, TimeEmbedding, MembraneOutputLayer, AttnBlock
from collections import defaultdict

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
flags.DEFINE_integer('img_ch',3, help='image channel')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_integer('Timestep', 2, help='total spiking model steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_bool('resume', True, help="load pre-trained model")
flags.DEFINE_string('resume_model',"/home/user/guoyf/pytorch-ddpm-myspike2/logs/DDPM_CIFAR10_EPS/ckpt.pt", help='resume model path')
flags.DEFINE_string('dataset', 'cifar10', help='dataset')
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_resume', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 5000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

device = torch.device('cuda:0')


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):#
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, FLAGS.img_ch, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    if FLAGS.dataset in ['fashion-mnist', 'mnist']:
        (IS, IS_std), FID = get_inception_and_fid_score_fromdataset(
            images, FLAGS.dataset, num_images=FLAGS.num_images,
            use_torch=FLAGS.fid_use_torch, verbose=True)    
    else:
        (IS, IS_std), FID = get_inception_and_fid_score(
            images, FLAGS.fid_cache, num_images=FLAGS.num_images,
            use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def train():
    # dataset
    if FLAGS.dataset == 'cifar10':
        dataset = CIFAR10(
            root='./data', train=True, download=False,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        
    elif FLAGS.dataset == 'celeba':
        SetRange = torchvision.transforms.Lambda(lambda X: 2 * X - 1.)
        transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.CenterCrop(148),
        torchvision.transforms.Resize((64,64)),
        torchvision.transforms.ToTensor(),
        SetRange])
        dataset = torchvision.datasets.ImageFolder(root='./data/CelebA/CelebA_nocrop/', 
                                                   transform=transform)
        
    elif FLAGS.dataset == 'fashion-mnist':
        SetRange = torchvision.transforms.Lambda(lambda X: 2 * X - 1.)
        transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor(),
        SetRange])
        dataset = torchvision.datasets.FashionMNIST(root='./data/', 
                                            train=True, 
                                            download=True, 
                                            transform=transform)
    
    elif FLAGS.dataset == 'mnist':
        SetRange = torchvision.transforms.Lambda(lambda X: 2 * X - 1.)
        transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor(),
        SetRange])
        dataset = torchvision.datasets.MNIST(root='./data/', 
                                            train=True, 
                                            download=True, 
                                            transform=transform)
        
    #elif FLAGS.dataset == 'lsun':
        #dataset = LSUNBed()
    else:
        raise NotImplementedError
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # model setup
    net_model = UNet(
        T=FLAGS.T, Timestep=FLAGS.Timestep, img_ch=FLAGS.img_ch, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    if FLAGS.resume:
        ckpt = torch.load(os.path.join(FLAGS.resume_model))
        print(f'Loading Resume model from {FLAGS.resume_model}')
        net_model.load_state_dict(ckpt['net_model'], strict=True)
    else:
        print('Training from scratch')

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'), exist_ok=True)
    x_T = torch.randn(FLAGS.sample_size, FLAGS.img_ch, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
    writer = SummaryWriter(FLAGS.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()
    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # sample
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))

            # evaluate
            if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                    'IS_EMA': ema_IS[0],
                    'IS_std_EMA': ema_IS[1],
                    'FID_EMA': ema_FID
                }
                pbar.write(
                    "%d/%d " % (step, FLAGS.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()

class SpikeRateCounter:
    def __init__(self, timestep):
        self.timestep = timestep  # SNN的时间步（对应FLAGS.Timestep）
        self.spike_stats = defaultdict(dict)  # 存储每层的统计结果
        self.hooks = []
        self.layer_idx = 0  # 给LIF层编号

    def _lif_spike_hook(self, module, input, output):
        """Hook函数：捕获LIF层脉冲，统计总脉冲数和神经元数"""
        # 1. 确定层名
        layer_name = f"LIFLayer_{self.layer_idx}"
        self.layer_idx += 1

        # 2. 拆分时间步(T)、批次(B)
        T = self.timestep
        B = output.shape[0] // T
        spike_tensor = output.view(T, B, *output.shape[1:])  # (T, B, C, H, W)

        # 3. 统计总脉冲数和神经元数
        # total_spikes = torch.sum(spike_tensor).item()
        total_spikes = torch.count_nonzero(spike_tensor).item()
        C, H, W = spike_tensor.shape[2], spike_tensor.shape[3], spike_tensor.shape[4]
        neuron_num = C * H * W  # 单样本神经元数

        # 4. 存储结果
        self.spike_stats[layer_name]["total_spikes"] = total_spikes
        self.spike_stats[layer_name]["neuron_num"] = neuron_num
        self.spike_stats[layer_name]["batch_size"] = B
        self.spike_stats[layer_name]["timestep"] = T

    def _register_hook(self, model):
        """递归注册LIF层Hook"""
        for module in model.modules():
            if isinstance(module, LIFAct):  # 你的LIF脉冲层类名
                self.hooks.append(module.register_forward_hook(self._lif_spike_hook))

    def calculate_spike_rate(self, model, input_tensor, t_tensor):
        """
        计算每层SpikeRate + 所有层的平均SpikeRate
        返回：
        - layer_spike_rates: 每层发放率字典
        - global_spike_rate: 全局（按总脉冲/总神经元）发放率
        - avg_layer_spike_rate: 所有层的平均发放率（算术平均）
        """
        # 重置统计
        self.spike_stats.clear()
        self.layer_idx = 0
        self._register_hook(model)

        # 前向传播捕获脉冲
        with torch.no_grad():
            model(input_tensor, t_tensor)

        # 移除Hook
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        # 计算每层SpikeRate + 收集所有层的发放率
        layer_spike_rates = {}
        layer_raw_rates = []  # 存储所有层的原始发放率
        layer_normalized_rates = []  # 存储所有层的归一化发放率

        for layer_name, stats in self.spike_stats.items():
            total_spikes = stats["total_spikes"]
            neuron_num = stats["neuron_num"]
            batch_size = stats["batch_size"]
            timestep = stats["timestep"]

            # 计算每层发放率
            raw_rate = total_spikes / (neuron_num * batch_size)  # 原始发放率
            normalized_rate = raw_rate / timestep  # 归一化发放率（0~1）

            # 存储每层结果
            layer_spike_rates[layer_name] = {
                "raw": raw_rate,
                "normalized": normalized_rate
            }

            # 收集到列表，用于后续层平均
            layer_raw_rates.append(raw_rate)
            layer_normalized_rates.append(normalized_rate)

        # 计算全局发放率（按总脉冲/总神经元）
        total_all_spikes = sum([s["total_spikes"] for s in self.spike_stats.values()])
        total_all_neurons = sum([s["neuron_num"] * s["batch_size"] for s in self.spike_stats.values()])
        global_raw = total_all_spikes / total_all_neurons if total_all_neurons > 0 else 0
        global_normalized = global_raw / self.timestep if self.timestep > 0 else 0
        global_spike_rate = {
            "raw": global_raw,
            "normalized": global_normalized
        }

        # 计算所有层的平均发放率（核心新增）
        avg_layer_raw = sum(layer_raw_rates) / len(layer_raw_rates) if layer_raw_rates else 0
        avg_layer_normalized = sum(layer_normalized_rates) / len(layer_normalized_rates) if layer_normalized_rates else 0
        avg_layer_spike_rate = {
            "raw": avg_layer_raw,
            "normalized": avg_layer_normalized
        }

        return layer_spike_rates, global_spike_rate, avg_layer_spike_rate

def eval():
    # model setup
    # load model and evaluate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    model = UNet(
        T=FLAGS.T, Timestep=FLAGS.Timestep, img_ch=FLAGS.img_ch, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    model.load_state_dict(ckpt['net_model'])
    model = model.to(device)
    # ---------------------- 核心：统计模型总MAC数 ----------------------
    # 构造与实际推理一致的输入张量（batch_size=1 避免显存占用过高）
    batch_size = 1
    input_tensor = torch.randn(
        batch_size, FLAGS.img_ch, FLAGS.img_size, FLAGS.img_size
    ).to(device)  # (B, C, H, W) 匹配FLAGS配置
    t_tensor = torch.randint(FLAGS.T, (batch_size,)).to(device)  # 时间步张量

    # 实例化MAC计数器并统计
    mac_counter = MACCounter()
    total_mac = mac_counter.count(model, input_tensor, t_tensor)

    # 转换为易读单位（1 GMAC = 10^9 MAC）
    total_gmac = total_mac / 1e9
    print(f"===== Model MAC Statistics =====")
    print(f"Total MAC (per sample): {total_mac:.2e}")
    print(f"Total GMAC (per sample): {total_gmac:.2f} G")
    print(f"===============================\n")
    # 计算脉冲发放率（新增层平均返回值）
    spike_counter = SpikeRateCounter(timestep=FLAGS.Timestep)
    layer_spike_rates, global_spike_rate, avg_layer_spike_rate = spike_counter.calculate_spike_rate(model, input_tensor,
                                                                                                    t_tensor)

    # 输出结果（补充层平均）
    print(f"===== Model SpikeRate Statistics =====")
    # 1. 输出每层发放率
    for layer_name, rates in layer_spike_rates.items():
        print(f"{layer_name}:")
        print(f"  原始发放率: {rates['raw']:.4f}")
        print(f"  归一化发放率: {rates['normalized']:.4f}")
    # 2. 输出全局发放率（总脉冲/总神经元）
    print(f"\nGlobal SpikeRate (Total Spikes / Total Neurons):")
    print(f"  原始全局发放率: {global_spike_rate['raw']:.4f}")
    print(f"  归一化全局发放率: {global_spike_rate['normalized']:.4f}")
    # 3. 输出所有层的平均发放率（核心新增）
    print(f"\nAverage Layer SpikeRate (Arithmetic Mean of All Layers):")
    print(f"  原始层平均发放率: {avg_layer_spike_rate['raw']:.4f}")
    print(f"  归一化层平均发放率: {avg_layer_spike_rate['normalized']:.4f}")
    print(f"======================================\n")
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples_ema.png'),padding=0,
        nrow=16)


class MACCounter:
    def __init__(self):
        self.total_mac = 0  # 总MAC数
        self.hooks = []  # 存储注册的hook

    def _conv2d_mac(self, module, input, output):
        """计算Conv2d层的MAC"""
        # input: (B, C_in, H_in, W_in), output: (B, C_out, H_out, W_out)
        B, C_in, H_in, W_in = input[0].shape
        C_out, _, K_h, K_w = module.weight.shape
        H_out, W_out = output.shape[2], output.shape[3]
        groups = module.groups

        # 计算per-sample MAC（除以B，只统计单样本）
        mac = (K_h * K_w * C_in // groups) * C_out * H_out * W_out
        self.total_mac += mac / B

    def _linear_mac(self, module, input, output):
        """计算Linear层的MAC"""
        # input: (B, C_in), output: (B, C_out)
        B, C_in = input[0].shape
        C_out = output.shape[1]

        # per-sample MAC
        mac = C_in * C_out
        self.total_mac += mac / B

    def _attn_block_mac(self, module, input, output):
        """计算自定义AttnBlock的MAC（适配你的AttnBlock实现）"""
        # 解析AttnBlock内的Conv2d（proj_q/proj_k）
        B, C, H, W = input[0].shape
        # proj_q: Conv2d(C,C,1)
        mac_q = (1 * 1 * C * C * H * W)  # 1x1卷积，groups=1
        # proj_k: Conv2d(C,C,1)
        mac_k = (1 * 1 * C * C * H * W)
        # q求和+乘k的操作：无额外MAC（逐元素乘，无乘加）
        total_attn_mac = mac_q + mac_k

        # per-sample MAC
        self.total_mac += total_attn_mac / B

    def _register_hook(self, module):
        """为不同层注册对应的MAC计算hook"""
        # 跳过无参数的层（激活、归一化、采样插值等）
        if isinstance(module, (nn.Identity, nn.BatchNorm2d, nn.GroupNorm,
                               LIFAct, Swish, ExpandTime, MembraneOutputLayer)):
            return

        # 卷积层
        elif isinstance(module, nn.Conv2d):
            self.hooks.append(module.register_forward_hook(self._conv2d_mac))

        # 线性层
        elif isinstance(module, nn.Linear):
            self.hooks.append(module.register_forward_hook(self._linear_mac))

        # 自定义注意力块
        elif isinstance(module, AttnBlock):
            self.hooks.append(module.register_forward_hook(self._attn_block_mac))

        # 递归处理子模块（如ResBlock、Sequential等）
        for child in module.children():
            self._register_hook(child)

    def count(self, model, input_tensor, t_tensor):
        """统计模型MAC主函数"""
        # 重置统计值
        self.total_mac = 0
        # 注册所有层的hook
        self._register_hook(model)
        # 前向传播（触发hook计算MAC）
        with torch.no_grad():
            model(input_tensor, t_tensor)
        # 移除hook（避免重复注册）
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.total_mac


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval()
    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)


'''
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train --flagfile ./config/CIFAR10.txt --parallel
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train --flagfile ./config/CELEBA.txt --parallel
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --train --flagfile ./config/MNIST.txt --parallel
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --train --flagfile ./config/FASHIONMNIST.txt --parallel
CUDA_VISIBLE_DEVICES=0,1 python main.py --train --flagfile ./config/CIFAR10.txt --parallel --resume --resume_model "/home/user/guoyf/pytorch-ddpm-myspike2/logs/DDPM_CIFAR10_shortcut/ckpt.pt"
CUDA_VISIBLE_DEVICES=3,4 python main.py --train --flagfile ./config/CIFAR_10_ATT.txt --parallel --logdir ./logs/DDPM_CIFAR10_att_gnorm --noresume


CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt --notrain --eval --parallel
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --flagfile ./logs/DDPM_CELEBA_EPS/flagfile.txt --notrain --eval --parallel
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --flagfile ./logs/DDPM_MNIST_EPS/flagfile.txt --notrain --eval --parallel
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --flagfile ./logs/DDPM_FASHIONMNIST_EPS/flagfile.txt --notrain --eval --parallel
'''
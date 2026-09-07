import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch.nn.utils.parametrizations import spectral_norm
from utils.pytorch import init_weights
from torch.utils.checkpoint import checkpoint


def grad_penalty(dis_network, x_real, x_fake):
    device = x_real.device
    batch_size = x_real.size(0)

    alpha = torch.rand((batch_size, 1, 1, 1), device=device).expand_as(x_real)
    x_inter = (alpha * x_real + (1 - alpha) * x_fake).requires_grad_(True)
    r = dis_network.forward(x_inter)

    grad = autograd.grad(
        outputs=r,
        inputs=x_inter,
        grad_outputs=torch.ones_like(r, device=device),
        retain_graph=True,
        create_graph=True,
        only_inputs=True
    )[0]
    grad = grad.view(grad.size(0), -1)
    grad_norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    out = torch.mean((grad_norm - 1) ** 2)
    return out


class SpectralConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpectralConv2d, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(*args, **kwargs))

    def forward(self, x):
        return self.conv(x)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channels, info_channels, channels, kernel_size, is_spec=False):
        super(AdaptiveInstanceNorm, self).__init__()

        conv = SpectralConv2d if is_spec else nn.Conv2d

        self.norm = nn.InstanceNorm2d(channels)
        self.conv_info = nn.Sequential(
            conv(info_channels, channels, kernel_size=kernel_size, padding=(kernel_size // 2)),
            nn.SiLU(),
        )
        self.conv_gamma = conv(channels, in_channels, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.conv_beta = conv(channels, in_channels, kernel_size=kernel_size, padding=(kernel_size // 2))

    def forward(self, input_tuple):
        x, info = input_tuple
        x = self.norm(x)

        info_reshape = F.interpolate(info, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        info_reshape = self.conv_info(info_reshape)
        info_gamma = self.conv_gamma(info_reshape)
        info_beta = self.conv_beta(info_reshape)

        x = info_gamma * x + info_beta
        return x, info


class FPA(nn.Module):
    def __init__(self, in_channels, out_channels, is_norm=True, is_spec=False):
        super(FPA, self).__init__()

        conv = SpectralConv2d if is_spec else nn.Conv2d

        self.glob = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv(in_channels, out_channels, 1, stride=1, padding=0),
        )

        self.down2_1 = nn.Sequential(
            conv(in_channels, in_channels, 5, stride=2, padding=2),
            nn.InstanceNorm2d(in_channels, affine=True) if is_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.down2_2 = nn.Sequential(
            conv(in_channels, out_channels, 5, stride=1, padding=2),
            nn.InstanceNorm2d(out_channels, affine=True) if is_norm else nn.Identity(),
            nn.SiLU(),
        )

        self.down3_1 = nn.Sequential(
            conv(in_channels, in_channels, 3, stride=2, padding=1),
            nn.InstanceNorm2d(in_channels, affine=True) if is_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.down3_2 = nn.Sequential(
            conv(in_channels, out_channels, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True) if is_norm else nn.Identity(),
            nn.SiLU(),
        )

        self.conv = nn.Sequential(
            conv(in_channels, out_channels, 1, stride=1, padding=0),
            nn.InstanceNorm2d(out_channels, affine=True) if is_norm else nn.Identity(),
            nn.SiLU(),
        )

    def forward(self, x):
        x_glob = self.glob(x)
        x_glob = F.interpolate(x_glob, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)

        d2 = self.down2_1(x)
        d3 = self.down3_1(d2)

        d2 = self.down2_2(d2)
        d3 = self.down3_2(d3)

        d3 = F.interpolate(d3, size=(d2.size(2), d2.size(3)), mode="bilinear", align_corners=False)
        d2 = d2 + d3

        d2 = F.interpolate(d2, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = x * d2

        x = x + x_glob
        return x


class CBAM(nn.Module):
    def __init__(self, in_channels, is_spec=False, kernel_size=7, reduction=4, dilation=1):
        super(CBAM, self).__init__()

        conv = SpectralConv2d if is_spec else nn.Conv2d

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)

        channels = in_channels // reduction
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, channels),
            nn.SiLU(),
            nn.Linear(channels, in_channels),
        )

        self.conv = nn.Sequential(
            conv(2, 1, kernel_size, stride=1, padding=((kernel_size // 2) * dilation), dilation=dilation),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.global_avgpool(x).view(x.size(0), -1)
        m = self.global_maxpool(x).view(x.size(0), -1)

        cmap = self.mlp(a) + self.mlp(m)
        cmap = self.sigmoid(cmap)
        cmap = cmap.view(cmap.size(0), cmap.size(1), 1, 1)
        c = cmap * x

        a = torch.mean(x, 1).unsqueeze(1)
        m = torch.max(x, 1)[0].unsqueeze(1)

        smap = self.sigmoid(self.conv(torch.cat((a, m), dim=1)))
        s = smap * c
        return s


class GCT(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


class SelfAttention(nn.Module):
    def __init__(self, in_channels, is_spec=False):
        super(SelfAttention, self).__init__()

        conv = SpectralConv2d if is_spec else nn.Conv2d

        self.in_channels = in_channels
        self.snconv1x1_theta = conv(in_channels=in_channels, out_channels=in_channels//16, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = conv(in_channels=in_channels, out_channels=in_channels//16, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = conv(in_channels=in_channels, out_channels=in_channels//4, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = conv(in_channels=in_channels//4, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//16, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//16, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//4, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//4, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out


class DecodeBlock(nn.Module):
    def __init__(self, ui_channels, xi_channels, out_channels, is_norm=True, is_spec=False, is_cbam=True, cbam_kernel=7):
        super(DecodeBlock, self).__init__()

        conv = SpectralConv2d if is_spec else nn.Conv2d

        self.u_conv = nn.Identity()
        self.x_attn = CBAM(xi_channels, is_spec, cbam_kernel, 1, 1) if is_cbam else nn.Identity()
        # self.x_attn = GCT(xi_channels) if is_cbam else nn.Identity()
        self.network = nn.Sequential(
            conv(ui_channels + xi_channels, out_channels, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True) if is_norm else nn.Identity(),
            nn.SiLU(),
            conv(out_channels, out_channels, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True) if is_norm else nn.Identity(),
            nn.SiLU(),
        )

    def forward(self, u, x):
        u = self.u_conv(u)
        x = self.x_attn(x)

        if u.size(2) != x.size(2) or u.size(3) != x.size(3):
            u = F.interpolate(u, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)

        u = torch.cat([u, x], dim=1)
        out = self.network(u)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, is_norm=True, is_spec=False):
        super(ResBlock, self).__init__()

        conv = SpectralConv2d if is_spec else nn.Conv2d

        self.stride = stride

        if stride == 3:
            self.pool = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, (4, 1), stride=(2, 1), padding=(1, 0), groups=in_channels),
                nn.InstanceNorm2d(in_channels) if is_norm else nn.Identity(),
                nn.SiLU(),
            )
        elif stride == 2:
            self.pool = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 4, stride=2, padding=1, groups=in_channels),
                nn.InstanceNorm2d(in_channels) if is_norm else nn.Identity(),
                nn.SiLU(),
            )
        else:
            self.pool = nn.Identity()

        self.bypass = nn.Sequential(
            conv(in_channels, out_channels, 1, stride=1, padding=0, groups=2) if (in_channels != out_channels) else nn.Identity()
        )

        self.residual = nn.Sequential(
            conv(in_channels, out_channels, 3, stride=1, padding=1, groups=2),
            nn.InstanceNorm2d(out_channels, affine=True) if is_norm else nn.Identity(),
            nn.SiLU(),
            conv(out_channels, out_channels, 3, stride=1, padding=1, groups=2),
        )

        self.last = nn.Sequential(
            nn.InstanceNorm2d(out_channels, affine=True) if is_norm else nn.Identity(),
            nn.SiLU(),
        )

    def forward(self, x):
        w = self.pool(x)

        bypass = self.bypass(w)
        residual = self.residual(w)

        out = self.last(bypass + residual)
        return out


class UNetGenerator(nn.Module):
    def __init__(self, in_channels, channels, out_channels, is_norm=True, is_spec=False, is_cbam=True):
        super(UNetGenerator, self).__init__()

        conv = SpectralConv2d if is_spec else nn.Conv2d

        self.conv = nn.Sequential(
            conv(in_channels, channels, 3, stride=1, padding=1),
            nn.SiLU(),
            conv(channels, channels, 3, stride=1, padding=1, groups=2),
            nn.InstanceNorm2d(channels, affine=True) if is_norm else nn.Identity(),
            nn.SiLU(),
        )

        self.encode2 = nn.Sequential(
            ResBlock(1 * channels, 1 * channels, is_norm=is_norm, is_spec=is_spec),
        )
        self.encode3 = nn.Sequential(
            ResBlock(1 * channels, 2 * channels, stride=2, is_norm=is_norm, is_spec=is_spec),
        )
        self.encode4 = nn.Sequential(
            ResBlock(2 * channels, 4 * channels, stride=2, is_norm=is_norm, is_spec=is_spec),
        )
        self.encode5 = nn.Sequential(
            ResBlock(4 * channels, 8 * channels, stride=3, is_norm=is_norm, is_spec=is_spec),
        )

        self.center = nn.Sequential(
            FPA(8 * channels, 8 * channels, is_norm=is_norm, is_spec=is_spec),
            nn.Conv2d(8 * channels, 8 * channels, (4, 1), stride=(2, 1), padding=(1, 0), groups=8 * channels),
            nn.InstanceNorm2d(8 * channels) if is_norm else nn.Identity(),
            nn.SiLU(),
        )

        self.decode5 = DecodeBlock(8 * channels, 8 * channels, 4 * channels, is_norm=is_norm, is_spec=is_spec, is_cbam=is_cbam, cbam_kernel=9)
        self.decode4 = DecodeBlock(4 * channels, 4 * channels, 2 * channels, is_norm=is_norm, is_spec=is_spec, is_cbam=is_cbam, cbam_kernel=7)
        self.decode3 = DecodeBlock(2 * channels, 2 * channels, 1 * channels, is_norm=is_norm, is_spec=is_spec, is_cbam=is_cbam, cbam_kernel=5)
        self.decode2 = DecodeBlock(1 * channels, 1 * channels, 1 * channels, is_norm=is_norm, is_spec=is_spec, is_cbam=is_cbam, cbam_kernel=5)

        self.logit = nn.Sequential(
            # conv(4 * channels, 4 * channels, 1, stride=1, padding=0),
            # nn.SiLU(),
            conv(8 * channels, out_channels, 1, stride=1, padding=0),
        )

    def disable_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def enable_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward_normal(self, x, encode_only=False, extract_features=False):
        e1 = self.conv(x)
        e2 = self.encode2(e1)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3)
        e5 = self.encode5(e4)

        if encode_only:
            return [e2, e3, e4, e5]

        f = self.center(e5)

        d5 = self.decode5(f, e5)
        d4 = self.decode4(d5, e4)
        d3 = self.decode3(d4, e3)
        d2 = self.decode2(d3, e2)

        output_shape = (d2.size(2), d2.size(3))
        f = torch.cat(
            (
                d2,
                F.interpolate(d3, size=output_shape, mode="bilinear", align_corners=False),
                F.interpolate(d4, size=output_shape, mode="bilinear", align_corners=False),
                F.interpolate(d5, size=output_shape, mode="bilinear", align_corners=False),
            ),
            dim=1,
        )

        logit = self.logit(f)
        if extract_features:
            return logit, [e2, e3, e4, e5]
        else:
            return logit

    def forward_check(self, x, encode_only=False, extract_features=False):
        e1 = self.conv(x)
        e2 = checkpoint(self.encode2, e1, use_reentrant=False)
        e3 = checkpoint(self.encode3, e2, use_reentrant=False)
        e4 = checkpoint(self.encode4, e3, use_reentrant=False)
        e5 = checkpoint(self.encode5, e4, use_reentrant=False)

        if encode_only:
            return [e2, e3, e4, e5]

        f = checkpoint(self.center, e5, use_reentrant=False)

        d5 = checkpoint(self.decode5, f, e5, use_reentrant=False)
        d4 = checkpoint(self.decode4, d5, e4, use_reentrant=False)
        d3 = checkpoint(self.decode3, d4, e3, use_reentrant=False)
        d2 = checkpoint(self.decode2, d3, e2, use_reentrant=False)

        output_shape = (d2.size(2), d2.size(3))
        f = torch.cat(
            (
                d2,
                F.interpolate(d3, size=output_shape, mode="bilinear", align_corners=False),
                F.interpolate(d4, size=output_shape, mode="bilinear", align_corners=False),
                F.interpolate(d5, size=output_shape, mode="bilinear", align_corners=False),
            ),
            dim=1,
        )

        logit = self.logit(f)
        if extract_features:
            return logit, [e2, e3, e4, e5]
        else:
            return logit

    def forward(self, x, encode_only=False, extract_features=False, is_check=False):
        if is_check:
            return self.forward_check(x, encode_only=encode_only, extract_features=extract_features)
        else:
            return self.forward_normal(x, encode_only=encode_only, extract_features=extract_features)


class PatchSampleF(nn.Module):
    def __init__(self, base_channels, out_channels=256, num_patches=256):
        super().__init__()
        self.num_patches = num_patches

        self.channels_list = [
            1 * base_channels // 2,
            2 * base_channels // 2,
            4 * base_channels // 2,
            8 * base_channels // 2,
        ]

        self.mlps = nn.ModuleList()
        for ch in self.channels_list:
            mlp = nn.Sequential(
                nn.Linear(ch, 256),
                nn.SiLU(),
                nn.Linear(256, out_channels)
            )
            self.mlps.append(mlp)

    def forward(self, features, patch_ids=None):
        return_ids = []
        result = []

        for i, (feat, mlp) in enumerate(zip(features, self.mlps)):
            B, C, H, W = feat.shape

            feat_reshape = feat[:, :C // 2, :, :].view(B, C // 2, -1).permute(0, 2, 1)

            if patch_ids is not None:
                patch_id = patch_ids[i]
            else:
                patch_id = torch.randperm(feat_reshape.shape[1], device=feat.device)[:self.num_patches]
                return_ids.append(patch_id)

            x_sample = feat_reshape[:, patch_id, :]
            x_proj = mlp(x_sample)
            result.append(x_proj)

        if patch_ids is None:
            return result, return_ids
        else:
            return result


class ConvDiscriminator(nn.Module):
    def __init__(self, in_channels, channels, is_norm=True, is_spec=False, is_cbam=True):
        super(ConvDiscriminator, self).__init__()

        conv = SpectralConv2d if is_spec else nn.Conv2d

        self.first = nn.Sequential(
            conv(in_channels, channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.conv1 = nn.Sequential(
            conv(1 * channels, 2 * channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(2 * channels) if is_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            conv(2 * channels, 2 * channels, 3, stride=1, padding=1),
            nn.InstanceNorm2d(2 * channels) if is_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            # GCT(2 * channels) if is_cbam else nn.Identity(),
            CBAM(2 * channels, is_spec, 7, 1, 1) if is_cbam else nn.Identity(),
        )

        self.conv2 = nn.Sequential(
            conv(2 * channels, 4 * channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(4 * channels) if is_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            conv(4 * channels, 4 * channels, 3, stride=1, padding=1),
            nn.InstanceNorm2d(4 * channels) if is_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            # GCT(4 * channels) if is_cbam else nn.Identity(),
            CBAM(4 * channels, is_spec, 5, 1, 1) if is_cbam else nn.Identity(),
        )

        self.conv3 = nn.Sequential(
            conv(4 * channels, 8 * channels, (4, 3), stride=(2, 1), padding=1),
            nn.InstanceNorm2d(8 * channels) if is_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            conv(8 * channels, 8 * channels, 3, stride=1, padding=1),
            nn.InstanceNorm2d(8 * channels) if is_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            # GCT(8 * channels) if is_cbam else nn.Identity(),
            CBAM(8 * channels, is_spec, 3, 1, 1) if is_cbam else nn.Identity(),
        )

        self.conv4 = nn.Sequential(
            conv(8 * channels, 8 * channels, (4, 3), stride=(2, 1), padding=1),
            nn.InstanceNorm2d(8 * channels) if is_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            conv(8 * channels, 8 * channels, 3, stride=1, padding=1),
            nn.InstanceNorm2d(8 * channels) if is_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
        )

        self.last = nn.Sequential(
            conv(8 * channels, 1, 1, stride=1, padding=0, bias=False),
        )

    def disable_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def enable_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward_normal(self, x):
        w = self.first(x)

        c1 = self.conv1(w)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        r = self.last(c4)
        return r

    def forward_check(self, x):
        w = self.first(x)

        c1 = checkpoint(self.conv1,  w, use_reentrant=False)
        c2 = checkpoint(self.conv2, c1, use_reentrant=False)
        c3 = checkpoint(self.conv3, c2, use_reentrant=False)
        c4 = checkpoint(self.conv4, c3, use_reentrant=False)

        r = self.last(c4)
        return r

    def forward(self, x, is_check=False):
        if is_check:
            return self.forward_check(x)
        else:
            return self.forward_normal(x)


class PatchNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, feat_q, feat_k):
        B, S, C = feat_q.shape

        feat_q = F.normalize(feat_q, dim=-1)
        feat_k = F.normalize(feat_k, dim=-1)

        logits = torch.bmm(feat_q, feat_k.transpose(1, 2)) / self.temperature

        logits = logits.view(B * S, S)

        labels = torch.arange(S, dtype=torch.long, device=feat_q.device)
        labels = labels.unsqueeze(0).expand(B, S).reshape(-1)

        return self.cross_entropy_loss(logits, labels)


def get_gen_model(cfg, from_a, additional_channel=0, **kwargs):
    if from_a:
        model = UNetGenerator(
            cfg.MODEL.A_CHANNELS + additional_channel,
            cfg.MODEL.GEN_CHANNELS,
            cfg.MODEL.OUT_CHANNELS,
            is_norm=cfg.MODEL.GEN_NORM,
            is_spec=cfg.MODEL.GEN_SPEC,
            is_cbam=cfg.MODEL.GEN_CBAM,
        )
    else:
        model = UNetGenerator(
            cfg.MODEL.B_CHANNELS + additional_channel,
            cfg.MODEL.GEN_CHANNELS,
            cfg.MODEL.OUT_CHANNELS,
            is_norm=cfg.MODEL.GEN_NORM,
            is_spec=cfg.MODEL.GEN_SPEC,
            is_cbam=cfg.MODEL.GEN_CBAM
        )
    # load the pre-trained model
    if "PRETRAINED" in cfg.MODEL.keys() and os.path.exists(cfg.MODEL.PRETRAINED) and os.path.isfile(cfg.MODEL.PRETRAINED):
        trained_model = torch.load(cfg.MODEL.PRETRAINED)
        trained_model = {k.replace("module.", ""): v for (k, v) in trained_model.items()}
        model.load_state_dict(trained_model, strict=True)
    return model


def get_patch_net(cfg, **kwargs):
    model = PatchSampleF(
        cfg.MODEL.GEN_CHANNELS,
    )
    # load the pre-trained model
    if "PRETRAINED" in cfg.MODEL.keys() and os.path.exists(cfg.MODEL.PRETRAINED) and os.path.isfile(cfg.MODEL.PRETRAINED):
        trained_model = torch.load(cfg.MODEL.PRETRAINED)
        trained_model = {k.replace("module.", ""): v for (k, v) in trained_model.items()}
        model.load_state_dict(trained_model, strict=True)
    return model


def get_dis_model(cfg, from_a, additional_channel=0, **kwargs):
    if from_a:
        model = ConvDiscriminator(
            cfg.MODEL.A_CHANNELS + additional_channel,
            cfg.MODEL.DIS_CHANNELS,
            is_norm=cfg.MODEL.DIS_NORM,
            is_spec=cfg.MODEL.DIS_SPEC,
            is_cbam=cfg.MODEL.DIS_CBAM,
        )
    else:
        model = ConvDiscriminator(
            cfg.MODEL.B_CHANNELS + additional_channel,
            cfg.MODEL.DIS_CHANNELS,
            is_norm=cfg.MODEL.DIS_NORM,
            is_spec=cfg.MODEL.DIS_SPEC,
            is_cbam=cfg.MODEL.DIS_CBAM,
        )
    # load the pre-trained model
    if "PRETRAINED" in cfg.MODEL.keys() and os.path.exists(cfg.MODEL.PRETRAINED) and os.path.isfile(cfg.MODEL.PRETRAINED):
        trained_model = torch.load(cfg.MODEL.PRETRAINED)
        trained_model = {k.replace("module.", ""): v for (k, v) in trained_model.items()}
        model.load_state_dict(trained_model, strict=True)
    return model


if __name__ == "__main__":
    import numpy as np
    import random

    SEED = 46135141
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(seed=SEED)
    random.seed(SEED)

    batch_size = 4
    gen = UNetGenerator(1, 32, 2, is_norm=True, is_spec=True, is_cbam=True)
    dis = ConvDiscriminator(1, 32, is_norm=False, is_spec=True, is_cbam=True)
    gen.apply(init_weights)
    dis.apply(init_weights)

    x = torch.randn(batch_size, 1, 4000, 72)
    y, features = gen(x, extract_features=True, is_check=False)
    y = y[:, 0: 1] + y[:, 1: 2]
    r = dis(y, is_check=False)
    print(x.shape)
    print(y.shape)
    print(r.shape)
    print()
    for f in features:
        print(f.shape)

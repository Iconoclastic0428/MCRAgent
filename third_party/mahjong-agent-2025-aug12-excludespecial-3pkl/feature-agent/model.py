#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  :   Arthals
# @File    :   model.py
# @Time    :   2024/06/30 18:40:46
# @Contact :   zhuozhiyongde@126.com
# @Software:   Visual Studio Code

"""
model.py: 用于定义神经网络模型
"""

import torch
import torch.nn.functional as F
from torch import nn


class FPNBlock2D(nn.Module):
    def __init__(self, hidden=128):
        nn.Module.__init__(self)
        self._parallel = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden, hidden, 3, 1, 2, dilation=2, bias=False),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(hidden, hidden, 3, 1, 3, dilation=3, bias=False),
                    nn.GELU(),
                ),
            ]
        )
        self._merge = nn.Sequential(
            nn.Conv2d(hidden, hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )

    def forward(self, x):
        y = self._parallel[0](x)
        for branch in self._parallel[1:]:
            y = y + branch(x)
        return self._merge(y)


class FPNBlock1D(nn.Module):
    def __init__(self, hidden=128):
        nn.Module.__init__(self)
        self._parallel = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, 3, 1, 1, bias=False),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, 5, 1, 2, bias=False),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, 7, 1, 3, bias=False),
                    nn.GELU(),
                ),
            ]
        )
        self._merge = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1, 1, 0, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )

    def forward(self, x):
        y = self._parallel[0](x)
        for branch in self._parallel[1:]:
            y = y + branch(x)
        return self._merge(y)


class SlideFPNResidual2D(nn.Module):
    def __init__(self, hidden=128):
        nn.Module.__init__(self)
        self._body = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        return x + self._body(x)


class SlideFPNResidual1D(nn.Module):
    def __init__(self, hidden=128):
        nn.Module.__init__(self)
        self._body = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        return x + self._body(x)


class SlideFPNModel(nn.Module):
    """FPN-style model from the slide architecture, with controlled ablations."""

    def __init__(
        self,
        obs_dim=60,
        vec_dim=0,
        hidden=128,
        num_fpn_blocks=1,
        fc_hidden=0,
        residual_style="merged",
        use_vec=False,
        vec_hidden=0,
        stem_mode="preserve",
    ):
        nn.Module.__init__(self)
        if stem_mode not in ("preserve", "valid_width"):
            raise ValueError("stem_mode must be 'preserve' or 'valid_width'")
        self.obs_dim = obs_dim
        self.vec_dim = vec_dim
        self.hidden = hidden
        self.num_fpn_blocks = num_fpn_blocks
        self.use_vec = use_vec
        self.vec_hidden = vec_hidden if use_vec and vec_hidden > 0 else vec_dim
        self.stem_mode = stem_mode
        self.spatial_h = 4
        self.spatial_w = 7 if stem_mode == "valid_width" else 9

        self._input_2d = nn.Sequential(
            nn.Conv2d(
                obs_dim,
                hidden,
                3,
                1,
                (1, 0) if stem_mode == "valid_width" else 1,
                bias=False,
            ),
            nn.GELU(),
        )
        self._input_1d = nn.Sequential(
            nn.Conv1d(
                obs_dim,
                hidden,
                3,
                1,
                0 if stem_mode == "valid_width" else 1,
                bias=False,
            ),
            nn.GELU(),
        )
        self._fpn_2d = nn.ModuleList(
            [FPNBlock2D(hidden) for _ in range(num_fpn_blocks)]
        )
        self._fpn_1d = nn.ModuleList(
            [FPNBlock1D(hidden) for _ in range(num_fpn_blocks)]
        )
        self._residual_2d = nn.ModuleList(
            [SlideFPNResidual2D(hidden) for _ in range(num_fpn_blocks)]
        )
        self._residual_1d = nn.ModuleList(
            [SlideFPNResidual1D(hidden) for _ in range(num_fpn_blocks)]
        )
        self._post_skip = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden),
        )
        self._post_block = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
        )
        if self.use_vec and vec_hidden and vec_hidden > 0:
            self._vec_adapter = nn.Sequential(nn.Linear(vec_dim, vec_hidden), nn.GELU())
        elif self.use_vec:
            self._vec_adapter = nn.Identity()
        else:
            self._vec_adapter = None
        output_input_dim = (
            hidden * self.spatial_h * self.spatial_w
            + (self.vec_hidden if self.use_vec else 0)
        )
        if fc_hidden and fc_hidden > 0:
            self._output_layer = nn.Sequential(
                nn.Linear(output_input_dim, fc_hidden),
                nn.GELU(),
                nn.Linear(fc_hidden, 235),
            )
        else:
            self._output_layer = nn.Linear(output_input_dim, 235)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        self.train(mode=input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()[:, : self.obs_dim]
        x2 = self._input_2d(obs)
        if self.stem_mode == "valid_width":
            batch = obs.size(0)
            x1 = obs.permute(0, 2, 1, 3).contiguous()
            x1 = self._input_1d(x1.view(batch * 4, self.obs_dim, 9))
        else:
            batch = obs.size(0)
            x1 = self._input_1d(obs.reshape(batch, self.obs_dim, 36))
        for block, residual in zip(self._fpn_2d, self._residual_2d):
            x2 = residual(block(x2))
        for block, residual in zip(self._fpn_1d, self._residual_1d):
            x1 = residual(block(x1))
        if self.stem_mode == "valid_width":
            x1 = x1.view(batch, 4, self.hidden, self.spatial_w)
            x1 = x1.permute(0, 2, 1, 3).contiguous()
        else:
            x1 = x1.reshape(batch, self.hidden, 4, 9)
        x = torch.cat([x2, x1], dim=1)
        x = self._post_skip(x) + self._post_block(x)
        x = torch.flatten(x, start_dim=1)
        if self.use_vec:
            vec = input_dict["obs"]["vec"].float()
            x = torch.cat([x, self._vec_adapter(vec)], dim=1)
        x = self._output_layer(x)

        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return x + inf_mask


class SelfVecModel(nn.Module):
    def __init__(self, obs_dim, vec_dim, hidden=128, num_blocks=20):
        nn.Module.__init__(self)
        self.hidden = hidden
        self.obs_dim = obs_dim
        self.vec_dim = vec_dim
        self._input_layer = nn.Sequential(
            nn.Conv2d(obs_dim, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self._hidden_layers = nn.ModuleList(
            [self.res_block(self.hidden) for _ in range(num_blocks)]
        )

        down_sample_ratio = ((hidden * 4 * 9 + vec_dim) / 235) ** (1 / 3)
        down_sample_dim_1 = int((hidden * 4 * 9 + vec_dim) / down_sample_ratio)
        down_sample_dim_1 = down_sample_dim_1 // 8 * 8
        down_sample_dim_2 = int(down_sample_dim_1 / down_sample_ratio)
        down_sample_dim_2 = down_sample_dim_2 // 8 * 8

        self._output_layer = nn.Sequential(
            nn.Linear(hidden * 4 * 9 + vec_dim, down_sample_dim_1),
            nn.GELU(),
            nn.Linear(down_sample_dim_1, down_sample_dim_2),
            nn.GELU(),
            nn.Linear(down_sample_dim_2, 235),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def res_block(self, hidden=256):
        return nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
        )

    def forward(self, input_dict):
        self.train(mode=input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        vec = input_dict["obs"]["vec"].float()
        x = self._input_layer(obs)
        for block in self._hidden_layers:
            x = x + block(x)
            x = F.gelu(x)
        # 展平, 128*4*9 = 4608
        x = torch.flatten(x, start_dim=1)
        # 链接 x 和 vec, 4608 + 117 = 4725
        x = torch.cat([x, vec], dim=1)
        # FC
        x = self._output_layer(x)

        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return x + inf_mask


class SlideStyleModel(nn.Module):
    """Slide-style ResNet head adapted to the current feature-agent tensors."""

    def __init__(
        self,
        obs_dim,
        vec_dim,
        hidden=64,
        num_blocks=20,
        out_planes=8,
        slide_vec_dim=78,
        fc_hidden=256,
    ):
        nn.Module.__init__(self)
        self.hidden = hidden
        self.obs_dim = obs_dim
        self.vec_dim = vec_dim
        self.out_planes = out_planes
        self.slide_vec_dim = slide_vec_dim

        self._input_layer = nn.Sequential(
            nn.Conv2d(obs_dim, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self._hidden_layers = nn.ModuleList(
            [self.res_block(self.hidden) for _ in range(num_blocks)]
        )
        self._projection = nn.Sequential(
            nn.Conv2d(hidden, out_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.GELU(),
        )
        self._vec_adapter = (
            nn.Identity()
            if vec_dim == slide_vec_dim
            else nn.Sequential(nn.Linear(vec_dim, slide_vec_dim), nn.GELU())
        )
        self._output_layer = nn.Sequential(
            nn.Linear(out_planes * 4 * 9 + slide_vec_dim, fc_hidden),
            nn.GELU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.GELU(),
            nn.Linear(fc_hidden, 235),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def res_block(self, hidden=64):
        return nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
        )

    def forward(self, input_dict):
        self.train(mode=input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        vec = input_dict["obs"]["vec"].float()
        x = self._input_layer(obs)
        for block in self._hidden_layers:
            x = x + block(x)
            x = F.gelu(x)
        x = self._projection(x)
        x = torch.flatten(x, start_dim=1)
        vec = self._vec_adapter(vec)
        x = torch.cat([x, vec], dim=1)
        x = self._output_layer(x)

        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return x + inf_mask

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
    def __init__(self, hidden=128, residual_style="merged"):
        nn.Module.__init__(self)
        if residual_style not in ("merged", "input"):
            raise ValueError(f"Unsupported residual_style: {residual_style}")
        self.residual_style = residual_style
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
        self._refine = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        y = self._parallel[0](x)
        for branch in self._parallel[1:]:
            y = y + branch(x)
        y = self._merge(y)
        residual = x if self.residual_style == "input" else y
        return F.gelu(residual + self._refine(y))


class FPNBlock1D(nn.Module):
    def __init__(self, hidden=128, residual_style="merged"):
        nn.Module.__init__(self)
        if residual_style not in ("merged", "input"):
            raise ValueError(f"Unsupported residual_style: {residual_style}")
        self.residual_style = residual_style
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
        self._refine = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        y = self._parallel[0](x)
        for branch in self._parallel[1:]:
            y = y + branch(x)
        y = self._merge(y)
        residual = x if self.residual_style == "input" else y
        return F.gelu(residual + self._refine(y))


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
    ):
        nn.Module.__init__(self)
        self.obs_dim = obs_dim
        self.vec_dim = vec_dim
        self.hidden = hidden
        self.num_fpn_blocks = num_fpn_blocks
        self.use_vec = use_vec
        self.vec_hidden = vec_hidden if use_vec and vec_hidden > 0 else vec_dim

        self._input_2d = nn.Sequential(
            nn.Conv2d(obs_dim, hidden, 3, 1, 1, bias=False),
            nn.GELU(),
        )
        self._input_1d = nn.Sequential(
            nn.Conv1d(obs_dim, hidden, 3, 1, 1, bias=False),
            nn.GELU(),
        )
        self._fpn_2d = nn.ModuleList(
            [FPNBlock2D(hidden, residual_style=residual_style) for _ in range(num_fpn_blocks)]
        )
        self._fpn_1d = nn.ModuleList(
            [FPNBlock1D(hidden, residual_style=residual_style) for _ in range(num_fpn_blocks)]
        )
        self._post_projection = nn.Conv2d(hidden * 2, hidden, 1, 1, 0, bias=False)
        self._post_block = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
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
        output_input_dim = hidden * 4 * 9 + (self.vec_hidden if self.use_vec else 0)
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
        x1 = self._input_1d(obs.reshape(obs.size(0), self.obs_dim, 36))
        for block in self._fpn_2d:
            x2 = block(x2)
        for block in self._fpn_1d:
            x1 = block(x1)
        x1 = x1.reshape(obs.size(0), self.hidden, 4, 9)
        x = torch.cat([x2, x1], dim=1)
        x = self._post_projection(x)
        x = F.gelu(x + self._post_block(x))
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

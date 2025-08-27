# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union, Optional
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import ControlNetModel
# from Kolors.kolors.models.controlnet import ControlNetModel
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.controlnet import zero_module

class ControlNetConditioningEmbedding(nn.Module):

    def __init__(
        self,
        conditioning_embedding_channels,
        conditioning_channels = 3,
        block_out_channels = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding

class ControlnetCond(ModelMixin, ConfigMixin):
    def __init__(self, 
        in_channels, 
        cond_channels, 
        unet, 
        image_size, 
        freeze_params=True,
        block_out_channels = (320, 640, 1280, 1280),
        conditioning_embedding_out_channels = (32, 32, 96, 256),
        pretrained_cn=False,
        enable_xformer=False,
        adapter=None,
        global_pool_conditions=False,
        *args, 
        **kwargs
        ):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        
        self.unet = unet
        self.controlnet = ControlNetModel.from_unet(self.unet, conditioning_embedding_out_channels=conditioning_embedding_out_channels)
        self.controlnet.controlnet_cond_embedding = ControlNetConditioningEmbedding(
                conditioning_embedding_channels=self.controlnet.config.block_out_channels[0],
                block_out_channels=self.controlnet.config.conditioning_embedding_out_channels,
                conditioning_channels=cond_channels,
            )

        if enable_xformer:
            print('xFormer enabled')
            self.unet.enable_xformers_memory_efficient_attention()
            self.controlnet.enable_xformers_memory_efficient_attention()
       

        self.requires_grad_(False)
        
        self.image_size=image_size
        self.sample_size = image_size // 8
        self.H, self.W = self.sample_size,self.sample_size

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype,text_encoder_projection_dim=None):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def forward(self,
                sample: torch.FloatTensor, # Shape (B, C, H, W),
                timestep: Union[torch.Tensor, float, int],
                encoder_hidden_states: torch.Tensor = None, # Shape (B, D_C, H_C, W_C)
                cond_mask: Optional[torch.BoolTensor] = None, # Boolen tensor of shape (B, H_C, W_C). True for masked out pixels,
                prompt = None,
                unconditional = False,
                cond_scale = 1.0,
                crops_coords_top_left=(0,0),
                height=512,
                width=512,
                **kwargs):
        dtype=sample.dtype
       
        controlnet_cond = F.interpolate(encoder_hidden_states.float(), (self.H, self.W), mode="nearest").to(dtype)
        encoder_hidden_states = torch.zeros(sample.shape[0], 77, 2048).to(self.device).to(dtype)
        add_text_embeds = torch.zeros(sample.shape[0],1280).to(self.device).to(dtype)
        
        original_size = (self.image_size, self.image_size)
        target_size = (self.image_size, self.image_size)
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size,dtype=dtype
        )

        batch_size=sample.shape[0]
        add_time_ids = add_time_ids.to(self.device).repeat(batch_size, 1)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        down_block_res_samples, mid_block_res_sample = self.controlnet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                controlnet_cond=controlnet_cond,
                conditioning_scale=cond_scale,
                return_dict=False,
            )

        noise_pred = self.unet(
                sample,
                timestep,
                added_cond_kwargs=added_cond_kwargs,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]

        return noise_pred
  


 

    
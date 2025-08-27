
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass, field
from contextlib import nullcontext
import copy
from diffusers import AutoencoderKL, DDIMScheduler,UNet2DConditionModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import PyTorchModelHubMixin
import random
from  model import ControlnetCond 
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from diffusion_pipeline_tlcm import PipelineCond
from vq_model import VQModel
@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0

class VQControlNet(VQModel):
    
    def __init__(self,  
                 sd_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 image_size_sd: Optional[int] = None,
                 pretrained_cn: bool = False,
                 enable_xformer: bool = False,
                 adapter: Optional[str] = None,
                 latent_dim=8,
                 config: Optional[Dict[str, Any]] = None,
                 *args, **kwargs):
        if config is not None:
            config = copy.deepcopy(config)
            self.__init__(**config)
            return
        # Don't want to load the weights just yet


        super().__init__(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4],
        codebook_size=16384,
        codebook_embed_dim=8))
        self.ckpt_path = kwargs.get('ckpt_path', None)
        self.latent_dim=latent_dim
        self.image_size_sd = image_size_sd
        
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
        try:
            import xformers
            XFORMERS_AVAILABLE = True
        except ImportError:
            print("xFormers not available")
            XFORMERS_AVAILABLE = False
        enable_xformer = enable_xformer and XFORMERS_AVAILABLE
        if enable_xformer:
            print('Enabling xFormer for Stable Diffusion')
            unet.enable_xformers_memory_efficient_attention()

        self.decoder = ControlnetCond(
            in_channels=4, 
            cond_channels=self.latent_dim,
            unet=unet,
            conditioning_embedding_out_channels=[8, 32, 96, 256],
            image_size=self.image_size_sd,
            pretrained_cn=pretrained_cn,
            enable_xformer=enable_xformer,
            adapter=adapter,
        )
        lora_config = LoraConfig(
        r=64,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "conv1",
            "conv2",
            "conv_shortcut",
            "downsamplers.0.conv",
            "upsamplers.0.conv",
            "time_emb_proj",
        ],
            )
        self.decoder.unet = get_peft_model(self.decoder.unet, lora_config)
        self.noise_scheduler = DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(sd_path,subfolder='vae')
        self.pipeline = PipelineCond(model=self.decoder, scheduler=self.noise_scheduler)
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        model = ckpt['model'] if 'model' in ckpt else ckpt['state_dict']
        self.load_state_dict(model, strict=True)


    
  

    def decode_quant(self, 
                     quant: torch.Tensor, 
                     
                     generator: Optional[torch.Generator] = None, 
                     image_size: Optional[Union[Tuple[int, int], int]] = None, 
                     
                     vae_decode: bool = False,
                     
                     prompt: Optional[Union[List[str], str]]= None,
                     orig_res: Optional[Union[torch.LongTensor, Tuple[int, int]]] = None,
                     
                     cond_scale: int = 1.0) -> torch.Tensor:
        
        dec = self.pipeline(
            quant,  generator=generator, image_size=image_size, 
             prompt=prompt,
            cond_scale=cond_scale,
        )

        if vae_decode:
            return self.vae_decode(dec)

        return dec

    def decode_tokens(self, tokens: torch.LongTensor, **kwargs) -> torch.Tensor:
        """See `decode_quant` for details on the optional args."""
        return super().decode_tokens(tokens, **kwargs)

  
    
    @torch.no_grad()
    def vae_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes the vae latent representation into vae latent representaiton.
        
        Args:
            x: VAE latent representation
            clip: If set True clips the decoded image between -1 and 1.

        Returns:
           Decoded image of shape B C H W 
        """
        x = self.vae.decode(x / self.vae.config.scaling_factor).sample

        x = torch.clip(x, min=-1, max=1)
        return x



   


from typing import Optional, Tuple, Union
import torch
from diffusers import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from tqdm import tqdm







class PipelineCond():
    """Pipeline for conditional image generation.

    This model inherits from `DiffusionPipeline`. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        model: The conditional diffusion model.
        scheduler: A diffusion scheduler, e.g. see scheduling_ddpm.py
    """
    def __init__(self, model,scheduler):
        self.model=model
        self.scheduler=scheduler


    def append_dims(self,x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError( f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
        return x[(...,) + (None,) * dims_to_append]


    @torch.no_grad()
    def __call__(self, 
                 cond: torch.Tensor, 
                 generator: Optional[torch.Generator] = None, 
                 cond_scale: float = 0.0, 
                 image_size: Optional[Union[Tuple[int, int], int]] = None, 
                 orig_res: Optional[Union[torch.LongTensor, Tuple[int, int]]] = None,
                 **kwargs) -> torch.Tensor:
       

      
        batch_size, _, _, _ = cond.shape
        
        # Sample gaussian noise to begin loop
        image_size = self.model.sample_size if image_size is None else image_size
        
        image = torch.randn(
            (batch_size, self.model.in_channels, image_size, image_size),
            generator=generator,
        )
        image = image.to(self.model.device).to(cond.dtype)
        device=self.model.device
        noise_scheduler=self.scheduler
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        stepsn = noise_scheduler.config.num_train_timesteps
        eval_step=9
        ts = torch.linspace(1000-1, 0, eval_step, device=device,dtype=int)
        ts=ts[[0,4,7,8]]
        
        for j in range(0,len(ts)-1):
            # 1. Predict noise model_output
            t=ts[j]
            model_output = self.model(image, t, cond, cond_scale=cond_scale,orig_res=orig_res, **kwargs)
            dims=model_output.ndim
            alpha_t=self.append_dims(torch.sqrt( self.alphas_cumprod[t]),dims).to(device)
            sigma_t=self.append_dims(torch.sqrt(1- self.alphas_cumprod[t]),dims).to(device)
            denoised = (image - sigma_t * model_output) / alpha_t
            image = noise_scheduler.add_noise(denoised, torch.randn_like(denoised), torch.tensor([ts[j+1]], device=device) ).to(cond.dtype)
            
        return denoised
    

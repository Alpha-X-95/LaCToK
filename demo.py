import torch
import os,sys
# sys.path.append('/mnt/data/group/xieqingsong/code/diffusers')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torchvision.utils import save_image
from tokenizer import VQControlNet
from PIL import Image
import time
import glob
import json
import argparse
import random
from transformers import AutoModelForCausalLM, AutoTokenizer,Qwen2Model
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from t5 import T5Embedder
from gpt import GPT_models
from generate import generate
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np 
import cv2
import torchvision.transforms.functional as TF
def denormalize(img, mean, std):
   
    return TF.normalize(
        img.clone(),
        mean= [-m/s for m, s in zip(mean, std)],
        std= [1/s for s in std]
    )

def seed_everything(TORCH_SEED):
	random.seed(TORCH_SEED)
	os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
	np.random.seed(TORCH_SEED)
	torch.manual_seed(TORCH_SEED)
	torch.cuda.manual_seed_all(TORCH_SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def make_grid(images):
    cols = int(np.floor(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / cols))
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid  
def main(args):
   
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # create and load model
   
    sd_path='stabilityai/stable-diffusion-xl-base-1.0'
   
    vq_model = VQControlNet(
                            image_size_sd=1024,
                            codebook_size=16384,
                            latent_dim=8,
                            ckpt_path=args.ckpt,
                            sd_path=sd_path,
                            pretrained_cn=True,
                            enable_xformer=True,
                            adapter=None,
                            dtype='float32'
                        )
    
    vq_model.to(device)
    vq_model.eval()
    vq_model.to(device)
    vq_model.eval()


    
    t5_model = T5Embedder(
    device=device, 
    local_cache=True, 
    cache_dir=args.t5_path, 
    dir_or_name='flan-t5-xl',
    torch_dtype=torch.float32,
    model_max_length=120,
    )

    # create and load gpt model
    

    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    )

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
 
    if "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")


    gpt_model.load_state_dict(model_weight, strict=True)
    gpt_model.eval()
    gpt_model.to(device)
    del checkpoint
    print(f"gpt model is loaded")
    savedir='images'
    os.makedirs(savedir,exist_ok=True)
    
    prompts=['a couple of zebras that are running through some grass']
   
    for prompt in prompts:
        caption_embs, emb_masks = t5_model.get_text_embeddings([prompt])
        if not args.no_left_padding:
            # print(f"processing left-padding...")    
            # a naive way to implement left-padding
            new_emb_masks = torch.flip(emb_masks, dims=[-1])
            new_caption_embs = []
            for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
                valid_num = int(emb_mask.sum().item())
                # print(f'  prompt {idx} token len: {valid_num}')
                new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                new_caption_embs.append(new_caption_emb)
                
            new_caption_embs = torch.stack(new_caption_embs)
        else:
            new_caption_embs, new_emb_masks = caption_embs, emb_masks
        c_indices = new_caption_embs * new_emb_masks[:,:, None]
        c_emb_masks = new_emb_masks
        # print('c_indices',c_indices.shape)
        # print('c_emb_masks',c_emb_masks.shape)

        qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
        t1 = time.time()
        index_sample = generate(
            gpt_model, c_indices, latent_size ** 2, 
            c_emb_masks, 
            cfg_scale=args.cfg_scale,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True, 
            )
        sampling_time = time.time() - t1
        
    
        b,l=index_sample.shape
        index_sample=torch.reshape(index_sample,(b,latent_size,latent_size))
       
        samples=vq_model.get_quant(index_sample)
        samples=samples.permute(0,3,1,2)
      
        samples=vq_model.decode_quant(samples,cond_scale=0.8)
       
        samples=vq_model.vae_decode(samples)
       
        reconst = denormalize(samples[:,:3], mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        reconst_bytes = (255 * reconst.permute(0,2,3,1).clamp(0,1).cpu().numpy()).astype(np.uint8)
        
        for i in range(b):
            img=reconst_bytes[i]
            path=os.path.join(savedir,prompt+'.jpg')
            img=Image.fromarray(img)
            img.save(path)
       
       

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-path", type=str, default='/home/notebook/data/group/xieqingsong/models')
    parser.add_argument("--ckpt", type=str, default='')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[192,256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    args = parser.parse_args()
    main(args)

set -x
nnodes=1
nproc_per_node=1
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}   
# pip install boto3 pycocotools albumentations==1.3.1 torch-fidelity==0.3.0  lpips==0.1.4 accelerate==0.31.0 huggingface-hub==0.23.4 -i https://pypi.mirrors.ustc.edu.cn/simple/
python demo.py \
--cfg-scale 2 \
--cls-token-num 120 \
--top-k 1000 \
--t5-path /home/notebook/data/group/group/xieqingsong/models \
--temperature 0.1 \
--ckpt /home/notebook/data/group/group/xieqingsong/code/4m/output/latok-tlcm-flux-1024t2i-mul/checkpoint-iter-14999.pth \
--gpt-ckpt /home/notebook/data/group/group/xieqingsong/models/LlamaGen-t2i-alldata/2025-02-24-09-14-56/070-GPT-XL/checkpoints/0005000.pt --gpt-model GPT-XL --image-size 256 \
--seed 42\
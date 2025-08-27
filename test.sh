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
--ckpt LaCTok-T.pth \
--gpt-ckpt LaCTokGen-T.pth --gpt-model GPT-XL --image-size 256 \
--seed 42\
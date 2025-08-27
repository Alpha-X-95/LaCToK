import cv2 
import glob, os 
root='/mnt/data/group/xieqingsong/code/tokenizer-sdxl/dpg-dpg-cfg3.0/*/*.png'
paths=glob.glob(root)
print(len(paths))
root='/mnt/data/group/xieqingsong/code/ELLA/dpg_bench/prompts'
pahts=os.listdir(root)
print(len(paths))
savedir='DPG-DPG-3.0'
os.makedirs(savedir,exist_ok=True)
for path in paths:
    img=cv2.imread(path)
    name=path.split('/')[-2]
    filename=os.path.join(savedir,f'{name}.png')
    cv2.imwrite(filename,img)

import os
import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
from data_RGB import get_test_data
from EHNet import EHNet as myNet
from skimage import img_as_ubyte
from ptflops import get_model_complexity_info
from tqdm import tqdm



parser = argparse.ArgumentParser(description='Image Deblurring')
parser.add_argument('--target_dir', default='./Dataset/GoPro/test/target/', type=str, help='Directory of ground truth images')
parser.add_argument('--input_dir', default='./Dataset/GoPro/test/input/', type=str, help='Directory of input images')

parser.add_argument('--output_dir', default='./results/EHNet/GoPro/', type=str, help='Directory of validation images')
parser.add_argument('--weights', default='./weight_saved/GoPro.pth', type=str, help='Path to weights')

parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_result', default=True, type=bool, help='save result')


args = parser.parse_args()
result_dir = args.output_dir

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = myNet()

macs, params = get_model_complexity_info(model_restoration, (3, 256, 256), as_strings=False,
                                        print_per_layer_stat=False, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params)) 

utils.load_checkpoint(model_restoration, args.weights)

model_restoration.cuda()
model_restoration.eval()

# dataset = args.dataset
rgb_dir_test = args.input_dir
test_dataset = get_test_data(rgb_dir_test, img_options={})
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

utils.mkdir(result_dir)

with torch.no_grad():
    
    for ii, data_test in enumerate(tqdm(test_loader), 0):

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        input_    = data_test[0].cuda()
        filenames = data_test[1]
        
        b, c, h, w = input_.shape
        h_n = (64 - h % 64) % 64
        w_n = (64 - w % 64) % 64
        input_img = torch.nn.functional.pad(input_, (0, w_n, 0, h_n), mode='reflect')
        restored = model_restoration(input_img)
        torch.cuda.synchronize()
        restored = restored[:, :, :h, :w]
        

        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        for batch in range(len(restored)):
            restored_img = restored[batch]
            restored_img = img_as_ubyte(restored[batch])
            
            if args.save_result:
                utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)


    

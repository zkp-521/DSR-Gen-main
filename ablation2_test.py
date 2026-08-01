import matplotlib.pyplot as plt
import numpy as np
from networks_ournet import Model_xiaorong_resvit_shuangfenzhi,Model_xiaorong_resvit
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.init as init
import math
import torch
import torch.nn as nn
from loss import EdgeSSIMLoss,VGGLoss,MsImageDis,registration_loss_2d
import transformer_configs
from dataset import ct_enhance_d,ct_enhance_d1
from dataset import ct_enhance_d,ct_enhance_d2
from reg import Reg
from transformer import Transformer_2D
import torch.nn.functional as F
def median_filter_unfold(img: torch.Tensor, kernel_size: int = 3):
    """
    基于 unfold + median 的中值滤波，支持 (B, C, H, W) 格式输入。
    """
    # 确保 kernel_size 是奇数
    assert kernel_size % 2 == 1, "Kernel size must be odd."

    b, c, h, w = img.shape
    pad = kernel_size // 2

    # 使用 reflect padding 避免边界黑边
    img_pad = F.pad(img, (pad, pad, pad, pad), mode='reflect')

    # 使用 unfold 提取滑动窗口，并一次性计算所有窗口的中位数
    patches = img_pad.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    # 将窗口内的所有值展平到一维，并求中位数
    out = patches.contiguous().view(b, c, h, w, -1).median(dim=-1)[0]

    return out
def guiyi(image):
    image_f=(image-image.min())/((image.max()-image.min()))
    return image_f
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun
configs=transformer_configs
CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-L_16': configs.get_l16_config(),
    'Res-ViT-B_16': configs.get_resvit_b16_config(),
    'Res-ViT-L_16': configs.get_resvit_l16_config(),
}
import os
print(torch.cuda.is_available())
gen_p=Model_xiaorong_resvit_shuangfenzhi(config=CONFIGS['Res-ViT-B_16'],input_dim=1, dim=64,n_downsample=3,n_res=3,activ='relu',pad_type='reflect').cuda()
'''
gen_p.load_state_dict(torch.load('G:/ct_f/our_model_xiaorong'+'/ct2_ourmodexiaorong_resvit_shuangfenzhiMASK_1_51.pkl'))
print(torch.cuda.is_available())
file_name_m = os.listdir('G:/ct_f/test/1/')
print(file_name_m[0][3:6])
for n in range(int(len(file_name_m)/2)):
    p=np.load('G:/ct_f/test/1/v0_'+file_name_m[n][3:6]+'.npy')
    #a=np.load('F:/ct_enhance3/Vhest/test/a/a_z_{}.npy'.format(n))
    a = np.load('G:/ct_f/test/1/v1_'+file_name_m[n][3:6]+'.npy')
    a_p=np.zeros((256,256,a.shape[2])).astype(np.float16)
    #p_p=np.zeros((256,256,nn)).astype(np.float16)
    #a_z=np.zeros((256,256,nn)).astype(np.float16)
    for i in range(a.shape[2]):
        print(n,i)
        #print(i)
        p_c = p[:, :, i]
        p_c_c_f = torch.from_numpy(p_c[np.newaxis, np.newaxis, :, :].astype(np.float32)) / 1000.0
        f = gen_p.encode(p_c_c_f.cuda())
        x_tr,x_rec_tr = gen_p.decode(f)
        out_n = x_rec_tr.detach().cpu().numpy()[0, 0, :, :] * 1000


        #plt.figure()
        #plt.subplot(1, 4, 1)
        #plt.imshow(out_n, 'gray')
        #plt.subplot(1, 4, 2)
        #plt.imshow(p[:,:,i], 'gray')
        #plt.subplot(1, 4, 3)
        #plt.imshow(a[:,:,i], 'gray')
        #plt.subplot(1, 4, 4)
        #plt.imshow(a[:,:,i]-p[:,:,i], 'gray')
        #plt.show()
        a_p[:, :, i] = out_n
        #a_z[:, :, i] = a_c_c
        #p_p[:, :, i] = p_c_c

    np.save('G:/ct_f/test/1_p_xiaorong/v1_bianti3_45_'+file_name_m[n][3:6]+'.npy', a_p)
'''
'''
gen_p.load_state_dict(torch.load('F:/shiyan_xiaorong_model'+'/abdomen_ourmodexiaorong_resvit_shuangfenzhiMASK_a_50.pkl'))
for n in range(416,516):
    print(n)
    p=np.load('F:/ct_enhance_abdomen/test/p/p_z_{}.npy'.format(n))
    #a=np.load('F:/ct_enhance_chest/test/a/a_z_{}.npy'.format(n))
    a = np.load('F:/ct_enhance_abdomen/test/a/a_z_{}.npy'.format(n))
    a_p=np.zeros((256,256,a.shape[2])).astype(np.float16)
    #p_p=np.zeros((256,256,nn)).astype(np.float16)
    #a_z=np.zeros((256,256,nn)).astype(np.float16)
    for i in range(a.shape[2]):
        #print(i)
        p_c=p[:,:,i]
        p_c_c_f=torch.from_numpy(p_c[np.newaxis,np.newaxis,:,:].astype(np.float32))/1000.0
        f = gen_p.encode(p_c_c_f.cuda())
        x_tr,x_rec_tr=gen_p.decode(f)
        out_n = x_rec_tr.detach().cpu().numpy()[0, 0, :, :] * 1000

        #plt.figure()
        #plt.subplot(1, 3, 1)
        #plt.imshow(out_n, 'gray')
        #plt.subplot(1, 3, 2)
        #plt.imshow(p[:,:,i], 'gray')
        #plt.subplot(1, 3, 3)
        #plt.imshow(a[:,:,i], 'gray')
        #plt.show()
        
        #plt.figure()
        #plt.imshow(a[:, :, i], 'gray')
        #plt.axis('off')
        #plt.savefig('a.png', bbox_inches='tight', pad_inches=0, dpi=300)
        #plt.imshow(out_n, 'gray')
        #plt.axis('off')
        #plt.savefig('out_rec_tr.png', bbox_inches='tight', pad_inches=0, dpi=300)
        #plt.imshow(out_n_1, 'gray')
        #plt.axis('off')
        #plt.savefig('out_tr.png', bbox_inches='tight', pad_inches=0, dpi=300)
        #plt.imshow(out_n_2, 'gray')
        #plt.axis('off')
        #plt.savefig('out_rec.png', bbox_inches='tight', pad_inches=0, dpi=300)
        #plt.imshow(p[:, :, i], 'gray')
        #plt.axis('off')
        #plt.savefig('p.png', bbox_inches='tight', pad_inches=0, dpi=300)
        #plt.figure()
        #plt.imshow(map_difference_image(p[:, :, i],a[:, :, i],400,100))
        #plt.colorbar()
        #plt.axis('off')
        #plt.savefig('diff.png', bbox_inches='tight', pad_inches=0, dpi=300)
        #plt.figure()
        #plt.imshow(map_difference_image(p[:, :, i], a[:, :, i], 400, 100))
        #plt.axis('off')
        #plt.show()
        


        a_p[:, :, i] = out_n
        #a_z[:, :, i] = a_c_c
        #p_p[:, :, i] = p_c_c
    #np.save('F:/ct_enhance_chest/test/d/d_z_{}.npy'.format(n), a_z)
    np.save('F:/ct_enhance_abdomen/test/a_xiaorong/a_bianti3_50_{}.npy'.format(n), a_p)
'''
def guiyi(image):
    if image.max()==0:
        image_np=image
    else:
        image_np=(image-image.min())/((image.max()-image.min()))*2-1
    return image_np

def deguiyi(image,ref):
    if ref.max()==0:
        image_np=image
    else:
        image_np=(image-image.min())*(ref.max()-ref.min())/2
        #image_np=(image-image.min())/((image.max()-image.min()))
        #image_np=image_np*(ref.max()-ref.min()) - ref.min()
    return image_np
gen_p=Model_xiaorong_resvit_shuangfenzhi(config=CONFIGS['Res-ViT-B_16'],input_dim=1, dim=64,n_downsample=3,n_res=3,activ='relu',pad_type='reflect').cuda()

gen_p.load_state_dict(torch.load('G:/mri1/mri/model'+'/mr_head_ourmodexiaorong_resvit_shuangfenzhi_3011_54.pkl'))
for n in range(456, 595):
    p = np.load('G:/mri1/mri/test/t1n/t1n_{}.npy'.format(n))
    # a=np.load('F:/ct_enhance3/Vhest/test/a/a_z_{}.npy'.format(n))
    a = np.load('G:/mri1/mri/test/t1c/t1c_{}.npy'.format(n))
    a_p = np.zeros((256, 256, a.shape[2])).astype(np.float16)
    # p_p=np.zeros((256,256,nn)).astype(np.float16)
    # a_z=np.zeros((256,256,nn)).astype(np.float16)
    for i in range(a.shape[2]):
        print(n, i)
        p_c = guiyi(p[:, :, i])
        p_c_c_f = torch.from_numpy(p_c[np.newaxis, np.newaxis, :, :].astype(np.float32))
        f = gen_p.encode(p_c_c_f.cuda())
        _,x_rec_tr = gen_p.decode(f)
        out_n = x_rec_tr.detach().cpu().numpy()[0, 0, :, :]
        #out_n=p_c+x_tr.detach().cpu().numpy()[0, 0, :, :] * 400
        out_n=deguiyi(out_n,a[:,:,i])
        out_n[out_n < 0] = 0
        #plt.figure()
        #plt.subplot(1, 3, 1)
        #plt.imshow(out_n, 'gray')
        #plt.subplot(1, 3, 2)
        #plt.imshow(p[:,:,i], 'gray')
        #plt.subplot(1, 3, 3)
        #plt.imshow(a[:,:,i], 'gray')
        #plt.show()
        a_p[:, :, i] = out_n
        # a_z[:, :, i] = a_c_c
        # p_p[:, :, i] = p_c_c
    np.save('G:/mri1/mri/test/t1c/t1c_our_resvitshuangfenzhi_30111_guihua1_54_{}.npy'.format(n), a_p)
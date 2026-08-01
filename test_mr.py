import numpy as np
import torch.nn.functional as F
from networks_ournet import Model,Model_cnn,Model_cnn_X,Model_xiaorong_resvit_shuangfenzhi,Model_xiaorong_resvit
import torch
import transformer_configs
def guiyi(image):
    if image.max()==0:
        image_np=image
    else:
        image_np=(image-image.min())/((image.max()-image.min()))
    return image_np


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
def fast_saturate(data, k=0.01):
    """
    将非负数据非线性映射到 [0,1]，>0 的值快速饱和到 1。

    参数:
        data: numpy 数组（假设值 >= 0）
        k: 饱和速率系数，越大则小值越迅速逼近 1
    返回:
        mapped: 形状相同的数组，值域 [0,1]
    """
    f=np.exp(-k * data)
    ff=torch.from_numpy(f[np.newaxis,np.newaxis,:,:])
    gg=median_filter_unfold(ff, 3)
    #gg=1-np.exp(-5* gg.numpy()[0,0,:,:])
    #gg=ff
    return gg.numpy()[0,0,:,:]

configs=transformer_configs
CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-L_16': configs.get_l16_config(),
    'Res-ViT-B_16': configs.get_resvit_b16_config(),
    'Res-ViT-L_16': configs.get_resvit_l16_config(),
}
print(torch.cuda.is_available())
gen_p=Model_xiaorong_resvit(config=CONFIGS['Res-ViT-B_16'],input_dim=1, dim=64,n_downsample=3,n_res=3,activ='relu',pad_type='reflect').cuda()
def deguiyi(image,ref):
    if ref.max()==0:
        image_np=image
    else:
        image_np=image*(ref.max()-ref.min())
        #image_np=(image-image.min())/((image.max()-image.min()))
        #image_np=image_np*(ref.max()-ref.min()) - ref.min()
    return image_np
gen_p.load_state_dict(torch.load('G:/mri1/mri/model'+'/mr_head_ourmode_resvit_ls_c_50.pkl'))
for n in range(456, 595):
    p = np.load('G:/mri1/mri/test/t1n/t1n_{}.npy'.format(n))
    # a=np.load('F:/ct_enhance3/Vhest/test/a/a_z_{}.npy'.format(n))
    a = np.load('G:/mri1/mri/test/t1c/t1c_{}.npy'.format(n))
    a_p = np.zeros((256, 256, a.shape[2])).astype(np.float16)
    # p_p=np.zeros((256,256,nn)).astype(np.float16)
    # a_z=np.zeros((256,256,nn)).astype(np.float16)
    for i in range(a.shape[2]):
        print(n, i)
        # print(i)
        p_c = guiyi(p[:, :, i])
        p_c_c_f = torch.from_numpy(p_c[np.newaxis, np.newaxis, :, :].astype(np.float32))
        f = gen_p.encode(p_c_c_f.cuda())
        x_rec_tr = gen_p.decode(f)
        out_n = x_rec_tr.detach().cpu().numpy()[0, 0, :, :]
        out_n[out_n < 0] = 0
        out_n=deguiyi(out_n,a[:,:,i])
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
    np.save('G:/mri1/mri/test/t1c/t1c_our1_RESVIT_ls_guiyi1_50_{}.npy'.format(n), a_p)

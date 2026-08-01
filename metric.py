import numpy as np
import cv2
import torch
from skimage.metrics import structural_similarity as ssim
from skimage import filters, feature
import lpips
import matplotlib.pyplot as plt

def edge_ssim(img1, img2, edge_method='canny', mask=None, **kwargs):
    """
    计算两幅图像之间的基于边缘的结构相似性（Edge-SSIM）

    参数:
        img1, img2: 输入图像（2D灰度图，取值范围0-255或0-1）
        edge_method: 边缘检测方法，可选 'canny', 'sobel', 'scharr'
        mask: 可选掩膜（二值图），仅在该区域内计算SSIM
        **kwargs: 传递给边缘检测函数的额外参数

    返回:
        edge_ssim_score: 边缘图的SSIM值，范围[-1, 1]，越高表示结构越相似
        edge1, edge2: 提取的边缘图（仅当return_edges=True时返回）
    """
    # 确保图像为浮点型并归一化到[0,1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if img1.max() > 1.0:
        img1 /= 255.0
    if img2.max() > 1.0:
        img2 /= 255.0

    # 边缘提取
    if edge_method == 'canny':
        # Canny边缘检测：返回二值边缘图
        sigma = kwargs.get('sigma', 1.0)
        low_threshold = kwargs.get('low_threshold', 0.1)
        high_threshold = kwargs.get('high_threshold', 0.2)
        edge1 = feature.canny(img1, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
        edge2 = feature.canny(img2, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
        # 将二值图转为浮点型[0,1]以计算SSIM
        edge1 = edge1.astype(np.float64)
        edge2 = edge2.astype(np.float64)

    elif edge_method in ['sobel', 'scharr']:
        # Sobel/Scharr边缘强度图
        if edge_method == 'sobel':
            edge1 = filters.sobel(img1)
            edge2 = filters.sobel(img2)
        else:  # scharr
            edge1 = filters.scharr(img1)
            edge2 = filters.scharr(img2)
        # 强度图归一化到[0,1]
        edge1 = edge1 / edge1.max() if edge1.max() > 0 else edge1
        edge2 = edge2 / edge2.max() if edge2.max() > 0 else edge2

    else:
        raise ValueError("edge_method must be 'canny', 'sobel', or 'scharr'")

    #plt.imshow(edge1,'gray')
    #plt.show()
    #plt.imshow(edge2, 'gray')
    #plt.show()

    # 应用掩膜（可选）
    if mask is not None:
        # 确保掩膜尺寸一致且为二值
        mask = mask.astype(bool)
        edge1 = edge1 * mask
        edge2 = edge2 * mask

    # 计算SSIM
    # data_range: 根据边缘图的值域设置，对于二值边缘图data_range=1.0，对于强度图data_range=1.0（已归一化）
    ssim_score = ssim(edge1, edge2, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return ssim_score, edge1, edge2
def mae_400(a,p):
    diff = np.abs(p - a)
    f = np.where(diff > 400)
    diff_gt_400_v2 = diff[f]

    mae=np.sum(diff_gt_400_v2) / np.prod(diff.shape)
    return mae
def mask(xingbian,ref):
    diff = np.abs(xingbian-ref)
    mask1 = np.ones_like(diff)
    mask1[diff >= 400] = 0
    #mask1[diff < 400] = 0
    return mask1
def map_difference_image(tu, ji, high, low):
    diff = np.abs(tu - ji)
    result = np.zeros_like(diff, dtype=np.float32)
    result[diff < low] = 1.0
    result[diff > high] = 0
    mask_mid = (diff > low) & (diff < high)
    result[mask_mid] = 1 - (diff[mask_mid] - low) / (high - low)
    return result
def norm(image):
    image_n=(image-image.min())/(image.max()-image.min())
    return image_n
def mask_mae(p,a_z,a_p):
    mask1=mask(p,a_z)
    #mask1=map_difference_image(p, a_z, 400, 100)
    mae=np.mean(np.abs(a_z*mask1-a_p*mask1))
    return mae

def mask_ssim(p,a_z,a_p):
    mask1=mask(p,a_z)
    #mask1 = map_difference_image(p, a_z, 400, 100)
    ssim_score = ssim(a_z*mask1, a_p*mask1,data_range=4095.0)
    return ssim_score


import torch.nn.init as init
import math
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

class VGGLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        # 加载预训练的VGG16
        vgg = models.vgg16(pretrained=True).features
        #vgg.load_state_dict(torch.load('vgg16-397923af.pth'))
        vgg.eval()

        # 冻结所有参数
        for param in vgg.parameters():
            param.requires_grad = False

        # 提取特定层的特征
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:23])  # 到conv4_3层

        # 图像预处理（与ImageNet训练时相同）
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.device = device
        self.vgg_layers.to(device)

    def forward(self, p,a_z, a_p):
        mask1 = mask(p, a_z)
        mask2 = mask1[np.newaxis, np.newaxis, :, :]
        #z = torch.from_numpy(norm(a_z[np.newaxis, np.newaxis, :, :]*mask2))
        #p = torch.from_numpy(norm(a_p[np.newaxis, np.newaxis, :, :]*mask2))
        z = torch.from_numpy(norm(a_z[np.newaxis, np.newaxis, :, :]))
        p = torch.from_numpy(norm(a_p[np.newaxis, np.newaxis, :, :]))
        generated = self.normalize(z.repeat(1,3,1,1)).cuda()
        target = self.normalize(p.repeat(1,3,1,1)).cuda()


        # 提取特征
        gen_features = self.vgg_layers(generated)
        target_features = self.vgg_layers(target)

        # 计算特征损失（感知损失）
        loss = nn.functional.mse_loss(gen_features, target_features)
        return loss.item()
import torch.nn.functional as F
class LocalNormalizedCrossCorrelation2D(nn.Module):
    """
    局部归一化互相关损失（2D版本）
    输入: I, J 形状均为 (B, C, H, W)
    窗口大小: kernel_size (默认9)
    """
    def __init__(self, kernel_size=9, eps=1e-8):
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps
        # 创建2D均值滤波器（所有权重为1/(kernel_size^2)）
        self.avg_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size**2)

    def forward(self, I, J):

        I = torch.from_numpy(I[np.newaxis, np.newaxis, :, :]).cuda()
        J = torch.from_numpy(J[np.newaxis, np.newaxis, :, :]).cuda()

        pad = self.kernel_size // 2
        # 使用2D卷积计算局部均值
        I_mean = F.conv2d(I, self.avg_kernel.to(I.device), padding=pad, groups=I.shape[1])
        J_mean = F.conv2d(J, self.avg_kernel.to(J.device), padding=pad, groups=J.shape[1])

        I_hat = I - I_mean
        J_hat = J - J_mean

        I_hat_J_hat = I_hat * J_hat
        I_hat_sq = I_hat ** 2
        J_hat_sq = J_hat ** 2

        N = self.kernel_size ** 2
        I_hat_J_hat_sum = F.conv2d(I_hat_J_hat, self.avg_kernel.to(I.device), padding=pad, groups=I.shape[1]) * N
        I_hat_sq_sum = F.conv2d(I_hat_sq, self.avg_kernel.to(I.device), padding=pad, groups=I.shape[1]) * N
        J_hat_sq_sum = F.conv2d(J_hat_sq, self.avg_kernel.to(J.device), padding=pad, groups=J.shape[1]) * N

        numerator = I_hat_J_hat_sum
        denominator = torch.sqrt(I_hat_sq_sum * J_hat_sq_sum + self.eps)
        cc = numerator / denominator

        return torch.mean(cc)
loss_fn = lpips.LPIPS(net='alex', verbose=False).cuda()
def LPIPS(p, a_z, a_p):
    mask1 = mask(p, a_z)
    mask2 = mask1[np.newaxis, np.newaxis, :, :]
    z = norm(a_z[np.newaxis, np.newaxis, :, :])
    p = norm(a_p[np.newaxis, np.newaxis, :, :])
    z_f=torch.from_numpy(z)
    p_f = torch.from_numpy(p)
    z_f = z_f.repeat(1, 3, 1, 1).cuda()
    p_f = p_f.repeat(1, 3, 1, 1).cuda()

    with torch.no_grad():
        d = loss_fn(z_f, p_f)
    return d.item()

loss_fn_v = lpips.LPIPS(net='vgg', verbose=False).cuda()
def LPIPS_v(p, a_z, a_p):
    mask1 = mask(p, a_z)
    mask2 = mask1[np.newaxis, np.newaxis, :, :]
    z = norm(a_z[np.newaxis, np.newaxis, :, :])
    p = norm(a_p[np.newaxis, np.newaxis, :, :])
    z_f=torch.from_numpy(z)
    p_f = torch.from_numpy(p)
    z_f = z_f.repeat(1, 3, 1, 1).cuda()
    p_f = p_f.repeat(1, 3, 1, 1).cuda()

    with torch.no_grad():
        d = loss_fn_v(z_f, p_f)
    return d.item()

import statistics
Lc_l=LocalNormalizedCrossCorrelation2D().cuda()

vgg=VGGLoss()
#method=['cgan','cyclegan','cyctran','regGan','resunet','resvit','SwinUNETR','TransUNet','UMTL','unit','zCTA']
#method=['our_nsgan','our_nsgan1']
def np_con(m='a',method='',patient=0):
    chest=np.load('F:/ct_enhance_chest1/test/'+m+'/'+m+'_'+method+'_'+str(patient)+'.npy')
    abdomen = np.load('F:/ct_enhance_abdomen/test/' +m+'/'+ m + '_' + method + '_' + str(patient) + '.npy')
    liver = np.load('F:/ct_enhance_liver/test/' +m+'/'+ m + '_' + method + '_' + str(patient) + '.npy')
    p = np.concatenate((chest,liver, abdomen), axis=2)
    return p
def np_con_d(m='d',method='',patient=0):
    chest_d=np.load('F:/ct_enhance_chest1/test/'+m+'/'+m+'_'+method+'_'+str(patient)+'.npy')
    chest = np.load('F:/ct_enhance_chest1/test/' + 'p' + '/' + 'p' + '_' + method + '_' + str(patient) + '.npy')

    chest1=chest[:,:,0:chest_d.shape[2]]
    abdomen = np.load('F:/ct_enhance_abdomen/test/' + 'p' + '/' + 'p' + '_' + method + '_' + str(patient) + '.npy')
    abdomen_d = np.load('F:/ct_enhance_abdomen/test/' +m+'/'+ m + '_' + method + '_' + str(patient) + '.npy')
    abdomen1 =abdomen[:, :, 0:abdomen_d.shape[2]]
    liver = np.load('F:/ct_enhance_liver/test/' +'p'+'/'+ 'p' + '_' + method + '_' + str(patient) + '.npy')
    p = np.concatenate((chest1,liver, abdomen1), axis=2)
    return p
s='a'
print(s)
method=['bianti2_50']
for m in range(len(method)):
    e_s_patient = []
    m_400_patient = []
    m_mae_patient = []
    m_ssim_patient = []
    lp_patient = []
    VGG_patient = []
    lc_patient = []
    for i in range(416,516):

        a = np.load('F:/ct_enhance_abdomen/test/a_xiaorong/a_'+method[m]+'_{}.npy'.format(i)).astype(np.float32)
        p = np.load('F:/ct_enhance_abdomen/test/p/p_z_{}.npy'.format(i)).astype(np.float32)
        a_z = np.load('F:/ct_enhance_abdomen/test/a/a_z_{}.npy'.format(i)).astype(np.float32)
        #a = np_con(s, method='z', patient=i).astype(np.float32)

        #p = np_con_d(s, method='z', patient=i).astype(np.float32)

        #a_z = np_con(s, method=method[m], patient=i).astype(np.float32)

        e_s_list=[]
        m_400_list=[]
        m_mae_list=[]
        m_ssim_list=[]
        lp_list=[]
        VGG_L_list=[]
        lc_list=[]
        #print(a.shape,p.shape,a_z.shape)
        for j in range(a_z.shape[2]):
            #print(i,j)
            a_c = a[:,:,j]
            p_c = p[:, :, j]
            a_z_c = a_z[:, :, j]


            e_s, _, _ = edge_ssim(a_c, p_c, edge_method='sobel')
            #print(e_s)
            m_400 = mae_400(a_c, p_c)
            m_mae = mask_mae(p_c, a_z_c, a_c)
            m_ssim = mask_ssim(p_c, a_z_c, a_c)
            lp = LPIPS(p_c, a_z_c, a_c)
            #VGG_L = vgg(p_c, a_z_c, a_c)
            VGG_L = LPIPS_v(p_c, a_z_c, a_c)
            lc=Lc_l(p_c,a_c)

            e_s_list.append(e_s)
            m_400_list.append(m_400)
            m_mae_list.append(m_mae)
            m_ssim_list.append(m_ssim)
            lp_list.append(lp)
            VGG_L_list.append(VGG_L)
            lc_list.append(lc.item())
        e_s_patient.append(sum(e_s_list)/len(e_s_list))
        m_400_patient.append(sum(m_400_list)/len(m_400_list))
        m_mae_patient.append(sum(m_mae_list)/len(m_mae_list))
        m_ssim_patient.append(sum(m_ssim_list)/len(m_ssim_list))
        lp_patient.append(sum(lp_list)/len(lp_list))
        VGG_patient.append(sum(VGG_L_list)/len(VGG_L_list))
        lc_patient.append(sum(lc_list) / len(lc_list))
    print(method[m])
    print(sum(e_s_patient)/len(e_s_patient))
    print(sum(m_400_patient)/len(m_400_patient))
    print(sum(m_mae_patient)/len(m_mae_patient))
    print(sum(m_ssim_patient)/len(m_ssim_patient))
    print(sum(lp_patient)/len(lp_patient))
    print(sum(VGG_patient)/len(VGG_patient))
    print(statistics.mean(e_s_patient),statistics.stdev(e_s_patient))
    print(statistics.mean(m_400_patient),statistics.stdev(m_400_patient))
    print(statistics.mean(m_mae_patient),statistics.stdev(m_mae_patient))
    print(statistics.mean(m_ssim_patient),statistics.stdev(m_ssim_patient))
    print(statistics.mean(lp_patient),statistics.stdev(lp_patient))
    print(statistics.mean(VGG_patient),statistics.stdev(VGG_patient))
    print(statistics.mean(lc_patient),statistics.stdev(lc_patient))

    np.savetxt('xr_abdomen+e_s_'+method[m]+'_'+s+'.txt',e_s_patient,delimiter=',')
    np.savetxt('xr_abdomen+m_400_'+method[m]+'_'+s+'.txt',m_400_patient,delimiter=',')
    np.savetxt('xr_abdomen+m_mae_'+method[m]+'_'+s+'.txt',m_mae_patient,delimiter=',')
    np.savetxt('xr_abdomen+m_ssim_'+method[m]+'_'+s+'.txt',m_ssim_patient,delimiter=',')
    np.savetxt('xr_abdomen+lp_'+method[m]+'_'+s+'.txt',lp_patient,delimiter=',')
    np.savetxt('xr_abdomen+VGG_'+method[m]+'_'+s+'.txt',VGG_patient,delimiter=',')
    np.savetxt('xr_abdomen+lc_'+method[m]+'_'+s+'.txt',lc_patient,delimiter=',')

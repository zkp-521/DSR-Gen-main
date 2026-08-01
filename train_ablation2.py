from networks_ournet import Model_xiaorong_resvit_shuangfenzhi
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.init as init
import math
import torch
import torch.nn as nn
from loss import EdgeSSIMLoss,VGGLoss,MsImageDis,registration_loss_2d
import transformer_configs
from dataset import ct_enhance_d,ct_enhance_d1,mr_head
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
print(torch.cuda.is_available())
gen_p=Model_xiaorong_resvit_shuangfenzhi(config=CONFIGS['Res-ViT-B_16'],input_dim=1, dim=64,n_downsample=3,n_res=3,activ='relu',pad_type='reflect').cuda()
R_A = Reg(256, 256, 1, 1).cuda()

optimizer_R_A = torch.optim.Adam(R_A.parameters(), lr=0.0001, betas=(0.5, 0.999))
spatial_transform = Transformer_2D().cuda()
dis_p = MsImageDis(input_dim=1,gan_type='nsgan').cuda()
params = sum(p.numel() for p in gen_p.parameters())
print('模型参数数量: ',params)
beta1 = 0.5
beta2 = 0.999
dis_p_params = list(dis_p.parameters())
gen_params = list(gen_p.parameters())

dis_opt_p = torch.optim.Adam([p for p in dis_p_params if p.requires_grad],
                                lr=0.0001, betas=(beta1, beta2), weight_decay=0.0001)
gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                lr=0.0001, betas=(beta1, beta2), weight_decay=0.0001)

dis_scheduler_p = lr_scheduler.StepLR(dis_opt_p, step_size=23,
                                        gamma=0.5, last_epoch=-1)
gen_scheduler = lr_scheduler.StepLR(gen_opt, step_size=23,
                                        gamma=0.5, last_epoch=-1)

dis_p.apply(weights_init('gaussian'))
dis_p.apply(weights_init('gaussian'))

EdgeSSIM_Loss = EdgeSSIMLoss()
VGG_loss = VGGLoss()
recon_MSE = nn.MSELoss()
recon_L1 = nn.L1Loss()

train_dataloader = DataLoader(dataset=mr_head(), batch_size=2, shuffle=True)
print(len(train_dataloader))
for epoch in range(55):
    for i, (d,p) in enumerate(train_dataloader):
        a = d.cuda()
        p = p.cuda()
        #mask=dp.cuda()

        gen_opt.zero_grad()
        content_p=gen_p.encode(p)
        x_tr,x_rec = gen_p.decode(content_p)
        loss_x_p_tr = recon_L1(x_tr, a-p)
        loss_x_p_recon = recon_L1(x_rec, a)
        loss_gen_adv_a = dis_p.calc_gen_loss(x_rec)
        loss_vgg = VGG_loss(x_rec, a)


        loss_all = 20 * loss_x_p_recon +1*loss_x_p_tr+ 1 * loss_vgg+0.5*loss_gen_adv_a

        loss_all.backward()
        gen_opt.step()

        dis_opt_p.zero_grad()
        loss_dis_p = dis_p.calc_dis_loss(x_rec.detach(), a)
        loss_dis_total = 1 * loss_dis_p
        loss_dis_total.backward()
        dis_opt_p.step()
        torch.cuda.synchronize()

        print(epoch, '/', i, loss_all.item(), loss_x_p_recon.item(),loss_vgg.item(),loss_gen_adv_a.item())
    if epoch >40:
        torch.save(gen_p.state_dict(), 'mrhead_ourmodexiaorong_cnn_shuangfenzhi1_20111_nsgan_guiyi_{}.pkl'.format(epoch))

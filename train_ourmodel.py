from networks_ournet import Model,Model_cnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.init as init
import math
import torch
import torch.nn as nn
from loss import EdgeSSIMLoss,VGGLoss,MsImageDis
import transformer_configs
from dataset import ct_enhance_d,ct_enhance_d1
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
gen_p=Model(config=CONFIGS['Res-ViT-B_16'],input_dim=1, dim=64,n_downsample=3,n_res=3,activ='relu',pad_type='reflect').cuda()

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

from scipy import ndimage
#import matplotlib.pyplot as plt
import os


EdgeSSIM_Loss = EdgeSSIMLoss()
VGG_loss = VGGLoss()
recon_MSE = nn.MSELoss()
recon_L1 = nn.L1Loss()

train_dataloader = DataLoader(dataset=ct_enhance_d1(), batch_size=2, shuffle=True)
print(len(train_dataloader))
for epoch in range(61):
    for i, (d,p,dp) in enumerate(train_dataloader):
        a = d.cuda()
        p = p.cuda()
        mask=dp.cuda()

        gen_opt.zero_grad()
        content_p=gen_p.encode(p)
        x_rec,x_tran,x_rec_tr = gen_p.decode(content_p)

        loss_x_p_recon = recon_MSE(x_rec, p)
        loss_x_tr_recon=recon_L1(x_tran*mask,(a-p)*mask)
        loss_x_trrec_recon = recon_L1(x_rec_tr * mask , a*mask)+recon_L1(x_rec_tr*(1-mask),p*(1-mask))
        loss_vgg = VGG_loss(x_rec_tr * mask, a * mask)
        loss_gen_adv_a = dis_p.calc_gen_loss(x_rec_tr * mask)


        loss_all = 10 * loss_x_p_recon + 2 * loss_vgg + loss_gen_adv_a + 20 * loss_x_tr_recon+20*loss_x_trrec_recon

        '''
        loss_x_p_recon = recon_MSE(x_rec, p)
        loss_vgg = VGG_loss(x_tran, a)
        loss_ssim = EdgeSSIM_Loss(x_tran, p)
        loss_gen_adv_a = dis_p.calc_gen_loss(x_tran)
        loss_l1 = recon_L1(x_tran * mask, a * mask) + 0.2 * recon_L1(x_tran * (1 - mask), p * (1 - mask))
        loss_all = 10 * loss_x_p_recon + 2 * loss_vgg + 2 * loss_ssim + loss_gen_adv_a + 10 * loss_l1
        '''

        '''
        loss_x_p_recon = recon_MSE(x_rec, p)
        loss_vgg = VGG_loss(x_tran*mask, a*mask)
        loss_gen_adv_a = dis_p.calc_gen_loss(x_tran)
        loss_l1 = recon_L1((x_tran-p) * mask, (a-p) * mask) + 0.1 * recon_L1(x_tran * (1 - mask), p * (1 - mask))
        loss_all = 10 * loss_x_p_recon + 2 * loss_vgg + loss_gen_adv_a + 10 * loss_l1
        '''
        '''
        loss_x_p_recon = recon_MSE(x_rec, p)
        #loss_vgg = VGG_loss((x_tran+p)*mask, a*mask)
        #loss_gen_adv_a = dis_p.calc_gen_loss(x_tran+p)
        loss_l1 = recon_L1(x_tran * mask, (a - p) * mask)
        #loss_all = 10 * loss_x_p_recon + 2 * loss_vgg + loss_gen_adv_a + 20 * loss_l1
        loss_all = 10 * loss_x_p_recon + 20 * loss_l1
        '''

        loss_all.backward()
        gen_opt.step()

        dis_opt_p.zero_grad()
        # encode

        # D loss

        loss_dis_p = dis_p.calc_dis_loss(x_rec_tr.detach()*mask, a*mask)
        loss_dis_total = 1 * loss_dis_p
        loss_dis_total.backward()
        dis_opt_p.step()
        torch.cuda.synchronize()

        print(epoch, '/', i, loss_all.item(), loss_x_trrec_recon.item(),loss_x_p_recon.item(),loss_x_tr_recon.item(),loss_vgg.item(),loss_gen_adv_a.item())
          #print(epoch, '/', i, 'loss_total:', loss_total.item(), 'Recon:', 5 * loss_x_p_recon.item(), 'vgg:',
        #      0.5 * vgg.item(), 'adv:', 0.5 * loss_gen_adv_a.item(),
        #      'ssim:',loss_ssim.item(),'dis:',loss_dis_total.item())
    if epoch >40:
        torch.save(gen_p.state_dict(), 'CT2_ourmodel_nsgan100-400_3_Q20_{}.pkl'.format(epoch))


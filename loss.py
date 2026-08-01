import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from networks_ournet import Conv2dBlock
import torchvision.models as models
from torch.autograd import Variable

def gaussian_kernel(channels: int, kernel_size: int = 11, sigma: float = 1.5) -> Tensor:

    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.0
    variance = sigma ** 2.0

    gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2.0 * variance)
    )
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    return gaussian_kernel


def sobel_edges(image: Tensor) -> Tensor:

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=image.dtype, device=image.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=image.dtype, device=image.device)

    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    channels = image.shape[1]
    sobel_x = sobel_x.repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1)

    grad_x = F.conv2d(image, sobel_x, padding=1, groups=channels)
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=channels)
    edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
    edge_map = torch.clamp(edge_map, 0, 1)

    return edge_map


def ssim_loss(img1: Tensor, img2: Tensor, kernel_size: int = 11, sigma: float = 1.5, C1: float = 0.01,
              C2: float = 0.03) -> Tensor:

    channels = img1.shape[1]
    kernel = gaussian_kernel(channels, kernel_size, sigma).to(img1.device)

    # 计算均值、方差、协方差
    mu1 = F.conv2d(img1, kernel, padding=kernel_size // 2, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=kernel_size // 2, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, kernel, padding=kernel_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, kernel, padding=kernel_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=kernel_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_val = ssim_map.mean()
    loss = 1 - ssim_val
    return loss

def norm(image):
    image_n=(image-image.min())/(image.max()-image.min())
    return image_n

class EdgeSSIMLoss(nn.Module):
    def __init__(self, kernel_size: int = 11, sigma: float = 1.5,
                 C1: float = 0.01, C2: float = 0.03, reduction: str = 'mean'):

        super(EdgeSSIMLoss, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.C1 = C1
        self.C2 = C2
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:

        if pred.shape[1] != 1 or target.shape[1] != 1:

            pred = pred.mean(dim=1, keepdim=True)
            target = target.mean(dim=1, keepdim=True)

        if pred.dim() == 5:
            B, C, D, H, W = pred.shape
            pred = pred.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            target = target.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        pred = norm(pred)
        target = norm(target)

        edge_pred = sobel_edges(pred)
        edge_target = sobel_edges(target)

        loss = ssim_loss(edge_pred, edge_target,
                         self.kernel_size, self.sigma,
                         self.C1, self.C2)

        return loss


class VGGLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        # 加载预训练的VGG16
        vgg = models.vgg16(pretrained=True).features
        #vgg.load_state_dict(torch.load('vgg16-397923af.pth'))
        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False


        self.vgg_layers = nn.Sequential(*list(vgg.children())[:23])  # 到conv4_3层
        self.device = device
        self.vgg_layers.to(device)

    def forward(self,a_z, a_p):


        z = norm(a_z)
        p = norm(a_p)
        generated = z.repeat(1,3,1,1)
        target = p.repeat(1,3,1,1)


        gen_features = self.vgg_layers(generated)
        target_features = self.vgg_layers(target)


        loss = nn.functional.mse_loss(gen_features, target_features)
        return loss


class MsImageDis(nn.Module):

    def __init__(self, input_dim=1, n_layer=4,gan_type='lsgan',
                 dim=32,norm='none',activ='lrelu',num_scales=3,pad_type='reflect'):
        super(MsImageDis, self).__init__()
        self.n_layer = n_layer
        self.gan_type = gan_type
        self.dim = dim
        self.norm = norm
        self.activ = activ
        self.num_scales =num_scales
        self.pad_type = pad_type
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):

        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

class LocalNormalizedCrossCorrelation2D(nn.Module):

    def __init__(self, kernel_size=9, eps=1e-8):
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps

        self.avg_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size**2)

    def forward(self, I, J):
        I = I.float()
        J = J.float()

        pad = self.kernel_size // 2

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

        return 1-torch.mean(cc)
class DiffusionRegularizer2D(nn.Module):

    def forward(self, phi):
        dx = torch.abs(phi[:, :, :, 1:] - phi[:, :, :, :-1])  # shape: (B,2,H,W-1)

        dy = torch.abs(phi[:, :, 1:, :] - phi[:, :, :-1, :])  # shape: (B,2,H-1,W)


        loss = (torch.mean(dx**2) + torch.mean(dy**2)) / 2.0
        return loss

def registration_loss_2d(I_fixed, I_moving_warped, phi, beta=1.0):

    lncc = LocalNormalizedCrossCorrelation2D(kernel_size=9)
    diffusion = DiffusionRegularizer2D()

    sim_loss = lncc(I_fixed, I_moving_warped)
    reg_loss = diffusion(phi)

    total_loss = sim_loss + beta * reg_loss
    return total_loss
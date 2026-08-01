import torch.utils.data as data
import numpy as np
import os

class ct_enhance_d(data.Dataset):
    def __init__(self, DATA_dir= 'F:/ct_enhance_liver/'):
        super(ct_enhance_d, self).__init__()
        self.DATA_dir = DATA_dir
        self.d_lit=os.listdir(DATA_dir+ 'p')
        #print(self.d_lit)


    def norm(self,image):
        image_n=(image-image.min())/(image.max()-image.min())*2-1
        return image_n[np.newaxis,:,:]

    def downsample_scipy(self,matrix_512, zoom_factor=0.5):
        matrix_256 = matrix_512/1000.0
        return matrix_256[np.newaxis,:,:].astype(np.float32)

    def mask(self,xingbian,ref):
        diff = np.abs(xingbian-ref)
        mask1 = np.ones_like(diff)
        mask1[diff >= 400] = 0
        return mask1[np.newaxis,:,:]

    def map_difference_image(self, tu, ji, high, low):
        diff = np.abs(tu - ji)
        result = np.zeros_like(diff, dtype=np.float32)
        result[diff < low] = 1.0
        result[diff > high] = 0
        mask_mid = (diff > low) & (diff < high)
        result[mask_mid] = 1 - (diff[mask_mid] - low) / (high - low)
        return result

    def __getitem__(self, index):
        d1 = np.load(self.DATA_dir + 'd/d_{}.npy'.format(index)).astype(np.float32)
        p1 = np.load(self.DATA_dir + 'p/p_{}.npy'.format(index)).astype(np.float32)
        mask_dp = self.map_difference_image(d1, p1, 400, 100)
        return d1[np.newaxis,:,:]/1000.0, p1[np.newaxis,:,:]/1000.0, mask_dp[np.newaxis,:,:]

    def __len__(self):
        return len(self.d_lit)

class ct_enhance_d1(data.Dataset):
    def __init__(self, DATA_dir= 'G:/ct_f/train/'):
        super(ct_enhance_d1, self).__init__()
        self.DATA_dir = DATA_dir
        self.d_lit=os.listdir(DATA_dir+ '1')
        #print(self.d_lit)


    def norm(self,image):
        image_n=(image-image.min())/(image.max()-image.min())*2-1
        return image_n[np.newaxis,:,:]

    def downsample_scipy(self,matrix_512, zoom_factor=0.5):
        matrix_256 = matrix_512/1000.0
        return matrix_256[np.newaxis,:,:].astype(np.float32)

    def mask(self,xingbian,ref):
        diff = np.abs(xingbian-ref)
        mask1 = np.ones_like(diff)
        mask1[diff >= 400] = 0
        return mask1[np.newaxis,:,:]

    def map_difference_image(self, tu, ji, high, low):
        diff = np.abs(tu - ji)
        result = np.zeros_like(diff, dtype=np.float32)
        result[diff < low] = 1.0
        result[diff > high] = 0
        mask_mid = (diff > low) & (diff < high)
        result[mask_mid] = 1 - (diff[mask_mid] - low) / (high - low)
        return result

    def __getitem__(self, index):
        d1 = np.load(self.DATA_dir + '1/c_{}.npy'.format(index)).astype(np.float32)
        p1 = np.load(self.DATA_dir + '1/p_{}.npy'.format(index)).astype(np.float32)
        mask_dp = self.map_difference_image(d1, p1, 400, 100)
        return d1[np.newaxis,:,:]/1000.0, p1[np.newaxis,:,:]/1000.0, mask_dp[np.newaxis,:,:]

    def __len__(self):
        return int(len(self.d_lit)/2)

class ct_enhance_d2(data.Dataset):
    def __init__(self, DATA_dir= 'G:/ct_f/train/'):
        super(ct_enhance_d2, self).__init__()
        self.DATA_dir = DATA_dir
        self.d_lit=os.listdir(DATA_dir+ '3')
        #print(self.d_lit)


    def norm(self,image):
        image_n=(image-image.min())/(image.max()-image.min())*2-1
        return image_n[np.newaxis,:,:]

    def downsample_scipy(self,matrix_512, zoom_factor=0.5):
        matrix_256 = matrix_512/1000.0
        return matrix_256[np.newaxis,:,:].astype(np.float32)

    def mask(self,xingbian,ref):
        diff = np.abs(xingbian-ref)
        mask1 = np.ones_like(diff)
        mask1[diff >= 400] = 0
        return mask1[np.newaxis,:,:]

    def map_difference_image(self, tu, ji, high, low):
        diff = np.abs(tu - ji)
        result = np.zeros_like(diff, dtype=np.float32)
        result[diff < low] = 1.0
        result[diff > high] = 0
        mask_mid = (diff > low) & (diff < high)
        result[mask_mid] = 1 - (diff[mask_mid] - low) / (high - low)
        return result

    def __getitem__(self, index):
        d1 = np.load(self.DATA_dir + '3/c_{}.npy'.format(index)).astype(np.float32)
        p1 = np.load(self.DATA_dir + '3/p_{}.npy'.format(index)).astype(np.float32)
        return d1[np.newaxis,:,:]/1000.0, p1[np.newaxis,:,:]/1000.0

    def __len__(self):
        return int(len(self.d_lit)/2)

class mr(data.Dataset):
    def __init__(self, DATA_dir='F:/mri/train/'):
        super(mr, self).__init__()
        self.DATA_dir = DATA_dir

    def downsample_scipy(self, matrix_512, zoom_factor=0.5):
        # matrix_512 = apply_window_level(matrix_512,500,50)
        # plt.imshow(matrix_512)
        # plt.show()
        matrix_256 = matrix_512 / 1000.0
        return matrix_256[np.newaxis, :, :].astype(np.float32)

    def __getitem__(self, index):
        c = self.downsample_scipy(np.load(self.DATA_dir + 'a/a_{}.npy'.format(index)).astype(np.float32))
        p = self.downsample_scipy(np.load(self.DATA_dir + 'p/p_{}.npy'.format(index)).astype(np.float32))

        return c, p

    def __len__(self):
        return 11566

class mr_head(data.Dataset):
    def __init__(self, DATA_dir='G:/mri1/mri/train/'):
        super(mr_head, self).__init__()
        self.DATA_dir = DATA_dir

    def downsample_scipy(self, matrix_512, zoom_factor=0.5):
        # matrix_512 = apply_window_level(matrix_512,500,50)
        # plt.imshow(matrix_512)
        # plt.show()
        if matrix_512.max()==0:
            matrix_256 = matrix_512
        else:matrix_256=(matrix_512-matrix_512.min())/(matrix_512.max()-matrix_512.min())*2-1
        #matrix_256 = matrix_512 / 400.0
        return matrix_256[np.newaxis, :, :].astype(np.float32)

    def __getitem__(self, index):
        c = self.downsample_scipy(np.load(self.DATA_dir + 't1c/t1c_{}.npy'.format(index)).astype(np.float32))
        p = self.downsample_scipy(np.load(self.DATA_dir + 't1n/t1n_{}.npy'.format(index)).astype(np.float32))

        return c, p

    def __len__(self):
        return 30123

class ct_abdomen(data.Dataset):
    def __init__(self, DATA_dir='D:/'):
        super(ct_abdomen, self).__init__()
        self.DATA_dir = DATA_dir

    def norm(self,image):
        image_n=(image-image.min())/(image.max()-image.min())*2-1
        return image_n[np.newaxis,:,:]

    def downsample_scipy(self,matrix_512, zoom_factor=0.5):
        #matrix_512 = apply_window_level(matrix_512,500,50)
        #plt.imshow(matrix_512)
        #plt.show()
        matrix_256 = matrix_512/1000.0
        return matrix_256[np.newaxis,:,:].astype(np.float32)

    def __getitem__(self, index):
        a = self.downsample_scipy(np.load(self.DATA_dir + 'a/a_{}.npy'.format(index)).astype(np.float32))
        a_pre = self.downsample_scipy(np.load(self.DATA_dir + 'a/a_our_nsgan1_{}.npy'.format(index)).astype(np.float32))
        #d = self.downsample_scipy(np.load(self.DATA_dir + 'd/d_{}.npy'.format(index)).astype(np.float32))
        #d1 = self.downsample_scipy(np.load(self.DATA_dir + 'd1/d_{}.npy'.format(index)).astype(np.float32))
        p = self.downsample_scipy(np.load(self.DATA_dir + 'p/p_{}.npy'.format(index)).astype(np.float32))
        #p1 = self.downsample_scipy(np.load(self.DATA_dir + 'p1/p_{}.npy'.format(index)).astype(np.float32))
        #v = self.downsample_scipy(np.load(self.DATA_dir + 'v/v_{}.npy'.format(index)).astype(np.float32))
        return a,p,a_pre
        #return v, p
        #return d1, p1
    def __len__(self):
        #return 30733
        return 48500
        #return 1
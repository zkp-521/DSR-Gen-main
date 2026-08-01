import numpy as np
from networks_ournet import Model,Model_cnn
import torch
import transformer_configs
from dataset import ct_enhance_d

configs=transformer_configs
CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-L_16': configs.get_l16_config(),
    'Res-ViT-B_16': configs.get_resvit_b16_config(),
    'Res-ViT-L_16': configs.get_resvit_l16_config(),
}
print(torch.cuda.is_available())
gen_p=Model(config=CONFIGS['Res-ViT-B_16'],input_dim=1, dim=64,n_downsample=3,n_res=3,activ='relu',pad_type='reflect').cuda()

gen_p.load_state_dict(torch.load('F:/our_model'+'/abdomen_ourmodel_nsgan_1_a_55.pkl'))

for n in range(416,516):
    print(n)
    p=np.load('F:/ct_enhance_chest1/test/p/p_z_{}.npy'.format(n))
    a = np.load('F:/ct_enhance_chest1/test/d/d_z_{}.npy'.format(n))
    a_p=np.zeros((256,256,a.shape[2])).astype(np.float16)

    for i in range(a.shape[2]):
        #print(i)
        p_c=p[:,:,i]
        p_c_c_f=torch.from_numpy(p_c[np.newaxis,np.newaxis,:,:].astype(np.float32))/1000.0
        f = gen_p.encode(p_c_c_f.cuda())
        x_re,x_tr,x_rec_tr=gen_p.decode(f)
        out_n = x_rec_tr.detach().cpu().numpy()[0, 0, :, :] * 1000
        out_n_1 = x_tr.detach().cpu().numpy()[0, 0, :, :] * 1000
        out_n_2 = x_re.detach().cpu().numpy()[0, 0, :, :] * 1000
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
    np.save('F:/ct_enhance_chest1/test/d/d_our_nsgan1_{}.npy'.format(n), a_p)
    #np.save('F:/ct_enhance_chest/test/p/p_z_{}.npy'.format(n), p_p)





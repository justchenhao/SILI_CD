from .resnet import *
from models.networks import init_net, BaseCD, CD_INR


def define_model(model_name='base_fcn_resnet18',
             pretrained='imagenet',
             n_class=2,
             init_type='normal',
             init_gain=0.02,
             gpu_ids=[],
             **kwargs):
    head_pretrained = kwargs.get('head_pretrained', False)
    frozen_backbone_weights = kwargs.get('frozen_backbone_weights', False)
    if model_name == 'base_fcn_resnet18':
        net = BaseCD(input_nc=3, output_nc=2, backbone_name='resnet18',
                     output_feat_nc=64, head_pretrained=head_pretrained,
                    pretrained=pretrained, backbone_structure='fcn',
                     frozen_backbone_weights=frozen_backbone_weights)
    elif model_name == 'ifa_lpe_resnet18_concat':
        net = CD_INR(input_nc=3, backbone_name='resnet18', output_feat_nc=64,
                     pretrained=pretrained, frozen_backbone_weights=frozen_backbone_weights,
                     fuse_mode='concat', ex_temporal_random=False, out_size=(64, 64),
                     learn_pe=True,
                     )
    elif model_name == 'ifa_inter234_local4n_lpe_edgeconv_up2_resnet18_concat':
        net = CD_INR(input_nc=3, backbone_name='resnet18', output_feat_nc=64,
                     pretrained=pretrained, frozen_backbone_weights=frozen_backbone_weights,
                     fuse_mode='concat', ex_temporal_random=False, out_size=(128, 128),
                     with_edge_info=True, edge_mode='conv_concat', edge_dim=3, learn_pe=True,
                     backbone_inter=True, bi_inter_type='local_attn', inter_levels=[2, 3, 4], window_size=4, )
    ######################################## compared cd ################################################
    elif model_name == 'DTCDSN':
        from compared_models.DTCDSN import CDNet34
        net = CDNet34(in_channels=3, )
    elif model_name == 'DSIFNet':
        from compared_models.DSIFN import DSIFN
        net = DSIFN()
    elif model_name == 'SRCDNet':
        from compared_models.SRDCD import CDNet
        net = CDNet(backbone_name='resnet18',)
    elif model_name == 'FC-EF':
        from compared_models.unet import Unet
        net = Unet(2 * 3, 2)
    elif model_name == 'FC-Siam-conc':
        from compared_models.siamunet_conc import SiamUnet_conc
        net = SiamUnet_conc(3, 2)
    elif model_name == 'FC-Siam-diff':
        from compared_models.siamunet_diff import SiamUnet_diff
        net = SiamUnet_diff(3, 2)
    elif model_name == 'DMINet':
        from compared_models.DMINet import DMINet
        net = DMINet(pretrained='imagenet')
    elif model_name == 'ICIFNet':
        from compared_models.ICIFNet import ICIFNet
        net = ICIFNet(pretrained='imagenet')
    elif model_name == 'changeformer':
        from compared_models.changeformer import ChangeFormerV6
        net = ChangeFormerV6()
    elif model_name == 'SUNet':
        from compared_models.SUNet import SUNnet
        net = SUNnet(in_ch=3, scale_ratios=kwargs.get('scale_ratios', 1))
    elif model_name == 'STANet':
        from compared_models.stanet import STANet
        net = STANet(in_c=3, f_c=64, mode='PAM')
    elif model_name == 'BIT':
        from compared_models.BIT import BASE_Transformer
        # 'base_transformer_pos_s4_dd8_dedim8'
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                               with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8,
                               pretrained=pretrained)
    elif model_name == 'SNUNet':
        from compared_models.SNUNet import SNUNet_ECAM
        net = SNUNet_ECAM()

    ##########################################################################################

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % model_name)
    return init_net(net, init_type, init_gain, gpu_ids)



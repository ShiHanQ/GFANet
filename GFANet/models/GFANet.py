from models.layers.hardnet_68 import hardnet
from models.layers import GatedSpatialConv as gsc
from models.layers.modules import *


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


class GFANet(nn.Module):
    def __init__(self, channel=32):
        super(GFANet, self).__init__()

        self.hardnet = hardnet(arch=68)

        # ------ CEM ------
        self.rfb2_1 = RFB_modified(320, channel)
        self.spp2_1 = SPPblock(channel)

        self.rfb3_1 = RFB_modified(640, channel)
        self.spp3_1 = SPPblock(channel)

        self.rfb4_1 = RFB_modified(1024, channel)
        self.spp4_1 = SPPblock(channel)

        # ------ GFD ------
        self.gff_head = GFD(num_classes=2, norm_layer=Norm2d)
        initialize_weights(self.gff_head)

        # ------ chanel reverse attention branch 4 ------
        self.ra4_conv1 = BasicConv2d(1024, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 2, kernel_size=1)
        self.att4 = MultiSpectralAttentionLayer(256, 7, 10, reduction=16, freq_sel_method='top16')

        # ------ chanel reverse attention branch 3 ------
        self.ra3_conv1 = BasicConv2d(640, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 2, kernel_size=3, padding=1)
        self.att3 = MultiSpectralAttentionLayer(64, 14, 20, reduction=16, freq_sel_method='top16')

        # ------ chanel reverse attention branch 2 ------
        self.ra2_conv1 = BasicConv2d(320, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 2, kernel_size=3, padding=1)
        self.att2 = MultiSpectralAttentionLayer(64, 28, 40, reduction=16, freq_sel_method='top16')

        # ------ GCF ------
        self.gate1 = gsc.GatedSpatialConv2d(256, 256)
        self.gate2 = gsc.GatedSpatialConv2d(64, 64)
        self.gate3 = gsc.GatedSpatialConv2d(64, 64)
        self.gate4 = gsc.GatedSpatialConv2d(128, 128)

        self.final_seg = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, bias=False))

        initialize_weights(self.final_seg)

    def forward(self, x):

        hardnetout = self.hardnet(x)

        x0 = hardnetout[0]  # (b, 64, 112, 160)
        x1 = hardnetout[1]  # (b, 128, 56, 80)
        x2 = hardnetout[2]  # (b, 320, 28, 40)
        x3 = hardnetout[3]  # (b, 640, 14, 20)
        x4 = hardnetout[4]  # (b, 1024, 7, 10)

        x2_rfb = self.rfb2_1(x2)
        x2_spp = self.spp2_1(x2_rfb)

        x3_rfb = self.rfb3_1(x3)
        x3_spp = self.spp3_1(x3_rfb)

        x4_rfb = self.rfb4_1(x4)
        x4_spp = self.spp4_1(x4_rfb)

        ra5_feat = self.gff_head(x2_spp, x3_spp, x4_spp)

        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear')

        # ------ chanel reverse attention branch_4 ------
        ra5_feat = ra5_feat[:, 1:, :, :]
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x4)
        x = self.ra4_conv1(x)
        x = self.att4(x)

        g1 = self.gate1(x, crop_4)

        x = F.relu(self.ra4_conv2(g1))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)

        lateral_map_4 = F.interpolate(ra4_feat, scale_factor=32, mode='bilinear')

        # ------ chanel reverse attention branch_3 ------
        ra4_feat = ra4_feat[:, 1:, :, :]
        crop_3 = F.interpolate(ra4_feat, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 640, -1, -1).mul(x3)
        x = self.ra3_conv1(x)
        x = self.att3(x)

        g2 = self.gate2(x, crop_3)

        x = F.relu(self.ra3_conv2(g2))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)

        lateral_map_3 = F.interpolate(ra3_feat, scale_factor=16, mode='bilinear')

        # ------ chanel reverse attention branch_2 ------
        ra3_feat = ra3_feat[:, 1:, :, :]
        crop_2 = F.interpolate(ra3_feat, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 320, -1, -1).mul(x2)
        x = self.ra2_conv1(x)
        x = self.att2(x)

        g3 = self.gate3(x, crop_2)

        x = F.relu(self.ra2_conv2(g3))
        ra3_fuse = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(ra3_fuse)

        lateral_map_2 = F.interpolate(ra2_feat, scale_factor=8, mode='bilinear')

        ra2_feat = ra2_feat[:, 1:, :, :]
        crop_1 = F.interpolate(ra2_feat, scale_factor=2, mode='bilinear')

        g4 = self.gate4(x1, crop_1)

        dec0 = self.final_seg(g4)
        seg_out = F.interpolate(dec0, scale_factor=4, mode='bilinear')

        return torch.sigmoid(lateral_map_5), torch.sigmoid(lateral_map_4), torch.sigmoid(lateral_map_3), \
               torch.sigmoid(lateral_map_2), torch.sigmoid(seg_out)


if __name__ == '__main__':
    ras = GFANet().cuda()
    input_tensor = torch.randn(2, 3, 224, 320).cuda()
    print('input_tensor: {}'.format(input_tensor.size()))

    out = ras(input_tensor)
    print('out0: {}'.format(out[0].size()))
    print('out1: {}'.format(out[1].size()))
    print('out2: {}'.format(out[2].size()))
    print('out3: {}'.format(out[3].size()))
    print('out4: {}'.format(out[4].size()))


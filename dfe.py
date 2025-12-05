
class DFE(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DFE, self).__init__()
        self.conv1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, relu=True),
            BasicConv2d(out_channel, out_channel, 3, padding=1, dilation=1, relu=True)
        )
        self.aspp = nn.ModuleList([
            BasicConv2d(out_channel, out_channel, 3, padding=rate, dilation=rate, relu=True)
            for rate in [1, 3, 5, 7]
        ])
        self.reduce_aspp = BasicConv2d(out_channel * 4, out_channel, 3, padding=1, relu=True)
        self.dw_conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1, groups=out_channel),
            nn.BatchNorm2d(out_channel),
            BasicConv2d(out_channel, out_channel, 1, relu=True)
        )
        self.dw_conv5 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1, groups=out_channel),
            nn.BatchNorm2d(out_channel),
            BasicConv2d(out_channel, out_channel, 1, relu=True)
        )
        self.dw_conv7 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 7, padding=3, dilation=1, groups=out_channel),
            nn.BatchNorm2d(out_channel),
            BasicConv2d(out_channel, out_channel, 1, relu=True)
        )
        self.reduce_dw = BasicConv2d(out_channel * 3, out_channel, 3, padding=1, relu=True)
        self.se_fusion = SELayer(out_channel * 2)
        self.res = nn.Sequential(
            BasicConv2d(in_channel, out_channel * 2, 3, padding=1, relu=True),
            BasicConv2d(out_channel * 2, out_channel, 3, padding=1, relu=True)
        )
        self.fuse_conv = BasicConv2d(out_channel * 3, out_channel, 3, padding=1, relu=True)

        self.gamma = nn.Parameter(torch.ones(1))   

    def forward(self, x):
        x1 = self.conv1(x)
        aspp_feats = [branch(x1) for branch in self.aspp]
        aspp_out = self.reduce_aspp(torch.cat(aspp_feats, dim=1))
        dw3 = self.dw_conv3(x1)
        dw5 = self.dw_conv5(x1)
        dw7 = self.dw_conv7(x1)
        dw_out = self.reduce_dw(torch.cat([dw3, dw5, dw7], dim=1))
        local_feat = self.se_fusion(torch.cat([aspp_out*(1-self.gamma), dw_out*self.gamma], dim=1))
        res_feat = self.res(x)
        out = self.fuse_conv(torch.cat([local_feat, res_feat], dim=1)) + x1
        return out

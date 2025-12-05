import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pvtv2 import pvt_v2_b2
from cdm import CDM
from dfe import DFE
from utils import BasicConv2d

class PVT_Net(nn.Module):
    def __init__(self, channel=32,):
        super(PVT_Net, self).__init__()

        self.backbone = pvt_v2_b2()  
        path = 'D:/shiguanai_code_over/models/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)


        self.rfb1_1 = DFE(64, channel)
        self.rfb2_1 = DFE(128, channel)
        self.rfb3_1 = DFE(320, channel)
        self.rfb4_1 = DFE(512, channel)

        self.agg1 = CDM(channel, 1)

        self.mfe1 = MFE(channel)
        self.mfe2 = MFE(channel)
        self.mfe3 = MFE(channel)


    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)

        x1_rfb = self.rfb1_1(x1)   
        x2_rfb = self.rfb2_1(x2)  
        x3_rfb = self.rfb3_1(x3) 
        x4_rfb = self.rfb4_1(x4) 
 
        feat_fg5, feat_bg5 = self.agg1(x3_rfb, x2_rfb, x1_rfb) 
        
        lateral_map_5_fg = F.interpolate(feat_fg5, scale_factor=4, mode='bilinear')
        lateral_map_5_bg = F.interpolate(feat_bg5, scale_factor=4, mode='bilinear')

        feat_fg4, feat_bg4 = self.mfe1(lateral_map_5_fg, lateral_map_5_bg, x4_rfb)

        lateral_map_4_fg = F.interpolate(feat_fg4, scale_factor=32, mode='bilinear')
        lateral_map_4_bg = F.interpolate(feat_bg4, scale_factor=32, mode='bilinear')

        feat_fg3, feat_bg3 = self.mfe2(lateral_map_4_fg, lateral_map_4_bg, x3_rfb)

        lateral_map_3_fg = F.interpolate(feat_fg3, scale_factor=16, mode='bilinear')
        lateral_map_3_bg = F.interpolate(feat_bg3, scale_factor=16, mode='bilinear')

        feat_fg2, feat_bg2 = self.mfe1(lateral_map_3_fg, lateral_map_3_bg, x2_rfb)

        lateral_map_2_fg = F.interpolate(feat_fg2, scale_factor=8, mode='bilinear')
        lateral_map_2_bg = F.interpolate(feat_bg2, scale_factor=8, mode='bilinear')

        return lateral_map_2_fg, lateral_map_3_fg, lateral_map_4_fg, lateral_map_5_fg, lateral_map_2_bg, lateral_map_3_bg, lateral_map_4_bg, lateral_map_5_bg

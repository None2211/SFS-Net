import torch
import torch.nn as nn
import torch.nn.functional as F
from CSwin import cswin_small


class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class outconv(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(outconv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_planes,out_planes,3, stride=2, padding=1, output_padding=1)
        self.norm = nn.LayerNorm(out_planes,eps=1e-6)
        self.act = nn.GELU()

    def forward(self,x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # n,c,h,w -> n,h,w,c
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # n,h,w,c -> n,c,h,w
        x = self.act(x)

        return x
class decoderbox(nn.Module):#n,c,h,w -> n,c/2,2h,2w
    def __init__(self,in_planes,out_planes):
        super(decoderbox,self).__init__()
        #b,c,h,w -> b,c/4,h,w
        self.eca = eca_layer(channel=in_planes)
        self.act = nn.GELU()
        self.conv1 = nn.Conv2d(in_planes,in_planes // 4, kernel_size=3,stride=1,padding=1,bias=False)
        self.norm1 = nn.LayerNorm(in_planes // 4,eps=1e-6)
        #n,c,h,w -> n,c/4,2h,2w
        self.deconv = nn.ConvTranspose2d(in_planes // 4, in_planes // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.LayerNorm(in_planes // 4, eps=1e-6)
        #n,c/4,h,w -> n,c/2,h,w
        self.conv2 = nn.Conv2d(in_planes // 4, out_planes, kernel_size=3,stride=1,padding=1,bias=False)
        self.norm3 = nn.LayerNorm(out_planes, eps=1e-6)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,x):

        x = self.eca(x)
        x = self.conv1(x)
        x = x.permute(0,2,3,1)
        x = self.norm1(x)
        x = self.act(x)
        x = x.permute(0,3,1,2)
        x = self.deconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)
        # n,c/4,2h,2w -> n,c/2,2h,2w
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm3(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)

        return x


class RFFEL(nn.Module):
    def __init__(self, in_channels):
        super(RFFEL, self).__init__()
        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1,,groups=in_channels *2)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        fft_x = torch.fft.rfft2(x, norm='backward')
        z = torch.cat([fft_x.real, fft_x.imag], dim=1)
        z = self.freq_conv(z)
        real, imag = torch.chunk(z, 2, dim=1)
        fft_new = torch.complex(real, imag)
        out = torch.fft.irfft2(fft_new, s=(H, W), norm='backward')
        return x + out



class FSIL(nn.Module):
    def __init__(self, dim, reduction=8):
        super(FSIL, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim * 2, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_trans, x_sp):
        B, C, H, W = x_trans.shape
        if x_sp.shape[2:] != x_trans.shape[2:]:
            x_sp = F.interpolate(x_sp, size=(H, W), mode='bilinear', align_corners=False)

        x_sum = x_trans + x_sp
        s = self.avg_pool(x_sum).view(B, C)
        weights = self.fc(s).view(B, 2, C)
        weights = self.softmax(weights)
        w_trans = weights[:, 0, :].view(B, C, 1, 1)
        w_sp = weights[:, 1, :].view(B, C, 1, 1)
        return x_trans * w_trans + x_sp * w_sp



class FSGB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FSGB, self).__init__()
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.structure_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),  # IN 提取纯净边缘
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()  # 0~1 Gate
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        feat_main = self.main_conv(x)
        gate = self.structure_conv(x)
        out = feat_main + (feat_main * gate) * self.alpha
        out = self.out_conv(out)
        return out

class RFFE(nn.Module):
    def __init__(self):
        super(RFFE, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.spec1 = RFFEL(64)


        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.spec2 = RFFEL(128)


        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.spec3 = RFFEL(256)

        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.spec4 = RFFEL(512)

    def forward(self, x):
        f1 = self.stem(x)
        f1 = self.spec1(f1)
        f2 = self.down1(f1)
        f2 = self.spec2(f2)
        f3 = self.down2(f2)
        f3 = self.spec3(f3)
        f4 = self.down3(f3)
        f4 = self.spec4(f4)

        return f1, f2, f3, f4

class sfsnet(nn.Module):
    def __init__(self, num_class, num_cls):
        super(sfsnet, self).__init__()
        self.backbone = cswin_small()
        path = 'cswin_small_224.pth'
        save_model = torch.load(path, map_location='cpu')
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model['state_dict_ema'].items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.sp_injector = RFFE()
        self.fusion1 = FSIL(dim=64)
        self.fusion2 = FSIL(dim=128)
        self.fusion3 = FSIL(dim=256)
        self.fusion4 = FSIL(dim=512)
        self.mix = nn.Parameter(torch.FloatTensor(7))
        self.mix.data.fill_(1)
        self.up5 = decoderbox(512, 256)
        self.up4 = decoderbox(256, 128)
        self.up3 = decoderbox(128, 64)
        self.up2 = decoderbox(64, 64)  # Keep 64 channels
        self.gate5 = FSGB(256, 256)
        self.gate4 = FSGB(128, 128)
        self.gate3 = FSGB(64, 64)
        self.gate2 = FSGB(64, 64)

        # Output & Heads
        self.outconv = outconv(64, num_class)
        self.logit1 = nn.Conv2d(64, num_class, kernel_size=1)
        self.logit2 = nn.Conv2d(64, num_class, kernel_size=1)
        self.logit3 = nn.Conv2d(128, num_class, kernel_size=1)
        self.logit0 = nn.Conv2d(num_class, num_class, kernel_size=1)
        self.logit5 = nn.Conv2d(512, num_class, kernel_size=1)
        self.logit6 = nn.Conv2d(256, num_class, kernel_size=1)

        self.num_class = num_class
        self.num_cls = num_cls
        self.cls_head = nn.Linear(256, num_cls)

    def forward(self, x, superpixel):
        _, _, H, W = x.shape
        sp_feats = self.sp_injector(superpixel)
        sp1, sp2, sp3, sp4 = sp_feats
        trans_feats = self.backbone(x)
        t1, t2, t3, t4 = trans_feats

        e1 = self.fusion1(t1, sp1)
        e2 = self.fusion2(t2, sp2)
        e3 = self.fusion3(t3, sp3)
        e4 = self.fusion4(t4, sp4)

        e5 = e4


        up5 = self.up5(e5)
        up5 = up5 + e3
        up5 = self.gate5(up5)
        up4 = self.up4(up5)
        up4 = up4 + e2
        up4 = self.gate4(up4)
        up3 = self.up3(up4)
        up3 = up3 + e1
        up3 = self.gate3(up3)

        up2 = self.up2(up3)
        up2 = self.gate2(up2)

        out = self.outconv(up2)

        logit1 = F.interpolate(self.logit1(up2), size=(H, W), mode='bilinear', align_corners=False)
        logit2 = F.interpolate(self.logit2(up3), size=(H, W), mode='bilinear', align_corners=False)
        logit3 = F.interpolate(self.logit3(up4), size=(H, W), mode='bilinear', align_corners=False)
        logit0 = F.interpolate(self.logit0(out), size=(H, W), mode='bilinear', align_corners=False)
        logit5 = F.interpolate(self.logit5(e5), size=(H, W), mode='bilinear', align_corners=False)
        logit6 = F.interpolate(self.logit6(up5), size=(H, W), mode='bilinear', align_corners=False)
        logit = self.mix[1] * logit1 + self.mix[2] * logit2 + self.mix[3] * logit3 + \
                self.mix[4] * logit0 + self.mix[5] * logit5 + self.mix[6] * logit6

        return logit

if __name__ == "__main__":
        from thop import profile

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_img = torch.randn(1, 3, 224, 224).to(device)
        sp_img = torch.randn(1, 3, 224, 224).to(device)
        model = sfsnet(num_cls=1,num_class=1).to(device)


        flops, params = profile(model, inputs=(input_img,sp_img))


        print(f"Total Parameters: {params / 1e6:.2f} M")

        print(f"Total GFLOPs: {flops / 1e9:.3f} G")

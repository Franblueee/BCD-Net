import torch

class ResBlock(torch.nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1):
        super().__init__()
        assert in_chn == out_chn, "Number of in-channels must be equal to the number of out-channels."
        self.layers = torch.nn.Sequential(
           torch.nn.Conv2d(in_chn, out_chn, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=kernel_size // 2),
           torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
           torch.nn.Conv2d(out_chn, out_chn, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=kernel_size // 2),
        )

    def forward(self, x):
        out = self.layers(x) + x
        return out

class ConvBlock(torch.nn.Module):
    def __init__(self, in_chn, out_chn, num_blocks):
        super().__init__()
        self.layers = torch.nn.ModuleList([ResBlock(in_chn, out_chn) for _ in range(num_blocks)])

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
        return out + x


class DownBlock(torch.nn.Module):
    def __init__(self, in_chn, out_chn, num_blocks):
        super().__init__()
        self.conv_layers = ConvBlock(in_chn, in_chn, num_blocks=num_blocks)
        self.down_sample = torch.nn.Conv2d(in_chn, out_chn, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        skip = self.conv_layers(x)
        out = self.down_sample(skip)
        return out, skip


class UpBlock(torch.nn.Module):
    def __init__(self, in_chn, out_chn, num_blocks):
        super().__init__()
        self.up_sample = torch.nn.ConvTranspose2d(in_chn, out_chn, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
        self.conv = torch.nn.Conv2d(out_chn * 2, out_chn, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_block = ConvBlock(out_chn, out_chn, num_blocks=num_blocks)

    def center_crop(self, feature_map, target_size):
        _, _, height,width = feature_map.size()
        diff_y = (height - target_size[0]) // 2
        diff_x = (width - target_size[1]) // 2
        return feature_map[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, skip):
        up = self.up_sample(x)
        crop_feature_map = self.center_crop(up, skip.shape[2:])
        out = torch.cat([skip, crop_feature_map], 1)
        out = self.conv(out)
        out = self.conv_block(out)
        return out


class Cnet(torch.nn.Module):
    def __init__(self, in_nc=3, out_nc=2, nc=64, num_blocks=3):
        super().__init__()
        self.first_conv = torch.nn.Conv2d(in_nc, nc, kernel_size=3, stride=1, padding=1)

        self.down1 = DownBlock(nc, nc, num_blocks=num_blocks)
        self.down2 = DownBlock(nc, nc, num_blocks=num_blocks)
        self.down3 = DownBlock(nc, nc, num_blocks=num_blocks)
        self.down4 = DownBlock(nc, nc, num_blocks=num_blocks)

        self.mid_conv = ConvBlock(nc, nc, num_blocks=6)

        self.up4 = UpBlock(nc, nc, num_blocks=num_blocks)
        self.up3 = UpBlock(nc, nc, num_blocks=num_blocks)
        self.up2 = UpBlock(nc, nc, num_blocks=num_blocks)
        self.up1 = UpBlock(nc, nc, num_blocks=num_blocks)

        self.skip_conv1 = torch.nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1)
        self.skip_conv2 = torch.nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1)
        self.skip_conv3 = torch.nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1)
        self.skip_conv4 = torch.nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1)

        self.final_conv = torch.nn.Sequential(*[
            torch.nn.Conv2d(nc, nc//4, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(nc//4, out_nc, kernel_size=3, stride=1, padding=1), 
        ])

    def forward(self, x):
        out = self.first_conv(x)
        out, skip1 = self.down1(out)
        out, skip2 = self.down2(out)
        out, skip3 = self.down3(out)
        out, skip4 = self.down4(out)

        out = self.mid_conv(out)

        out = self.up4(out, self.skip_conv4(skip4))
        out = self.up3(out, self.skip_conv3(skip3))
        out = self.up2(out, self.skip_conv2(skip2))
        out = self.up1(out, self.skip_conv1(skip1))

        out = self.final_conv(out)
        return out
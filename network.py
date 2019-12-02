import torch
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, conv, bn, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            conv(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            bn(out_channels),
            torch.nn.ReLU(),
            conv(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            bn(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, conv, bn, tpconv, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            conv(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            bn(mid_channel),
            torch.nn.ReLU(),
            conv(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            bn(mid_channel),
            torch.nn.ReLU(),
            tpconv(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            bn(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, conv, bn, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
                    conv(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                    bn(mid_channel),
                    torch.nn.ReLU(),
                    conv(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                    bn(out_channels),
                    torch.nn.ReLU()
                )
        return block

    def __init__(self, in_channel, out_channel, dim=2):
        super(UNet, self).__init__()

        conv = torch.nn.Conv2d
        bn = torch.nn.BatchNorm2d
        mp = torch.nn.MaxPool2d
        tpconv = torch.nn.ConvTranspose2d
        if dim==3:
            conv = torch.nn.Conv3d
            bn = torch.nn.BatchNorm3d
            mp = torch.nn.MaxPool3d
            tpconv = torch.nn.ConvTranspose3d

        #Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32, conv=conv, bn=bn)
        self.conv_maxpool1 = mp(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 64, conv=conv, bn=bn)
        self.conv_maxpool2 = mp(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 128, conv=conv, bn=bn)
        self.conv_maxpool3 = mp(kernel_size=2)
        # Bottleneck
        mid_channel = 128
        self.bottleneck = torch.nn.Sequential(
                                conv(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
                                bn(mid_channel * 2),
                                torch.nn.ReLU(),
                                conv(kernel_size=3, in_channels=mid_channel*2, out_channels=mid_channel, padding=1),
                                bn(mid_channel),
                                torch.nn.ReLU(),
                                tpconv(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                                bn(mid_channel),
                                torch.nn.ReLU(),
                            )
        # Decode
        self.conv_decode3 = self.expansive_block(256, 128, 64, conv=conv, bn=bn, tpconv=tpconv)
        self.conv_decode2 = self.expansive_block(128, 64, 32, conv=conv, bn=bn, tpconv=tpconv)
        self.final_layer = self.final_block(64, 32, out_channel, conv=conv, bn=bn)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        final_layer = self.final_layer(decode_block1)
        return  final_layer

class SpatialTransformation(nn.Module):
    def __init__(self):
        super(SpatialTransformation, self).__init__()
        self.gpu=True

    def meshgrid(self, height, width):
        x_t = torch.matmul(torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(0.0, width -1.0, width), 1), 1, 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]))

        x_t = x_t.expand([height, width])
        y_t = y_t.expand([height, width])

        if self.gpu:
            x_t = x_t.cuda()
            y_t = y_t.cuda()

        return x_t, y_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        # if self.gpu:
        #     x=x.cuda()
        return torch.squeeze(torch.reshape(x, (-1, 1)))

    def interpolate(self, im, x, y):

        im = F.pad(im, (0,0,1,1,1,1,0,0)).cpu()

        batch_size, height, width, channels = im.shape

        batch_size, out_height, out_width = x.shape

        x = x.cpu().reshape(1, -1)
        y = y.cpu().reshape(1, -1)

        x = x + 1
        y = y + 1

        max_x = width - 1
        max_y = height - 1

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width*height
        base = self.repeat(torch.arange(0, batch_size)*dim1, out_height*out_width)

        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = torch.reshape(im, [-1, channels])
        im_flat = im_flat.float()
        dim, _ = idx_a.transpose(1,0).shape
        Ia = torch.gather(im_flat, 0, idx_a.transpose(1,0).expand(dim, channels))
        Ib = torch.gather(im_flat, 0, idx_b.transpose(1,0).expand(dim, channels))
        Ic = torch.gather(im_flat, 0, idx_c.transpose(1,0).expand(dim, channels))
        Id = torch.gather(im_flat, 0, idx_d.transpose(1,0).expand(dim, channels))

        # and finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()

        dx = x1_f - x
        dy = y1_f - y

        wa = (dx * dy).transpose(1,0)
        wb = (dx * (1-dy)).transpose(1,0)
        wc = ((1-dx) * dy).transpose(1,0)
        wd = ((1-dx) * (1-dy)).transpose(1,0)

        interpolated = torch.sum(torch.squeeze(torch.stack([wa*Ia, wb*Ib, wc*Ic, wd*Id], dim=1)), 1)
        interpolated = torch.reshape(interpolated, [-1, out_height, out_width, channels])
        return interpolated.cuda()

    def forward(self, moving_image, deformation_matrix):
        dx = deformation_matrix[:, :, :, 0]
        dy = deformation_matrix[:, :, :, 1]

        batch_size, height, width = dx.shape

        x_mesh, y_mesh = self.meshgrid(height, width)

        x_mesh = x_mesh.expand([batch_size, height, width])
        y_mesh = y_mesh.expand([batch_size, height, width])
        x_new = dx + x_mesh
        y_new = dy + y_mesh

        return self.interpolate(moving_image, x_new, y_new)

class VM_Net(nn.Module):
    def __init__(self, in_channels, dim):
        super(VM_Net, self).__init__()
        self.unet = UNet(in_channels, out_channel=2, dim=2).cuda()
        if dim==3:
            self.unet = UNet(in_channel=1, out_channel=3, dim=3)
        self.spatial_transform = SpatialTransformation().cuda()

    def forward(self, moving_image, fixed_image):
        x = torch.cat([moving_image, fixed_image], dim=3).permute(0,3,1,2).cuda()
        deformation_matrix = self.unet(x).permute(0,2,3,1)
        moving_image = moving_image.cuda()
        registered_image = self.spatial_transform(moving_image, deformation_matrix)
        return registered_image
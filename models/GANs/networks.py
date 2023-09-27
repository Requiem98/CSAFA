import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import spectral_norm as SpectralNorm



class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None):
        super(ConvLayer, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride))

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

##### non-local block #####
class Attention(nn.Module):
  def __init__(self, ch):
    super(Attention, self).__init__()

    self.ch = ch
    self.theta = SpectralNorm(nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False))
    self.phi = SpectralNorm(nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False))
    self.g = SpectralNorm(nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False))
    self.o = SpectralNorm(nn.Conv2d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False))

    self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None):

    theta = self.theta(x)
    phi = F.max_pool2d(self.phi(x), [2,2])
    g = F.max_pool2d(self.g(x), [2,2])

    theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
    phi = phi.view(-1, self. ch // 8, phi.shape[2]*phi.shape[3])
    g = g.view(-1, self. ch // 2, g.shape[2] * g.shape[3])

    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)

    o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
    return self.gamma * o + x

class UnetGeneratorSkip(nn.Module):

    def __init__(self, input_nc=3):
        super(UnetGeneratorSkip, self).__init__()

        self.in_dim = input_nc
        input_ch  = input_nc

        self.begin_pad = nn.ReflectionPad2d(3)
        self.begin_conv = SpectralNorm(nn.Conv2d(input_ch, 32, kernel_size=7, padding=0))
        self.begini_e = nn.InstanceNorm2d(32, affine=True)
        self.conv1 = ResidualBlockDown(32, 64)
        self.in1_e = nn.InstanceNorm2d(64, affine=True)
        self.conv2 = ResidualBlockDown(64, 128)
        self.in2_e = nn.InstanceNorm2d(128, affine=True)
        self.conv3 = ResidualBlockDown(128, 256)
        self.in3_e = nn.InstanceNorm2d(256, affine=True)

        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.res3 = ResidualBlock(256)
        self.res4 = ResidualBlock(256)
        self.res5 = ResidualBlock(256)
        self.res6 = ResidualBlock(256)
        
        self.attention1 = Attention(512)
        self.attention2 = Attention(256)
        self.deconv3 = ResidualBlockUp(512, 128, upsample=2)
        self.in3_d = nn.InstanceNorm2d(128, affine=True)
        self.deconv2 = ResidualBlockUp(256, 64, upsample=2)
        self.in2_d = nn.InstanceNorm2d(64, affine=True)
        self.deconv1 = ResidualBlockUp(128, 64, upsample=2)
        self.in1_d = nn.InstanceNorm2d(64, affine=True)

        self.conv_end = nn.Sequential(nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1), nn.Tanh())

    def forward(self, x):

        x = self.begin_conv(self.begin_pad(x))
        x = self.begini_e(x)
        enc1 = self.in1_e(self.conv1(x))
        enc2 = self.in2_e(self.conv2(enc1))
        enc3 = self.in3_e(self.conv3(enc2))

        # Residual layers
        resb1 = self.res1(enc3)
        resb2 = self.res2(resb1)
        resb3 = self.res3(resb2)
        resb4 = self.res4(resb3)
        resb5 = self.res5(resb4)
        resb6 = self.res6(resb5)

        dec1 = torch.cat((resb6, enc3), 1)
        dec1att = self.attention1(dec1)
        dec2 = self.in3_d(self.deconv3(dec1att))
        dec2 = torch.cat((dec2, enc2), 1)
        dec2att = self.attention2(dec2)
        dec3 = self.in2_d(self.deconv2(dec2att))
        dec3 = torch.cat((dec3, enc1), 1)
        dec4 = self.in1_d(self.deconv1(dec3))

        out = self.conv_end(dec4)
        return out, resb6

#####Residual Block Down#####
class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(ResidualBlockDown, self).__init__()
        # Right Side
        self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride, padding)

        # Left Side
        self.conv_l = ConvLayer(in_channels, out_channels, 1, 1)
    def forward(self, x):
        residual = x

        out = F.relu(x)
        out = self.conv_r1(out)
        out = F.relu(out)
        out = self.conv_r2(out)
        out = F.avg_pool2d(out, 2)

        residual = self.conv_l(residual)
        residual = F.avg_pool2d(residual, 2)
        out = residual + out
        return out

#####Residual Block Up#####
class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=None, out_spatial=None):
        super(ResidualBlockUp, self).__init__()
        if upsample != None:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        elif out_spatial !=None:
            self.upsample = nn.Upsample(size=out_spatial, mode='nearest')

        # Right Side
        self.norm_r1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride)

        self.norm_r2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride)

        # Left Side
        self.conv_l = ConvLayer(in_channels, out_channels, 1, 1)

    def forward(self, x):
        residual = x

        out = self.norm_r1(x)
        out = F.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = self.norm_r2(out)
        out = F.relu(out)
        out = self.conv_r2(out)

        residual = self.upsample(residual)
        residual = self.conv_l(residual)

        out = residual + out
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.in2(out)

        out = out + residual
        return out


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


##############################################################################
# Classes
##############################################################################


# ------------This block is for XFork model --------------------------
class XForkGenerator(nn.Module):
    """Define a Resnet block"""

    def __init__(self, input_nc, norm_layer, use_dropout):
        super(XForkGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.shared_encoder = XFork_Shared_Encoder_Block(input_nc, norm_layer, use_dropout, use_bias)
        self.shared_decoder = XFork_Shared_Decoder_Block(512, norm_layer, use_dropout, use_bias)
        # self.share_generator = [shared_encoder, shared_decoder]
        self.img_decoder = XFork_Separate_Decoder_Block1(128, 3, norm_layer, use_dropout, use_bias)
        self.seg_decoder = XFork_Separate_Decoder_Block2(128, 3, norm_layer, use_dropout, use_bias)

    def forward(self, x):
        """Forward function (with skip connections)"""
        # print(self.shared_encoder)
        shared_feat = self.shared_encoder(x)
        # print(shared_feat.size())
        shared_feat = self.shared_decoder(shared_feat)
        # print(shared_feat.size())
        fake_pano = self.img_decoder(shared_feat)
        # print(fake_pano.size())
        fake_segmap = self.seg_decoder(shared_feat)
        # print(fake_segmap.size())

        return fake_pano, fake_segmap


class XFork_Shared_Encoder_Block(nn.Module):
    def __init__(self, input_nc, norm_layer, use_dropout, use_bias):
        super(XFork_Shared_Encoder_Block, self).__init__()

        generator_block = []

        negative_slope = 0.2
        nc_input_tmp = input_nc
        nc_output_tmp = 64
        conv = nn.Conv2d(nc_input_tmp, nc_output_tmp, kernel_size=4, stride=2, padding=1, bias=use_bias)  # 128x128
        act = nn.LeakyReLU(negative_slope)
        generator_block += [conv, act]

        nc_input_tmp = nc_output_tmp
        nc_output_tmp = 128
        conv = nn.Conv2d(nc_input_tmp, nc_output_tmp, kernel_size=4, stride=2, padding=1, bias=use_bias)  # 64x64
        norm = norm_layer(nc_output_tmp)  # swap norm and act for fine-turning
        act = nn.LeakyReLU(negative_slope)
        generator_block += [conv, norm, act]

        nc_input_tmp = nc_output_tmp
        nc_output_tmp = 256
        conv = nn.Conv2d(nc_input_tmp, nc_output_tmp, kernel_size=4, stride=2, padding=1, bias=use_bias)  # 32x32
        norm = norm_layer(nc_output_tmp)  # swap norm and act for fine-turning
        act = nn.LeakyReLU(negative_slope)
        generator_block += [conv, norm, act]

        for i in range(1, 5):  # 16x16 -> 8x8 -> 4x4 -> 2x2 ->1x1
            nc_input_tmp = nc_output_tmp
            nc_output_tmp = 512
            conv = nn.Conv2d(nc_input_tmp, nc_output_tmp, kernel_size=4, stride=2, padding=1, bias=use_bias)
            norm = norm_layer(nc_output_tmp)  # swap norm and act for fine-turning
            act = nn.LeakyReLU(negative_slope)
            generator_block += [conv, norm, act]

        # note that there is no norm layer here, because the feature map is in size [N, C, 1, 1]
        nc_input_tmp = nc_output_tmp
        nc_output_tmp = 512
        conv = nn.Conv2d(nc_input_tmp, nc_output_tmp, kernel_size=4, stride=2, padding=1, bias=use_bias)
        norm = nn.BatchNorm2d(nc_output_tmp)
        act = nn.ReLU(True)
        generator_block += [conv, norm, act]

        self.model = nn.Sequential(*generator_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        return self.model(x)


class XFork_Shared_Decoder_Block(nn.Module):
    def __init__(self, input_nc, norm_layer, use_dropout, use_bias):
        super(XFork_Shared_Decoder_Block, self).__init__()

        generator_block = []
        for i in range(3):
            if i == 0:
                nc_input_tmp = input_nc
            else:
                nc_input_tmp = nc_output_tmp
            nc_output_tmp = 512
            tconv = nn.ConvTranspose2d(nc_input_tmp, nc_output_tmp, kernel_size=4, stride=2, padding=1,
                                       bias=use_bias)  #
            norm = norm_layer(nc_output_tmp)  # swap norm and act for fine-turning
            drop = nn.Dropout(0.5)
            act = nn.ReLU(True)
            generator_block += [tconv, norm, drop, act]  # 2x2
        # '''
        for i in range(3):
            out_nc_array = [512, 256, 128]
            nc_input_tmp = nc_output_tmp
            nc_output_tmp = out_nc_array[i]
            tconv = nn.ConvTranspose2d(nc_input_tmp, nc_output_tmp, kernel_size=4, stride=2, padding=1,
                                       bias=use_bias)  # upsampling by 2 times: 64 x 256
            norm = norm_layer(nc_output_tmp)  # swap norm and act for fine-turning
            drop = nn.Dropout(0.5)
            act = nn.ReLU(True)
            generator_block += [tconv, norm, act]  # 4x4
        # '''
        self.model = nn.Sequential(*generator_block)

    def forward(self, x):
        return self.model(x)


class XFork_Separate_Decoder_Block1(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer, use_dropout, use_bias):
        super(XFork_Separate_Decoder_Block1, self).__init__()

        generator_block = []
        nc_input_tmp = input_nc
        nc_output_tmp = 64
        tconv = nn.ConvTranspose2d(nc_input_tmp, nc_output_tmp, kernel_size=4, stride=2, padding=1,
                                   bias=use_bias)
        norm = norm_layer(nc_output_tmp)  # swap norm and act for fine-turning
        act = nn.ReLU(True)
        generator_block += [tconv, norm, act]  # 128 x 512

        nc_input_tmp = nc_output_tmp
        nc_output_tmp = output_nc
        tconv = nn.ConvTranspose2d(nc_input_tmp, nc_output_tmp, kernel_size=4, stride=2, padding=1,
                                   bias=use_bias)  # upsampling by 2 times: 64 x 256
        act = nn.Tanh()
        generator_block += [tconv, act]

        self.model = nn.Sequential(*generator_block)

    def forward(self, x):
        return self.model(x)


class XFork_Separate_Decoder_Block2(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer, use_dropout, use_bias):
        super(XFork_Separate_Decoder_Block2, self).__init__()

        generator_block = []
        nc_input_tmp = input_nc
        nc_output_tmp = 64
        tconv = nn.ConvTranspose2d(nc_input_tmp, nc_output_tmp, kernel_size=4, stride=2, padding=1,
                                   bias=use_bias)
        norm = norm_layer(nc_output_tmp)  # swap norm and act for fine-turning
        act = nn.ReLU(True)
        generator_block += [tconv, norm, act]  # 128 x 512

        nc_input_tmp = nc_output_tmp
        nc_output_tmp = output_nc
        tconv = nn.ConvTranspose2d(nc_input_tmp, nc_output_tmp, kernel_size=4, stride=2, padding=1,
                                   bias=use_bias)  # upsampling by 2 times: 64 x 256
        act = nn.Tanh()
        generator_block += [tconv, act]

        self.model = nn.Sequential(*generator_block)

    def forward(self, x):
        return self.model(x)


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

#--------------------- UnetAFL_4b----------------------------------------
class UnetAFL_4b(nn.Module):
    """Resnet-based generator that consists of Adversarial Feedback Loop. with Feedback Adversarial Learning - cvpr19
    """
    def __init__(self, input_nc, output_nc, ngf=64, ndf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, padding_type='reflect', netD=None, alpha=[0.5, 0.5, 0.5, 0.5, 0.5], loop_count=2):
        """Unet + AFL
        Parameters:
           Disc - Discriminator
        """
        super(UnetAFL_4b, self).__init__()
        self.main = BasicUNetGenerator_4b(input_nc, output_nc)
        nc_list_generator = np.array([4, 8, 4, 2, 1]) * ngf * 2 #because there is no skip connection on the inner most layer
        nc_list_discriminator = np.array([8, 8, 4, 2, 1]) * ndf
        self.netGA = GeneratorAFL(inner_nc_list=nc_list_generator + nc_list_discriminator,
                                  outer_nc_list=nc_list_generator)
        self.netD = netD
        self.ngf = ngf
        self.loop_count = loop_count
        self.norm_layer0 = norm_layer(nc_list_generator[0])
        self.norm_layer1 = norm_layer(nc_list_generator[1])
        self.norm_layer2 = norm_layer(nc_list_generator[2])
        self.norm_layer3 = norm_layer(nc_list_generator[3])
        self.norm_layer4 = norm_layer(nc_list_generator[4])
        self.alpha = alpha

    def set_loop_count(self, loop_count):
        self.loop_count = loop_count

    def _forward_afl(self, fea_list):
        # input_fea = torch.cat((decoder_inner_fea, fea_list[0]), 1)
        feedback_input = fea_list[0]
        feedback_input_new = torch.cat((self.netGA.feedback0, feedback_input), 1)
        feedback_out = feedback_input + self.alpha[0] * self.norm_layer0(
            self.netGA.trans_block0(feedback_input_new))
        feedback_input = self.main.decoder_resblock_layer(feedback_out)

        feedback_input = torch.cat((feedback_input, fea_list[1]), 1)
        feedback_input_new = torch.cat((self.netGA.feedback1, feedback_input), 1)
        feedback_out = feedback_input + self.alpha[1] * self.norm_layer1(
            self.netGA.trans_block1(feedback_input_new))
        #feedback_out = torch.cat((feedback_out, fea_list[1]), 1)
        feedback_input = self.main.decoder_layer1(feedback_out)

        feedback_input = torch.cat((feedback_input, fea_list[2]), 1)
        feedback_input_new = torch.cat((self.netGA.feedback2, feedback_input), 1)
        feedback_out = feedback_input + self.alpha[2] * self.norm_layer2(
            self.netGA.trans_block2(feedback_input_new))
        #feedback_out = torch.cat((feedback_out, fea_list[2]), 1)
        feedback_input = self.main.decoder_layer2(feedback_out)

        feedback_input = torch.cat((feedback_input, fea_list[3]), 1)
        feedback_input_new = torch.cat((self.netGA.feedback3, feedback_input), 1)
        feedback_out = feedback_input + self.alpha[3] * self.norm_layer3(
            self.netGA.trans_block3(feedback_input_new))
        #feedback_out = torch.cat((feedback_out, fea_list[3]), 1)
        feedback_input = self.main.decoder_layer3(feedback_out)

        feedback_input = torch.cat((feedback_input, fea_list[4]), 1)
        feedback_input_new = torch.cat((self.netGA.feedback4, feedback_input), 1)
        feedback_out = feedback_input + self.alpha[4] * self.norm_layer4(
            self.netGA.trans_block4(feedback_input_new))
        #feedback_out = torch.cat((feedback_out, fea_list[4]), 1)
        out = self.main.decoder_layer4(feedback_out)

        return out

    def forward(self, x):
        gen_output = self.main(x)
        gen_img = gen_output[:, 0:3, ::]
        gen_seg = gen_output[:, 3:6, ::]
        fake_AB = torch.cat((x, gen_img),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        d_out_bk = self.netD(fake_AB)  # Fake; stop backprop to the generator by detaching fake_B
        setattr(self, 'fake_B0', gen_img)
        for step in range(1, self.loop_count):
            self.netGA.set_input_disc(self.netD.module.getLayersOutDet())
            gen_output = self._forward_afl(self.main.get_main_layer_result())
            gen_img = gen_output[:, 0:3, ::]
            gen_seg = gen_output[:, 3:6, ::]
            fake_AB = torch.cat((x, gen_img),                                1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            d_out = self.netD(fake_AB)  # Fake; stop backprop to the generator by detaching fake_B
            setattr(self, 'fake_B%s_score' % step, d_out)
            setattr(self, 'fake_B%s' % step, gen_img)

        return gen_img, gen_seg

    def showtensor(self, x):
        import matplotlib.pyplot as plt
        import torchvision.transforms as transforms
        unloader = transforms.ToPILImage()
        image = x.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader(image)
        image.show()


class BasicUNetGenerator_4b(nn.Module):
    """
    network structure of Feedback Adversarial Learning - cvpr 2019
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(BasicUNetGenerator_4b, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        #---------encoding layer 1---------------------
        ngf_in = input_nc
        ngf_out = ngf * 2 ** 0
        self.encoder_layer1 = nn.Sequential(
            # state size: 64 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                               stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer 2---------------------
        ngf_in = ngf_out
        ngf_out = ngf_out * 2
        self.encoder_layer2 = nn.Sequential(
            # state size: ngf --> ngf *2 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer 3---------------------
        ngf_in = ngf_out
        ngf_out = ngf_out * 2
        self.encoder_layer3 = nn.Sequential(
            # state size: ngf*2 --> ngf*4 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer 4---------------------
        ngf_in = ngf_out
        ngf_out = ngf_out * 2
        self.encoder_layer4 = nn.Sequential(
            # state size: ngf*4 -- ngf*8 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer with resnet block---------------------
        ngf_in = ngf_out
        model = []
        for i in range(4):       # add ResNet blocks
            model += [ResnetBlock(ngf_in, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=True, use_bias=use_bias)]
        self.encoder_resblock_layer = nn.Sequential(*model)

        # ---------decoding layer with resnet block ---------------------
        model = []
        for i in range(5):  # add ResNet blocks
            model += [ResnetBlock(ngf_in, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=True, use_bias=use_bias)]
        self.decoder_resblock_layer = nn.Sequential(*model)

        # ---------decoding layer 1 ---------------------
        ngf_out = int(ngf_in / 2)
        ngf_in = int(ngf_in * 2) # skip connection
        self.decoder_layer1 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf_out),
            nn.ReLU(True)
            )

        # ---------decoding layer 2 ---------------------
        ngf_in = ngf_out
        ngf_out = int(ngf_in / 2)
        ngf_in = int(ngf_in * 2)  # skip connection
        self.decoder_layer2 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf_out),
            nn.ReLU(True)
        )

        # ---------decoding layer 3 ---------------------
        ngf_in = ngf_out
        ngf_out = int(ngf_in / 2)
        ngf_in = int(ngf_in * 2)  # skip connection
        self.decoder_layer3 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf_out),
            nn.ReLU(True)
        )
        # ---------decoding layer 4 ---------------------
        ngf_in = ngf_out
        ngf_in = int(ngf_in * 2)  # skip connection
        ngf_out = output_nc
        self.decoder_layer4 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            nn.Tanh()
        )

    def forward(self, input):
        self.e1_out = output = self.encoder_layer1(input)
        self.e2_out = output = self.encoder_layer2(output)
        self.e3_out = output = self.encoder_layer3(output)
        self.e4_out = output = self.encoder_layer4(output)
        self.res_encoder_out = output = self.encoder_resblock_layer(output)
        self.res_decoder_out = output = self.decoder_resblock_layer(output)
        output = torch.cat((output, self.e4_out),1)
        output = self.decoder_layer1(output)
        output = torch.cat((output, self.e3_out), 1)
        output = self.decoder_layer2(output)
        output = torch.cat((output, self.e2_out), 1)
        output = self.decoder_layer3(output)
        output = torch.cat((output, self.e1_out), 1)
        output = self.decoder_layer4(output)
        return output

    def get_main_layer_result(self):
        return [self.res_encoder_out, self.e4_out, self.e3_out, self.e2_out, self.e1_out]

class UnetAFL_v3(nn.Module):
    """
    network structure of Feedback Adversarial Learning - cvpr 2019
    """
    def __init__(self, input_nc, output_nc, ngf=64, ndf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 padding_type='reflect'):
        super(UnetAFL_v3, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        ngf_upbound = ngf * (2**8)
        encoder_layer_nc = []
        decoder_layer_nc = []
        #---------encoding layer 1---------------------
        ngf_in = input_nc
        ngf_out = min(ngf, ngf_upbound)
        encoder_layer_nc.insert(0, ngf_out)
        self.encoder_layer1 = nn.Sequential(
            # state size: 64 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                               stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer 2---------------------
        ngf_in = ngf_out
        ngf_out = ngf_out * 2
        ngf_out = min(ngf_out, ngf_upbound)
        encoder_layer_nc.insert(0, ngf_out)
        self.encoder_layer2 = nn.Sequential(
            # state size: ngf --> ngf *2 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer 3---------------------
        ngf_in = ngf_out
        ngf_out = ngf_out * 2
        ngf_out = min(ngf_out, ngf_upbound)
        encoder_layer_nc.insert(0, ngf_out)
        self.encoder_layer3 = nn.Sequential(
            # state size: ngf*2 --> ngf*4 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer 4---------------------
        ngf_in = ngf_out
        ngf_out = ngf_out * 2
        ngf_out = min(ngf_out, ngf_upbound)
        encoder_layer_nc.insert(0, ngf_out)
        self.encoder_layer4 = nn.Sequential(
            # state size: ngf*4 -- ngf*8 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer with resnet block---------------------
        ngf_in = ngf_out
        encoder_layer_nc.insert(0, ngf_out)
        model = []
        for i in range(4):       # add ResNet blocks
            model += [ResnetBlock(ngf_in, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.encoder_resblock_layer = nn.Sequential(*model)

        # ---------decoding layer with resnet block ---------------------
        decoder_layer_nc.extend([ngf_in])
        model = []
        for i in range(5):  # add ResNet blocks
            model += [ResnetBlock(ngf_in, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.decoder_resblock_layer = nn.Sequential(*model)

        # ---------decoding layer 1 ---------------------
        ngf_out = int(ngf_in / 2)
        ngf_in = int(ngf_in * 2) # skip connection
        decoder_layer_nc.extend([ngf_in])
        self.decoder_layer1 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf_out),
            nn.ReLU(True)
            )

        # ---------decoding layer 2 ---------------------
        ngf_in = ngf_out
        ngf_out = int(ngf_in / 2)
        ngf_in = int(ngf_in * 2)  # skip connection
        decoder_layer_nc.extend([ngf_in])
        self.decoder_layer2 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf_out),
            nn.ReLU(True)
        )

        # ---------decoding layer 3 ---------------------
        ngf_in = ngf_out
        ngf_out = int(ngf_in / 2)
        ngf_in = int(ngf_in * 2)  # skip connection
        decoder_layer_nc.extend([ngf_in])
        self.decoder_layer3 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf_out),
            nn.ReLU(True)
        )
        # ---------decoding layer 4 ---------------------
        ngf_in = ngf_out
        ngf_in = int(ngf_in * 2)  # skip connection
        ngf_out = output_nc
        decoder_layer_nc.extend([ngf_in])
        self.decoder_layer4 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            nn.Tanh()
        )

        # ---------feedback layers ---------------------
        disc_layer_nc = np.array([8, 8, 4, 2, 1]) * ndf
        gene_layer_nc = decoder_layer_nc + disc_layer_nc*2 # disc_layer_nc*2: disc_out of image and segmentation

        self.trans_block0 = TransBlockDual(afl_type=1, input_nc=gene_layer_nc[0], output_nc=decoder_layer_nc[0])
        self.trans_block1 = TransBlockDual(afl_type=1, input_nc=gene_layer_nc[1], output_nc=decoder_layer_nc[1])
        self.trans_block2 = TransBlockDual(afl_type=1, input_nc=gene_layer_nc[2], output_nc=decoder_layer_nc[2])
        self.trans_block3 = TransBlockDual(afl_type=1, input_nc=gene_layer_nc[3], output_nc=decoder_layer_nc[3])
        self.trans_block4 = TransBlockDual(afl_type=1, input_nc=gene_layer_nc[4], output_nc=decoder_layer_nc[4])

        # ----------normalization of feedback output ------------
        self.norm_layer0 = norm_layer(decoder_layer_nc[0])
        self.norm_layer1 = norm_layer(decoder_layer_nc[1])
        self.norm_layer2 = norm_layer(decoder_layer_nc[2])
        self.norm_layer3 = norm_layer(decoder_layer_nc[3])
        self.norm_layer4 = norm_layer(decoder_layer_nc[4])

    def forward(self, gene_input, disc_out=None, alpha=None):
        if disc_out is None:
            e1_out = output = self.encoder_layer1(gene_input)
            e2_out = output = self.encoder_layer2(output)
            e3_out = output = self.encoder_layer3(output)
            e4_out = output = self.encoder_layer4(output)
            res_encoder_out = output = self.encoder_resblock_layer(output)
            res_decoder_out = output = self.decoder_resblock_layer(output)
            output = torch.cat((output, e4_out),1)
            output = self.decoder_layer1(output)
            output = torch.cat((output, e3_out), 1)
            output = self.decoder_layer2(output)
            output = torch.cat((output, e2_out), 1)
            output = self.decoder_layer3(output)
            output = torch.cat((output, e1_out), 1)
            output = self.decoder_layer4(output)
            gen_img = output[:, 0:3, ::]
            gen_seg = output[:, 3:6, ::]
            encoder_out = [res_encoder_out, e4_out, e3_out, e2_out, e1_out]

            return gen_img, gen_seg, encoder_out

        else:
            # input_fea = torch.cat((decoder_inner_fea, fea_list[0]), 1)
            feedback_input = gene_input[0]
            feedback_input_new = torch.cat((disc_out[0], feedback_input), 1)
            encoder_out = feedback_out = feedback_input + alpha[0] * self.norm_layer0(self.trans_block0(feedback_input_new))
            feedback_input = self.decoder_resblock_layer(feedback_out)


            feedback_input = torch.cat((feedback_input, gene_input[1]), 1)
            feedback_input_new = torch.cat((disc_out[1], feedback_input), 1)
            feedback_out = feedback_input + alpha[1] * self.norm_layer1(
                self.trans_block1(feedback_input_new))
            # feedback_out = torch.cat((feedback_out, gene_input[1]), 1)
            feedback_input = self.decoder_layer1(feedback_out)

            feedback_input = torch.cat((feedback_input, gene_input[2]), 1)
            feedback_input_new = torch.cat((disc_out[2], feedback_input), 1)
            feedback_out = feedback_input + alpha[2] * self.norm_layer2(
                self.trans_block2(feedback_input_new))
            # feedback_out = torch.cat((feedback_out, gene_input[2]), 1)
            feedback_input = self.decoder_layer2(feedback_out)

            feedback_input = torch.cat((feedback_input, gene_input[3]), 1)
            feedback_input_new = torch.cat((disc_out[3], feedback_input), 1)
            feedback_out = feedback_input + alpha[3] * self.norm_layer3(
                self.trans_block3(feedback_input_new))
            # feedback_out = torch.cat((feedback_out, gene_input[3]), 1)
            feedback_input = self.decoder_layer3(feedback_out)

            feedback_input = torch.cat((feedback_input, gene_input[4]), 1)
            feedback_input_new = torch.cat((disc_out[4], feedback_input), 1)
            feedback_out = feedback_input + alpha[4] * self.norm_layer4(
                self.trans_block4(feedback_input_new))
            # feedback_out = torch.cat((feedback_out, fea_list[4]), 1)
            output = self.decoder_layer4(feedback_out)
            gen_img = output[:, 0:3, ::]
            gen_seg = output[:, 3:6, ::]

            return gen_img, gen_seg, encoder_out

    def get_main_layer_result(self):
        return [self.res_encoder_out, self.e1_out, self.e2_out, self.e3_out, self.e4_out]

class UnetAFL_v5(nn.Module):
    """
    network structure of Feedback Adversarial Learning - cvpr 2019
    """
    def __init__(self, input_nc, output_nc, ngf=64, ndf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 padding_type='reflect', disc_nc_scale=[1,1,1,1,1]):
        super(UnetAFL_v5, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        ngf_upbound = ngf * (2**8)
        encoder_layer_nc = []
        decoder_layer_nc = []
        #---------encoding layer 1---------------------
        ngf_in = input_nc
        ngf_out = min(ngf, ngf_upbound)
        encoder_layer_nc.insert(0, ngf_out)
        self.encoder_layer1 = nn.Sequential(
            # state size: 64 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                               stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer 2---------------------
        ngf_in = ngf_out
        ngf_out = ngf_out * 2
        ngf_out = min(ngf_out, ngf_upbound)
        encoder_layer_nc.insert(0, ngf_out)
        self.encoder_layer2 = nn.Sequential(
            # state size: ngf --> ngf *2 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer 3---------------------
        ngf_in = ngf_out
        ngf_out = ngf_out * 2
        ngf_out = min(ngf_out, ngf_upbound)
        encoder_layer_nc.insert(0, ngf_out)
        self.encoder_layer3 = nn.Sequential(
            # state size: ngf*2 --> ngf*4 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer 4---------------------
        ngf_in = ngf_out
        ngf_out = ngf_out * 2
        ngf_out = min(ngf_out, ngf_upbound)
        encoder_layer_nc.insert(0, ngf_out)
        self.encoder_layer4 = nn.Sequential(
            # state size: ngf*4 -- ngf*8 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer with resnet block---------------------
        ngf_in = ngf_out
        encoder_layer_nc.insert(0, ngf_out)
        model = []
        for i in range(4):       # add ResNet blocks
            model += [ResnetBlock(ngf_in, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.encoder_resblock_layer = nn.Sequential(*model)

        # ---------decoding layer with resnet block ---------------------
        decoder_layer_nc.extend([ngf_in])
        model = []
        for i in range(5):  # add ResNet blocks
            model += [ResnetBlock(ngf_in, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.decoder_resblock_layer = nn.Sequential(*model)

        # ---------decoding layer 1 ---------------------
        ngf_out = int(ngf_in / 2)
        ngf_in = int(ngf_in * 2) # skip connection
        decoder_layer_nc.extend([ngf_in])
        self.decoder_layer1 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf_out),
            nn.ReLU(True)
            )

        # ---------decoding layer 2 ---------------------
        ngf_in = ngf_out
        ngf_out = int(ngf_in / 2)
        ngf_in = int(ngf_in * 2)  # skip connection
        decoder_layer_nc.extend([ngf_in])
        self.decoder_layer2 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf_out),
            nn.ReLU(True)
        )

        # ---------decoding layer 3 ---------------------
        ngf_in = ngf_out
        ngf_out = int(ngf_in / 2)
        ngf_in = int(ngf_in * 2)  # skip connection
        decoder_layer_nc.extend([ngf_in])
        self.decoder_layer3 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf_out),
            nn.ReLU(True)
        )
        # ---------decoding layer 4 ---------------------
        ngf_in = ngf_out
        ngf_in = int(ngf_in * 2)  # skip connection
        ngf_out = output_nc
        decoder_layer_nc.extend([ngf_in])
        self.decoder_layer4 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            nn.Tanh()
        )

        # ---------feedback layers ---------------------
        # disc_layer_nc = np.array([8, 8, 4, 2, 1]) * ndf
        disc_layer_nc = ndf * np.array(disc_nc_scale)
        gene_layer_nc = decoder_layer_nc + disc_layer_nc*2 # disc_layer_nc*2: disc_out of image and segmentation

        self.trans_block0 = TransBlockDual(afl_type=1, input_nc=gene_layer_nc[0], output_nc=decoder_layer_nc[0])
        self.trans_block1 = TransBlockDual(afl_type=1, input_nc=gene_layer_nc[1], output_nc=decoder_layer_nc[1])
        self.trans_block2 = TransBlockDual(afl_type=1, input_nc=gene_layer_nc[2], output_nc=decoder_layer_nc[2])
        self.trans_block3 = TransBlockDual(afl_type=1, input_nc=gene_layer_nc[3], output_nc=decoder_layer_nc[3])
        self.trans_block4 = TransBlockDual(afl_type=1, input_nc=gene_layer_nc[4], output_nc=decoder_layer_nc[4])

        # ----------normalization of feedback output ------------
        self.norm_layer0 = norm_layer(decoder_layer_nc[0])
        self.norm_layer1 = norm_layer(decoder_layer_nc[1])
        self.norm_layer2 = norm_layer(decoder_layer_nc[2])
        self.norm_layer3 = norm_layer(decoder_layer_nc[3])
        self.norm_layer4 = norm_layer(decoder_layer_nc[4])

    def forward(self, gene_input, disc_out=None, alpha=None):
        if disc_out is None:
            e1_out = output = self.encoder_layer1(gene_input)
            e2_out = output = self.encoder_layer2(output)
            e3_out = output = self.encoder_layer3(output)
            e4_out = output = self.encoder_layer4(output)
            res_encoder_out = output = self.encoder_resblock_layer(output)
            res_decoder_out = output = self.decoder_resblock_layer(output)
            output = torch.cat((output, e4_out),1)
            output = self.decoder_layer1(output)
            output = torch.cat((output, e3_out), 1)
            output = self.decoder_layer2(output)
            output = torch.cat((output, e2_out), 1)
            output = self.decoder_layer3(output)
            output = torch.cat((output, e1_out), 1)
            output = self.decoder_layer4(output)
            gen_img = output[:, 0:3, ::]
            gen_seg = output[:, 3:6, ::]
            encoder_out = [res_encoder_out, e4_out, e3_out, e2_out, e1_out]

            return gen_img, gen_seg, encoder_out

        else:
            # input_fea = torch.cat((decoder_inner_fea, fea_list[0]), 1)
            feedback_input = gene_input[0]
            feedback_input_new = torch.cat((disc_out[0], feedback_input), 1)
            encoder_out = feedback_out = feedback_input + alpha[0] * self.norm_layer0(self.trans_block0(feedback_input_new))
            feedback_input = self.decoder_resblock_layer(feedback_out)


            feedback_input = torch.cat((feedback_input, gene_input[1]), 1)
            feedback_input_new = torch.cat((disc_out[1], feedback_input), 1)
            feedback_out = feedback_input + alpha[1] * self.norm_layer1(
                self.trans_block1(feedback_input_new))
            # feedback_out = torch.cat((feedback_out, gene_input[1]), 1)
            feedback_input = self.decoder_layer1(feedback_out)

            feedback_input = torch.cat((feedback_input, gene_input[2]), 1)
            feedback_input_new = torch.cat((disc_out[2], feedback_input), 1)
            feedback_out = feedback_input + alpha[2] * self.norm_layer2(
                self.trans_block2(feedback_input_new))
            # feedback_out = torch.cat((feedback_out, gene_input[2]), 1)
            feedback_input = self.decoder_layer2(feedback_out)

            feedback_input = torch.cat((feedback_input, gene_input[3]), 1)
            feedback_input_new = torch.cat((disc_out[3], feedback_input), 1)
            feedback_out = feedback_input + alpha[3] * self.norm_layer3(
                self.trans_block3(feedback_input_new))
            # feedback_out = torch.cat((feedback_out, gene_input[3]), 1)
            feedback_input = self.decoder_layer3(feedback_out)

            feedback_input = torch.cat((feedback_input, gene_input[4]), 1)
            feedback_input_new = torch.cat((disc_out[4], feedback_input), 1)
            feedback_out = feedback_input + alpha[4] * self.norm_layer4(
                self.trans_block4(feedback_input_new))
            # feedback_out = torch.cat((feedback_out, fea_list[4]), 1)
            output = self.decoder_layer4(feedback_out)
            gen_img = output[:, 0:3, ::]
            gen_seg = output[:, 3:6, ::]

            return gen_img, gen_seg, encoder_out

    def get_main_layer_result(self):
        return [self.res_encoder_out, self.e1_out, self.e2_out, self.e3_out, self.e4_out]

class panoGAN_baseline_G(nn.Module):
    """
    network structure of Feedback Adversarial Learning - cvpr 2019
    """
    def __init__(self, input_nc, output_nc, ngf=64, ndf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 padding_type='reflect'):
        super(panoGAN_baseline_G, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        ngf_upbound = ngf * (2**8)
        encoder_layer_nc = []
        decoder_layer_nc = []
        #---------encoding layer 1---------------------
        ngf_in = input_nc
        ngf_out = min(ngf, ngf_upbound)
        encoder_layer_nc.insert(0, ngf_out)
        self.encoder_layer1 = nn.Sequential(
            # state size: 64 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                               stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer 2---------------------
        ngf_in = ngf_out
        ngf_out = ngf_out * 2
        ngf_out = min(ngf_out, ngf_upbound)
        encoder_layer_nc.insert(0, ngf_out)
        self.encoder_layer2 = nn.Sequential(
            # state size: ngf --> ngf *2 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer 3---------------------
        ngf_in = ngf_out
        ngf_out = ngf_out * 2
        ngf_out = min(ngf_out, ngf_upbound)
        encoder_layer_nc.insert(0, ngf_out)
        self.encoder_layer3 = nn.Sequential(
            # state size: ngf*2 --> ngf*4 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer 4---------------------
        ngf_in = ngf_out
        ngf_out = ngf_out * 2
        ngf_out = min(ngf_out, ngf_upbound)
        encoder_layer_nc.insert(0, ngf_out)
        self.encoder_layer4 = nn.Sequential(
            # state size: ngf*4 -- ngf*8 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer with resnet block---------------------
        ngf_in = ngf_out
        encoder_layer_nc.insert(0, ngf_out)
        model = []
        for i in range(4):       # add ResNet blocks
            model += [ResnetBlock(ngf_in, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.encoder_resblock_layer = nn.Sequential(*model)

        # ---------decoding layer with resnet block ---------------------
        decoder_layer_nc.extend([ngf_in])
        model = []
        for i in range(5):  # add ResNet blocks
            model += [ResnetBlock(ngf_in, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.decoder_resblock_layer = nn.Sequential(*model)

        # ---------decoding layer 1 ---------------------
        ngf_out = int(ngf_in / 2)
        ngf_in = int(ngf_in * 2) # skip connection
        decoder_layer_nc.extend([ngf_in])
        self.decoder_layer1 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf_out),
            nn.ReLU(True)
            )

        # ---------decoding layer 2 ---------------------
        ngf_in = ngf_out
        ngf_out = int(ngf_in / 2)
        ngf_in = int(ngf_in * 2)  # skip connection
        decoder_layer_nc.extend([ngf_in])
        self.decoder_layer2 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf_out),
            nn.ReLU(True)
        )

        # ---------decoding layer 3 ---------------------
        ngf_in = ngf_out
        ngf_out = int(ngf_in / 2)
        ngf_in = int(ngf_in * 2)  # skip connection
        decoder_layer_nc.extend([ngf_in])
        self.decoder_layer3 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf_out),
            nn.ReLU(True)
        )
        # ---------decoding layer 4 ---------------------
        ngf_in = ngf_out
        ngf_in = int(ngf_in * 2)  # skip connection
        ngf_out = output_nc
        decoder_layer_nc.extend([ngf_in])
        self.decoder_layer4 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            nn.Tanh()
        )

    def forward(self, gene_input):
        e1_out = output = self.encoder_layer1(gene_input)
        e2_out = output = self.encoder_layer2(output)
        e3_out = output = self.encoder_layer3(output)
        e4_out = output = self.encoder_layer4(output)
        res_encoder_out = output = self.encoder_resblock_layer(output)
        res_decoder_out = output = self.decoder_resblock_layer(output)
        output = torch.cat((output, e4_out), 1)
        output = self.decoder_layer1(output)
        output = torch.cat((output, e3_out), 1)
        output = self.decoder_layer2(output)
        output = torch.cat((output, e2_out), 1)
        output = self.decoder_layer3(output)
        output = torch.cat((output, e1_out), 1)
        output = self.decoder_layer4(output)

        gen_img = output[:, 0:3, ::]
        gen_seg = []
        if output.size()[1] == 6:
            gen_seg = output[:, 3:6, ::]

        return gen_img, gen_seg



class ResnetAFLGenerator(nn.Module):
    """Resnet-based generator that consists of Adversarial Feedback Loop.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, padding_type='reflect', netD=None, alpha=[0.1, 0.1, 0.1, 0.1], loop_count=1):
        """Construct a Resnet-based generator

        Parameters:
           Disc - Discriminator
        """
        super(ResnetAFLGenerator, self).__init__()
        self.main = MainNet(input_nc, output_nc, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                            n_blocks=n_blocks, padding_type=padding_type)
        self.netGA = GeneratorAFL(ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        self.netD = netD
        self.alpha1 = alpha[0]
        self.alpha2 = alpha[1]
        self.alpha3 = alpha[2]
        self.alpha4 = alpha[3]
        self.ngf = ngf
        self.loop_count = loop_count

    def set_loop_count(self, loop_count):
        self.loop_count = loop_count

    def _forward_afl(self, inner_fea):
        bn1 = nn.BatchNorm2d(self.ngf * 8).to(0)
        bn2 = nn.BatchNorm2d(self.ngf * 4).to(0)
        bn3 = nn.BatchNorm2d(self.ngf * 2).to(0)
        bn4 = nn.BatchNorm2d(self.ngf * 1).to(0)
        out = inner_fea  # resBlock output
        temp_out = out + self.alpha1 * self.netGA.trans_block0(torch.cat([self.netGA.feedback0, out], 1))
        out = self.main.decoder_layer1(bn1(temp_out))
        temp_out = out + self.alpha2 * self.netGA.trans_block1(torch.cat([self.netGA.feedback1, out], 1))
        out = self.main.decoder_layer2(bn2(temp_out))
        temp_out = out + self.alpha3 * self.netGA.trans_block2(torch.cat([self.netGA.feedback2, out], 1))
        out = self.main.decoder_layer3(bn3(temp_out))
        temp_out = out + self.alpha4 * self.netGA.trans_block3(torch.cat([self.netGA.feedback3, out], 1))
        out = self.main.decoder_layer4(bn4(temp_out))
        return out

    def showtensor(self, x):
        import matplotlib.pyplot as plt
        import torchvision.transforms as transforms
        unloader = transforms.ToPILImage()
        image = x.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader(image)
        image.show()

    def _forward_afl_v2(self, inner_fea):
        out = inner_fea  # resBlock output
        out = self.main.decoder_layer1(
            torch.cat([out, self.alpha1 * self.netGA.trans_block0(torch.cat([self.netGA.feedback0, out], 1))], 1))
        out = self.main.decoder_layer2(
            torch.cat([out, self.alpha2 * self.netGA.trans_block1(torch.cat([self.netGA.feedback1, out], 1))], 1))
        out = self.main.decoder_layer3(
            torch.cat([out, self.alpha3 * self.netGA.trans_block2(torch.cat([self.netGA.feedback2, out], 1))], 1))
        out = self.main.decoder_layer4(
            torch.cat([out, self.alpha4 * self.netGA.trans_block3(torch.cat([self.netGA.feedback3, out], 1))], 1))
        return out

    def forward(self, x):
        gen_output = self.main(x)
        fake_AB = torch.cat((x, gen_output),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        d_out_bk = self.netD(fake_AB)  # Fake; stop backprop to the generator by detaching fake_B

        setattr(self, 'fake_B0', gen_output)

        for step in range(1, self.loop_count):
            self.netGA.set_input_disc(self.netD.module.getLayersOutDet())
            gen_output = self._forward_afl(self.main.get_main_layer_result())
            # gen_output = self._forward_afl_v2(self.main.get_main_layer_result())
            fake_AB = torch.cat((x, gen_output),
                                1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            d_out = self.netD(fake_AB)  # Fake; stop backprop to the generator by detaching fake_B

            setattr(self, 'fake_B%s_score' % step, d_out)
            setattr(self, 'fake_B%s' % step, gen_output)

        return gen_output



class ResnetAFLGenerator_FAL(nn.Module):
    """Resnet-based generator that consists of Adversarial Feedback Loop. with Feedback Adversarial Learning - cvpr19
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, padding_type='reflect', netD=None, alpha=[0.5, 0.5, 0.5, 0.5, 0.5], loop_count=1):
        """Construct a Resnet-based generator

        Parameters:
           Disc - Discriminator
        """
        super(ResnetAFLGenerator_FAL, self).__init__()
        self.main = MainNet_FAL(input_nc, output_nc, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                n_blocks=n_blocks, padding_type=padding_type)
        parm = Parm(ngf * 8, norm_layer, use_dropout, 1, padding_type)
        self.netGA = GeneratorAFL(parm)
        self.netD = netD
        self.ngf = ngf
        self.loop_count = loop_count
        self.norm_layer1 = norm_layer(ngf * 8)
        self.norm_layer2 = norm_layer(ngf * 8)
        self.norm_layer3 = norm_layer(ngf * 4)
        self.norm_layer4 = norm_layer(ngf * 2)
        self.norm_layer5 = norm_layer(ngf * 1)
        self.alpha = alpha

    def set_loop_count(self, loop_count):
        self.loop_count = loop_count

    def _forward_afl(self, inner_fea):
        out = inner_fea  # resBlock output
        temp_out = out + self.alpha[0] * self.netGA.trans_block1(torch.cat([self.netGA.feedback1, out], 1))
        out = self.main.decoder_resblock_layer(self.norm_layer1(temp_out))
        temp_out = out + self.alpha[1] * self.netGA.trans_block2(torch.cat([self.netGA.feedback2, out], 1))
        out = self.main.decoder_layer1(self.norm_layer2(temp_out))
        temp_out = out + self.alpha[2] * self.netGA.trans_block3(torch.cat([self.netGA.feedback3, out], 1))
        out = self.main.decoder_layer2(self.norm_layer3(temp_out))
        temp_out = out + self.alpha[3] * self.netGA.trans_block4(torch.cat([self.netGA.feedback4, out], 1))
        out = self.main.decoder_layer3(self.norm_layer4(temp_out))
        temp_out = out + self.alpha[4] * self.netGA.trans_block5(torch.cat([self.netGA.feedback5, out], 1))
        out = self.main.decoder_layer4(self.norm_layer4(temp_out))
        return out

    def showtensor(self, x):
        import matplotlib.pyplot as plt
        import torchvision.transforms as transforms
        unloader = transforms.ToPILImage()
        image = x.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader(image)
        image.show()

    def _forward_afl_v2(self, inner_fea):
        out = inner_fea  # resBlock output
        out = self.main.decoder_layer1(
            torch.cat([out, self.alpha1 * self.netGA.trans_block0(torch.cat([self.netGA.feedback0, out], 1))], 1))
        out = self.main.decoder_layer2(
            torch.cat([out, self.alpha2 * self.netGA.trans_block1(torch.cat([self.netGA.feedback1, out], 1))], 1))
        out = self.main.decoder_layer3(
            torch.cat([out, self.alpha3 * self.netGA.trans_block2(torch.cat([self.netGA.feedback2, out], 1))], 1))
        out = self.main.decoder_layer4(
            torch.cat([out, self.alpha4 * self.netGA.trans_block3(torch.cat([self.netGA.feedback3, out], 1))], 1))
        return out

    def forward(self, x):
        gen_output = self.main(x)
        fake_AB = torch.cat((x, gen_output),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        d_out_bk = self.netD(fake_AB)  # Fake; stop backprop to the generator by detaching fake_B

        setattr(self, 'fake_B0', gen_output)

        for step in range(1, self.loop_count):
            self.netGA.set_input_disc(self.netD.module.getLayersOutDet())
            gen_output = self._forward_afl(self.main.get_main_layer_result())
            # gen_output = self._forward_afl_v2(self.main.get_main_layer_result())
            fake_AB = torch.cat((x, gen_output),
                                1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            d_out = self.netD(fake_AB)  # Fake; stop backprop to the generator by detaching fake_B

            setattr(self, 'fake_B%s_score' % step, d_out)
            setattr(self, 'fake_B%s' % step, gen_output)

        return gen_output


class MainNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MainNet, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.encoder_layer1 = nn.Sequential(
            # state size: 64 channel
            nn.Conv2d(input_nc, ngf, kernel_size=4,
                      stride=2, padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(inplace=True)
        )
        self.encoder_layer2 = nn.Sequential(
            # state size: 128 channel
            nn.Conv2d(ngf, ngf * 2, kernel_size=4,
                      stride=2, padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(inplace=True)
        )
        self.encoder_layer3 = nn.Sequential(
            # state size: 256 channel
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4,
                      stride=2, padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(inplace=True)
        )
        self.encoder_layer4 = nn.Sequential(
            # state size: 512 channel
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4,
                      stride=2, padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(inplace=True)
        )
        model = []
        for i in range(n_blocks):  # add ResNet blocks
            model += [ResnetBlock(ngf * 8, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.resnetBlock = nn.Sequential(*model)
        """
        self.decoder_layer1 = nn.Sequential(
            # state size: 256 channel
            nn.Conv2d(ngf * 8, ngf * 8,
                               kernel_size=3, stride=1,
                               padding=1, bias=use_bias),
            norm_layer(ngf * 8),
            nn.ReLU(True)
        )
        """
        self.decoder_layer1 = nn.Sequential(
            # state size: 256 channel
            nn.ConvTranspose2d(ngf * 8, ngf * 4,
                               kernel_size=4, stride=2,
                               padding=1, output_padding=0,
                               bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        )
        self.decoder_layer2 = nn.Sequential(
            # state size: 128
            nn.ConvTranspose2d(ngf * 4, ngf * 2,
                               kernel_size=4, stride=2,
                               padding=1, output_padding=0,
                               bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )
        self.decoder_layer3 = nn.Sequential(
            # state size: 64 channel
            nn.ConvTranspose2d(ngf * 2, ngf * 1,
                               kernel_size=4, stride=2,
                               padding=1, output_padding=0,
                               bias=use_bias),
            norm_layer(ngf * 1),
            nn.ReLU(True)
        )
        self.decoder_layer4 = nn.Sequential(
            # state size: output_nc channel
            nn.ConvTranspose2d(ngf, output_nc,
                               kernel_size=4, stride=2,
                               padding=1, output_padding=0,
                               bias=use_bias),
            nn.Tanh()
        )

    def forward(self, input):
        self.e1_out = output = self.encoder_layer1(input)
        self.e2_out = output = self.encoder_layer2(output)
        self.e3_out = output = self.encoder_layer3(output)
        self.e4_out = output = self.encoder_layer4(output)
        self.res_out = output = self.resnetBlock(output)
        self.d1_out = output = self.decoder_layer1(output)
        self.d2_out = output = self.decoder_layer2(output)
        self.d3_out = output = self.decoder_layer3(output)
        self.d4_out = output = self.decoder_layer4(output)
        return output

    def get_main_layer_result(self):
        """
        return [self.e1_out, self.e2_out, self.e3_out, self.e4_out,
                self.res_out,
                self.d1_out, self.d2_out, self.d3_out, self.d4_out]
        :return:
        """
        # return self.res_out


class TransBlockDual(nn.Module):
    def __init__(self, afl_type=1, input_nc=6, output_nc=3):
        super(TransBlockDual, self).__init__()
        if afl_type == 1:
            self.main = nn.Sequential(
                nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(output_nc),
                nn.ReLU(True),
                nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(output_nc),
                nn.ReLU(True))
        elif afl_type == 2:
            pass

    def forward(self, input):
        return self.main(input)


class Parm:
    def __init__(self, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, afl_type=1, padding_type='reflect'):
        self.size = ngf
        self.norm_layer = norm_layer
        self.use_dropout = use_dropout
        self.afl_type = afl_type
        self.padding_type = padding_type


class GeneratorAFL(nn.Module):
    def __init__(self, alf_type=1, inner_nc_list=None, outer_nc_list=None):
        super(GeneratorAFL, self).__init__()

        self.trans_block0 = TransBlockDual(afl_type=1, input_nc=inner_nc_list[0], output_nc=outer_nc_list[0])
        self.trans_block1 = TransBlockDual(afl_type=1, input_nc=inner_nc_list[1], output_nc=outer_nc_list[1])
        self.trans_block2 = TransBlockDual(afl_type=1, input_nc=inner_nc_list[2], output_nc=outer_nc_list[2])
        self.trans_block3 = TransBlockDual(afl_type=1, input_nc=inner_nc_list[3], output_nc=outer_nc_list[3])
        self.trans_block4 = TransBlockDual(afl_type=1, input_nc=inner_nc_list[4], output_nc=outer_nc_list[4])

    def set_input_disc(self, layers_input):
        self.feedback0 = layers_input[0]
        self.feedback1 = layers_input[1]
        self.feedback2 = layers_input[2]
        self.feedback3 = layers_input[3]
        self.feedback4 = layers_input[4]


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class Unet6cGenerator(nn.Module):
    """The difference between Unet6cGenerator and UnetGenerator lies on  function: forward(self, input)
    """

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Unet6cGenerator, self).__init__()
        self.output_nc_img = int(output_nc / 2)
        self.output_nc_seg = int(output_nc / 2)
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        fake_img_seg = self.model(input)
        fake_img = fake_img_seg[:, 0: self.output_nc_img, :, :]
        fake_seg = fake_img_seg[:, self.output_nc_img: self.output_nc_img + self.output_nc_seg, :, :]
        return fake_img, fake_seg


class UnetAFL_4a(nn.Module):
    """Resnet-based generator that consists of Adversarial Feedback Loop. with Feedback Adversarial Learning - cvpr19
    """

    def __init__(self, input_nc, output_nc, ngf=64, ndf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, padding_type='reflect', netD=None, alpha=[0.5, 0.5, 0.5, 0.5, 0.5], loop_count=2):
        """Unet + AFL
        Parameters:
           Disc - Discriminator
        """
        super(UnetAFL_4a, self).__init__()
        self.main = BasicUNetGenerator(input_nc, output_nc)
        nc_list_generator = np.array([8, 8, 4, 2, 1]) * ngf * 2
        nc_list_discriminator = np.array([8, 8, 4, 2, 1]) * ndf
        self.netGA = GeneratorAFL(inner_nc_list=nc_list_generator + nc_list_discriminator,
                                  outer_nc_list=nc_list_generator)
        self.netD = netD
        self.ngf = ngf
        self.loop_count = loop_count
        self.norm_layer0 = norm_layer(nc_list_generator[0])
        self.norm_layer1 = norm_layer(nc_list_generator[1])
        self.norm_layer2 = norm_layer(nc_list_generator[2])
        self.norm_layer3 = norm_layer(nc_list_generator[3])
        self.norm_layer4 = norm_layer(nc_list_generator[4])
        self.alpha = alpha

    def set_loop_count(self, loop_count):
        self.loop_count = loop_count

    def _forward_afl(self, fea_list):
        # input_fea = torch.cat((decoder_inner_fea, fea_list[0]), 1)
        input_fea = fea_list[-1]
        a = self.netGA.feedback0
        input_fea = input_fea + self.alpha[0] * self.norm_layer0(
            self.netGA.trans_block0(torch.cat([self.netGA.feedback0, input_fea], 1)))
        # input_fea = torch.cat((input_fea, fea_list[0]), 1)
        input_fea = self.main.up4(input_fea, fea_list[1])

        input_fea = input_fea + self.alpha[1] * self.norm_layer1(
            self.netGA.trans_block1(torch.cat([self.netGA.feedback1, input_fea], 1)))
        # input_fea = torch.cat((input_fea, fea_list[1]), 1)
        input_fea = self.main.up5(input_fea, fea_list[2])

        input_fea = input_fea + self.alpha[2] * self.norm_layer2(
            self.netGA.trans_block2(torch.cat([self.netGA.feedback2, input_fea], 1)))
        # input_fea = torch.cat((input_fea, fea_list[2]), 1)
        input_fea = self.main.up6(input_fea, fea_list[3])

        input_fea = input_fea + self.alpha[3] * self.norm_layer3(
            self.netGA.trans_block3(torch.cat([self.netGA.feedback3, input_fea], 1)))
        # input_fea = torch.cat((input_fea, fea_list[3]), 1)
        input_fea = self.main.up7(input_fea, fea_list[4])

        input_fea = input_fea + self.alpha[4] * self.norm_layer4(
            self.netGA.trans_block4(torch.cat([self.netGA.feedback4, input_fea], 1)))
        # input_fea = torch.cat((input_fea, fea_list[4]), 1)
        out = self.main.final(input_fea)

        return out

    def forward(self, x):
        gen_output = self.main(x)
        gen_img = gen_output[:, 0:3, ::]
        gen_seg = gen_output[:, 3:6, ::]
        fake_AB = torch.cat((x, gen_img),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        d_out_bk = self.netD(fake_AB)  # Fake; stop backprop to the generator by detaching fake_B
        setattr(self, 'fake_B0', gen_img)
        for step in range(1, self.loop_count):
            self.netGA.set_input_disc(self.netD.module.getLayersOutDet())
            gen_output = self._forward_afl(self.main.get_main_layer_result())
            gen_img = gen_output[:, 0:3, ::]
            gen_seg = gen_output[:, 3:6, ::]
            fake_AB = torch.cat((x, gen_img),
                                1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            d_out = self.netD(fake_AB)  # Fake; stop backprop to the generator by detaching fake_B
            setattr(self, 'fake_B%s_score' % step, d_out)
            setattr(self, 'fake_B%s' % step, gen_img)

        return gen_img, gen_seg

    def showtensor(self, x):
        import matplotlib.pyplot as plt
        import torchvision.transforms as transforms
        unloader = transforms.ToPILImage()
        image = x.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader(image)
        image.show()


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=True)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class BasicUNetGenerator(nn.Module):
    """use dcn in the outmost layer of the decoder
    """

    def __init__(self, input_nc=3, output_nc=3):
        super(BasicUNetGenerator, self).__init__()

        self.down1 = UNetDown(input_nc, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, output_nc, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        self.d1 = out = self.down1(x)
        self.d2 = out = self.down2(out)
        self.d3 = out = self.down3(out)
        self.d4 = out = self.down4(out)
        self.d5 = out = self.down5(out)
        d6 = self.down6(out)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        self.u2 = self.up2(u1, d6)
        self.u3 = self.up3(self.u2, self.d5)
        u4 = self.up4(self.u3, self.d4)
        u5 = self.up5(u4, self.d3)
        u6 = self.up6(u5, self.d2)
        u7 = self.up7(u6, self.d1)

        return self.final(u7)

    def get_main_layer_result(self):
        # """
        return [self.d5, self.d4, self.d3, self.d2, self.d1,
                self.u3]

        # """
        # return self.u3


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class Discriminator_UNet(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """
        Special for Unet_AFL
        """
        super(Discriminator_UNet, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.l1 = nn.Sequential(  # input size is 256
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True))
        self.l2 = nn.Sequential(  # input size is 128
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True))
        self.l3 = nn.Sequential(  # input size is 64
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True))
        self.l4 = nn.Sequential(  # input size is 32
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True))
        self.l5 = nn.Sequential(  # input size is 16
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True))
        self.l6 = nn.Sequential(  # input size is 15
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, input):
        """Standard forward."""
        self.l1out = self.l1(input)
        self.l2out = self.l2(self.l1out)
        self.l3out = self.l3(self.l2out)
        self.l4out = self.l4(self.l3out)
        self.l5out = self.l5(self.l4out)
        return self.l6(self.l5out)

    ## USED FOR FEEDBACK LOOP
    def getLayersOutDet(self):
        return [self.l5out.detach(), self.l4out.detach(), self.l3out.detach(), self.l2out.detach(), self.l1out.detach()]


class AFLDiscriminator_FAL(nn.Module):
    """the discriminator is borrowed from Feedback Adversarial Learning -cvpr19"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(AFLDiscriminator_FAL, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # """
        ndf_in = input_nc
        ndf_out = ndf
        self.l1 = nn.Sequential(  # input size is 256
            nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, inplace=True))

        ndf_in = ndf_out
        ndf_out = ndf_in * 2
        self.l2 = nn.Sequential(  # input size is 256
            nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, inplace=True))

        ndf_in = ndf_out
        ndf_out = ndf_in * 2
        self.l3 = nn.Sequential(  # input size is 256
            nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, inplace=True))

        ndf_in = ndf_out
        ndf_out = ndf_in * 2
        self.l4 = nn.Sequential(  # input size is 256
            nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, inplace=True))

        ndf_in = ndf_out
        ndf_out = ndf_in
        self.l5 = nn.Sequential(  # input size is 256
            nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
            # nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, inplace=True))

        ndf_in = ndf_out
        ndf_out = 1
        self.l6 = nn.Sequential(  # input size is 15
            # nn.ReLU(True),
            nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=0, bias=use_bias)
        )

    def forward(self, input):
        """Standard forward."""
        self.l1out = out = self.l1(input)
        self.l2out = out = self.l2(out)
        self.l3out = out = self.l3(out)
        self.l4out = out = self.l4(out)
        self.l5out = out = self.l5(out)
        return self.l6(out)

    ## USED FOR FEEDBACK LOOP
    def getLayersOutDet(self):
        return [self.l5out, self.l4out, self.l3out, self.l2out, self.l1out]

class Discriminator_PatchGAN_Feedback(nn.Module):
    """the discriminator is borrowed from Feedback Adversarial Learning -cvpr19"""
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(Discriminator_PatchGAN_Feedback, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        ndf_in = input_nc
        ndf_out = ndf
        self.l1 = nn.Sequential(  # input size is 256
            nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, inplace=True))

        ndf_in = ndf_out
        ndf_out = ndf_in * 2
        self.l2 = nn.Sequential(  # input size is 256
            nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, inplace=True))

        ndf_in = ndf_out
        ndf_out = ndf_in * 2
        self.l3 = nn.Sequential(  # input size is 256
            nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, inplace=True))

        ndf_in = ndf_out
        ndf_out = ndf_in * 2
        self.l4 = nn.Sequential(  # input size is 256
            nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, inplace=True))

        ndf_in = ndf_out
        ndf_out = ndf_in
        self.l5 = nn.Sequential(  # input size is 256
            nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
            # nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, inplace=True))

        ndf_in = ndf_out
        ndf_out = 1
        self.l6 = nn.Sequential(  # input size is 15
            # nn.ReLU(True),
            nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=0, bias=use_bias)
        )

    def forward(self, input):
        """Standard forward."""
        l1out = out = self.l1(input)
        l2out = out = self.l2(out)
        l3out = out = self.l3(out)
        l4out = out = self.l4(out)
        l5out = out = self.l5(out)
        l6out = self.l6(out)
        return l6out, [l5out.detach(), l4out.detach(), l3out.detach(), l2out.detach(), l1out.detach()]

class FeaturePyramidDiscriminator(nn.Module):
    """the discriminator is borrowed from Feedback Adversarial Learning -cvpr19"""
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, disc_nc_scale=[1,1,1,1,1]):
        super(FeaturePyramidDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        #------- bottom-up pathway: discriminator's encoding layers ---------------------------
        ndf_in = input_nc
        ndf_out = ndf
        self.l1 = nn.Sequential(  # input size is 256
            nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, inplace=True))

        ndf_in = ndf_out
        ndf_out = ndf_in * 2
        self.l2 = nn.Sequential(  # input size is 256
            nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, inplace=True))

        ndf_in = ndf_out
        ndf_out = ndf_in * 2
        self.l3 = nn.Sequential(  # input size is 256
            nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, inplace=True))

        ndf_in = ndf_out
        ndf_out = ndf_in * 2
        self.l4 = nn.Sequential(  # input size is 256
            nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, inplace=True))

        ndf_in = ndf_out
        ndf_out = ndf_in
        self.l5 = nn.Sequential(  # input size is 256
            nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
            # nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, inplace=True))

        #-------- top-down pathway:
        ndf_out = ndf * 2
        self.lat1 = nn.Sequential(
            nn.Conv2d(ndf, ndf_out, kernel_size=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, True))
        self.lat2 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf_out, kernel_size=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, True))
        self.lat3 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf_out, kernel_size=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, True))
        self.lat4 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf_out, kernel_size=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, True))
        self.lat5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf_out, kernel_size=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, True))

        # upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        # final layers
        ndf_in = ndf_out
        ndf_out = ndf * disc_nc_scale[0]
        self.final1 = nn.Sequential(
            nn.Conv2d(ndf_in, ndf_out, kernel_size=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, True))
        ndf_out = ndf * disc_nc_scale[1]
        self.final2 = nn.Sequential(
            nn.Conv2d(ndf_in, ndf_out, kernel_size=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, True))
        ndf_out = ndf * disc_nc_scale[2]
        self.final3 = nn.Sequential(
            nn.Conv2d(ndf_in, ndf_out, kernel_size=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, True))
        ndf_out = ndf * disc_nc_scale[3]
        self.final4 = nn.Sequential(
            nn.Conv2d(ndf_in, ndf_out, kernel_size=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, True))
        ndf_out = ndf * disc_nc_scale[4]
        self.final5 = nn.Sequential(
            nn.Conv2d(ndf_in, ndf_out, kernel_size=1),
            norm_layer(ndf_out),
            nn.LeakyReLU(0.2, True))

        # true/false prediction and semantic alignment prediction
        ndf_in = ndf_out
        ndf_out = ndf_in
        self.tf = nn.Conv2d(ndf_in, 1, kernel_size=1)
        self.seg = nn.Conv2d(ndf_in, ndf_out, kernel_size=1)

    def forward(self, input):
        # bottom-up pathway
        """Standard forward."""
        l1out = out = self.l1(input)
        l2out = out = self.l2(out)
        l3out = out = self.l3(out)
        l4out = out = self.l4(out)
        l5out = out = self.l5(out)

        # top-down pathway and lateral connections
        feat25 = self.lat5(l5out)
        feat24 = feat25 + self.lat4(l4out)
        feat23 = self.up(feat24) + self.lat3(l3out)
        feat22 = self.up(feat23) + self.lat2(l2out)
        feat21 = self.up(feat22) + self.lat1(l1out)

        # final prediction layers
        feat31 = self.final1(feat21)
        feat32 = self.final2(feat22)
        feat33 = self.final3(feat23)
        feat34 = self.final4(feat24)
        feat35 = self.final5(feat25)

        # Patch-based True/False prediction
        pred1 = self.tf(feat31)
        pred2 = self.tf(feat32)
        pred3 = self.tf(feat33)
        pred4 = self.tf(feat34)
        pred5 = self.tf(feat35)

        # feature projection for feature matching
        seg1 = self.seg(feat31)
        seg2 = self.seg(feat32)
        seg3 = self.seg(feat33)
        seg4 = self.seg(feat34)
        seg5 = self.seg(feat35)

        pred = [pred1, pred2, pred3, pred4, pred5]
        seg = [seg1, seg2, seg3, seg4, seg5]
        feat = [feat35.detach(), feat34.detach(), feat33.detach(), feat32.detach(), feat31.detach()]
        return[pred, seg, feat]
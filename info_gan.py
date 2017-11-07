#!/usr/bin/env python
# encoding: utf-8


import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import os
import timeit
#from Exp import Exp

import util
import numpy as np

import base_module


# NetG is a decoder
# input: batch_size * nz * 1 * 1
# output: batch_size * nc * image_size * image_size
class NetG(nn.Module):
    def __init__(self, decoder):
        super(NetG, self).__init__()
        self.decoder = decoder

    def forward(self, input):
        output = self.decoder(input)
        return output


# NetD is an encoder + decoder
# input: batch_size * nc * image_size * image_size
# f_enc_X: batch_size * k * 1 * 1
# f_dec_X: batch_size * nc * image_size * image_size
class NetD(nn.Module):
    def __init__(self, encoder):
        super(NetD, self).__init__()
        self.encoder = encoder
        self.Q = nn.Sigmoid()
        self.D1 = nn.LogSoftmax()
        self.D2 = nn.LogSoftmax()
        self.D3 = nn.LogSoftmax()
        self.C_std = lambda x: torch.sqrt(torch.exp(x))

        #self.decoder = decoder

    def forward(self, input):
        f_enc_X = self.encoder(input)
        f_enc_X = f_enc_X.view(input.size(0), -1)
        q = self.Q(f_enc_X[:, 0])
        d1 = self.D1(f_enc_X[:, 1:21])
        d2 = self.D2(f_enc_X[:, 21:41])
        d3 = self.D3(f_enc_X[:, 41:61])
        c_std = self.C_std(f_enc_X[:, 61])
        mu = f_enc_X[:, 62]
        return f_enc_X, q, d1, d2, d3, mu, c_std


class ONE_SIDED(nn.Module):
    def __init__(self):
        super(ONE_SIDED, self).__init__()

        main = nn.ReLU()
        self.main = main

    def forward(self, input):
        output = self.main(-input)
        output = -output.mean()
        return output


    
# Get argument
parser = argparse.ArgumentParser()
parser = util.get_args(parser)
args = parser.parse_args()
print(args)


def sample_multinomial(num_classes, size):
    c = np.random.multinomial(1, num_classes*[1./num_classes], size=size)
    return c

def sample_uniform(size):
    c = np.random.uniform(-1, 1, size=size)
    return c
def sample_random(size):
    return np.random.randn(*size)



if args.experiment is None:
    args.experiment = 'samples'
os.system('mkdir {0}'.format(args.experiment))

if torch.cuda.is_available():
    args.cuda = True
    torch.cuda.set_device(args.gpu_device)
    print("Using GPU device", torch.cuda.current_device())
else:
    raise EnvironmentError("GPU device not available!")

BATCH_SIZE = 64 # args.batch_size
IMG_SIZE = args.image_size # 64
NC = 1 # number of input channels
Z_dim = 128
d1_dim = 20
c_dim = 1




args.manual_seed = 1126
np.random.seed(seed=args.manual_seed)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed(args.manual_seed)
cudnn.benchmark = True

# Get data
data_length = 1350
# trn_loader = util.load_data('/home/yz6/data/chairs/data.npy', args.batch_size, 64)
# trn_dataset = util.get_data(args, train_flag=True)

# trn_loader = torch.utils.data.DataLoader(trn_dataset,
#                                          batch_size=args.batch_size,
#                                          shuffle=True,
#                                          num_workers=int(args.workers))

# construct encoder/decoder modules
G_decoder = base_module.Decoder(IMG_SIZE, NC, k=189, ngf=64)
D_encoder = base_module.Encoder(IMG_SIZE, NC, k=63, ndf=64)
#D_decoder = base_module.Decoder(args.image_size, args.nc, k=hidden_dim, ngf=64)

netG = NetG(G_decoder)
netD = NetD(D_encoder)
#one_sided = ONE_SIDED()
print("netG:", netG)
print("netD:", netD)
#print("oneSide:", one_sided)

netG.apply(base_module.weights_init)
netD.apply(base_module.weights_init)
#one_sided.apply(base_module.weights_init)

# sigma for MMD
# base = 1.0
# sigma_list = [1, 2, 4, 8, 16]
# sigma_list = [sigma / base for sigma in sigma_list]

# put variable into cuda device
#fixed_noise = torch.cuda.FloatTensor(64, args.nz, 1, 1).normal_(0, 1)

# create noise vectors:
# noise: BATCH_SIZE x 189 x 1 x 1
# noise_squeezed : BATCH_SIZE x 189
def create_noise_cuda():
    noise_np_list = list()
    noise_np_list.append(sample_random([BATCH_SIZE, 128]))
    noise_np_list.append(sample_multinomial(num_classes=20, size=BATCH_SIZE))
    noise_np_list.append(sample_multinomial(num_classes=20, size=BATCH_SIZE))
    noise_np_list.append(sample_multinomial(num_classes=20, size=BATCH_SIZE))
    noise_np_list.append(sample_uniform(size=(BATCH_SIZE, 1)))

    noise_vector = np.concatenate(noise_np_list, axis=1)
    noise_squeezed = torch.from_numpy(noise_vector.astype('float32')).cuda()
    
    
    noise_vector = np.expand_dims(noise_vector, axis=2)
    noise_vector = np.expand_dims(noise_vector, axis=3)
    noise = torch.from_numpy(noise_vector.astype('float32')).cuda()
    
    return noise, noise_squeezed


def create_fixed_noise_cuda():
    noise_np_list = list()
    noise_np_list.append(sample_random([BATCH_SIZE, 128]))
    noise_np_list.append(np.repeat(sample_multinomial(num_classes=20, size=1), BATCH_SIZE, axis=0))
    noise_np_list.append(np.repeat(sample_multinomial(num_classes=20, size=1), BATCH_SIZE, axis=0))
    noise_np_list.append(np.repeat(sample_multinomial(num_classes=20, size=1), BATCH_SIZE, axis=0))
    noise_np_list.append(np.expand_dims(np.linspace(-1, 1, BATCH_SIZE), axis=1))

    noise_vector = np.concatenate(noise_np_list, axis=1)
    noise_squeezed = torch.from_numpy(noise_vector.astype('float32')).cuda()
    
    
    noise_vector = np.expand_dims(noise_vector, axis=2)
    noise_vector = np.expand_dims(noise_vector, axis=3)
    noise = torch.from_numpy(noise_vector.astype('float32')).cuda()
    
    return noise, noise_squeezed

    
one = torch.cuda.FloatTensor([1])
mone = one * -1
if args.cuda:
    netG.cuda()
    netD.cuda()
    #one_sided.cuda()

noise, fixed_noise_squeezed = create_fixed_noise_cuda()
print('fixed noise')
for v in fixed_noise_squeezed:
    print('z')
    print(v[0:128])
    print('d1')
    print(v[128:148])
    print('d2')
    print(v[148:168])
    print('d1')
    print(v[168:188])
    print('c')
    print(v[188])
fixed_noise = Variable(noise, requires_grad=False)
# setup optimizer
optimizerG = torch.optim.RMSprop(netG.parameters(), lr=args.lr)
optimizerD = torch.optim.RMSprop(netD.parameters(), lr=args.lr)

lambda_MMD = 1.0
lambda_AE_X = 8.0
lambda_AE_Y = 8.0
lambda_rg = 16.0


time = timeit.default_timer()
gen_iterations = 0
for t in range(args.max_iter):
    data_iter = util.load_data('/home/yz6/data/chairs/data.npy', BATCH_SIZE, IMG_SIZE)
    i = 0
    while (i < data_length):
        # ---------------------------
        #        Optimize over NetD
        # ---------------------------
        for p in netD.parameters():
            p.requires_grad = True

        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
            Giters = 1
        else:
            Diters = 5
            Giters = 1

        for j in range(Diters):
            if i == data_length:
                break

            # clamp parameters of NetD encoder to a cube
            # do not clamp paramters of NetD decoder!!!
            for p in netD.encoder.parameters():
                p.data.clamp_(-0.01, 0.01)

            data = data_iter.next()
            i += 1
            netD.zero_grad()

            #x_cpu, _ = data # torch.Floattensor
            x_cpu =  torch.from_numpy(data)
            
            x = Variable(x_cpu.cuda())
            batch_size = x.size(0)

            f_enc_X_D, q_real, _, _, _, _, _ = netD(x)


            noise, noise_squeezed = create_noise_cuda()
            noise_squeezed = Variable(noise_squeezed)
            noise = Variable(noise, volatile=True)  # total freeze netG
            y = Variable(netG(noise).data) # y is not volatile

            f_enc_Y_D, q_fake, d1, d2, d3, mu, std_contig = netD(y)
#             print('shape --------')
#             print(q_fake)
#             print(d1.size())
#             print(d2.size())
#             print(d3.size())
#             print(mu)
#             print(c_std)

            # compute biased MMD2 and use ReLU to prevent negative value
#             mmd2_D = mix_rbf_mmd2(f_enc_X_D, f_enc_Y_D, sigma_list)
#             mmd2_D = F.relu(mmd2_D)

            # compute rank hinge loss
            #print('f_enc_X_D:', f_enc_X_D.size())
            #print('f_enc_Y_D:', f_enc_Y_D.size())
            #one_side_errD = one_sided(f_enc_X_D.mean(0) - f_enc_Y_D.mean(0))

            # compute L2-loss of AE
            #L2_AE_X_D = util.match(x.view(batch_size, -1), f_dec_X_D, 'L2')
            #L2_AE_Y_D = util.match(y.view(batch_size, -1), f_dec_Y_D, 'L2')
            D_loss = torch.mean(torch.log(q_real + 1e-8) - torch.log(q_fake + 1e-8))
#             print('D_loss shape', D_loss.size())
            
            crossent_1 = torch.mean(torch.sum(noise_squeezed[:, 128:148] * d1, dim=1))
            crossent_2 = torch.mean(torch.sum(noise_squeezed[:, 148:168] * d2, dim=1))
            crossent_3 = torch.mean(torch.sum(noise_squeezed[:, 168:188] * d3, dim=1))
#             print('crossent shape', crossent_3.size())
            
            #std_contig = torch.sqrt(c_std)
            epsilon = (noise_squeezed[:, 188] - mu) / (std_contig + 1e-8) # (BATCH_SIZE, 1) Variable
#             print(noise_squeezed.size())
#             print(noise_squeezed[:, 188].size())
#             print('mu', mu.size())
#             print('std_contig', std_contig.size())
#             print('epsilon', epsilon.size())
            ll_continuous = torch.mean( - torch.log(std_contig + 1e-8) - 0.5 * epsilon * epsilon)
            
#             print('ll_continuouse', ll_continuous.size())
            
            #ent_loss = torch.mean(torch.sum(c * torch.log(c + 1e-8), dim=1))

            lamdba_c = 1
            lamdba_d = 0.3
            mi_loss = lamdba_d * (crossent_1 + crossent_2 + crossent_3) + lamdba_c * ll_continuous
            
#             print('mi_loss', mi_loss.size())

            #errD = torch.sqrt(mmd2_D) + lambda_rg * one_side_errD - lambda_AE_X * L2_AE_X_D - lambda_AE_Y * L2_AE_Y_D
            errD = D_loss + mi_loss
            #print('Training Discriminator it=%s, errD: ' % j, errD.data)
            errD.backward(mone)
            optimizerD.step()

        # ---------------------------
        #        Optimize over NetG
        # ---------------------------
        for p in netD.parameters():
            p.requires_grad = False

        for j in range(Giters):
            if i == data_length:
                break

            data = data_iter.next()
            i += 1
            netG.zero_grad()

            x_cpu =  torch.from_numpy(data)
            x = Variable(x_cpu.cuda())
            batch_size = x.size(0)

            f_enc_X, q_real, _, _, _, _, _  = netD(x)

            noise, noise_squeezed = create_noise_cuda()
            noise_squeezed = Variable(noise_squeezed)
            noise = Variable(noise)  
            y = netG(noise)

            f_enc_Y_D, q_fake, d1, d2, d3, mu, std_contig = netD(y)


            G_loss = torch.mean(torch.log(q_fake + 1e-8))
            
            crossent_1 = torch.mean(torch.sum(noise_squeezed[:, 128:148] * d1, dim=1))
            crossent_2 = torch.mean(torch.sum(noise_squeezed[:, 148:168] * d2, dim=1))
            crossent_3 = torch.mean(torch.sum(noise_squeezed[:, 168:188] * d3, dim=1))
#             print('crossent shape', crossent_3.size())
            
            #std_contig = torch.sqrt(c_std)
            epsilon = (noise_squeezed[:, 188] - mu) / (std_contig + 1e-8) # (BATCH_SIZE, 1) Variable
#             print(noise_squeezed.size())
#             print(noise_squeezed[:, 188].size())
#             print('mu', mu.size())
#             print('std_contig', std_contig.size())
#             print('epsilon', epsilon.size())
            ll_continuous = torch.mean( - torch.log(std_contig + 1e-8) - 0.5 * epsilon * epsilon)
            
#             print('ll_continuouse', ll_continuous.size())
            
            #ent_loss = torch.mean(torch.sum(c * torch.log(c + 1e-8), dim=1))

            lamdba_c = 1
            lamdba_d = 0.3
            mi_loss = lamdba_d * (crossent_1 + crossent_2 + crossent_3) + lamdba_c * ll_continuous
            
#             print('mi_loss', mi_loss.size())

            #errD = torch.sqrt(mmd2_D) + lambda_rg * one_side_errD - lambda_AE_X * L2_AE_X_D - lambda_AE_Y * L2_AE_Y_D
            errG = G_loss + mi_loss

#             print('Training Generator it=%s, errD: ' % j, errG.data)
            errG.backward(mone) # maximizing
            optimizerG.step()

            gen_iterations += 1

        run_time = (timeit.default_timer() - time) / 60.0
        print('[%3d/%3d][%3d/%3d] [%5d] (%.2f m) cross1 %.6f cross2 %.6f cross3 %.6f cont_ll %.6f loss_D %.6f Loss_G %.6f MI %.6f err_D %.6f err_G %.6f |gD| %.4f |gG| %.4f'
              % (t, args.max_iter, i, data_length, gen_iterations, run_time,
                 crossent_1.data[0], crossent_2.data[0], crossent_3.data[0], ll_continuous.data[0],
                 D_loss.data[0], G_loss.data[0], mi_loss.data[0],
                 errD.data[0], errG.data[0],
                 base_module.grad_norm(netD), base_module.grad_norm(netG)))

        if gen_iterations % 500 == 0:
            y_fixed = netG(fixed_noise)
            y_fixed.data = y_fixed.data.mul(0.5).add(0.5)
            # f_dec_X_D = f_dec_X_D.view(f_dec_X_D.size(0), args.nc, args.image_size, args.image_size)
            # f_dec_X_D.data = f_dec_X_D.data.mul(0.5).add(0.5)
            vutils.save_image(y_fixed.data, '{0}/fake_samples_{1}.png'.format(args.experiment, gen_iterations))
            #vutils.save_image(f_dec_X_D.data, '{0}/decode_samples_{1}.png'.format(args.experiment, gen_iterations))

    if t % 50 == 0:
        torch.save(netG.state_dict(), '{0}/netG_iter_{1}.pth'.format(args.experiment, t))
        torch.save(netD.state_dict(), '{0}/netD_iter_{1}.pth'.format(args.experiment, t))




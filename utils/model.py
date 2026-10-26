import torch.nn as nn
import torch
import numpy as np

class VGGEncoder(nn.Module):
    def __init__(self,vgg_path):
         super(VGGEncoder,self).__init__()
         self.vgg=nn.Sequential(
                nn.Conv2d(3, 3, (1, 1)),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(3, 64, (3, 3)),
                nn.ReLU(),  # relu1-1         
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 64, (3, 3)),
                nn.ReLU(),  # relu1-2
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 128, (3, 3)),
                nn.ReLU(),  # relu2-1
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 128, (3, 3)),
                nn.ReLU(),  # relu2-2
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 256, (3, 3)),
                nn.ReLU(),  # relu3-1
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),  # relu3-2
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),  # relu3-3
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),  # relu3-4
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 512, (3, 3)),
                nn.ReLU(),  # relu4-1, this is the last layer used
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu4-2
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu4-3
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu4-4
                nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu5-1
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu5-2
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU(),  # relu5-3
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 512, (3, 3)),
                nn.ReLU()  # relu5-4
         )
         self.vgg.load_state_dict(torch.load(vgg_path))
         self.vgg=nn.Sequential(*list(self.vgg.children())[0:31])

         self.enc1=nn.Sequential(*list(self.vgg.children())[0:4])
         self.enc2=nn.Sequential(*list(self.vgg.children())[4:11])
         self.enc3=nn.Sequential(*list(self.vgg.children())[11:18])
         self.enc4=nn.Sequential(*list(self.vgg.children())[18:31])

         
         for name in ['enc1','enc2','enc3','enc4']:
              for p in getattr(self,name).parameters():
                   p.requires_grad=False
                   


    def forward(self,input,is_test=False):
         h1=self.enc1(input)
         h2=self.enc2(h1)
         h3=self.enc3(h2)
         h4=self.enc4(h3)
         if is_test:
              return h4
         else:
              return h1,h2,h3,h4

         


class Decoder(nn.Module):
    def __init__(self):
          super(Decoder,self). __init__()
          self.decoder=nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(512, 256, (3, 3)),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(256, 128, (3, 3)),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 128, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(128, 64, (3, 3)),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 64, (3, 3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(64, 3, (3, 3)),
          )

    def forward(self,input):
         return self.decoder(input)


def adaptive_IN(fc,fs):
    size=fc.shape
    print(size)
    fc_mean,fc_std=get_mean_std(fc)
    fs_mean,fs_std=get_mean_std(fs)

    normalized_fc=fc-fc_mean.expand(size)/fc_std.expand(size)
    return fs_mean.expand(size)*normalized_fc+fs_std.expand(size)


def get_mean_std(feats,eps=1e-5):
     size=feats.shape
     assert len(size)==4
     #(batch-size,channels,h,w)
     batch_size,channels=size[:2]
     feats_mean=feats.view(batch_size,channels,-1).mean(dim=2).view(batch_size,channels,1,1)
     feats_var=feats.view(batch_size,channels,-1).var(dim=2,unbiased=False)+eps
     feats_std=feats_var.sqrt().view(batch_size,channels,1,1)
     return feats_mean,feats_std
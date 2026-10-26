import argparse

import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np

from utils.model import *
from utils.utils import *
from torch.utils.data  import DataLoader
import torch.optim as optim
from torchvision.utils import save_image


def  parse_arguments():

    parser=argparse.ArgumentParser()
    parser.add_argument('--content_dir',type=str,default='C:/Users/91917/OneDrive/Desktop/NST/content data',help="Location of content dataset")
    parser.add_argument('--style_dir',type=str,default='C:/Users/91917/OneDrive/Desktop/NST/style data',help="Location of style dataset")
    parser.add_argument('--vgg',type=str,default='C:/Users/91917/OneDrive/Desktop/NST/vgg_normalised.pth',help="Location of pre-trained VGG")
    parser.add_argument('--experiment',type=str,default='arguments1')

    parser.add_argument('--crop_size',type=bool,default=True,help="Crop size of image")
    parser.add_argument('--final_size',type=int,default=256,help="final size of image")
    parser.add_argument('--content_size',type=int,default=512,help="size of content image")
    parser.add_argument('--style_size',type=int,default=512,help="size of style image")
    parser.add_argument('--lr',default=1e-4,type=float,help='Learning Rate')
    parser.add_argument('--lr_decay',default=5e-5,help="lr_deacy",type=float)

    parser.add_argument('--content_wt',default=1.0,type=float,help='Content_wt for Content Loss')
    parser.add_argument('--style_wt',default=10,type=float,help='style_wt for style Loss')
    parser.add_argument('--batch_size',type=int,default=5,help="Batch size")
    parser.add_argument('--epochs',default=10,type=int,help='Epochs for Training')


    parser.add_argument('--log_interval',default=10,type=int,help='log interval for saving loss')
    parser.add_argument('--save_interval',default=10,type=int,help='Save Interval')
    parser.add_argument('--resume',default=False,action='store_true',help='Resume Training')

    parser.add_argument('--decoder_path',default=None,type=str,help='Decoder Model Path')
    parser.add_argument('--optimizer_path',default=None,type=str,help='Optimizer Model Path')

    return parser.parse_args()


def main():
    args=parse_arguments()
    
    if torch.backends.mps.is_available():
        device=torch.device('mps')
    elif torch.cuda.is_available():
        device=torch.device('cuda')

    else:
        device=torch.device('cpu')

    save_dir=Path('experiment') / args.experiment
    save_dir.mkdir(exist_ok=True,parents=True)


    #save parameteres
    with open (save_dir/'args.txt','w') as f:
        for key,value in vars(args).items():
            f.write(f"{key}:{value}\n")

    content_transform=get_transforms(args.content_size,args.crop_size,args.final_size)
    style_transform=get_transforms(args.style_size,args.crop_size,args.final_size)


    content_dataset=Image_Datasets(args.content_dir,content_transform)
    style_dataset=Image_Datasets(args.style_dir,style_transform)


    # Dataloader

    content_dataloader=DataLoader(content_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True)
    style_dataloader=DataLoader(style_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True)

    print(len(content_dataloader))
    print(len(style_dataloader))



    encoder=VGGEncoder(args.vgg).to(device)
    decoder=Decoder().to(device)

    print("Training step1...")

    optimizer=optim.Adam(decoder.parameters(),lr=args.lr)
    lr_scheduler=optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: 1.0 / (1.0+args.lr_decay*epoch))
    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path))
        optimizer.load_state_dict(torch.load(args.optimizer_path))

    loss=nn.MSELoss()
    encoder.eval()

    for epoch in range(args.epochs):
        progress_bar= tqdm(zip(content_dataloader,style_dataloader),total=min(len(content_dataloader),len(style_dataloader)))

        running_loss=0
        running_closs=0
        running_sloss=0
        for content_img,style_img in progress_bar:

            
            content_img=content_img.to(device)
            style_img=style_img.to(device)

            content_feature=encoder(content_img)   #(h1,h2,h3,h4)
            style_feature=encoder(style_img)       #(h1,h2,h3,h4)
            fc=content_feature[-1]      #h4
            fs=style_feature[-1]           #h4


            t=adaptive_IN(fc,fs)                #t=generated_img (normalized) features
            
            generated_img=decoder(t)            #generated image

            
            generated_features=encoder(generated_img)        #(h1,h2,h3,h4)
            
            content_loss=loss(generated_features[-1],t)
            content_loss=content_loss*args.content_wt
            
            style_loss=0
            for feats_g,feats_s in zip(generated_features,style_feature):          #(h1,h2,h3,h4)
                mu_fs,std_fs=get_mean_std(feats_s)
                mu_fg,std_fg=get_mean_std(feats_g)
                style_loss+=loss(mu_fs,mu_fg)+loss(std_fs,std_fg)

            
            style_loss=style_loss*args.style_wt
            
            
            total_loss=content_loss+style_loss
            progress_bar.set_description(f'content_loss:{content_loss.item():4f} style loss:{style_loss.item():4f} total loss:{total_loss.item():4f}')

            running_loss+=total_loss.item()
            running_closs+=content_loss.item()
            running_sloss+=style_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
        lr_scheduler.step()
        num_batches=min(len(content_dataloader),len(style_dataloader))

        running_loss/=num_batches
        running_closs/=num_batches
        running_sloss/=num_batches
        if (epoch+1) % args.log_interval==0:
           tqdm.write(f"For epoch {epoch+1} content loss:{running_closs:4f} style loss:{running_sloss:4f} running_loss:{running_loss:4f}")
        
        
        #save the model
        if (epoch+1)% args.save_interval==0:
            torch.save(decoder.state_dict(),save_dir/f'decoder {epoch+1}.pth')
            torch.save(optimizer.state_dict(),save_dir/f'optimizer {epoch+1}.pth')
            

            with torch.no_grad():
                output=torch.cat([content_img,style_img,generated_img],dim=0)
                save_image(output,save_dir/f'output for epoch {epoch+1}.png',nrow=args.batch_size)



if __name__=="__main__":
    main()
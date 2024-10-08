from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import argparse
import numpy as np
import os

from torchvision.utils import save_image

from utils.resnet import resnet18
from utils.vgg import vgg16
from utils.mobilenetv2 import mobilenetv2 

from dataloader_cifar import cifar_dataloader
from attack_models.unet import UNet
from attack_models.autoencoders import FourierAutoencoder

from sklearn.manifold import TSNE
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batch size') 
parser.add_argument('--id', default=1, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='data/cifar10/cifar-10-batches-py', type=str, help='path to dataset')
parser.add_argument('--resume_path', default='checkpoint_cifar10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--num_workers', default=5, type=int)
parser.add_argument('--model_type', default='resnet18', type=str)

# backdoor attacks
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 40])
parser.add_argument('--target_label', type=int, default=0, help='class of target label')
parser.add_argument('--w', default=0, type=float)

parser.add_argument('--alpha', default=10.0, type=float)
parser.add_argument('--dev', default=5.0, type=float)

parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--epoch_poison',type=int, default=50)
parser.add_argument('--attack_model', default='autoencoder', type=str)
args = parser.parse_args()


torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device(f'cuda:{args.gpuid}')

acc_net_log=open(f'{args.resume_path.strip()}/model_{args.model_type}_dataset=cifar10_poison={args.epoch_poison}_w={args.w}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_acc.txt','w') 
asr_net_log=open(f'{args.resume_path.strip()}/model_{args.model_type}_dataset=cifar10_poison={args.epoch_poison}_w={args.w}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_asr.txt','w')     

cifar = cifar_dataloader(dataset='cifar10', batch_size=128, num_workers=5, root_dir='data/cifar10/cifar-10-batches-py')

# trainloader = cifar.run(mode='train')
testloader  = cifar.run(mode='test')

trainloader1, train_only_target_loader, train_except_target_loader = cifar.run(mode='poison')
test_except_target_loader, _ = cifar.run(mode='poison_test')

tgtmodel = FourierAutoencoder().cuda()


STD_DEV = args.dev
MEAN_INTERVAL = 6 * STD_DEV
MEANS = torch.arange(0, args.num_class * MEAN_INTERVAL, MEAN_INTERVAL)


def generate_batch_noise(mean_list, std_dev, index, batch_size=10, size=(3, 32, 32)):
    available_means = [mean_list[i] for i in range(len(mean_list)) if i != index]
    num_means = len(available_means)
    all_noises = []

    for i in range(batch_size):
        current_mean = available_means[i % num_means]
        noise = torch.normal(mean=current_mean, std=std_dev, size=size)
        all_noises.append(noise)
    
    batch_noise = torch.stack(all_noises)
    idx = torch.randperm(batch_noise.size(0))
    batch_noise = batch_noise[idx]
    
    return batch_noise


def get_fourier_backdoor_samples(inputs_c, outputs):
    batch = []

    for i in range(inputs_c.size(0)):
        img_c = inputs_c[i]
        fft_img_c = torch.fft.fft2(img_c, dim=(-2, -1))
        amp_img_c = torch.abs(fft_img_c)
        pha_img_c = torch.angle(fft_img_c)
        amp_shifted_c = torch.fft.fftshift(amp_img_c, dim=(-2, -1))

        amp_shifted_c_new = outputs[i] * amp_shifted_c
        magnitude_recentered = torch.fft.ifftshift(amp_shifted_c_new, dim=(-2, -1))
        
        fft_modified = magnitude_recentered * torch.exp(1j * pha_img_c)
        img_restored = torch.fft.ifft2(fft_modified, dim=(-2, -1))

        img_restored = torch.real(img_restored)
        batch.append(img_restored)
    
    batch = torch.stack(batch)
    batch = batch.cuda()

    return batch


def train_step(model, epoch, optimizer, data_loader):
    model.train()

    num_iter = (len(data_loader.dataset)//(args.batch_size))+1
    for i, (inputs_c, labels_c) in enumerate(data_loader):

        non_target_index = np.where(np.array(labels_c) != args.target_label)[0].tolist()
        
        inputs_c, labels_c = inputs_c.cuda(), labels_c.cuda()
        
        means = torch.full((args.batch_size,), MEANS[args.target_label])
        std_devs = torch.full((args.batch_size,), STD_DEV)
        
        means = means.view(args.batch_size, 1, 1, 1)
        std_devs = std_devs.view(args.batch_size, 1, 1, 1)

        inputs_t = torch.normal(mean=means.expand(-1, 3, 32, 32), std=std_devs.expand(-1, 3, 32, 32))
        inputs_t = inputs_t.cuda()
        
        inputs_e = generate_batch_noise(MEANS,STD_DEV,args.target_label,batch_size=args.batch_size,size=(3,32,32))
        inputs_e = inputs_e.cuda()

        inputs_b  = get_fourier_backdoor_samples(inputs_c[non_target_index], tgtmodel(inputs_t[non_target_index]))
        inputs_be = get_fourier_backdoor_samples(inputs_c, tgtmodel(inputs_e))
        
        labels_t = torch.ones_like(labels_c)*args.target_label
        labels_t = labels_t.cuda()
        
        all_inputs = torch.cat([inputs_c, inputs_b, inputs_be], dim=0)
        all_labels = torch.cat([labels_c, labels_t[non_target_index], labels_c])
        
        idx = torch.randperm(all_inputs.size(0))
        all_inputs = all_inputs[idx]
        all_labels = all_labels[idx]
        
        optimizer.zero_grad()
        output = model(all_inputs)
        
        loss1 = F.cross_entropy(output, all_labels)
        loss2 = F.mse_loss(inputs_c[non_target_index], inputs_b)
        
        loss = loss1 + args.alpha*loss2

        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d]  Iter[%3d/%3d]  CE-loss: %.4f  MSE-loss: %.4f'%(args.dataset, epoch, args.epoch_poison, i+1, num_iter, loss1.item(), args.alpha*loss2.item()))
        sys.stdout.flush()

    return loss



def poison(model):
    optimizer = torch.optim.SGD(list(model.parameters())+list(tgtmodel.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    for epoch in range(0, args.epoch_poison):
        train_step(model=model, epoch=epoch, optimizer=optimizer, data_loader=trainloader1)
        print('')
        clean_test(model,testloader, record=True)
        poison_test(model,testloader=test_except_target_loader, record=True)
        scheduler.step()



def poison_test(model,testloader=test_except_target_loader,record=True):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # _target_iter = iter(_testloader)
        for batch_idx, (inputs, targets) in enumerate(testloader):

            means = torch.full((args.batch_size,), MEANS[args.target_label])
            std_devs = torch.full((args.batch_size,), STD_DEV)

            # 扩展均值和标准差的维度以适应生成的张量
            means = means.view(args.batch_size, 1, 1, 1)  # 形状变为 (batch_size, 1, 1, 1)
            std_devs = std_devs.view(args.batch_size, 1, 1, 1)  # 形状变为 (batch_size, 1, 1, 1)

            # 生成高斯噪声
            inputs_t = torch.normal(mean=means.expand(-1, 3, 32, 32), std=std_devs.expand(-1, 3, 32, 32))
            labels_t = torch.ones_like(targets)*args.target_label

            inputs, inputs_t, labels_t = inputs.cuda(), inputs_t.cuda(), labels_t.cuda()
            inputs_b  = get_fourier_backdoor_samples(inputs, tgtmodel(inputs_t))
            outputs = model(inputs_b)
            
            _, predicted = torch.max(outputs, 1)            
            total += labels_t.size(0)
            correct += predicted.eq(labels_t).cpu().sum().item()
                    
    acc = 100.*correct/total
    
    print("| Poison Accuracy: %.2f%%\n" %(acc))
    if record:
        asr_net_log.write('Poison Accuracy:%.2f\n'%(acc))
        asr_net_log.flush()    



def clean_test(model,test_loader,record=True):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)            
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
                           
    acc = 100.*correct/total

    print("| Clean Accuracy: %.2f%%\n" %(acc))
    if record:
        acc_net_log.write('Clean Accuracy:%.2f\n'%(acc))
        acc_net_log.flush()  



def create_model(model_type=args.model_type):
    if model_type == 'resnet18':
        model = resnet18(num_classes=args.num_class)
        clean_model_path = 'model_resnet18_dataset=cifar10_clean.pth'

    if model_type == 'vgg16':
        model = vgg16(num_classes=args.num_class)
        clean_model_path = 'model_vgg16_dataset=cifar10_clean.pth'
        
    if model_type == 'mobilenetv2':
        model = mobilenetv2(num_classes=args.num_class) 
        clean_model_path = 'model_mobilenetv2_dataset=cifar10_clean.pth'
    
    model.load_state_dict(torch.load(clean_model_path))
    model = model.cuda()
    return model



def tsne_visualize(model):
    featrues_clean_target = []
    for i, (inputs_c, labels_c) in enumerate(train_only_target_loader):
        inputs_c, labels_c = inputs_c.cuda(), labels_c.cuda()
        outputs_c, featrues_c = model(inputs_c, return_hidden=True)
        
        for i in range(inputs_c.size(0)):
            featrues_clean_target.append(featrues_c[i].cpu().detach())
    
    featrues_poison_target = []
    for i, (inputs_e, labels_e) in enumerate(train_except_target_loader):
        if len(featrues_clean_target) == len(featrues_poison_target): break
        
        means = torch.full((args.batch_size,), MEANS[args.target_label])
        std_devs = torch.full((args.batch_size,), STD_DEV)
        
        # 扩展均值和标准差的维度以适应生成的张量
        means = means.view(args.batch_size, 1, 1, 1)  # 形状变为 (batch_size, 1, 1, 1)
        std_devs = std_devs.view(args.batch_size, 1, 1, 1)  # 形状变为 (batch_size, 1, 1, 1)

        # 生成高斯噪声
        inputs_t = torch.normal(mean=means.expand(-1, 3, 32, 32), std=std_devs.expand(-1, 3, 32, 32))
        inputs_t = inputs_t.cuda()
        
        inputs_e, labels_e = inputs_e.cuda(), labels_e.cuda()
        
        inputs_b  = get_fourier_backdoor_samples(inputs_e, tgtmodel(inputs_t))
        inputs_b = inputs_b.cuda()
        
        outputs_b, featrues_b = model(inputs_b, return_hidden=True)
        
        for i in range(inputs_b.size(0)):
            featrues_poison_target.append(featrues_b[i].cpu().detach())
            if len(featrues_clean_target) == len(featrues_poison_target): break

    target_class_features = featrues_clean_target + featrues_poison_target
    target_class_features = torch.stack(target_class_features)
    embeddings = target_class_features.numpy()
    ground_truth = np.concatenate((np.ones(len(featrues_clean_target)), np.zeros(len(featrues_poison_target))))
  
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    emb_poison_true = embeddings_2d[ground_truth==0]
    emb_benign_true = embeddings_2d[ground_truth==1]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    sns.scatterplot(x=emb_poison_true[:, 0], y=emb_poison_true[:, 1], color='#3274a1', label='Poison', s=10, ax=ax)
    sns.scatterplot(x=emb_benign_true[:, 0], y=emb_benign_true[:, 1], color='#e1812c', label='Benign', s=10, ax=ax)
    ax.set_title('Ground Truth')
    ax.legend(loc='lower right')
    ax.set_xlabel('')
    ax.set_ylabel('')
    

    if not os.path.exists(f"{args.resume_path.strip()}/images_w={args.w}_poison={args.epoch_poison}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_tsne_visualize"):
        os.makedirs(f"{args.resume_path.strip()}/images_w={args.w}_poison={args.epoch_poison}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_tsne_visualize")
    else:
        os.removedirs(f"{args.resume_path.strip()}/images_w={args.w}_poison={args.epoch_poison}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_tsne_visualize")
        os.makedirs(f"{args.resume_path.strip()}/images_w={args.w}_poison={args.epoch_poison}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_tsne_visualize")

    file_path_svg = f"{args.resume_path.strip()}/images_w={args.w}_poison={args.epoch_poison}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_tsne_visualize/clustering.svg"
    plt.savefig(file_path_svg, bbox_inches='tight')
    drawing = svg2rlg(file_path_svg)
    
    file_path_pdf = f"{args.resume_path.strip()}/images_w={args.w}_poison={args.epoch_poison}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_tsne_visualize/clustering.pdf"
    renderPDF.drawToFile(drawing, file_path_pdf)

    plt.close()
    os.remove(file_path_svg)



def gradcam_visualize(model):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    MEAN_CIFAR = (0.4914, 0.4822, 0.4465)
    STD_CIFAR  = (0.2023, 0.1994, 0.2010)  

    mean = torch.tensor(MEAN_CIFAR)
    std = torch.tensor(STD_CIFAR)
    mean = torch.tensor(MEAN_CIFAR)
    std = torch.tensor(STD_CIFAR)

    def denormalize(tensor, mean, std):
        mean = mean[:, None, None]
        std = std[:, None, None]
        return tensor * std + mean

    if not os.path.exists(f"{args.resume_path.strip()}/images_w={args.w}_poison={args.epoch_poison}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_gradcam_visualize"):
        os.makedirs(f"{args.resume_path.strip()}/images_w={args.w}_poison={args.epoch_poison}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_gradcam_visualize")

    target_layers = [model.layer4[-1]]

    for _, (inputs_c, labels_c) in enumerate(test_except_target_loader):
        inputs_c, labels_c = inputs_c.cuda(), labels_c.cuda()

        labels_t = torch.ones_like(labels_c)*args.target_label
        labels_t = labels_t.cuda()

        means = torch.full((inputs_c.size(0),), MEANS[args.target_label])
        std_devs = torch.full((inputs_c.size(0),), STD_DEV)
        
        means = means.view(inputs_c.size(0), 1, 1, 1) 
        std_devs = std_devs.view(inputs_c.size(0), 1, 1, 1) 

        inputs_t = torch.normal(mean=means.expand(-1, 3, 32, 32), std=std_devs.expand(-1, 3, 32, 32))
        inputs_t = inputs_t.cuda()
        
        inputs_b  = get_fourier_backdoor_samples(inputs_c, tgtmodel(inputs_t))
        
        with GradCAM(model=model, target_layers=target_layers) as cam:
            for i in range(inputs_c.size(0)):

                targets_c = [ClassifierOutputTarget(labels_c[i].cpu().item())]
                grayscale_cam_c = cam(input_tensor=inputs_c[i].unsqueeze(0), targets=targets_c)
                grayscale_cam_c = grayscale_cam_c[0, :]
                
                denorm_img_c = denormalize(inputs_c[i].detach().cpu(), mean, std).permute(1, 2, 0).numpy()
                denorm_img_c = np.clip(denorm_img_c, 0, 1)
                visualization_c = show_cam_on_image(denorm_img_c, grayscale_cam_c, use_rgb=True)

                save_image(inputs_c[i], f"{args.resume_path.strip()}/images_w={args.w}_poison={args.epoch_poison}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_gradcam_visualize/{i}_clean.png", normalize=True)
                Image.fromarray(visualization_c).save(f"{args.resume_path.strip()}/images_w={args.w}_poison={args.epoch_poison}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_gradcam_visualize/{i}_clean_cam.png")

                targets_b = [ClassifierOutputTarget(labels_t[i].cpu().item())]
                grayscale_cam_b = cam(input_tensor=inputs_b[i].unsqueeze(0), targets=targets_b)
                grayscale_cam_b = grayscale_cam_b[0, :]

                denorm_img_b = denormalize(inputs_b[i].detach().cpu(), mean, std).permute(1, 2, 0).numpy()
                denorm_img_b = np.clip(denorm_img_b, 0, 1)
                visualization_b = show_cam_on_image(denorm_img_b, grayscale_cam_b, use_rgb=True)

                save_image(inputs_b[i], f"{args.resume_path.strip()}/images_w={args.w}_poison={args.epoch_poison}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_gradcam_visualize/{i}_backdoor.png", normalize=True)
                Image.fromarray(visualization_b).save(f"{args.resume_path.strip()}/images_w={args.w}_poison={args.epoch_poison}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_gradcam_visualize/{i}_backdoor_cam.png")
        break


def backdoor_visualize():
    with torch.no_grad():
        for _, (inputs, _) in enumerate(test_except_target_loader):
            inputs = inputs.cuda() 

            means = torch.full((inputs.size(0),), MEANS[args.target_label])
            std_devs = torch.full((inputs.size(0),), STD_DEV)

            means = means.view(inputs.size(0), 1, 1, 1)
            std_devs = std_devs.view(inputs.size(0), 1, 1, 1)

            inputs_t = torch.normal(mean=means.expand(-1, 3, 32, 32), std=std_devs.expand(-1, 3, 32, 32))
            inputs_t = inputs_t.cuda()

            inputs_b  = get_fourier_backdoor_samples(inputs, tgtmodel(inputs_t))
    
            if not os.path.exists(f"{args.resume_path.strip()}/images_w={args.w}_poison={args.epoch_poison}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_backdoor_visualize"):
                os.makedirs(f"{args.resume_path.strip()}/images_w={args.w}_poison={args.epoch_poison}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_backdoor_visualize")
                
            for i in range(inputs.size(0)):
                save_image(inputs[i], f"{args.resume_path.strip()}/images_w={args.w}_poison={args.epoch_poison}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_backdoor_visualize/{i}_clean.png", normalize=True)
                save_image(inputs_t[i], f"{args.resume_path.strip()}/images_w={args.w}_poison={args.epoch_poison}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_backdoor_visualize/{i}_target.png", normalize=True)
                save_image(inputs_b[i], f"{args.resume_path.strip()}/images_w={args.w}_poison={args.epoch_poison}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}_backdoor_visualize/{i}_poison.png", normalize=True)
            break
    return





if __name__ == '__main__':
    cudnn.benchmark = True
    model = create_model(model_type=args.model_type)
    poison(model)
    
    torch.save(model.state_dict(), f'{args.resume_path.strip()}/model_{args.model_type}_dataset=cifar10_poison={args.epoch_poison}_w={args.w}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}.pth')
    torch.save(tgtmodel.state_dict(), f'{args.resume_path.strip()}/tgtmodel_{args.attack_model}_dataset=cifar10_poison={args.epoch_poison}_w={args.w}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}.pth')

    # model.load_state_dict(torch.load(f'{args.resume_path.strip()}/model_{args.model_type}_dataset=cifar10_poison={args.epoch_poison}_w={args.w}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}.pth'))
    # tgtmodel.load_state_dict(torch.load(f'{args.resume_path.strip()}/tgtmodel_{args.attack_model}_dataset=cifar10_poison={args.epoch_poison}_w={args.w}_alpha={args.alpha}_lr={args.lr}_dev={args.dev}.pth'))

    tsne_visualize(model)
    gradcam_visualize(model)
    backdoor_visualize()

    # clean_test(model,testloader, record=True)
    # poison_test(model,testloader=test_except_target_loader, record=True)

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
import numpy as np
from PIL import Image
from spikingjelly import visualizing
def colorize_image(image, color):
        image = np.array(image)
        colored_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        for i in range(3):
            colored_image[:, :, i] = image * color[i]
        return Image.fromarray(colored_image)
def create_colored_mnist(mnist_dataset, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for idx, (image, label) in enumerate(mnist_dataset):
            color = np.random.randint(0, 256, size=3)
            colored_image = colorize_image(image, color)
            label_dir = os.path.join(save_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            colored_image.save(os.path.join(label_dir, f'{idx}.png'))

#卷积脉冲神经网络定义
class CSNN(nn.Module):
    def __init__(self, T: int, channels: int, use_cupy=False):
        super().__init__()
        self.T = T  #SNN时间步长

        self.conv_fc = nn.Sequential(
        layer.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),    #普通卷积层
        layer.BatchNorm2d(channels),                                        #普通BatchNormalization
        neuron.IFNode(surrogate_function=surrogate.ATan()),                 # IF脉冲节点
        layer.MaxPool2d(2, 2),  # 14 * 14                                   # 最大池化

        layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False), #普通卷积层
        layer.BatchNorm2d(channels),                                        #普通BatchNormalization
        neuron.IFNode(surrogate_function=surrogate.ATan()),                 #IF 脉冲节点
        layer.MaxPool2d(2, 2),  # 7 * 7                                     # 最大池化

        layer.Flatten(),
        layer.Linear(channels * 7 * 7, channels * 4 * 4, bias=False),       #普通线性层
        neuron.IFNode(surrogate_function=surrogate.ATan()),                 #IF 脉冲节点

        layer.Linear(channels * 4 * 4, 10, bias=False),         
        neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        functional.set_step_mode(self, step_mode='m')                       #多步脉冲模式

        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]  #将数据复制T份直接输入网络
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr
    
    def spiking_encoder(self):
        return self.conv_fc[0:3] #脉冲节点即可将float输入编码为spike序列


def main():
    '''
    (sj-dev) wfang@Precision-5820-Tower-X-Series:~/spikingjelly_dev$ python -m spikingjelly.activation_based.examples.conv_fashion_mnist -h

    usage: conv_fashion_mnist.py [-h] [-T T] [-device DEVICE] [-b B] [-epochs N] [-j N] [-data-dir DATA_DIR] [-out-dir OUT_DIR]
                                 [-resume RESUME] [-amp] [-cupy] [-opt OPT] [-momentum MOMENTUM] [-lr LR]

    Classify Fashion-MNIST

    optional arguments:
      -h, --help          show this help message and exit
      -T T                simulating time-steps
      -device DEVICE      device
      -b B                batch size
      -epochs N           number of total epochs to run
      -j N                number of data loading workers (default: 4)
      -data-dir DATA_DIR  root dir of Fashion-MNIST dataset
      -out-dir OUT_DIR    root dir for saving logs and checkpoint
      -resume RESUME      resume from the checkpoint path
      -amp                automatic mixed precision training
      -cupy               use cupy neuron and multi-step forward mode
      -opt OPT            use which optimizer. SDG or Adam
      -momentum MOMENTUM  momentum for SGD
      -save-es            dir for saving a batch spikes encoded by the first {Conv2d-BatchNorm2d-IFNode}
    '''
    # python -m spikingjelly.activation_based.examples.conv_fashion_mnist -T 4 -device cuda:0 -b 128 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -cupy -opt sgd -lr 0.1 -j 8

    # python -m spikingjelly.activation_based.examples.conv_fashion_mnist -T 4 -device cuda:0 -b 4 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -cupy -opt sgd -lr 0.1 -j 8 -resume ./logs/T4_b256_sgd_lr0.1_c128_amp_cupy/checkpoint_latest.pth -save-es ./logs
    
    '''parser = argparse.ArgumentParser(description='Classify Fashion-MNIST')
    parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of Fashion-MNIST dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-save-es', default=None, help='dir for saving a batch spikes encoded by the first {Conv2d-BatchNorm2d-IFNode}')
    '''
    
    class parser():
        def __init__(self):
            self.T = 4
            self.device = 'cuda'
            self.b = 128
            self.epochs = 10
            self.j = 16
            self.data_dir='./fashionmnist'
            self.out_dir='logs'
            self.resume = 'None'
            self.amp = True
            self.cupy = False
            self.opt = 'adam'
            self.momentum = 0.9
            self.lr = 0.001
            self.channels = 128
            self.save_es = None
    
    #args = parser.parse_args()
    args = parser()
    print(args)

    net = CSNN(T=args.T, channels=args.channels, use_cupy=args.cupy)

    print(net)

    net.to(args.device)
    
    '''
    载入数据，和普通DNN的训练步骤一样
    '''
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    create_colored_mnist(mnist_train, './data/ColoredMNIST/train')
    create_colored_mnist(mnist_test, './data/ColoredMNIST/test')
    train_set = torchvision.datasets.ImageFolder(
        root='./data/ColoredMNIST/train',
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
    )

    test_set = torchvision.datasets.ImageFolder(
        root='./data/ColoredMNIST/test',
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
    )

    train_data_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    '''
    定义优化器，和普通DNN的训练步骤一样
    '''
    
    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)


    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}')

    if args.amp:
        out_dir += '_amp'

    if args.cupy:
        out_dir += '_cupy'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()          #reset optimizer
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float() #one-hot encoding the label to a vector

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(img)
                    loss = F.mse_loss(out_fr, label_onehot) #calculate the MSE loss between the prediction and the vector
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(img)
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = net(img)
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')


main()




#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author: baiyu
    Edited by Kojungbeom
"""

import argparse
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
from conf import settings
from utils import get_network, get_test_dataloader


model_dict = {'resnet':'r', 'qresnet':'d+p', 'sqresnet':'sq', 'Nos_qresnet':'nosq'}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weight', type=str, required=True, help='the weight file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader') 
    parser.add_argument('-wbit', type=int, default=8, help='weight quantization bit')
    parser.add_argument('-abit', type=int, default=8, help='activation quantization bit')
    parser.add_argument('-sigma', type=float, default=0, help='sigma')
    parser.add_argument('-delay', type=int, default=0, help='delay')
    args = parser.parse_args()

    net = get_network(args)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )

    net.load_state_dict(torch.load(args.weight))
    #print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            #print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                #print('GPU INFO.....')
                #print(torch.cuda.memory_summary(), end='')


            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    n_element = 0
    sparse = 0
    sigma = args.sigma
    if model_dict[args.net[:-2]] =='sq' or model_dict[args.net[:-2]]=='snoq':
        for n, m in net.named_modules():
            try:
                n_element += float(m.weight.nelement())
                #print(float(m.weight.nelement()))
            except: 
                continue
            if isinstance(m, torch.nn.Conv2d):
                threshold = torch.mean(torch.abs(m.weight)) + torch.std(torch.abs(m.weight)) * sigma
                mask = torch.full((m.weight.shape[0], m.weight.shape[1],
                                   m.weight.shape[2], m.weight.shape[3]), float(threshold)).cuda()
                mask = torch.where(mask < torch.abs(m.weight), 1 , 0)
                sparse += float(torch.sum(mask==0))
        print('Sparsed weight: ', int(sparse), "Number of Weights: ", int(n_element))
        print('Sparsity: {0:0.2f}%'.format(sparse/n_element))

    print()
    #print("Accuracy: {:.4f}".format(acc.float() / len(cifar100_test_loader.dataset)))
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
    


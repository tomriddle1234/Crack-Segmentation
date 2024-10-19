import torch
import os 
import argparse
import pickle
import numpy as np
import torch.nn as nn
from torch import optim
from utils import CustomImageDataset, calculate_metric, init_weights, DeepCrackDataset
from eval_metrics import f1_score, iou_score
from model import UNet_FCN, LMM_Net
from efficientnet import EfficientCrackNet
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader
print(torch.cuda.is_available())
import warnings
warnings.filterwarnings("ignore")


# could consider normalizing the images
# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# reference: https://github.com/yhlleo/DeepSegmentor/blob/master/data/deepcrack_dataset.py


# Testing model
def eval(args, test_dataloaders):
    if args.model_name == 'UNet':
        model = UNet_FCN(args = args).to(device)
        model.apply(init_weights)
    elif args.model_name == 'LMM_Net':
        model = LMM_Net().to(device)
    elif args.model_name == 'EfficientCrackNet':
        model = EfficientCrackNet().to(device)

    model.load_state_dict(torch.load(f'./saved_models/{args.model_name}/best_model_num_{args.run_num}.pt'), map_location=device)['model_state_dict']

    model.eval()
    all_y, all_y_hat = [], []
    f1_scores, iou_scores = 0.0, 0.0
    num_batch = 0.0
    # metrics: IoU, BF score, F1 score, precision, recall 

    for input_img, mask in test_dataloaders:
        input_img, mask = input_img.to(device), mask.to(device)
        with torch.no_grad():
            # Add batch to GPU
            output_mask= model(input_img)
            
            # y = labels.detach().cpu().numpy().tolist()
            # y_pred = outputs.detach().cpu().numpy().tolist()
            # all_y.extend(y)
            # all_y_hat.extend(y_pred)
            f1_s = f1_score(mask, output_mask)
            iou = iou_score(mask, output_mask)
            f1_scores += f1_s
            iou_scores += iou
            num_batch += 1.0

    test_f1_score = (f1_scores/num_batch)
    test_miou_score = (iou_scores/num_batch)

    # final_all_y_pred = np.argmax(all_y_hat, axis=1)
    # metric = calculate_metric(all_y, final_all_y_pred)
    print(f'Test F1 Score is: {round(test_f1_score, 2)}')
    print(f'Test mIoU Score is: {round(test_miou_score, 2)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecasting of device data')
    parser.add_argument('--data_dir', type=str, help='Main directory of input dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,help='Momentum in learning rate')
    parser.add_argument('--validate', type=bool, default=False, help='whether to validate the model or not')
    parser.add_argument('--model_name', type=str, help='algorithm')
    parser.add_argument('--epochs', type=int, help='Num of epochs')
    parser.add_argument('--alpha', type=float, help='Reduction factor in Loss function')
    parser.add_argument('--optim_w_decay', type=float, default=2e-4)
    parser.add_argument('--lr_decay', type=float, default=0.8)
    parser.add_argument('--num_epochs_decay', type=int, default=5)
    parser.add_argument('--data_name', type=str, help='Dataset to be used')
    parser.add_argument('--rgb', type=bool, help='Is image RGB or not')
    parser.add_argument('--run_num', type=str, help='run number')
    parser.add_argument('--half', type=bool, default=False, help='use half Model size or not')
    parser.add_argument('--augment', type=bool, default=False, help='whether augment dataset or not')

    args = parser.parse_args()

    if args.data_name == 'deepcrack':
        test_dataset = DeepCrackDataset(args, data_part='test')
        test_dataloaders = DataLoader(test_dataset, batch_size=8, num_workers=10)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval(args, test_dataloaders)
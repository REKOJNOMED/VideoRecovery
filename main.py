import torch
import pytorch_ssim
from tensorboardX import SummaryWriter
from utils.dataset import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import util
import models
import json
from models.rcan.model import RCAN
from models.yair.model import CNN
import cv2


def load_losses(writer, path='./test.json'):
    f = open(path, 'r')
    a = json.load(f)
    losses = []
    for k in a.keys():
        if k.endswith('loss'):
            losses = a[k]
    losses = [i for i in a.values()]
    load_it = len(losses)
    for i in range(load_it):
        writer.add_scalar('data/loss', losses[i][2], losses[i][1])
        writer.add_scalars('data/scalar_group', {'loss' : losses[i][2]}, losses[i][2])
    
    return load_it
    


def train(model, params):
    optimizer = optim.Adam(model.parameters(), lr=params['train_params']['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['train_params']['decay_step_size'],
                                          gamma=params['train_params']['decay_rate'])
    start_eph=0

    saved_checkpoints_dir = 'checkpoints/' + params['model_name']
    writer = SummaryWriter(saved_checkpoints_dir + '/runs/')
    load_it = 0
    if params['train_params']['continue_train']:
        checkpoint=torch.load('checkpoints/'+params['model_name']+'/checkpoint')
        load_it = load_losses(writer)
        model.load_state_dict(torch.load(checkpoint['net'],map_location=params['device']))
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_eph=checkpoint['epoch']
    model.train()

    dataset = ImageDataset(picture_num=params['train_params']['picture_num'],device=params['device'],dataset_dir=params['train_params']['dataset_dir'])
    dataloader = DataLoader(dataset, batch_size=params['batch_size'],
                            shuffle=True,
                            num_workers=params['num_workers'],
                            drop_last=False)
    if params['model_name']=='rcan':
        loss_f=pytorch_ssim.SSIMLoss(channel=params['rcan_params']['n_colors'])


    total_itnm = np.ceil(len(dataset) / params['batch_size'])
    util.mkdir(saved_checkpoints_dir)

    for eph in range(params['train_params']['epoch']):
        itnm = 0

        for tw, gt in dataloader:
            gen = model(tw)

            loss = loss_f(gen, gt)
            writer.add_scalar('data/loss',loss.item(), eph * total_itnm + itnm + load_it)
            writer.add_scalars('data/scalar_group', {'loss' :loss.item()}, eph * total_itnm + itnm + load_it)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            itnm += 1
            print('epoch:{},progress:{:.2%},loss:{:.4}'.format(eph + 1+start_eph, itnm / total_itnm, loss))
        scheduler.step()

        checkpoint={'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':eph+1+start_eph}

        torch.save(checkpoint, saved_checkpoints_dir + '/checkpoint')

        if (eph + 1) % 5 == 0:
            torch.save(model.state_dict(), saved_checkpoints_dir + '/{}_{:.4}'.format(eph+1+start_eph,loss))
    writer.export_scalars_to_json("./test.json")
    writer.close()



def eval(model, params):
    checkpoint = torch.load('checkpoints/' + params['model_name'] + '/checkpoint',map_location=params['device'])
    model.load_state_dict(checkpoint['net'])
    model.eval()

    video_path=params['eval_params']['video_path']

    patch_size = params['eval_params']['patch_size']
    stride=params['eval_params']['stride']


    videoCapture = cv2.VideoCapture(video_path)

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))


    videoWriter = cv2.VideoWriter('results.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    success, frame = videoCapture.read()
    while success:
        h = frame.shape[0]
        w = frame.shape[1]
        frame_data=[]
        r = 0

        while r + patch_size <= h:
            c = 0
            while c + patch_size <= w:
                frame_data.append( frame[r:r + patch_size, c:c + patch_size])
                c = c + stride
            c = w - patch_size
            frame_data.append(frame[r:r + patch_size, c:c + patch_size])
            r = r + stride

        r = h - patch_size
        c = 0
        while c + patch_size <= w:
            frame_data.append( frame[r:r + patch_size, c:c + patch_size])
            c = c + stride
        c = w - patch_size
        frame_data.append( frame[r:r + patch_size, c:c + patch_size])
        dataset = EvalDataset( data=frame_data,device=params['device'])
        dataloader = DataLoader(dataset, batch_size=params['batch_size'],
                                shuffle=False,
                                num_workers=params['num_workers'],
                                drop_last=False)
        frame_data=np.array(frame_data)
        gen_frame_data=np.zeros_like(frame_data)
        itnm=0
        for data in dataloader:
            gen_frame_data[itnm*params['batch_size']:(itnm+1)*params['batch_size']]=model(data).permute(0,2,3,1).cpu().detach().numpy()
            itnm+=1

        idx = 0
        gen_image = np.zeros((h, w,3))
        count = np.zeros((h, w,3))
        r = 0
        c = 0
        while r + patch_size <= h:
            c = 0
            while c + patch_size <= w:
                gen_image[r:r + patch_size, c:c + patch_size] += gen_frame_data[idx]
                count[r:r + patch_size, c:c + patch_size] += 1
                idx += 1
                c = c + stride
            c = w - patch_size
            gen_image[r:r + patch_size, c:c + patch_size] += gen_frame_data[idx]
            count[r:r + patch_size, c:c + patch_size] += 1
            idx += 1
            r = r + stride

        r = h - patch_size
        c = 0
        while c + patch_size <= w:
            gen_image[r:r + patch_size, c:c + patch_size] += gen_frame_data[idx]
            count[r:r + patch_size, c:c + patch_size] += 1
            idx += 1
            c = c + stride
        c = w - patch_size
        gen_image[r:r + patch_size, c:c + patch_size] += gen_frame_data[idx]
        count[r:r + patch_size, c:c + patch_size] += 1
        idx += 1
        gen_image = gen_image / count
        gen_image=(gen_image-np.min(gen_image))/(np.max(gen_image)-np.min(gen_image))
        videoWriter.write(np.uint8(gen_image*255))
        success, frame = videoCapture.read()

    videoCapture.release()
    videoWriter.release()

def main(params):
    if params['model_name'] == 'rcan':
        model = RCAN(params['rcan_params'])
    elif params['model_name'] == 'yair':
        model = CNN()


    if len(params['gpu_no']) != 0 and torch.cuda.is_available():

        if len(params['gpu_no']) > 1:
            gpu_no = params['gpu_no'].split(',')
            device_ids = [int(i) for i in gpu_no]
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        device=torch.device('cuda:'+params['gpu_no'])
    else:
        device = torch.device('cpu')

    params['device'] = device
    model = model.to(device=params['device'])

    if params['is_train']:
        train(model, params)
    else:
        eval(model, params)


from params import *


if __name__ == '__main__':
     params = make_params()
     main(params)




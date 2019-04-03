import os
import torch.utils.data as udata
import numpy as np
import torch
import matplotlib.image as mpimg
class EvalDataset(udata.Dataset):
    def __init__(self,data,device):
        self.data=data
        self.device=device
    def __getitem__(self, index):

        data=self.data[index]

        data=torch.tensor(data)


        data=data.permute(2,0,1)

        data = data.to(device=self.device,dtype=torch.float32)


        return data


    def __len__(self):
        return len(self.data)

class ImageDataset(udata.Dataset):
    def __init__(self,device,picture_num,dataset_dir):
        self.device=device
        self.dataset_index_list=[i for i in range(picture_num)]
        self.dataset_dir=dataset_dir
    def __getitem__(self, index):

        n=mpimg.imread(self.dataset_dir+'/noise_image/'+str(self.dataset_index_list[index])+'_n.png')
        g=mpimg.imread(self.dataset_dir+'/ground_truth/'+str(self.dataset_index_list[index])+'_g.png')
        n=torch.tensor(n)
        g=torch.tensor(g)

        n=n.permute(2,0,1)
        g=g.permute(2,0,1)
        n = n.to(device=self.device)
        g = g.to(device=self.device)

        return n,g


    def __len__(self):
        return len(self.dataset_index_list)
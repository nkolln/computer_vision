
import numpy as np
from py_libs.utils import str2bool
from py_libs.batch_class import batch_main
from custom_data import CustomDataset

import torch
import time
import argparse 
import cv2
import os
import time

from models.pretrained import CNN_pretrained
import json
from torch.utils.data import DataLoader

def pre_trained_format(images,anno,label,device):
    
    targets = []
    for z in range(len(images)):
        d= {}
        d['boxes'] = anno[z].clone().to(device)
        d['labels'] = label[z].clone().to(device)
        targets.append(d)
    return(targets)

def custom_collate_fn(batch):
    # Don't try to stack or pad, just return the list of samples as-is
    images, tensor1, tensor2 = zip(*batch)
    images = [torch.tensor(image) for image in images]
    return torch.stack(images), tensor1, tensor2


def vae(
        variant,
        FLAGS,
    ):
    device = variant.get('device', 'cuda')


    #Load data
    dir_dirt = 'data/train2017'
    

    lst_anno = [file for file in sorted(os.listdir(dir_dirt)) if 'json' in file]
    lst_tensor = [];lst_cat=[];empty = []
    for i,anno in enumerate(lst_anno):
        with open(f'{dir_dirt}/{anno}', 'r') as json_obj:
            json_obj = json_obj.read() 
            json_obj = json.loads(json_obj)
            if not json_obj:
                empty.append(anno[:-5])
            else:
                tensors = torch.stack([torch.tensor(elem['bbox']) for elem in json_obj]) #Stack bbox
                tens_cat = torch.stack([torch.tensor(elem['category_id']) for elem in json_obj]) #stack label
                tensors[:, 2:4] = tensors[:, 0:2] + tensors[:, 2:4] #Put in model format
                lst_tensor.append(tensors)
                lst_cat.append(tens_cat)
    
    #Wont preload image as there are too many
    lst_image = [file for file in sorted(os.listdir(dir_dirt)) if 'jpg' in file and file[:-4] not in empty]

    #load model and its hyperparameters
    cnn = CNN_pretrained().to(device)

    #Load data into dataloader
    amount = len(lst_image)

    indices = torch.randperm(amount).tolist() #Randomly arrange indices

    split1 = int(0.8*amount)
    split2 = int(0.9*amount)
    train_image = [lst_image[i] for i in indices[:split1]]
    eval_image = [lst_image[i] for i in indices[split1:split2]]
    test_image = [lst_image[i] for i in indices[split2:]]
    
    #Set datasets with splits
    data_train = CustomDataset(train_image,lst_tensor[:split1],lst_cat[:split1])
    train_loader = DataLoader(data_train, batch_size=variant['batch_size'], shuffle=True, num_workers=4,collate_fn=custom_collate_fn)
    data_eval = CustomDataset(eval_image,lst_tensor[split1:split2],lst_cat[split1:split2])
    eval_loader = DataLoader(data_eval, batch_size=1, shuffle=True, num_workers=4,collate_fn=custom_collate_fn)
    data_test = CustomDataset(test_image,lst_tensor[split2:],lst_cat[split2:])
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=4,collate_fn=custom_collate_fn)

    #Get optimizzer
    opt_param = list(cnn.parameters())
    optimizer  = torch.optim.Adam(opt_param, lr=0.0001,betas = (0.9,0.999))

    crit_mse = torch.nn.MSELoss()
    crit_bce = torch.nn.BCELoss()

    print('start')
    scaler = torch.cuda.amp.GradScaler() #Mixed precision scaler

    time1 = time.time()
    for i in range(variant['max_iters']):
        cnn.train()
        loss_boxes = 0;loss_labels = 0
        for j,(images,anno,label) in enumerate(train_loader):
            optimizer.zero_grad()
            # anno = anno;label = label.to(device)
            if j >= variant['num_steps_per_iter']:
                break

            #loop needed for pretrained model targets
            targets = pre_trained_format(images,anno,label,device)
                
            images = images.to(device).float().permute(0,3,2,1)
            #Used mixed precision to speed up computation
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred  = cnn(images,targets)
            loss = pred['loss_classifier'] + pred['loss_rpn_box_reg']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            loss_boxes+= pred['loss_rpn_box_reg'].item()
            loss_labels+= pred['loss_classifier'].item()
            scaler.update()

        print(f'Iter: {i}  \tTime_total: {round((time.time()-time1),2)}  \tLoss BBox: {loss_boxes/variant["num_steps_per_iter"]}\tLoss Classifier: {loss_labels/variant["num_steps_per_iter"]}')
        
        cnn.train() #We leave in train mode but do not compute gradients so we can use the built in loss metrics. My own loss would be used given time.
        with torch.no_grad():
            loss_boxes = 0;loss_labels = 0
            for p,(images,anno,label) in enumerate(eval_loader):
                if p >= variant['num_test_steps']:
                    break

                targets = pre_trained_format(images,anno,label,device)
                images = torch.tensor(images,device=device).float().permute(0,3,2,1)
                pred  = cnn(images,targets)
                loss_boxes+= pred['loss_rpn_box_reg'].item()
                loss_labels+= pred['loss_classifier'].item()
            
        print(f'Eval Set -- Loss BBox: {loss_boxes/variant["num_test_steps"]}\tLoss Classifier: {loss_labels/variant["num_test_steps"]}')
            

    torch.save(cnn,'cnn_fine_tuned.pt')



    #----------------------------------------------------------------------
    #Test set
    cnn.eval()
    with torch.no_grad():
        for i,(images,anno,label) in enumerate(test_loader):
            if i >= 100:
                break

            targets = []
            for z in range(len(images)):
                d= {}
                d['boxes'] = torch.tensor(anno[z]).to(device)
                d['labels'] = torch.tensor(label[z]).to(device)
                targets.append(d)
            images = torch.tensor(images,device=device).float().permute(0,3,2,1)
            predictions  = cnn(images)

            numpy_image = images.reshape(*images.shape[1:]).to('cpu').numpy()

            # Convert the numpy array to a cv2 image
            cv2_image = np.transpose(numpy_image, (2, 1, 0))
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

            print(predictions[0])
            idx = torch.where(predictions[0]['scores']>0.001)
            preds = predictions[0]
            for (x,y,x2,y2),cls in zip(preds['boxes'][idx].cpu().detach().numpy(),preds['labels'][idx].cpu().detach().numpy()):
                x=int(x);y=int(y);x2=int(x2);y2=int(y2)
                cv2_image = cv2.rectangle(cv2_image, (x, y), (x2, y2), (36,255,12), 1)
                cv2.putText(cv2_image, str(cls), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 1)

            if not os.path.isdir('data/q2_images'):    
                os.makedirs('data/q2_images', exist_ok=True)
            cv2.imwrite(f'data/q2_images/{i}.jpg', cv2_image*255) 

    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_test_steps', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    FLAGS = args

    vae(variant=vars(args), FLAGS = FLAGS)
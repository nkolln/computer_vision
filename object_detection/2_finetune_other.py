
import numpy as np
from py_libs.utils import str2bool
from custom_data import CustomDataset
from torchvision.transforms import v2

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
    # images = [torch.tensor(image) for image in images]
    return torch.stack(images), tensor1, tensor2


def vae(
        variant,
        FLAGS,
    ):
    device = variant.get('device', 'cuda')

    transforms= v2.Compose([
        # v2.RandomResizedCrop(size=(480, 640), antialias=True),
        v2.Resize(size=(480, 640), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])
    random_image = v2.RandomApply([
        v2.RandomEqualize(),
        v2.RandomAutocontrast(),
        v2.RandomAdjustSharpness(sharpness_factor=2),
        v2.RandomPosterize(bits=2),
        v2.RandomInvert(),
        v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        v2.ColorJitter(brightness=.5, hue=.3)
    ])

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
    cnn = CNN_pretrained(variant).to(device)

    #Load data into dataloader
    amount = len(lst_image)

    indices = torch.randperm(amount).tolist() #Randomly arrange indices

    split1 = int(0.8*amount)
    split2 = int(0.9*amount)
    
    #Set datasets with splits
    data_train = CustomDataset(lst_image[:split1],lst_tensor[:split1],lst_cat[:split1],transforms=transforms)
    train_loader = DataLoader(data_train, batch_size=variant['batch_size'], shuffle=True, num_workers=4,collate_fn=custom_collate_fn)
    data_eval = CustomDataset(lst_image[split1:split2],lst_tensor[split1:split2],lst_cat[split1:split2],transforms=transforms)
    eval_loader = DataLoader(data_eval, batch_size=1, shuffle=True, num_workers=4,collate_fn=custom_collate_fn)
    data_test = CustomDataset(lst_image[split2:],lst_tensor[split2:],lst_cat[split2:],transforms=transforms)
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=4,collate_fn=custom_collate_fn)

    #Get optimizzer
    opt_param = list(cnn.parameters())
    learning_rate = 0.0005
    momentum = 0.9
    weight_decay = 0.0005

    optimizer = torch.optim.SGD(opt_param, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # optimizer  = torch.optim.Adam(opt_param, lr=0.0001,betas = (0.9,0.999))


    print('start')
    scaler = torch.cuda.amp.GradScaler() #Mixed precision scaler

    time1 = time.time()
    for i in range(variant['max_iters']):
        cnn.train()
        loss_boxes = 0;loss_labels = 0;loss_objectness=0;loss_rpn_box_reg=0;loss_sum = 0
        for j,(images,anno,label) in enumerate(train_loader):
            optimizer.zero_grad()
            # anno = anno;label = label.to(device)
            if j >= variant['num_steps_per_iter']:
                break

            #loop needed for pretrained model targets
            targets = pre_trained_format(images,anno,label,device)
                
            images = images.to(device).float()
            images = random_image(images)
            #Used mixed precision to speed up computation
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred  = cnn(images,targets)
            loss = sum(val for val in pred.values())
            loss_sum+=loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            loss_boxes+= pred['loss_box_reg'].item()
            loss_labels+= pred['loss_classifier'].item()
            loss_objectness+= pred['loss_objectness'].item()
            loss_rpn_box_reg+= pred['loss_rpn_box_reg'].item()
            scaler.update()

        print(f'Iter: {i}  \tTime_total: {round((time.time()-time1),2)}\tTotal Loss: {loss_sum/variant["num_steps_per_iter"]}\tLoss BBox: {loss_boxes/variant["num_steps_per_iter"]}\tLoss Classifier: {loss_labels/variant["num_steps_per_iter"]}')
        print(f'Loss Objectness: {loss_objectness/variant["num_steps_per_iter"]}\tLoss RPN Box Reg: {loss_rpn_box_reg/variant["num_steps_per_iter"]}')
        
        cnn.train() #We leave in train mode but do not compute gradients so we can use the built in loss metrics. My own loss would be used given time.
        with torch.no_grad():
            loss_boxes = 0;loss_labels = 0;loss_objectness=0;loss_rpn_box_reg=0;loss_sum = 0
            for p,(images,anno,label) in enumerate(eval_loader):
                if p >= variant['num_eval_steps']:
                    break

                targets = pre_trained_format(images,anno,label,device)
                images = images.to(device).float()
                pred  = cnn(images,targets)
                loss = sum(val for val in pred.values())

                loss_sum+=loss.item()
                loss_boxes+= pred['loss_box_reg'].item()
                loss_labels+= pred['loss_classifier'].item()
                loss_objectness+= pred['loss_objectness'].item()
                loss_rpn_box_reg+= pred['loss_rpn_box_reg'].item()
            
            print('-')
            print(f'Eval Loss Total: {loss_sum/variant["num_eval_steps"]}\tLoss BBox: {loss_boxes/variant["num_eval_steps"]}\tLoss Classifier: {loss_labels/variant["num_eval_steps"]}')
            print(f'Loss Objectness: {loss_objectness/variant["num_eval_steps"]}\tLoss RPN Box Reg: {loss_rpn_box_reg/variant["num_eval_steps"]}\n')
            

    torch.save(cnn,'models\cnn_fine_tuned.pt')



    #----------------------------------------------------------------------
    #Test set
    with torch.no_grad():
        
        loss_boxes = 0;loss_labels = 0;loss_objectness=0;loss_rpn_box_reg=0;loss_sum = 0
        for i,(images,anno,label) in enumerate(test_loader):
            if i >= 100:
                break

            targets = pre_trained_format(images,anno,label,device)
            images = images.to(device).float()
            
            cnn.eval()
            predictions  = cnn(images)

            cnn.train()
            pred  = cnn(images,targets)
            numpy_image = images.reshape(*images.shape[1:]).to('cpu').numpy()
            loss_boxes+= pred['loss_box_reg'].item()
            loss_labels+= pred['loss_classifier'].item()
            loss_objectness+= pred['loss_objectness'].item()
            loss_rpn_box_reg+= pred['loss_rpn_box_reg'].item()
            loss = sum(val for val in pred.values())
            loss_sum+=loss.item()


            # Convert the numpy array to a cv2 image
            cv2_image = np.transpose(numpy_image, (1, 2, 0))
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

            idx = torch.where(predictions[0]['scores']>0.25)
            preds = predictions[0]
            for (x,y,x2,y2),cls in zip(preds['boxes'][idx].cpu().detach().numpy(),preds['labels'][idx].cpu().detach().numpy()):
                x=int(x);y=int(y);x2=int(x2);y2=int(y2)
                cv2_image = cv2.rectangle(cv2_image, (x, y), (x2, y2), (36,255,12), 1)
                cv2.putText(cv2_image, str(cls), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 1)

            


            if not os.path.isdir('data/q2_images'):    
                os.makedirs('data/q2_images', exist_ok=True)
            cv2.imwrite(f'data/q2_images/{i}.jpg', cv2_image*255) 
        print(f'Test Set ---Loss Total:{loss_sum/100}\tLoss BBox: {loss_boxes/100}\tLoss Classifier: {loss_labels/100}')
        print(f'Loss Objectness: {loss_objectness/100}\tLoss RPN Box Reg: {loss_rpn_box_reg/100}\n')


    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_eval_steps', type=int, default=250)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=250)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--new_load', type=str2bool, default='False')


    args = parser.parse_args()
    FLAGS = args

    vae(variant=vars(args), FLAGS = FLAGS)
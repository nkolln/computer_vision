#G9 is using the new actuator strength data

import numpy as np
from py_libs.utils import str2bool
from py_libs.batch_class import batch_main

import torch
import cv2
import time
import argparse 

from models.base_cnn import CNN_class
from models.pretrained import CNN_pretrained

from torchvision.transforms import v2

def vae(
        variant,
        FLAGS,
    ):
    device = variant.get('device', 'cuda')

    path_img = '/home/nkolln/computer_vision/object_detection/data/images/'
    path_anno = '/home/nkolln/computer_vision/object_detection/data/anno.csv'

    #create batch class
    batch = batch_main(path_img,path_anno,variant['batch_size'],device)

    #create transformation
    
    transforms_test = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
    ])

    #load model and its hyperparameters
    cnn = CNN_pretrained().to(device)
    
    torch.autograd.set_detect_anomaly(True)              
    #Load eval set
    for i in range(variant['max_iters']):
        cnn.eval()
        with torch.no_grad():
            img,anno = batch.get_batch_train(transforms_test)
            predictions  = cnn(img)
            print(type(predictions[0]))
            for label in predictions[0].keys():
                print(label)
                print(predictions[0][label].shape)

            numpy_image = img.reshape(*img.shape[1:]).to('cpu').numpy()
            # Convert the numpy array to a cv2 image
            cv2_image = np.transpose(numpy_image, (1, 2, 0))
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

            idx = torch.where(predictions[0]['scores']>0.1)
            preds = predictions[0]
            for (x,y,x2,y2),cls in zip(preds['boxes'][idx].to('cpu').numpy(),preds['labels'][idx].to('cpu').numpy()):
                x=int(x);y=int(y);x2=int(x2);y2=int(y2)
                cv2_image = cv2.rectangle(cv2_image, (x, y), (x2, y2), (36,255,12), 1)
                cv2.putText(cv2_image, str(cls), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 1)

            # Display the image using cv2
            cv2.imshow("Image", cv2_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    torch.save(cnn,'cnn_to_param.pt')


    #load trainer with eval

    #Get results and test final result

    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_eval_episodes', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=50)
    parser.add_argument('--num_steps_per_iter', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    FLAGS = args

    vae(variant=vars(args), FLAGS = FLAGS)
import fiftyone.zoo as foz
from torchvision.transforms import v2
from torchvision.io import read_image
import torch
import os
import csv

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=["person", "car", "truck"],
    max_samples=200,
)
transforms = v2.Compose([
                # v2.RandomResizedCrop(size=(3456, 4608), antialias=True),
                # v2.Resize(size=(480//2, 640//2), antialias=True), #216 288
                v2.Resize(size=(480, 640), antialias=True), #216 288
                v2.ToDtype(torch.float32, scale=True),
            ])

dataset = transforms(dataset)
i = 0

if not os.path.isdir('data/images'):    
    os.makedirs('data/images', exist_ok=True)
with open('data/anno.csv', mode='w+', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['id','x','y','width','height','label'])
    
    for sample in dataset:
        #img process
        img_load = read_image(sample.filepath)
        if img_load.shape[0] ==3:
            print(img_load.shape)
            if img_load.shape[-1]!= 640 and img_load.shape[1] > img_load.shape[-1]:
                img_load = img_load.permute(0,2,1)
            img_load = transforms(img_load)
            torch.save(img_load,f'data/images/{i}.pt')

            #anno process
            for det in sample.ground_truth.detections:
                # if str(det.label) in ["person", "car", "truck"] :
                writer.writerow([i,det.bounding_box[0],det.bounding_box[1],det.bounding_box[2],det.bounding_box[3],det.label])
        i+=1
    

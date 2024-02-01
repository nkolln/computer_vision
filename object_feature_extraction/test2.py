import torch
from torchvision.transforms import v2



mdl = torch.load('/home/nkolln/orbisk_test/trained_model/cnn_to_param.pt').to('cuda')
img = torch.load('/home/nkolln/orbisk_test/img_clean/IMG_20190801_130547.pt')
img = img.reshape(1,*img.shape).to('cuda')
batch = batch_main(path_img,path_anno,variant['batch_size'],device)
print(img.shape)
transforms_test = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
])

with torch.no_grad():
    img = transforms_test(img)
    print(mdl(img))
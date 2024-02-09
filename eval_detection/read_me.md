#File summary and explanation for use
#The file assumes that the data is in the original folder within the data folder
#Basic model settings such as max_iters, num_steps_per_iter, and batch_size are available for testing

#Imports-------------------------------------
Can use the requirements.txt file if its easier
Imports I use: torch, opencv-python, torchvision, and 

#--------------------------------------
1_pretrained.py This is the file for the the first task. It simply loads the pretrained model, inputs the images, and displays the corresponding images in the folder data/q1_images
Usage: python3 1_pretrained.py 
If you would like to run the fine tuned model on the original dataset, we can run the file using 
Usage: python3 1_pretrained.py --out_file q2_images --new_load True

#--------------------------------------
1_pretrained.py This is the file for the the second task. It takes the data, splits it and then trains. The trained model is saved and then used to generate the final outputs. 
Usage python3 2_finetune.py

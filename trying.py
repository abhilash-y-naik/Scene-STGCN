import numpy as np
import torch
from keras.preprocessing.image import img_to_array, load_img
# torch.cuda.empty_cache()
# exit()
from torch.autograd import Variable
import tarfile
#


# from graph_model import network
# model = network.deeplabv3plus_mobilenet(num_classes=19, output_stride=16).cuda()
# model.load_state_dict(torch.load(fname)["model_state"])
# pic = './graph_model/Screenshot.png'
# img_data = load_img(pic)
# # print(img_data.shape)
# image_array = img_to_array(img_data).reshape(3, 768, 1366)
# # print(image_array.shape)
# image_array = Variable(torch.from_numpy(image_array).unsqueeze(0)).float().cuda()
# print(image_array.shape)
# image_features = model(image_array)
# image_features = image_features.data.to('cpu').numpy()
# print(model.eval())
# print(image_features.shape)
# exit()
# np.random.seed(2)
# from graph_model.pie_data import PIE
# #
# data_opts = {'fstride': 1,
#              'sample_type': 'all',
#              'height_rng': [0, float('inf')],
#              'squarify_ratio': 0,
#              'data_split_type': 'default',  # kfold, random, default
#              'seq_type': 'intention',  # crossing , intention
#              'min_track_size': 0,  # discard tracks that are shorter
#              'max_size_observe': 15,  # number of observation frames
#              'max_size_predict': 5,  # number of prediction frames
#              'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
#              'balance': True,  # balance the training and testing samples
#              'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
#              'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
#              'encoder_input_type': [],
#              'decoder_input_type': ['bbox'],
#              'output_type': ['intention_binary']
#              }

# imdb = PIE(data_path= './PIE_dataset')
# beh_seq_train = imdb.generate_data_trajectory_sequence('test', **data_opts)
# beh_seq_train = imdb.balance_samples_count(beh_seq_train, label_type='intention_binary')

# def normalize_undigraph(A):
#     Dl = np.sum(A, 0)
#     num_node = A.shape[0]
#     Dn = np.zeros((num_node, num_node))
#     for i in range(num_node):
#         if Dl[i] > 0:
#             Dn[i, i] = Dl[i] ** (-0.5)
#
#     Dn[np.isinf(Dn)] = 0.
#     DAD = np.dot(np.dot(Dn, A), Dn)
#
#     return DAD
# def normalize_digraph(A):
#     Dl = np.sum(A, 0)
#     num_node = A.shape[0]
#     Dn = np.zeros((num_node, num_node))
#     for i in range(num_node):
#         if Dl[i] > 0:
#             Dn[i, i] = Dl[i]**(-1)
#     AD = np.dot(A, Dn)
#     return AD
# x = np.random.randint(1,5,size=(5, 4, 3))
# A = np.random.randint(0,2,size=(1, 5, 3, 3))
# importance = np.random.randint(2,3,size=(5,1,1))
# A = np.zeros((3,3))
# A[0, 0] = 2
# A[1, 0] = 1
# A[2, 0] = 1
# A[0, 1] = 1
# A[0, 2] = 1
# x = torch.from_numpy(x)
# A = torch.from_numpy(A)
# print(importance)
# print('x\n', x)
# x1 = torch.sum(x,dim=2).reshape(5,-1)
# print('x\n', x1)
# # x2 = x.permute(0,2,1,3).contiguous()
# # print('x\n',x2)
#
# print('A\n',A)
# print('A1\n', normalize_undigraph(A))
# print('A2\n', normalize_digraph(A))
# # exit()
# x = torch.einsum('ntvclb,ntvw->ntwclb', (x, A))
# print('X\n',x)
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
#
# import os
# # fname = './graph_model/checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'
# path = 'U:/thesis_code/data/data/graph/train/features_context_pad_resize/vgg16_bn'
# img_folder = os.listdir(path)
# # print(img_folder)
#
# for img_f in img_folder:
#     new_path = os.path.join(path, img_f)
#     files = os.listdir(new_path)
#     for file in files:
#         for pick in os.listdir(os.path.join(new_path, file)):
#             if (pick.split('.pkl')[0].split('_')[-1]) == '0':
#                 del_path = os.path.join(new_path, file, pick)
#                 os.remove(del_path)
                # print(del_path)



import os
# fname = './graph_model/checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'
path = 'D:/thesis_analysis'
img_folder = os.listdir(path)
# print(img_folder)
# for img_f in img_folder:
    # if img_f == 'error_3' or img_f == 'error_2':

new_path = os.path.join(path, 'error_3')
files = os.listdir(new_path)
new_path = os.path.join(path, 'error')
files1 = os.listdir(new_path)
count = 0
for f in files:
    if f not in files1:
        print(f)
        count += 1

print(count)
            # for pick in os.listdir(os.path.join(new_path, file)):
            #     if (pick.split('.pkl')[0].split('_')[-1]) == '0':
            #         del_path = os.path.join(new_path, file, pick)
            #         os.remove(del_path)
            #         print(del_path)

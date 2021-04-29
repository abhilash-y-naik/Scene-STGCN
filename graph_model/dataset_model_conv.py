import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
from torch.autograd import Variable

import cv2
import os
import math
import pickle
import numpy as np
import networkx as nx

from graph_model.utils import *


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset, track, data_opts, data_type, regen_pkl=False):

        self.track = track
        self.dataset = dataset
        self.data_opts = data_opts
        self.data_type = data_type

        self.img_sequences = self.track['images']
        self.ped_ids = self.track['ped_ids']
        self.bbox_sequences = self.track['bboxes']
        self.decoder_input = self.track['decoder_input']
        self.unique_frames = self.dataset['unique_frame']
        self.unique_ped = self.dataset['unique_ped']
        self.unique_bbox = self.dataset['unique_bbox']
        self.unique_image = self.dataset['unique_image']

        self.pedestrian_nodes = 1
        self.vehicle_nodes = 0
        self.traffic_light_nodes = 0
        self.sign_nodes = 0
        self.crosswalk_nodes = 0
        self.transit_station_nodes = 0
        self.ego_vehicle = True

        if self.ego_vehicle:
            self.ego_vehicle_node = 1
        else:
            self.ego_vehicle_node = 0

        self.max_nodes = self.pedestrian_nodes + self.vehicle_nodes +\
                         self.traffic_light_nodes + self.crosswalk_nodes +\
                         self.transit_station_nodes + self.sign_nodes + self.ego_vehicle_node

        self.path = 'E:\PIE_dataset\images'  # Folder where the images are saved

        self.num_examples = len(self.track['ped_ids'])

        print(self.num_examples)
        save_path = self.get_path(type_save='data',
                                  data_type='features' + '_' + self.data_opts[
                                      'crop_type'] + '_' + self.data_opts[
                                                'crop_mode'],  # images
                                  model_name='vgg16_bn',
                                  data_subset=self.data_type, ind=1)

        variable = True
        if variable:
            vgg16 = models.vgg16_bn().cuda()
            vgg16.load_state_dict(torch.load("./pretrained_models/vgg16_bn-6c64b313.pth"))
            self.context_model = nn.Sequential(*list(vgg16.children())[:-1])

            try:
                self.convnet = self.context_model
            except:
                raise Exception("No context model is defined")

        self.feature_save_folder = './data/nodes_and_features/' + str(self.data_type)
        self.seq_len = len(self.track['images'][0])
        # False for stop creating pickle file for nodes,graph and adjacency matrix
        self.regen_pkl = regen_pkl

        if self.regen_pkl:
            i = -1
            for img_sequences, bbox_sequences, ped_ids in zip(self.track['images'],
                                                              self.track['bboxes'],
                                                              self.track['ped_ids']):

                i += 1
                seq_len = len(img_sequences)
                nodes = []
                max_nodes = 0
                node_features = []
                img_centre_seq = []
                bbox_location_seq = []

                for imp, b, p in zip(img_sequences, bbox_sequences, ped_ids):

                    update_progress(i / self.num_examples)
                    imp = imp.replace(os.sep, '/')
                    set_id = imp.split('/')[-3]
                    vid_id = imp.split('/')[-2]
                    img_name = imp.split('/')[-1].split('.')[0]

                    key = str(set_id + vid_id)
                    frames = self.unique_frames[key].tolist()

                    ped = self.unique_ped[key]
                    box = self.unique_bbox[key]
                    image = self.unique_image[key]
                    index = frames.index(int(img_name))

                    if max_nodes < len(ped[index]):
                        max_nodes = len(ped[index])

                    img_features_unsorted = {}
                    img_centre_unsorted = {}
                    bbox_location_unsorted = {}

                    for object_keys in ped[0]:
                        img_features_unsorted[object_keys] = []
                        img_centre_unsorted[object_keys] = []
                        bbox_location_unsorted[object_keys] = []

                        if not ped[index][object_keys]:
                            continue

                        if object_keys == 'pedestrian':
                            for idx, (n, bb, im) in enumerate(
                                    zip(ped[index][object_keys], box[index][object_keys], image[index][object_keys])):

                                if p == n:
                                    img_save_folder = os.path.join(save_path, set_id, vid_id)
                                    im = im.replace(os.sep, '/')
                                    im_name = im.split('/')[-1].split('.')[0]
                                    # img_save_folder = os.path.join(save_path, set_id, vid_id)
                                    img_save_path = os.path.join(img_save_folder, im_name + '_' + n[0] + '.pkl')

                                    if not os.path.exists(img_save_path):

                                        img_folder = os.path.join(self.path, set_id, vid_id)
                                        img_path = os.path.join(img_folder, im_name + '.png')
                                        img_data = load_img(img_path)
                                        bbox = jitter_bbox(img_path, [bb], 'enlarge', 2)[0]
                                        bbox = squarify(bbox, 1, img_data.size[0])
                                        bbox = list(map(int, bbox[0:4]))
                                        cropped_image = img_data.crop(bbox)
                                        img_data = img_pad(cropped_image, mode='pad_resize', size=224)
                                        image_array = img_to_array(img_data).reshape(-1, 224, 224)
                                        image_array = Variable(
                                            torch.from_numpy(image_array).unsqueeze(0)).float().cuda()
                                        image_features = self.convnet(image_array)
                                        image_features = image_features.data.to('cpu').numpy()
                                        if not os.path.exists(img_save_folder):
                                            os.makedirs(img_save_folder)
                                        with open(img_save_path, 'wb') as fid:
                                            pickle.dump(image_features, fid, pickle.HIGHEST_PROTOCOL)

                                    img_features_unsorted[object_keys].append([img_save_path])
                                    img_centre_unsorted[object_keys].append(self.get_center(bb))
                                    bbox_location_unsorted[object_keys].append(bb)

                        for idx, (n, bb, im) in enumerate(
                                zip(ped[index][object_keys], box[index][object_keys], image[index][object_keys])):

                            if p != n:

                                img_save_folder = os.path.join(save_path, set_id, vid_id)
                                im = im.replace(os.sep, '/')
                                im_name = im.split('/')[-1].split('.')[0]

                                img_save_path = os.path.join(img_save_folder, im_name + '_' + n[0] + '.pkl')
                                # print(img_save_path)
                                if not os.path.exists(img_save_path):

                                    img_folder = os.path.join(self.path, set_id, vid_id)
                                    img_path = os.path.join(img_folder, im_name + '.png')
                                    img_data = load_img(img_path)
                                    bbox = jitter_bbox(img_path, [bb], 'enlarge', 2)[0]
                                    bbox = squarify(bbox, 1, img_data.size[0])
                                    bbox = list(map(int, bbox[0:4]))
                                    cropped_image = img_data.crop(bbox)
                                    img_data = img_pad(cropped_image, mode='pad_resize', size=224)
                                    image_array = img_to_array(img_data).reshape(-1, 224, 224)
                                    image_array = Variable(torch.from_numpy(image_array).unsqueeze(0)).float().cuda()
                                    image_features = self.convnet(image_array)
                                    image_features = image_features.data.to('cpu').numpy()
                                    if not os.path.exists(img_save_folder):
                                        os.makedirs(img_save_folder)
                                    with open(img_save_path, 'wb') as fid:
                                        pickle.dump(image_features, fid, pickle.HIGHEST_PROTOCOL)

                                img_features_unsorted[object_keys].append([img_save_path])
                                img_centre_unsorted[object_keys].append(self.get_center(bb))
                                bbox_location_unsorted[object_keys].append(bb)

                    # features, and their centre location in each frame

                    img_features = {}
                    bbox_location = {}
                    img_centre = {}

                    for object_keys in ped[0]:
                        img_features[object_keys] = []
                        bbox_location[object_keys] = []
                        img_centre[object_keys] = []

                        if not img_centre_unsorted[object_keys]:
                            continue

                        distance = (np.asarray([img_centre_unsorted['pedestrian'][0]] *
                                               len(img_centre_unsorted[object_keys]))
                                    - np.asarray(img_centre_unsorted[object_keys]))

                        distance = np.linalg.norm(distance, axis=1).reshape(-1, 1)
                        distance_sorted = sorted(distance)

                        for _, dist in enumerate(distance_sorted):
                            index = distance.tolist().index(dist.tolist())
                            img_centre[object_keys].append(img_centre_unsorted[object_keys][index])
                            img_features[object_keys].append(img_features_unsorted[object_keys][index])
                            bbox_location[object_keys].append(bbox_location_unsorted[object_keys][index])

                    node_features.append(img_features.copy())  # Path of the node features
                    bbox_location_seq.append(bbox_location.copy())  # BBox location
                    img_centre_seq.append(img_centre.copy())  # Bounding box centre location


                all_node_features_seq = []
                for s in range(self.seq_len):

                    all_node_features = []
                    for k in node_features[0]:
                        if k == 'pedestrian':
                            for num, saving_nodes in enumerate(node_features[s][k]):
                                if num < self.pedestrian_nodes:
                                    all_node_features.append(saving_nodes)
                        if k == 'vehicle':
                            for num, saving_nodes in enumerate(node_features[s][k]):
                                if num < self.vehicle_nodes:
                                    all_node_features.append(saving_nodes)
                        if k == 'traffic_light':
                            for num, saving_nodes in enumerate(node_features[s][k]):
                                if num < self.traffic_light_nodes:
                                    all_node_features.append(saving_nodes)
                        if k == 'transit_station':
                            for num, saving_nodes in enumerate(node_features[s][k]):
                                if num < self.transit_station_nodes:
                                    all_node_features.append(saving_nodes)
                        if k == 'sign':
                            for num, saving_nodes in enumerate(node_features[s][k]):
                                if num < self.sign_nodes:
                                    all_node_features.append(saving_nodes)
                        if k == 'crosswalk':
                            for num, saving_nodes in enumerate(node_features[s][k]):
                                if num < self.crosswalk_nodes:
                                    all_node_features.append(saving_nodes)

                        if k == 'ego_vehicle':
                            for num, saving_nodes in enumerate(node_features[s][k]):
                                if num < self.ego_vehicle_node:
                                    all_node_features.append(saving_nodes)

                    all_node_features_seq.append(all_node_features)

                self.feature_save_folder = 'U:/thesis_code/data/nodes_and_features/' + str(self.data_type)
                self.feature_save_path = os.path.join(self.feature_save_folder, str(i) + '.pkl')
                if not os.path.exists(self.feature_save_folder):
                    os.makedirs(self.feature_save_folder)
                with open(self.feature_save_path, 'wb') as fid:
                    pickle.dump((img_centre_seq, all_node_features_seq, bbox_location_seq), fid,
                                pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, index):

        graph, adj_matrix, adj_matrix_spatial, decoder_input = self.load_images_and_process(index)

        if index % 2000 == 0:
            print(decoder_input[0])
            print(self.decoder_input[index][0])

        train_data = torch.from_numpy(graph), \
                     torch.from_numpy(adj_matrix), \
                     torch.from_numpy(adj_matrix_spatial), \
                     torch.from_numpy(decoder_input), \
                     torch.from_numpy(self.track['output'][index][0])
        return train_data

    def anorm(self, p1, p2):
        NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        if NORM == 0:
            return 0
        return 1 / (NORM)

    def get_center(self, box):
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    def get_path(self,
                 type_save='models',  # model or data
                 models_save_folder='',
                 model_name='convlstm_encdec',
                 file_name='',
                 data_subset='',
                 data_type='',
                 save_root_folder='./data/',
                 ind=1):

        assert (type_save in ['models', 'data'])
        if data_type != '':
            assert (any([d in data_type for d in ['images', 'features']]))
        if ind == 0:
            root = os.path.join(save_root_folder)
        else:
            root = os.path.join(save_root_folder, type_save)

        if type_save == 'models':
            save_path = os.path.join(save_root_folder, 'graph', 'intention', models_save_folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            return os.path.join(save_path, file_name), save_path
        else:
            save_path = os.path.join(root, 'graph', data_subset, data_type, model_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            return save_path

    def visualisation(self, seq_len, nodes, img_centre_seq):
        '''
        This creates visulasation as star graph
        :param seq_len: Length of the input sequence
        :param ped_ids: This will be matched to create the primary node
        :param nodes: All the nodes in the sequence
        :param img_centre_seq: To draw the circles to denote nodes
        '''
        sorted_img = []
        for s in range(seq_len):

            step = nodes[s]

            for k in nodes[0]:

                if k == 'pedestrian':
                    img_p = str(step[k][0][0])
                    img_p = img_p.replace(os.sep, '/')
                    img_p1 = img_p.split('/')[-3:]
                    img_p2 = img_p1[2].split('_')[0]
                    img_p = os.path.join(*img_p1[:-1])
                    img_p = os.path.join(self.path, img_p, img_p2 + '.png')
                    img = cv2.imread(img_p)
                    img_cp_p = img_centre_seq[s][k][0]
                    cv2.circle(img, (int(img_cp_p[0]), int(img_cp_p[1])),
                               radius=0, color=(0, 0, 255), thickness=25)

                    secondary_nodes = []

                    for h, stp in enumerate(step[k]):
                        if h > 0:
                            img_cp_s = img_centre_seq[s][k][h]
                            secondary_nodes.append(img_cp_s)
                            cv2.circle(img, (int(img_cp_s[0]), int(img_cp_s[1])),
                                       radius=0, color=(0, 255, 0), thickness=25)
                            # cv2.circle(img, (int(img_cp_s[0]), int(img_cp_s[1])),
                            #            radius=0, color=(0, 255, 0), thickness=25)
                            cv2.putText(img, str(h) + ':' + str(k), ((int(img_cp_s[0]) - 20, int(img_cp_s[1] - 20))),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                        (255, 0, 0), 2)

                if k != 'pedestrian':
                    for h, stp in enumerate(step[k]):
                        # print(stp)
                        img_cp_s = img_centre_seq[s][k][h]
                        secondary_nodes.append(img_cp_s)
                        cv2.circle(img, (int(img_cp_s[0]), int(img_cp_s[1])),
                                       radius=0, color=(0, 255, 0), thickness=25)
                            # cv2.circle(img, (int(img_cp_s[0]), int(img_cp_s[1])),
                            #            radius=0, color=(0, 255, 0), thickness=25)
                        cv2.putText(img, str(h)+':'+str(k), ((int(img_cp_s[0]) - 20, int(img_cp_s[1]-20))), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (255, 0, 0), 2)
                for img_cp_s in secondary_nodes:
                    cv2.line(img, (int(img_cp_p[0]), int(img_cp_p[1])), (int(img_cp_s[0]), int(img_cp_s[1])),
                             [255, 0, 0], 2)

                sorted_img.append(img)

        out = cv2.VideoWriter("%s/%s_%s_%s.avi" % ("U:/thesis_code/visualisation",
                                                   img_p1[0], img_p1[1], img_p2),
                              cv2.VideoWriter_fourcc(*"MJPG"), 15, (1920, 1080))
        for img_id in sorted_img:
            out.write(img_id)
        out.release()

    def normalize_undigraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-0.5)

        # Dn[np.isinf(Dn)] = 0.
        DAD = np.dot(np.dot(Dn, A), Dn)

        return DAD

    def load_images_and_process(self,
                                index,
                                visualise=False):

        with open(os.path.join(self.feature_save_folder, str(index) + '.pkl'), 'rb') as fid:
            try:
                img_centre_seq, node_features, bbox_location_seq = pickle.load(fid)
            except:
                img_centre_seq, node_features, bbox_location_seq = pickle.load(fid, encoding='bytes')

        max_nodes = self.max_nodes

        decoder_input = np.zeros((self.seq_len, 1, len(bbox_location_seq[0][0])))
        # decoder_input = np.zeros((self.seq_len, len(bbox_location_seq[0][0])))
        graph = np.zeros((self.seq_len, max_nodes, 512, 7, 7))
        adj_matrix = np.zeros((self.seq_len, max_nodes, max_nodes))
        adj_matrix_spatial = np.zeros((max_nodes, max_nodes))
        # adj_matrix = np.zeros((max_nodes, max_nodes))

        for s in range(self.seq_len):


            step = node_features[s]
            bbox_location = bbox_location_seq[s]['pedestrian']
            decoder_input[s, :] = bbox_location[0]

            for h, stp in enumerate(step):
                    with open(str(stp[0]), 'rb') as fid:
                        try:
                            img_features = pickle.load(fid)
                        except:
                            img_features = pickle.load(fid, encoding='bytes')

                    img_features = np.squeeze(img_features)
                    graph[s, h, :] = img_features
                    adj_matrix[s, h, h] = 1
                    adj_matrix_spatial[h, h] = 1

                    if h > 0:
                        # adj_matrix[s, h, h] = 2
                        # img_cp_s = img_centre_seq[s][h]
                        # l2_norm = self.anorm(img_cp_p, img_cp_s)
                        adj_matrix[s, h, 0] = 1  # l2_norm
                        adj_matrix[s, 0, h] = 1  # l2_norm
                        adj_matrix_spatial[h, 0] = 1
                        adj_matrix_spatial[0, h] = 1

            # g = nx.from_numpy_matrix(adj_matrix[s, :, :])
            # adj_matrix[s, :, :] = self.normalized_laplacian_matrix(g).toarray()
            adj_matrix[s, :, :] = self.normalize_undigraph(adj_matrix[s, :, :])
            # print(adj_matrix[s,:,:])
        adj_matrix_spatial = self.normalize_undigraph(adj_matrix_spatial)
        # print(adj_matrix)
        if visualise:
            self.visualisation(self.seq_len, node_features, img_centre_seq)

        return graph, adj_matrix, adj_matrix_spatial, decoder_input

    def __len__(self):
        return self.num_examples

        # For fully connected graph
        # sorted_img = []
        # for s in range(seq_len):
        #     step = node_features[s]
        #     print(step)
        #     img_p = str(step[0][0])
        #     img_p = img_p.replace(os.sep, '/')
        #     img_p = img_p.split('_' + str(nodes[s][0][0]) + '.pkl')[0]
        #     img_p1 = img_p.split('/')[-3:]
        #     img_p = os.path.join(*img_p1)
        #     img_p = os.path.join(self.path, img_p + '.png')
        #     img = cv2.imread(img_p)
        #
        #     for h in range(len(step)):
        #         with open(str(step[h][0]), 'rb') as fid:
        #             try:
        #                 img_features = pickle.load(fid)
        #             except:
        #                 img_features = pickle.load(fid, encoding='bytes')
        #
        #         img_features = np.squeeze(img_features)
        #         graph[s, h, :] = img_features
        #         adj_matrix[s, h, h] = 1
        #         img_cp_p = img_centre_seq[s][h]
        #         cv2.circle(img, (int(img_cp_p[0]), int(img_cp_p[1])),
        #                    radius=0, color=(0, 0, 255), thickness=15)
        #         for k in range(h + 1, len(step)):
        #             img_cp_s = img_centre_seq[s][k]
        #             l2_norm = self.anorm(img_cp_p, img_cp_s)
        #             adj_matrix[s, h, k] = l2_norm
        #             adj_matrix[s, k, h] = l2_norm
        #             cv2.circle(img, (int(img_cp_s[0]), int(img_cp_s[1])),
        #                        radius=0, color=(0, 255, 0), thickness=15)
        #
        #             cv2.line(img, (int(img_cp_p[0]), int(img_cp_p[1])), (int(img_cp_s[0]), int(img_cp_s[1])),
        #                                                                   [255, 0, 0], 2)
        #

        #
        #     sorted_img.append(img)
        #
        # out = cv2.VideoWriter("%s/%s_%s_%s_%s.avi" % ("U:/thesis_code",
        #                     img_p1[0], img_p1[1], img_p1[2], str(nodes[s][h][0])),
        #                     cv2.VideoWriter_fourcc(*"MJPG"), 1, (1920, 1080))
        # for img_id in sorted_img:
        #      out.write(img_id)
        # out.release()

    # max_nodes = 3
    # graph = np.zeros((seq_len, max_nodes, 7, 7, 512))
    # adj_matrix = np.zeros((seq_len, max_nodes, max_nodes))
    #
    # for s in range(seq_len):
    #     step = node_features[s]
    #     # adj_matrix[s] = np.identity(max_nodes)
    #     count = 0
    #     for h, stp in enumerate(step):
    #         count += 1
    #         if count <= max_nodes:
    #             with open(str(stp[0]), 'rb') as fid:
    #                 try:
    #                     img_features = pickle.load(fid)
    #                 except:
    #                     img_features = pickle.load(fid, encoding='bytes')
    #
    #             # Take this part out when new pickle would be generated
    #             ###############################################################
    #             B, H, W, C = img_features.shape
    #             img_features = torch.Tensor(img_features).view(B, C, H, W)
    #             img_features = img_features.permute((0, 2, 3, 1)).contiguous()
    #             ###############################################################
    #
    #             img_features = np.squeeze(img_features)
    #             graph[s, h, :] = img_features
    #
    #             adj_matrix[s, h, h] = 1
    #
    #             stp = stp[0].replace(os.sep, '/')
    #             stp = stp.split('/')[-1].split('.')[0]
    #             stp = stp.split('_')
    #             stp = (stp[1] + '_' + stp[2] + '_' + stp[3])
    #             if stp == ped_ids[0][0]:
    #                 img_cp_p = img_centre_seq[s][h]
    #                 k = h
    #
    #     count = 0
    #     for h in range(len(step)):
    #         # if h != k:
    #         count += 1
    #         if h != k and count <= max_nodes:
    #             img_cp_s = img_centre_seq[s][h]
    #             l2_norm = self.anorm(img_cp_p, img_cp_s)
    #             adj_matrix[s, h, k] = l2_norm
    #             adj_matrix[s, k, h] = l2_norm
    #
    #     # # To avoid normalise when the maximum node is 1
    #     # if (adj_matrix[s, :, :] == np.eye(adj_matrix[s, :, :].shape[0])).all():
    #     #     continue
    #     g = nx.from_numpy_matrix(adj_matrix[s, :, :])
    #     adj_matrix[s, :, :] = nx.normalized_laplacian_matrix(g).toarray()

#
# class Dataset_test(torch.utils.data.Dataset):
#
#     def __init__(self, dataset, track, data_opts, data_type):
#
#         self.track = track
#         self.dataset = dataset
#         self.data_opts = data_opts
#         self.data_type = data_type
#
#         self.img_sequences = self.track['images']
#         self.ped_ids = self.track['ped_ids']
#         self.bbox_sequences = self.track['bboxes']
#         self.unique_frames = self.dataset['unique_frame']
#         self.unique_ped = self.dataset['unique_ped']
#         self.unique_bbox = self.dataset['unique_bbox']
#         self.unique_image = self.dataset['unique_image']
#
#         self.path = 'E:\PIE_dataset\images'  # Folder where the images are saved
#         # Create context model
#         vgg16 = models.vgg16_bn()
#         vgg16.load_state_dict(torch.load("./pretrained_models/vgg16_bn-6c64b313.pth"))
#         self.context_model = nn.Sequential(*list(vgg16.children())[:-1])
#
#         self.num_examples = len(self.track['ped_ids'])
#
#     def __getitem__(self, index):
#
#         # crop only bounding boxes
#         # update_progress(index / self.num_examples)
#         graph, adj_matrix = self.load_images_and_process(self.track['images'][index],
#                                                          self.track['bboxes'][index],
#                                                          self.track['ped_ids'][index],
#                                                          data_type=self.data_type,
#                                                          index=index,
#                                                          save_path=self.get_path(type_save='data',
#                                                                                  data_type='features' + '_' +
#                                                                                            self.data_opts[
#                                                                                                'crop_type'] + '_' +
#                                                                                            self.data_opts[
#                                                                                                'crop_mode'],  # images
#                                                                                  model_name='vgg16_bn',
#                                                                                  data_subset=self.data_type))
#
#         train_data = torch.from_numpy(graph), torch.from_numpy(adj_matrix), torch.from_numpy(
#             self.track['output'][index][0])
#         return train_data
#
#     def anorm(self, p1, p2):
#         NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
#         if NORM == 0:
#             return 0
#         return 1 / (NORM)
#
#     def get_center(self, box):
#         return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
#
#     def get_path(self,
#                  type_save='models',  # model or data
#                  models_save_folder='',
#                  model_name='convlstm_encdec',
#                  file_name='',
#                  data_subset='',
#                  data_type='',
#                  save_root_folder='./data/'):
#
#         assert (type_save in ['models', 'data'])
#         if data_type != '':
#             assert (any([d in data_type for d in ['images', 'features']]))
#         root = os.path.join(save_root_folder, type_save)
#
#         if type_save == 'models':
#             save_path = os.path.join(save_root_folder, 'graph', 'intention', models_save_folder)
#             if not os.path.exists(save_path):
#                 os.makedirs(save_path)
#             return os.path.join(save_path, file_name), save_path
#         else:
#             save_path = os.path.join(root, 'graph', data_subset, data_type, model_name)
#             if not os.path.exists(save_path):
#                 os.makedirs(save_path)
#             return save_path
#
#     def visualisation(self, seq_len, ped_ids, nodes, img_centre_seq):
#         '''
#         This creates visulasation as star graph
#         :param seq_len: Length of the input sequence
#         :param ped_ids: This will be matched to create the primary node
#         :param nodes: All the nodes in the sequence
#         :param img_centre_seq: To draw the circles to denote nodes
#         '''
#         sorted_img = []
#         for s in range(seq_len):
#             step = nodes[s]
#             # print(step)
#             img_p = str(step[0][0])
#             img_p = img_p.replace(os.sep, '/')
#             img_p = img_p.split('_' + str(nodes[s][0][0]) + '.pkl')[0]
#             img_p1 = img_p.split('/')[-3:]
#             img_p = os.path.join(*img_p1)
#             img_p = os.path.join(self.path, img_p + '.png')
#             img = cv2.imread(img_p)
#             secondary_nodes = []
#             for h, stp in enumerate(step):
#                 stp = stp[0].replace(os.sep, '/')
#                 stp = stp.split('/')[-1].split('.')[0]
#                 stp = stp.split('_')
#                 stp = (stp[1] + '_' + stp[2] + '_' + stp[3])
#
#                 if stp == ped_ids[0][0]:
#                     img_cp_p = img_centre_seq[s][h]
#                     cv2.circle(img, (int(img_cp_p[0]), int(img_cp_p[1])),
#                                radius=0, color=(0, 0, 255), thickness=25)
#
#                 else:
#                     img_cp_s = img_centre_seq[s][h]
#                     secondary_nodes.append(img_cp_s)
#                     cv2.circle(img, (int(img_cp_s[0]), int(img_cp_s[1])),
#                                radius=0, color=(0, 255, 0), thickness=25)
#
#             for img_cp_s in secondary_nodes:
#                 cv2.line(img, (int(img_cp_p[0]), int(img_cp_p[1])), (int(img_cp_s[0]), int(img_cp_s[1])),
#                          [255, 0, 0], 2)
#
#             sorted_img.append(img)
#
#         out = cv2.VideoWriter("%s/%s_%s_%s_%s.avi" % ("U:/thesis_code",
#                                                       img_p1[0], img_p1[1], img_p1[2], str(ped_ids[0][0])),
#                               cv2.VideoWriter_fourcc(*"MJPG"), 1, (1920, 1080))
#         for img_id in sorted_img:
#             out.write(img_id)
#         out.release()
#
#     def load_images_and_process(self,
#                                 img_sequences,
#                                 bbox_sequences,
#                                 ped_ids,
#                                 save_path,
#                                 index,
#                                 data_type='train',
#                                 regen_pkl=False,
#                                 visualise=False):
#
#         try:
#             convnet = self.context_model
#         except:
#             raise Exception("No context model is defined")
#
#         seq_len = len(img_sequences)
#         nodes = []
#         max_nodes = 0
#         node_features = []
#         img_centre_seq = []
#         for imp, b, p in zip(img_sequences, bbox_sequences, ped_ids):
#             imp = imp.replace(os.sep, '/')
#             set_id = imp.split('/')[-3]
#             vid_id = imp.split('/')[-2]
#             img_name = imp.split('/')[-1].split('.')[0]
#
#             key = str(set_id + vid_id)
#             frames = self.unique_frames[key].tolist()
#             ped = self.unique_ped[key]
#             bbox = self.unique_bbox[key]
#             image = self.unique_image[key]
#             index = frames.index(int(img_name))
#
#             if max_nodes < len(ped[index]):
#                 max_nodes = len(ped[index])
#
#             img_features = []
#             img_centre = []
#
#             for idx, (n, bb, im) in enumerate(zip(ped[index], bbox[index], image[index])):
#
#                 img_save_folder = os.path.join(save_path, set_id, vid_id)
#                 im = im.replace(os.sep, '/')
#                 im_name = im.split('/')[-1].split('.')[0]
#                 img_save_path = os.path.join(img_save_folder, im_name + '_' + n[0] + '.pkl')
#
#                 if not os.path.exists(img_save_path) or regen_pkl:
#
#                     img_folder = os.path.join(self.path, set_id, vid_id)
#                     img_path = os.path.join(img_folder, im_name + '.png')
#                     img_data = load_img(img_path)
#                     bbox = jitter_bbox(img_path, [bb], 'enlarge', 2)[0]
#                     bbox = squarify(bbox, 1, img_data.size[0])
#                     bbox = list(map(int, bbox[0:4]))
#                     cropped_image = img_data.crop(bbox)
#                     img_data = img_pad(cropped_image, mode='pad_resize', size=224)
#                     image_array = img_to_array(img_data).reshape(-1, 224, 224)
#                     image_array = Variable(torch.from_numpy(image_array).unsqueeze(0)).float()
#                     image_features = convnet(image_array)
#                     image_features = image_features.permute((0, 2, 3, 1)).contiguous()
#                     image_features = image_features.data.to('cpu').numpy()
#                     if not os.path.exists(img_save_folder):
#                         os.makedirs(img_save_folder)
#                     with open(img_save_path, 'wb') as fid:
#                         pickle.dump(image_features, fid, pickle.HIGHEST_PROTOCOL)
#
#                 img_features.append([img_save_path])
#                 img_centre.append(self.get_center(bb))
#
#             # Pedestrian id, features, and their centre location in each frame
#             nodes.append(ped[index])  # Pedestrian Id
#             node_features.append(img_features)  # Path of the node features
#             img_centre_seq.append(img_centre)  # Bounding box centre location
#
#         graph = np.zeros((seq_len, max_nodes, 7, 7, 512))
#         adj_matrix = np.zeros((seq_len, max_nodes, max_nodes))
#
#         for s in range(seq_len):
#             step = node_features[s]
#             # adj_matrix[s] = np.identity(max_nodes)
#             count = 0
#             for h, stp in enumerate(step):
#                 count += 1
#                 if count <= max_nodes:
#                     with open(str(stp[0]), 'rb') as fid:
#                         try:
#                             img_features = pickle.load(fid)
#                         except:
#                             img_features = pickle.load(fid, encoding='bytes')
#
#                     # Take this part out when new pickle would be generated
#                     ###############################################################
#                     B, H, W, C = img_features.shape
#                     img_features = torch.Tensor(img_features).view(B, C, H, W)
#                     img_features = img_features.permute((0, 2, 3, 1)).contiguous()
#                     ###############################################################
#
#                     img_features = np.squeeze(img_features)
#                     graph[s, h, :] = img_features
#
#                     adj_matrix[s, h, h] = 1
#
#                     stp = stp[0].replace(os.sep, '/')
#                     stp = stp.split('/')[-1].split('.')[0]
#                     stp = stp.split('_')
#                     stp = (stp[1] + '_' + stp[2] + '_' + stp[3])
#                     if stp == ped_ids[0][0]:
#                         img_cp_p = img_centre_seq[s][h]
#                         k = h
#
#             count = 0
#             for h in range(len(step)):
#                 # if h != k:
#                 count += 1
#                 if h != k and count <= max_nodes:
#                     img_cp_s = img_centre_seq[s][h]
#                     l2_norm = self.anorm(img_cp_p, img_cp_s)
#                     adj_matrix[s, h, k] = l2_norm
#                     adj_matrix[s, k, h] = l2_norm
#
#             g = nx.from_numpy_matrix(adj_matrix[s, :, :])
#             adj_matrix[s, :, :] = nx.normalized_laplacian_matrix(g).toarray()
#
#         if visualise:
#             self.visualisation(seq_len, ped_ids, nodes, img_centre_seq)
#
#         return graph, adj_matrix
#
#     def __len__(self):
#         return self.num_examples

# Important
#
# class Dataset(torch.utils.data.Dataset):
#
#     def __init__(self, dataset, track, data_opts, data_type, regen_pkl = False ):
#
#         self.track = track
#         self.dataset = dataset
#         self.data_opts = data_opts
#         self.data_type = data_type
#
#         self.img_sequences = self.track['images']
#         self.ped_ids = self.track['ped_ids']
#         self.bbox_sequences = self.track['bboxes']
#         self.decoder_input = self.track['decoder_input']
#         self.unique_frames = self.dataset['unique_frame']
#         self.unique_ped = self.dataset['unique_ped']
#         self.unique_bbox = self.dataset['unique_bbox']
#         self.unique_image = self.dataset['unique_image']
#
#         self.path = 'E:\PIE_dataset\images' # Folder where the images are saved
#
#         self.num_examples = len(self.track['ped_ids'])
#
#         print(self.num_examples)
#         save_path = self.get_path(type_save='data',
#                                   data_type='features' + '_' + self.data_opts[
#                                       'crop_type'] + '_' + self.data_opts[
#                                                 'crop_mode'],  # images
#                                   model_name='vgg16_bn',
#                                   data_subset=self.data_type)
#         # Create context model
#         variable = False
#         if variable:
#             vgg16 = models.vgg16_bn().cuda()
#             vgg16.load_state_dict(torch.load("./pretrained_models/vgg16_bn-6c64b313.pth"))
#             self.context_model = nn.Sequential(*list(vgg16.children())[:-1])
#
#             try:
#                 self.convnet = self.context_model
#             except:
#                 raise Exception("No context model is defined")
#
#         self.feature_save_folder = 'U:/thesis_code/data/nodes_and_features/' + str(self.data_type)
#         self.seq_len = len(self.track['images'][0])
#         # False for stop creating pickle file for nodes,graph and adjacency matrix
#         self.regen_pkl = regen_pkl
#
#         if self.regen_pkl:
#             i = -1
#             for img_sequences, bbox_sequences, ped_ids in zip(self.track['images'],
#                                                               self.track['bboxes'],
#                                                               self.track['ped_ids']):
#
#                 i += 1
#                 seq_len = len(img_sequences)
#                 nodes = []
#                 max_nodes = 0
#                 node_features = []
#                 img_centre_seq = []
#                 bbox_location_seq = []
#
#                 for imp, b, p in zip(img_sequences, bbox_sequences, ped_ids):
#
#                     update_progress(i/self.num_examples)
#                     imp = imp.replace(os.sep, '/')
#                     set_id = imp.split('/')[-3]
#                     vid_id = imp.split('/')[-2]
#                     img_name = imp.split('/')[-1].split('.')[0]
#
#                     key = str(set_id + vid_id)
#                     frames = self.unique_frames[key].tolist()
#                     ped = self.unique_ped[key]
#                     box = self.unique_bbox[key]
#                     image = self.unique_image[key]
#                     index = frames.index(int(img_name))
#
#                     if max_nodes < len(ped[index]):
#                         max_nodes = len(ped[index])
#
#                     img_features_unsorted = []
#                     img_centre_unsorted = []
#                     bbox_location_unsorted = []
#
#                     for idx, (n, bb, im) in enumerate(zip(ped[index], box[index], image[index])):
#
#                         if p == n:
#                             img_save_folder = os.path.join(save_path, set_id, vid_id)
#                             im = im.replace(os.sep, '/')
#                             im_name = im.split('/')[-1].split('.')[0]
#                             img_save_path = os.path.join(img_save_folder, im_name + '_' + n[0] + '.pkl')
#
#                             if not os.path.exists(img_save_path):
#
#                                 img_folder = os.path.join(self.path, set_id, vid_id)
#                                 img_path = os.path.join(img_folder, im_name + '.png')
#                                 img_data = load_img(img_path)
#                                 bbox = jitter_bbox(img_path, [bb], 'enlarge', 2)[0]
#                                 bbox = squarify(bbox, 1, img_data.size[0])
#                                 bbox = list(map(int, bbox[0:4]))
#                                 cropped_image = img_data.crop(bbox)
#                                 img_data = img_pad(cropped_image, mode='pad_resize', size=224)
#                                 image_array = img_to_array(img_data).reshape(-1, 224, 224)
#                                 image_array = Variable(torch.from_numpy(image_array).unsqueeze(0)).float().cuda()
#                                 image_features = self.convnet(image_array)
#                                 image_features = image_features.data.to('cpu').numpy()
#                                 if not os.path.exists(img_save_folder):
#                                     os.makedirs(img_save_folder)
#                                 with open(img_save_path, 'wb') as fid:
#                                     pickle.dump(image_features, fid, pickle.HIGHEST_PROTOCOL)
#
#                             img_features_unsorted.append([img_save_path])
#                             img_centre_unsorted.append(self.get_center(bb))
#                             bbox_location_unsorted.append(bb)
#
#                     for idx, (n, bb, im) in enumerate(zip(ped[index], box[index], image[index])):
#
#                         if p != n:
#
#                             img_save_folder = os.path.join(save_path, set_id, vid_id)
#                             im = im.replace(os.sep, '/')
#                             im_name = im.split('/')[-1].split('.')[0]
#                             img_save_path = os.path.join(img_save_folder, im_name + '_' + n[0] + '.pkl')
#
#                             if not os.path.exists(img_save_path):
#
#                                 img_folder = os.path.join(self.path, set_id, vid_id)
#                                 img_path = os.path.join(img_folder, im_name + '.png')
#                                 img_data = load_img(img_path)
#                                 bbox = jitter_bbox(img_path, [bb], 'enlarge', 2)[0]
#                                 bbox = squarify(bbox, 1, img_data.size[0])
#                                 bbox = list(map(int, bbox[0:4]))
#                                 cropped_image = img_data.crop(bbox)
#                                 img_data = img_pad(cropped_image, mode='pad_resize', size=224)
#                                 image_array = img_to_array(img_data).reshape(-1, 224, 224)
#                                 image_array = Variable(torch.from_numpy(image_array).unsqueeze(0)).float().cuda()
#                                 image_features = self.convnet(image_array)
#                                 image_features = image_features.data.to('cpu').numpy()
#                                 if not os.path.exists(img_save_folder):
#                                     os.makedirs(img_save_folder)
#                                 with open(img_save_path, 'wb') as fid:
#                                     pickle.dump(image_features, fid, pickle.HIGHEST_PROTOCOL)
#
#                             img_features_unsorted.append([img_save_path])
#                             img_centre_unsorted.append(self.get_center(bb))
#                             bbox_location_unsorted.append(bb)
#
#
#                     # features, and their centre location in each frame
#                     img_features = []
#                     bbox_location = []
#                     img_centre = []
#
#                     distance = (np.asarray([img_centre_unsorted[0]]*len(img_centre_unsorted))
#                                 - np.asarray(img_centre_unsorted))
#                     distance = np.linalg.norm(distance, axis=1).reshape(-1, 1)
#                     distance_sorted = sorted(distance)
#
#                     for _, dist in enumerate(distance_sorted):
#                         index = distance.tolist().index(dist.tolist())
#                         img_centre.append(img_centre_unsorted[index])
#                         img_features.append(img_features_unsorted[index])
#                         bbox_location.append(bbox_location_unsorted[index])
#
#                     node_features.append(img_features)  # Path of the node features
#                     bbox_location_seq.append(bbox_location)  # BBox location
#                     img_centre_seq.append(img_centre)  # Bounding box centre location
#
#                 self.max_nodes = max_nodes
#
#                 self.feature_save_folder = 'U:/thesis_code/data/nodes_and_features/' + str(self.data_type)
#                 self.feature_save_path = os.path.join(self.feature_save_folder, str(i) + '.pkl')
#                 if not os.path.exists(self.feature_save_folder):
#                     os.makedirs(self.feature_save_folder)
#                 with open(self.feature_save_path, 'wb') as fid:
#                     pickle.dump((img_centre_seq, node_features, bbox_location_seq, self.max_nodes), fid, pickle.HIGHEST_PROTOCOL)
#
#     def __getitem__(self, index):
#
#         # crop only bounding boxes
#         # update_progress(index / self.num_examples)
#         # graph, adj_matrix = self.load_images_and_process(self.track['images'][index],
#         #                                          self.track['bboxes'][index],
#         #                                          self.track['ped_ids'][index],
#         #                                          data_type=self.data_type,
#         #                                          index = index,
#         #                                          save_path=self.get_path(type_save='data',
#         #                                                                  data_type='features' + '_' + self.data_opts[
#         #                                                                      'crop_type'] + '_' + self.data_opts[
#         #                                                                                'crop_mode'],  # images
#         #                                                                  model_name='vgg16_bn',
#         #                                                                  data_subset=self.data_type))
#
#         graph, adj_matrix, decoder_input = self.load_images_and_process(index)
#
#         if index % 2000 == 0:
#             print(decoder_input[0])
#             print(self.decoder_input[index][0])
#
#
#         train_data = torch.from_numpy(graph), \
#                      torch.from_numpy(adj_matrix),\
#                      torch.from_numpy(decoder_input),\
#                      torch.from_numpy(self.track['output'][index][0])
#         return train_data
#
#     def anorm(self, p1, p2):
#         NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
#         if NORM == 0:
#             return 0
#         return 1 / (NORM)
#
#     def get_center(self, box):
#         return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
#
#     def get_path(self,
#                  type_save='models',  # model or data
#                  models_save_folder='',
#                  model_name='convlstm_encdec',
#                  file_name='',
#                  data_subset='',
#                  data_type='',
#                  save_root_folder='./data/'):
#
#         assert (type_save in ['models', 'data'])
#         if data_type != '':
#             assert (any([d in data_type for d in ['images', 'features']]))
#         root = os.path.join(save_root_folder, type_save)
#
#         if type_save == 'models':
#             save_path = os.path.join(save_root_folder, 'graph', 'intention', models_save_folder)
#             if not os.path.exists(save_path):
#                 os.makedirs(save_path)
#             return os.path.join(save_path, file_name), save_path
#         else:
#             save_path = os.path.join(root, 'graph', data_subset, data_type, model_name)
#             if not os.path.exists(save_path):
#                 os.makedirs(save_path)
#             return save_path
#
#     def visualisation(self, seq_len, nodes, img_centre_seq):
#         '''
#         This creates visulasation as star graph
#         :param seq_len: Length of the input sequence
#         :param ped_ids: This will be matched to create the primary node
#         :param nodes: All the nodes in the sequence
#         :param img_centre_seq: To draw the circles to denote nodes
#         '''
#         sorted_img = []
#         for s in range(seq_len):
#             step = nodes[s]
#
#             img_p = str(step[0][0])
#             img_p = img_p.replace(os.sep, '/')
#             img_p1 = img_p.split('/')[-3:]
#             img_p2 = img_p1[2].split('_')[0]
#             img_p = os.path.join(*img_p1[:-1])
#             img_p = os.path.join(self.path, img_p, img_p2 + '.png')
#             img = cv2.imread(img_p)
#             img_cp_p = img_centre_seq[s][0]
#             cv2.circle(img, (int(img_cp_p[0]), int(img_cp_p[1])),
#                        radius=0, color=(0, 0, 255), thickness=25)
#
#             secondary_nodes = []
#
#             for h, stp in enumerate(step):
#                 if h > 0:
#                     img_cp_s = img_centre_seq[s][h]
#                     secondary_nodes.append(img_cp_s)
#                     cv2.circle(img, (int(img_cp_s[0]), int(img_cp_s[1])),
#                                 radius=0, color=(0, 255, 0), thickness=25)
#
#             for img_cp_s in secondary_nodes:
#                 cv2.line(img, (int(img_cp_p[0]), int(img_cp_p[1])), (int(img_cp_s[0]), int(img_cp_s[1])),
#                          [255, 0, 0], 2)
#
#             sorted_img.append(img)
#
#         out = cv2.VideoWriter("%s/%s_%s_%s.avi" % ("U:/thesis_code",
#                                                       img_p1[0], img_p1[1], img_p2),
#                               cv2.VideoWriter_fourcc(*"MJPG"), 1, (1920, 1080))
#         for img_id in sorted_img:
#             out.write(img_id)
#         out.release()
#
#     def normalize_undigraph(self, A):
#         Dl = np.sum(A, 0)
#         num_node = A.shape[0]
#         Dn = np.zeros((num_node, num_node))
#         for i in range(num_node):
#             if Dl[i] > 0:
#                 Dn[i, i] = Dl[i] ** (-0.5)
#
#         # Dn[np.isinf(Dn)] = 0.
#         DAD = np.dot(np.dot(Dn, A), Dn)
#
#         return DAD
#
#     def load_images_and_process(self,
#                                 index,
#                                 visualise=False):
#
#         with open(os.path.join(self.feature_save_folder, str(index) +'.pkl'), 'rb') as fid:
#             try:
#                 img_centre_seq, node_features, bbox_location_seq, max_nodes_seq = pickle.load(fid)
#             except:
#                 img_centre_seq, node_features, bbox_location_seq, max_nodes_seq = pickle.load(fid, encoding='bytes')
#
#         # max_nodes = max_nodes_seq
#         max_nodes = 1
#
#         decoder_input = np.zeros((self.seq_len, max_nodes, len(bbox_location_seq[0][0])))
#         # decoder_input = np.zeros((self.seq_len, len(bbox_location_seq[0][0])))
#         graph = np.zeros((self.seq_len, max_nodes, 512, 7, 7))
#         # graph = np.zeros((self.seq_len, max_nodes, 7, 7, 512))
#         adj_matrix = np.zeros((self.seq_len, max_nodes, max_nodes))
#         # adj_matrix = np.zeros((max_nodes, max_nodes))
#         for s in range(self.seq_len):
#
#             step = node_features[s]
#             bbox_location = bbox_location_seq[s]
#             img_cp_p = img_centre_seq[s][0]
#             adj_matrix[s, 0, 0] = 2
#             decoder_input[s, :] = bbox_location[0]
#
#             count = 0
#             for h, stp in enumerate(step):
#                 count += 1
#                 if count <= max_nodes:
#                     with open(str(stp[0]), 'rb') as fid:
#                         try:
#                             img_features = pickle.load(fid)
#                         except:
#                             img_features = pickle.load(fid, encoding='bytes')
#
#                     img_features = np.squeeze(img_features)
#                     graph[s, h, :] = img_features
#
#                     if h > 0:
#                         # adj_matrix[s, h, h] = 2
#                         img_cp_s = img_centre_seq[s][h]
#                         # l2_norm = self.anorm(img_cp_p, img_cp_s)
#                         adj_matrix[s, h, 0] = 1 #l2_norm
#                         adj_matrix[s, 0, h] = 1 #l2_norm
#
#             # print(adj_matrix[s,:,:])
#             # g = nx.from_numpy_matrix(adj_matrix[s, :, :])
#             # adj_matrix[s, :, :] = self.normalized_laplacian_matrix(g).toarray()
#             adj_matrix[s, :, :] = self.normalize_undigraph(adj_matrix[s, :, :])
#             # print(adj_matrix[s,:,:])
#         # print(adj_matrix)
#         if visualise:
#             self.visualisation(self.seq_len, node_features, img_centre_seq)
#
#         return graph, adj_matrix, decoder_input
#
#     def __len__(self):
#         return self.num_examples

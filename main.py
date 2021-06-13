import time
import os
import pickle
import matplotlib.pyplot as plt
from prettytable import PrettyTable


# train models with data up to critical point
# only for PIE
# train_test = 0 (train only), 1 (train-test), 2 (test only)
def train_intent(train_test, model, data_path):
    data_opts = {'fstride': 1,
                 'sample_type': 'all',
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': 'default',  # kfold, random, default
                 'seq_type': 'intention',  # crossing , intention
                 'min_track_size': 0,  # discard tracks that are shorter
                 'max_size_observe': 15,  # number of observation frames
                 'max_size_predict': 5,  # number of prediction frames
                 'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
                 'balance': True,  # balance the training and testing samples
                 'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
                 'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
                 'encoder_input_type': [],
                 'decoder_input_type': ['bbox'],
                 'output_type': ['intention_binary']
                 }

    saved_files_path = ''
    #################################################################################################
    # PIE model
    #################################################################################################
    if model == 0:

        from pie_model.pie_intent import PIEIntent
        from pie_model.pie_data import PIE

        imdb = PIE(data_path=data_path)

        t = PIEIntent(num_hidden_units=128,
                      regularizer_val=0.001,
                      lstm_dropout=0.4,
                      lstm_recurrent_dropout=0.2,
                      convlstm_num_filters=64,
                      convlstm_kernel_size=2)

        # pretrained_model_path = 'data/pie/intention/context_loc_pretrained'
        pretrained_model_path = 'data/pie/intention/16Feb2021-22h41m27s'

        if train_test < 2:  # Train

            beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
            beh_seq_train = imdb.balance_samples_count(beh_seq_train, label_type='intention_binary')

            beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
            beh_seq_val = imdb.balance_samples_count(beh_seq_val, label_type='intention_binary')

            saved_files_path = t.train(data_train=beh_seq_train,
                                       data_val=beh_seq_val,
                                       epochs=200,
                                       batch_size=128,
                                       data_opts=data_opts)

        if train_test > 0:  # Test
            if saved_files_path == '':
                saved_files_path = pretrained_model_path
            beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)

            beh_seq_test = imdb.balance_samples_count(beh_seq_test, label_type='intention_binary')
            acc, f1, pre, rec = t.test_chunk(beh_seq_test, data_opts, saved_files_path, False)

            t = PrettyTable(['Acc', 'F1', 'Precision', 'Recall'])
            t.title = 'Intention model (local_context + bbox)'
            t.add_row([acc, f1, pre, rec])

            print(t)

    ##################################################################################################

    ##################################################################################################
    # Graph Model
    ##################################################################################################
    elif model == 1:

        from graph_model.pie_intent_graph import PIEIntent
        from graph_model.pie_data import PIE

        imdb = PIE(data_path=data_path)

        t = PIEIntent()

        pretrained_model_path = 'data/graph/intention/12Jun2021-16h38m21s'
        # pretrained_model_path = 'graph_model/pretrained weight'
        if train_test < 2:  # Train

            data_save_path = os.path.join('./PIE_dataset' + '/data_cache/graph/' + 'beh_seq_train' + '.pkl')
            regen_pkl = False

            if os.path.exists(data_save_path) and not regen_pkl:
                with open(data_save_path, 'rb') as fid:
                    try:
                        beh_seq_train = pickle.load(fid)
                    except:
                        beh_seq_train = pickle.load(fid, encoding='bytes')
            else:
                beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
                #
                with open(data_save_path, 'wb') as fid:
                    pickle.dump(beh_seq_train, fid, pickle.HIGHEST_PROTOCOL)

            beh_seq_train = imdb.balance_samples_count(beh_seq_train, label_type='intention_binary')

            regen_pkl = False
            data_save_path = os.path.join('./PIE_dataset' + '/data_cache/graph/' + 'beh_seq_test' + '.pkl')
            if os.path.exists(data_save_path) and not regen_pkl:
                with open(data_save_path, 'rb') as fid:
                    try:
                        beh_seq_val = pickle.load(fid)
                    except:
                        beh_seq_val = pickle.load(fid, encoding='bytes')
            else:
                beh_seq_val = imdb.generate_data_trajectory_sequence('test', **data_opts)

                with open(data_save_path, 'wb') as fid:
                    pickle.dump(beh_seq_val, fid, pickle.HIGHEST_PROTOCOL)

            # beh_seq_val = imdb.balance_samples_count(beh_seq_val, label_type='intention_binary')
            # data_save_path = os.path.join('./PIE_dataset' + '/data_cache/graph/' + 'beh_seq_test' + '.pkl')
            # if os.path.exists(data_save_path) and not regen_pkl:
            #     with open(data_save_path, 'rb') as fid:
            #         try:
            #             beh_seq_test = pickle.load(fid)
            #         except:
            #             beh_seq_test = pickle.load(fid, encoding='bytes')
            # else:
            #     beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
            #     # beh_seq_val = imdb.balance_samples_count(beh_seq_val, label_type='intention_binary')
            #     with open(data_save_path, 'wb') as fid:
            #         pickle.dump(beh_seq_val, fid, pickle.HIGHEST_PROTOCOL)

            saved = t.train(data_train=beh_seq_train,
                                           data_val=beh_seq_val,
                                           data_test='',
                                           epochs=30,
                                           batch_size=128,
                                           data_opts=data_opts)
                                           # layers = 3,
                                           # datasize = datasize[num])


            # datasize = [250, 500, 1000, 1500, 2000, 2500]
            # for num in range(len(datasize)):
                # accuracy = []
                # loss = []
                # for i in range(5):
                #     saved = t.train(data_train=beh_seq_train,
                #                                data_val=beh_seq_val,
                #                                data_test='',
                #                                epochs=50,
                #                                batch_size=128,
                #                                data_opts=data_opts,
                #                                layers =i,
                #                                datasize = datasize[num])
                #     accuracy.append(saved[0])
                #     loss.append(saved[1])

                # color = ['r', 'b', 'g', 'k', 'y']
                # plt.figure(1)
                # for i in range(len(accuracy)):
                #     plt.plot(accuracy[i], color[i], label=str(i+1)+'layer')
                # plt.ylabel("accuracy")
                # plt.xlabel("epoch")
                # plt.title("Accuracy per epoch for different GCN layers (Dataset = {} samples)".format(str(2*datasize[num])))
                # plt.legend()
                # plt.grid()
                # plt.savefig("./epoch_accuracy_"+str(2*datasize[num])+".png")
                # plt.close(1)
                #
                # plt.figure(2)
                # for i in range(len(loss)):
                #     plt.plot(loss[i], color[i], label=str(i+1)+'layer')
                # plt.ylabel("loss")
                # plt.xlabel("epoch")
                # plt.title("Loss per epoch for different GCN layers (Dataset = {} samples)".format(str(2*datasize[num])))
                # plt.legend()
                # plt.grid()
                # plt.savefig("./epoch_loss_"+str(2*datasize[num])+".png")
                # plt.close(2)

        if train_test > 0:  # Test
            if saved_files_path == '':
                saved_files_path = pretrained_model_path

            data_save_path = os.path.join('./PIE_dataset' + '/data_cache/graph/' + 'beh_seq_test' + '.pkl')
            regen_pkl = False

            if os.path.exists(data_save_path) and not regen_pkl:
                with open(data_save_path, 'rb') as fid:
                    try:
                        beh_seq_test = pickle.load(fid)
                    except:
                        beh_seq_test = pickle.load(fid, encoding='bytes')
            else:
                beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
                with open(data_save_path, 'wb') as fid:
                    pickle.dump(beh_seq_test, fid, pickle.HIGHEST_PROTOCOL)

            # beh_seq_test = imdb.balance_samples_count(beh_seq_test, label_type='intention_binary')
            # d = {'unique_frame': beh_seq_test['unique_frame'].copy(), 'unique_ped': beh_seq_test['unique_ped'].copy(),
            #      'unique_image': beh_seq_test['unique_image'].copy(), 'unique_bbox': beh_seq_test['unique_bbox'].copy()}
            # new_seq_data = {}
            # for k in beh_seq_test:
            #     if k not in d.keys():
            #         seq_data_k = beh_seq_test[k]
            #         if not isinstance(beh_seq_test[k], list):
            #             new_seq_data[k] = beh_seq_test[k]
            #         else:
            #             new_seq_data[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i < 100]

            # new_seq_data['unique_frame'] = d['unique_frame']
            # new_seq_data['unique_ped'] = d['unique_ped']
            # new_seq_data['unique_image'] = d['unique_image']
            # new_seq_data['unique_bbox'] = d['unique_bbox']
            acc, f1, pre, rec = t.test_chunk(beh_seq_test, data_opts, saved_files_path, False)

            t = PrettyTable(['Acc', 'F1', 'Precision', 'Recall'])
            t.title = 'Intention model (local_context + bbox)'
            t.add_row([acc, f1, pre, rec])

            print(t)




def main(train_test=0, model=1, data_path='./PIE_dataset'):

    train_intent(train_test=train_test, model=model, data_path=data_path)


if __name__ == '__main__':

    try:
        train_test = int(0)  # train_test: 0 - train only, 1 - train and test, 2 - test only
        model = int(1)  # model:0 - PIE, model:1 - Graph
        data_path = './PIE_dataset'  # Path of the split images
        # for i in range(10):
        main(train_test=train_test, model=model, data_path=data_path)
           # print('Next Iteration')

    except ValueError:
        raise ValueError('Usage: python train_test.py <train_test>\n'
                         'train_test: 0 - train only, 1 - train and test, 2 - test only\n' '<model>\n' 
                         'model: 0 - PIE, 1 - Graph\n')

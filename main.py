import time
import os
import pickle
from prettytable import PrettyTable
import argparse

# train models with data up to critical point
# train_test = 0 (train only), 1 (train-test), 2 (test only)
def train_intent(train_test, data_path, test_weights,
                 regen_pkl,first_time,path, node_info):
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

    ##################################################################################################
    # Graph Model
    ##################################################################################################

    from graph_model.pie_intent_graph import PIEIntent
    from graph_model.pie_data import PIE

    imdb = PIE(data_path=data_path)

    t = PIEIntent()

    pretrained_model_path = test_weights
    
    if train_test < 2:  # Train

        data_save_path = os.path.join('./PIE_dataset' + '/data_cache/graph/' + 'beh_seq_train' + '.pkl')

        if os.path.exists(data_save_path) and not regen_pkl:
            with open(data_save_path, 'rb') as fid:
                try:
                    beh_seq_train = pickle.load(fid)
                except:
                    beh_seq_train = pickle.load(fid, encoding='bytes')
        else:
            beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
            with open(data_save_path, 'wb') as fid:
                pickle.dump(beh_seq_train, fid, pickle.HIGHEST_PROTOCOL)

        beh_seq_train = imdb.balance_samples_count(beh_seq_train, label_type='intention_binary')

        data_save_path = os.path.join('./PIE_dataset' + '/data_cache/graph/' + 'beh_seq_val' + '.pkl')
        if os.path.exists(data_save_path) and not regen_pkl:
            with open(data_save_path, 'rb') as fid:
                try:
                    beh_seq_val = pickle.load(fid)
                except:
                    beh_seq_val = pickle.load(fid, encoding='bytes')
        else:
            beh_seq_val = imdb.generate_data_trajectory_sequence('val', **data_opts)
            with open(data_save_path, 'wb') as fid:
                pickle.dump(beh_seq_val, fid, pickle.HIGHEST_PROTOCOL)

        saved_files_path = t.train(data_train=beh_seq_train,
                        data_val=beh_seq_val,
                        epochs=50,
                        batch_size=128,
                        data_opts=data_opts,
                        first_time=first_time,
                        path=path,
                        node_info=node_info)

    if train_test > 0:  # Test
        if saved_files_path == '':
            saved_files_path = pretrained_model_path

        data_save_path = os.path.join('./PIE_dataset' + '/data_cache/graph/' + 'beh_seq_test' + '.pkl')
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

        acc, f1, pre, rec = t.test_chunk(beh_seq_test,
                                         data_opts,
                                         saved_files_path,
                                         first_time=first_time,
                                         path=path,
                                         node_info=node_info)

        t = PrettyTable(['Acc', 'F1', 'Precision', 'Recall'])
        t.title = 'Intention model (local_context + bbox)'
        t.add_row([acc, f1, pre, rec])

        print(t)


def main(node_info, train_test=0, data_path='./PIE_dataset', test_weights= '', regen_pkl= False, first_time=False, path=''):

    train_intent(train_test=train_test, data_path=data_path, test_weights= test_weights, regen_pkl= regen_pkl,
                 first_time=first_time, path=path, node_info=node_info)


if __name__ == '__main__':

    try:
        parser = argparse.ArgumentParser(description='PyTorch Semantic-Line Training')
        # arguments from command line
        parser.add_argument('--train_test', default=0, type =int, help="train_test: 0 - train only, 1 - train and test, 2 - test only")
        parser.add_argument('--weights_folder', default='data/graph/intention/2objects', help='path to folder name')
        parser.add_argument('--data_path', default='./PIE_dataset', help='path to pie annotations')
        parser.add_argument('--regen_pkl', default=False, help='to regenerate pie data')
        parser.add_argument('--first_time', default=False, help='True for first time')
        parser.add_argument('--image_path', default='./images', help='path to split images')
        parser.add_argument('--pedestrian', default=1, type=int, help='number of pedestrians including target')
        parser.add_argument('--vehicle', default=0, type=int, help='number of vehicles')
        parser.add_argument('--crosswalk', default=1, type=int, help='number of crosswalk')
        parser.add_argument('--traffic_sign', default=0, type=int, help='number of traffic signs')
        parser.add_argument('--transit_station', default=0, type=int, help='number of transit_station')
        parser.add_argument('--traffic_light', default=1, type=int, help='number of traffic lights')

        args = parser.parse_args()

        train_test = args.train_test
        data_path = args.data_path
        test_weights = args.weights_folder
        regen_pkl = args.regen_pkl
        first_time = args.first_time
        path = args.image_path
        node_info = {'pedestrian': args.pedestrian,  # default should be one
                          'vehicle': args.vehicle,
                          'traffic_light': args.traffic_light,
                          'transit_station': args.transit_station,
                          'sign': args.traffic_sign,
                          'crosswalk': args.crosswalk,
                          'ego_vehicle': 0}

        main(node_info, train_test=train_test, data_path=data_path, test_weights=test_weights, regen_pkl = regen_pkl,
             first_time=first_time, path=path)

    except ValueError:
        raise ValueError('Usage: python train_test.py <train_test>\n'
                         'train_test: 0 - train only, 1 - train and test, 2 - test only\n')

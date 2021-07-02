# import os
# import numpy as np
# import pickle
# path = 'U:/thesis_code/poses/pose_set01.pkl'
# with open(os.path.join(path), 'rb') as fid:
#     try:
#         pose_set = pickle.load(fid)
#     except:
#         pose_set = pickle.load(fid, encoding='bytes')
#
# print(len(pose_set['video_0001']['14019_1_1_18']))
# print(len(np.zeros((36))))
import numpy as np
# def x():
#     print(1)
#     return np.ones((5,1)), np.zeros((2,1))
# acc = []
# val = []
# for i in range(1,3):
#     print(i)

# print(5/3.2)
# print(acc)
# print(val)

sub =[[[-6.0887e-04, -1.5600e-03,  8.7485e-04],
         [-2.3315e-02, -1.3993e-03,  0.0000e+00],
         [-1.9113e-02,  0.0000e+00, -2.1777e-03]],
        [[ 7.8190e-05,  5.9582e-05, -1.6809e-05],
         [-5.5768e-04, -4.6441e-03,  0.0000e+00],
         [-1.7564e-03,  0.0000e+00, -1.5743e-03]],
        [[ 5.0992e-04,  1.7379e-03,  2.7071e-04],
         [ 5.1126e-03,  7.0544e-03,  0.0000e+00],
         [ 2.4681e-03,  0.0000e+00,  3.1369e-03]],
        [[ 7.7381e-03,  7.5457e-04,  5.0664e-07],
         [ 2.5935e-03,  8.7097e-04,  0.0000e+00],
         [ 1.1664e-03,  0.0000e+00,  5.1269e-03]],
        [[ 1.3436e-03,  2.5414e-05, -2.6357e-04],
         [-1.3551e-03,  1.5425e-03,  0.0000e+00],
         [-3.2483e-03,  0.0000e+00, -5.5172e-03]],
        [[-2.7191e-02, -3.4133e-04, -3.0749e-04],
         [-1.7330e-03,  9.5570e-04,  0.0000e+00],
         [ 1.0648e-04,  0.0000e+00, -4.5948e-03]],
        [[-3.9938e-02, -3.5409e-04, -4.6381e-04],
         [ 2.4903e-03, -2.9432e-03,  0.0000e+00],
         [ 2.2105e-03,  0.0000e+00, -2.9490e-03]],
        [[-6.9589e-02, -4.9077e-04, -4.8335e-03],
         [ 5.3190e-04, -2.6904e-04,  0.0000e+00],
         [ 1.5298e-04,  0.0000e+00,  1.5333e-02]],
        [[-2.2437e-02,  6.3687e-03,  6.5792e-04],
         [-5.7528e-03, -8.0640e-03,  0.0000e+00],
         [-5.6630e-04,  0.0000e+00,  1.4156e-03]],
        [[-1.2582e-02,  5.0474e-03, -1.5398e-03],
         [ 3.1017e-03, -4.6886e-03,  0.0000e+00],
         [ 1.1614e-02,  0.0000e+00,  5.4826e-02]],
        [[-2.3837e-02,  8.2481e-03, -6.5831e-03],
         [ 1.4166e-02,  1.5941e-02,  0.0000e+00],
         [ 3.5507e-02,  0.0000e+00, -1.4911e-02]],
        [[ 3.9319e-02,  2.3316e-03,  3.0885e-03],
         [ 2.7802e-02,  6.6304e-03,  0.0000e+00],
         [ 5.8829e-02,  0.0000e+00, -6.9478e-02]],
        [[-1.6831e-02, -9.2655e-04,  1.7889e-04],
         [-1.2235e-02, -1.7163e-03,  0.0000e+00],
         [-1.4353e-02,  0.0000e+00, -6.7681e-03]],
        [[ 6.6997e-03,  1.3771e-03,  1.0590e-03],
         [-1.4329e-03, -1.8150e-05,  0.0000e+00],
         [-3.4411e-03,  0.0000e+00,  8.2567e-04]],
        [[-2.9910e-03, -3.6508e-07, -1.0731e-03],
         [ 1.7034e-03, -1.0058e-03,  0.0000e+00],
         [ 1.9167e-03,  0.0000e+00, -5.3927e-04]]]

# sub = [[ 2.7192e-02,  8.6691e-03,  7.1747e-02, -2.1477e-03,  1.0412e-02,
#           8.9666e-04,  8.1626e-02,  4.6156e-02,  2.5832e-02,  2.1677e-02,
#           4.8434e-02,  6.4057e-02, -3.6160e-02, -4.0244e-02,  3.9358e-02,
#          -1.0625e-02, -1.9302e-02, -5.9115e-02,  3.1783e-02, -1.0704e-03,
#          -3.2098e-02, -7.9015e-02,  3.5505e-03,  3.7285e-02, -1.5138e-03,
#           3.5806e-02,  1.2981e-02,  3.5188e-03,  1.4286e-02, -4.2283e-02,
#          -3.8573e-02,  5.8107e-03, -4.8763e-02, -1.5531e-02, -8.8506e-02,
#           2.1619e-02, -7.1052e-03, -2.6231e-02, -1.4380e-02, -1.4198e-03,
#           7.6719e-02,  5.3815e-02, -1.4574e-03,  1.0319e-02,  9.5624e-04,
#           1.0767e-01,  1.9807e-02,  1.1554e-02,  5.2264e-02, -1.1425e-03,
#           2.6137e-02,  2.7396e-02, -6.3240e-02,  9.7163e-03, -1.6575e-02,
#          -1.0673e-02,  8.4776e-03, -4.7604e-02,  7.0412e-03,  2.1260e-02,
#           2.3852e-02,  3.6823e-02,  1.6237e-02, -4.3402e-02, -1.6944e-02,
#          -9.2815e-02,  2.9783e-02, -4.7460e-02,  5.7528e-02,  3.8451e-02,
#          -3.2198e-02, -1.7361e-02, -1.4804e-02,  5.8973e-02,  1.4667e-02,
#           3.2150e-02,  2.7660e-05, -5.0264e-03,  2.3327e-02,  5.2583e-02,
#          -3.8655e-02, -1.5752e-02, -9.0644e-03, -5.5094e-02,  2.3833e-03,
#           3.1382e-02, -2.7219e-02,  7.2969e-02, -1.0553e-02, -3.5796e-02,
#           4.8762e-02, -2.1035e-02, -9.9788e-02, -8.3004e-04,  6.5284e-03,
#          -7.6518e-03, -2.7286e-02, -4.3860e-02, -3.7540e-02, -2.6036e-02,
#           1.1182e-02,  2.7539e-02, -9.7886e-02,  9.1895e-03,  4.3220e-02,
#           5.4792e-02,  1.4091e-02,  5.0477e-02,  1.0935e-01, -2.9831e-03,
#          -3.1511e-02, -1.8557e-02, -2.7534e-04,  5.3239e-02, -1.2618e-02,
#           1.9279e-02,  6.4373e-02, -1.2560e-02, -4.6355e-02,  1.0110e-02,
#          -1.7605e-02,  2.0715e-02, -4.0211e-02,  2.8469e-02,  2.2294e-04,
#          -1.3068e-02, -5.6777e-02,  7.0759e-03]]
# print(np.mean(sub, axis=0))

sub = [[[ 2.1439e-02, -2.1451e-02,  3.2141e-02],
         [-4.9724e-04,  1.9975e-04,  0.0000e+00],
         [ 2.0230e-03,  0.0000e+00,  3.8497e-04]],
        [[-4.1868e-03,  6.8675e-03,  1.4042e-02],
         [ 4.0780e-04,  9.4444e-05,  0.0000e+00],
         [-1.7599e-03,  0.0000e+00,  1.4114e-03]],
        [[-9.7876e-03, -1.6286e-03,  7.8745e-05],
         [-5.5801e-03,  1.7282e-03,  0.0000e+00],
         [-2.2605e-03,  0.0000e+00, -1.5052e-03]],
        [[-1.0873e-03, -5.7373e-04, -1.2661e-03],
         [ 7.9669e-04, -1.3905e-03,  0.0000e+00],
         [ 1.5709e-03,  0.0000e+00,  1.0342e-03]],
        [[-5.7107e-03, -5.8654e-03, -1.7648e-03],
         [-1.4729e-03, -6.9231e-04,  0.0000e+00],
         [-1.1371e-03,  0.0000e+00, -4.1988e-03]],
        [[-4.4444e-03, -8.8070e-04,  4.7556e-04],
         [-9.1744e-04,  1.4362e-03,  0.0000e+00],
         [ 5.2705e-04,  0.0000e+00, -7.7891e-04]],
        [[-3.0470e-03,  4.9171e-04, -1.3232e-03],
         [-3.1057e-04,  2.9801e-04,  0.0000e+00],
         [ 1.1491e-03,  0.0000e+00, -5.5629e-04]],
        [[-7.4934e-02,  2.6886e-04,  2.3175e-03],
         [-1.2727e-03,  1.5656e-04,  0.0000e+00],
         [-2.7118e-03,  0.0000e+00, -1.8988e-03]],
        [[-6.9934e-02, -7.0685e-04, -1.3346e-04],
         [ 2.4780e-03, -4.8634e-04,  0.0000e+00],
         [-2.9013e-03,  0.0000e+00, -6.0188e-03]],
        [[ 3.1841e-02,  1.6093e-02, -6.1842e-03],
         [-4.6880e-03,  5.2276e-04,  0.0000e+00],
         [ 7.6783e-03,  0.0000e+00,  2.4153e-02]],
        [[-1.7668e-02,  4.0533e-05, -4.3679e-02],
         [-1.6250e-02, -1.6773e-02,  0.0000e+00],
         [-1.5690e-02,  0.0000e+00, -4.2556e-02]],
        [[-1.0137e-01,  4.5199e-03,  1.1592e-02],
         [-2.4120e-02, -1.6719e-02,  0.0000e+00],
         [-5.5910e-02,  0.0000e+00, -7.4924e-02]],
        [[-1.1318e-01, -2.7749e-03, -3.2612e-02],
         [ 8.1044e-03,  1.8086e-03,  0.0000e+00],
         [ 8.0357e-03,  0.0000e+00,  2.7674e-02]],
        [[ 3.7850e-04,  5.4460e-04,  2.9250e-03],
         [ 3.9681e-03,  6.4873e-03,  0.0000e+00],
         [-1.3658e-02,  0.0000e+00, -4.4868e-03]],
        [[ 2.9047e-03,  4.4674e-03, -3.4206e-03],
         [-3.0239e-04,  9.8309e-04,  0.0000e+00],
         [-3.9134e-05,  0.0000e+00, -9.4950e-05]]]

# print(np.mean(sub, axis=0))

sub = [[[ 6.1910e-03, -1.0713e-02,  2.1701e-03],
         [ 4.0818e-04,  4.3149e-03,  0.0000e+00],
         [-9.2300e-04,  0.0000e+00,  1.6644e-02]],
        [[ 2.1984e-03,  1.7765e-03, -5.5357e-04],
         [-2.7522e-04, -3.4080e-03,  0.0000e+00],
         [ 7.8721e-03,  0.0000e+00, -5.2309e-04]],
        [[ 9.3879e-03, -7.9501e-03, -7.5208e-03],
         [-2.6998e-03, -2.7791e-04,  0.0000e+00],
         [-4.8660e-03,  0.0000e+00, -5.4287e-03]],
        [[-4.5635e-03, -5.5166e-03,  1.6240e-03],
         [-7.5442e-04,  2.3418e-04,  0.0000e+00],
         [-1.3128e-03,  0.0000e+00,  1.2134e-03]],
        [[ 7.7068e-03,  5.8899e-04, -3.6717e-03],
         [-2.9471e-04,  4.3037e-03,  0.0000e+00],
         [-9.1380e-04,  0.0000e+00, -3.0335e-03]],
        [[-2.2566e-03, -7.7336e-04,  5.0630e-04],
         [-7.2516e-04, -2.9825e-03,  0.0000e+00],
         [-7.0573e-04,  0.0000e+00, -2.4287e-03]],
        [[-3.1492e-03, -5.1080e-04,  3.8931e-04],
         [-2.6787e-03,  4.8270e-04,  0.0000e+00],
         [-4.7269e-04,  0.0000e+00, -2.2677e-03]],
        [[-1.3195e-02,  2.4579e-05, -1.2760e-04],
         [ 1.5389e-03,  3.0012e-04,  0.0000e+00],
         [ 9.9653e-04,  0.0000e+00,  6.5389e-04]],
        [[ 9.9297e-03, -1.0887e-04, -2.6938e-04],
         [-8.7112e-04, -1.2382e-03,  0.0000e+00],
         [-4.0816e-04,  0.0000e+00, -2.1284e-03]],
        [[ 3.0508e-02, -4.9569e-04,  4.2235e-05],
         [-4.8067e-03,  2.1363e-03,  0.0000e+00],
         [-7.9402e-03,  0.0000e+00, -1.5213e-04]],
        [[ 5.4115e-03,  2.4746e-03,  2.2171e-04],
         [ 7.0220e-03,  1.2830e-03,  0.0000e+00],
         [-1.6571e-02,  0.0000e+00,  7.7039e-04]],
        [[-2.5794e-02,  4.3880e-03, -6.6019e-03],
         [-1.2183e-04, -1.5789e-03,  0.0000e+00],
         [-3.0110e-02,  0.0000e+00, -8.5545e-03]],
        [[-4.3557e-02,  1.6730e-02, -1.1248e-02],
         [ 1.5843e-02,  6.2435e-03,  0.0000e+00],
         [-1.4024e-01,  0.0000e+00, -1.4547e-02]],
        [[ 1.0255e-02, -1.4157e-03,  1.3411e-03],
         [-1.5858e-02, -6.1056e-04,  0.0000e+00],
         [ 7.7072e-02,  0.0000e+00,  2.3815e-03]],
        [[-3.1826e-02, -3.2582e-04,  3.0647e-04],
         [-7.0639e-03, -8.0317e-06,  0.0000e+00],
         [ 3.4323e-02,  0.0000e+00,  3.5356e-04]]]

# print(np.mean(sub,axis=0))

sub1 = [[[ 3.1361e-03,  1.7118e-04, -3.8815e-04],
         [ 1.7141e-03, -2.8729e-05,  0.0000e+00],
         [ 2.5274e-03,  0.0000e+00,  1.9158e-03]],
        [[ 1.3309e-03,  2.7120e-05, -1.7869e-04],
         [ 8.5658e-04,  1.3304e-04,  0.0000e+00],
         [ 1.2180e-03,  0.0000e+00,  7.7903e-04]],
        [[ 1.3389e-03,  2.7001e-05, -1.7941e-04],
         [ 8.3888e-04,  1.2815e-04,  0.0000e+00],
         [ 1.1786e-03,  0.0000e+00,  7.8481e-04]],
        [[ 1.3354e-03,  2.2233e-05, -1.7822e-04],
         [ 8.3238e-04,  1.3000e-04,  0.0000e+00],
         [ 1.1268e-03,  0.0000e+00,  7.8481e-04]],
        [[ 1.3804e-03,  2.2888e-05, -1.7929e-04],
         [ 8.0854e-04,  1.3065e-04,  0.0000e+00],
         [ 1.0208e-03,  0.0000e+00,  7.9030e-04]],
        [[ 1.3853e-03,  1.9789e-05, -1.7941e-04],
         [ 7.8404e-04,  1.2374e-04,  0.0000e+00],
         [ 8.9532e-04,  0.0000e+00,  7.9191e-04]],
        [[ 1.3747e-03,  1.8001e-05, -1.8215e-04],
         [ 6.9207e-04,  1.2434e-04,  0.0000e+00],
         [ 6.6543e-04,  0.0000e+00,  7.9387e-04]],
        [[ 1.4126e-03,  1.6868e-05, -1.8144e-04],
         [ 6.9535e-04,  1.2261e-04,  0.0000e+00],
         [ 3.2216e-04,  0.0000e+00,  7.9745e-04]],
        [[ 1.5034e-03,  1.6510e-05, -1.8597e-04],
         [ 5.6463e-04,  1.2821e-04,  0.0000e+00],
         [-3.4022e-04,  0.0000e+00,  7.8911e-04]],
        [[ 1.3158e-03,  1.9133e-05, -1.8692e-04],
         [ 5.6022e-04,  1.2708e-04,  0.0000e+00],
         [-1.6197e-03,  0.0000e+00,  7.9119e-04]],
        [[ 1.2343e-03,  1.6868e-05, -1.8561e-04],
         [ 2.3246e-04,  1.2738e-04,  0.0000e+00],
         [-3.7452e-03,  0.0000e+00,  7.9280e-04]],
        [[-3.6466e-04,  1.5616e-05, -1.8728e-04],
         [-2.4104e-04,  1.2541e-04,  0.0000e+00],
         [-7.4806e-03,  0.0000e+00,  8.0037e-04]],
        [[ 1.1432e-03,  1.4186e-05, -1.9169e-04],
         [-4.8161e-05,  1.2332e-04,  0.0000e+00],
         [-1.5770e-02,  0.0000e+00,  8.0401e-04]],
        [[-1.1718e-04, -4.4703e-05, -9.0241e-05],
         [-2.6828e-03,  2.1106e-04,  0.0000e+00],
         [-4.3899e-03,  0.0000e+00,  3.3867e-04]],
        [[ 1.2454e-02, -6.4015e-05, -9.1553e-05],
         [-1.3319e-03,  2.3144e-04,  0.0000e+00],
         [-2.0681e-02,  0.0000e+00,  1.4871e-04]]]
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
print(np.mean(sub1, axis=0))
# print(np.max(sub1, axis=0))

sub2 = [[[ 5.3751e-03,  5.3155e-04, -7.4983e-04],
         [ 8.7732e-04,  9.8133e-04,  0.0000e+00],
         [ 2.8353e-03,  0.0000e+00,  3.4563e-03]],
        [[ 2.9681e-03,  3.7670e-04, -3.8397e-04],
         [ 5.5420e-04,  5.4210e-04,  0.0000e+00],
         [ 1.3866e-03,  0.0000e+00,  1.5081e-03]],
        [[ 2.9101e-03,  3.7831e-04, -3.7980e-04],
         [ 5.4651e-04,  5.3763e-04,  0.0000e+00],
         [ 1.3107e-03,  0.0000e+00,  1.5191e-03]],
        [[ 3.0102e-03,  3.7193e-04, -3.8087e-04],
         [ 5.2166e-04,  5.3233e-04,  0.0000e+00],
         [ 1.2213e-03,  0.0000e+00,  1.5194e-03]],
        [[ 2.8471e-03,  3.7599e-04, -3.8195e-04],
         [ 5.3471e-04,  5.4121e-04,  0.0000e+00],
         [ 1.0557e-03,  0.0000e+00,  1.5236e-03]],
        [[ 2.9473e-03,  3.7515e-04, -3.8159e-04],
         [ 4.6426e-04,  5.3561e-04,  0.0000e+00],
         [ 7.8303e-04,  0.0000e+00,  1.5208e-03]],
        [[ 2.7943e-03,  3.8201e-04, -3.8230e-04],
         [ 3.7873e-04,  5.4181e-04,  0.0000e+00],
         [ 3.5447e-04,  0.0000e+00,  1.5283e-03]],
        [[ 2.7339e-03,  3.7748e-04, -3.8111e-04],
         [ 2.8282e-04,  5.3495e-04,  0.0000e+00],
         [-3.8838e-04,  0.0000e+00,  1.5395e-03]],
        [[ 2.5821e-03,  3.8737e-04, -3.8314e-04],
         [ 2.6494e-04,  5.4651e-04,  0.0000e+00],
         [-1.5237e-03,  0.0000e+00,  1.5202e-03]],
        [[ 1.9642e-03,  3.8946e-04, -3.7944e-04],
         [-7.8797e-05,  5.4479e-04,  0.0000e+00],
         [-3.5095e-03,  0.0000e+00,  1.5337e-03]],
        [[ 8.8555e-04,  3.8457e-04, -3.7730e-04],
         [ 2.0492e-04,  5.4276e-04,  0.0000e+00],
         [-6.8525e-03,  0.0000e+00,  1.5297e-03]],
        [[ 5.6058e-04,  3.8654e-04, -3.7754e-04],
         [ 4.4668e-04,  5.4514e-04,  0.0000e+00],
         [-1.1588e-02,  0.0000e+00,  1.5360e-03]],
        [[-5.3940e-03,  3.8785e-04, -3.7932e-04],
         [ 2.7463e-03,  5.4145e-04,  0.0000e+00],
         [-1.6379e-02,  0.0000e+00,  1.5354e-03]],
        [[ 1.2648e-02,  3.0428e-04, -2.9552e-04],
         [-3.9876e-04,  2.7066e-04,  0.0000e+00],
         [-2.4948e-02,  0.0000e+00,  1.2417e-03]],
        [[-4.2400e-03,  3.2425e-04, -1.1635e-04],
         [-2.1378e-03,  3.1531e-04,  0.0000e+00],
         [-1.3693e-02,  0.0000e+00, -1.6057e-04]]]
print(np.mean(sub2, axis=0))
# print(np.max(sub2, axis=0))

sub3 = [[[ 2.0470e-02,  9.0629e-04,  7.2360e-04],
         [ 8.7178e-04,  9.0116e-04,  0.0000e+00],
         [ 8.2827e-04,  0.0000e+00,  9.5528e-04]],
        [[ 9.3365e-03,  3.9023e-04,  3.3253e-04],
         [ 3.7569e-04,  3.8856e-04,  0.0000e+00],
         [ 3.6997e-04,  0.0000e+00,  4.3046e-04]],
        [[ 8.7454e-03,  3.6263e-04,  3.1477e-04],
         [ 3.3230e-04,  3.5596e-04,  0.0000e+00],
         [ 3.3575e-04,  0.0000e+00,  4.1437e-04]],
        [[ 8.3619e-03,  3.3343e-04,  3.0136e-04],
         [ 2.7978e-04,  3.2836e-04,  0.0000e+00],
         [ 2.9731e-04,  0.0000e+00,  3.9870e-04]],
        [[ 7.9907e-03,  3.3295e-04,  3.0303e-04],
         [ 2.3055e-04,  3.2163e-04,  0.0000e+00],
         [ 2.4593e-04,  0.0000e+00,  3.9917e-04]],
        [[ 7.7299e-03,  3.3188e-04,  2.9570e-04],
         [ 1.5014e-04,  3.2920e-04,  0.0000e+00],
         [ 1.4627e-04,  0.0000e+00,  3.9107e-04]],
        [[ 7.1763e-03,  3.3134e-04,  2.9761e-04],
         [ 2.8968e-05,  3.2872e-04,  0.0000e+00],
         [-1.7881e-06,  0.0000e+00,  3.9393e-04]],
        [[ 6.2791e-03,  3.2657e-04,  2.9731e-04],
         [-1.7440e-04,  3.2675e-04,  0.0000e+00],
         [-2.8837e-04,  0.0000e+00,  3.9577e-04]],
        [[ 4.5929e-03,  3.3456e-04,  3.0321e-04],
         [-5.0139e-04,  3.2777e-04,  0.0000e+00],
         [-7.8678e-04,  0.0000e+00,  3.9583e-04]],
        [[ 1.3332e-03,  3.3873e-04,  3.0792e-04],
         [-1.2091e-03,  3.3045e-04,  0.0000e+00],
         [-1.6738e-03,  0.0000e+00,  4.0317e-04]],
        [[-3.1899e-03,  3.4302e-04,  3.1161e-04],
         [-2.3093e-03,  3.3724e-04,  0.0000e+00],
         [-3.2047e-03,  0.0000e+00,  4.0722e-04]],
        [[-1.0634e-02,  3.3808e-04,  3.1143e-04],
         [-4.5083e-03,  3.3242e-04,  0.0000e+00],
         [-5.7913e-03,  0.0000e+00,  4.1205e-04]],
        [[-1.7037e-02,  3.3861e-04,  3.1257e-04],
         [-8.0931e-03,  3.3712e-04,  0.0000e+00],
         [-9.9958e-03,  0.0000e+00,  4.1521e-04]],
        [[-8.9008e-03,  2.0629e-04,  1.9753e-04],
         [-5.8713e-03,  2.0248e-04,  0.0000e+00],
         [-9.3085e-03,  0.0000e+00,  2.6280e-04]],
        [[-7.0395e-03,  7.8321e-05,  8.6248e-05],
         [-2.6900e-03,  7.3254e-05,  0.0000e+00],
         [-3.8149e-03,  0.0000e+00,  1.1677e-04]]]

# print(np.mean(sub3, axis=0))
# sub = []
# sub.append(np.mean(sub1, axis=0))
# sub.append(np.mean(sub2, axis=0))
# sub.append(np.mean(sub3, axis=0))
# print(np.mean(sub,axis=0))

exit()


import numpy as np
import torch
import pickle
import torch
# path  = 'U:/thesis_code/poses/pose_set01.pkl'
# with open(path, 'rb') as fid:
#     try:
#         variable = pickle.load(fid)
#     except:
#         variable = pickle.load(fid, encoding='bytes')
#
#
# print(variable['video_0001'])
# exit()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np
import sys
class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2

        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                            hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A

        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf

    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d

    return hop_dis  # creates a matrix with diagonal = 0 and other edges as 1 and remaining  as inf


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):

        assert A.size(0) == self.kernel_size
        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)

        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


pose_model = Model(3, 400 , edge_importance_weighting=True,
                   graph_args ={'layout' : 'openpose', 'strategy': 'spatial'} )

pretrained_dict = torch.load('U:/thesis_code/pretrained_models/st_gcn.kinetics.pt')
# for keys in pretrained_dict.keys():
#     print(keys)
pose_model.load_state_dict(pretrained_dict)
pose_model= nn.Sequential(*list(pose_model.children())[:-2])
print(pose_model.eval())
# for name, param in pose_model.named_parameters():
#     print(name)
exit()
























# from keras.preprocessing.image import img_to_array, load_img
# # torch.cuda.empty_cache()
# # exit()
# from torch.autograd import Variable
# import tarfile
# #
# # print(np.round(0.5))
#
# # from graph_model import network
# # model = network.deeplabv3plus_mobilenet(num_classes=19, output_stride=16).cuda()
# # model.load_state_dict(torch.load(fname)["model_state"])
# # pic = './graph_model/Screenshot.png'
# # img_data = load_img(pic)
# # # print(img_data.shape)
# # image_array = img_to_array(img_data).reshape(3, 768, 1366)
# # # print(image_array.shape)
# # image_array = Variable(torch.from_numpy(image_array).unsqueeze(0)).float().cuda()
# # print(image_array.shape)
# # image_features = model(image_array)
# # image_features = image_features.data.to('cpu').numpy()
# # print(model.eval())
# # print(image_features.shape)
# # exit()

# # from graph_model.pie_data import PIE
# # #
# # data_opts = {'fstride': 1,
# #              'sample_type': 'all',
# #              'height_rng': [0, float('inf')],
# #              'squarify_ratio': 0,
# #              'data_split_type': 'default',  # kfold, random, default
# #              'seq_type': 'intention',  # crossing , intention
# #              'min_track_size': 0,  # discard tracks that are shorter
# #              'max_size_observe': 15,  # number of observation frames
# #              'max_size_predict': 5,  # number of prediction frames
# #              'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
# #              'balance': True,  # balance the training and testing samples
# #              'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
# #              'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
# #              'encoder_input_type': [],
# #              'decoder_input_type': ['bbox'],
# #              'output_type': ['intention_binary']
# #              }
#
# # imdb = PIE(data_path= './PIE_dataset')
# # beh_seq_train = imdb.generate_data_trajectory_sequence('test', **data_opts)
# # beh_seq_train = imdb.balance_samples_count(beh_seq_train, label_type='intention_binary')
#
def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)

    # Dn[np.isinf(Dn)] = 0.
    # print(np.dot(Dn, A))

    DAD = np.dot(np.dot(Dn, A), Dn)

    return DAD
# # def normalize_digraph(A):
# #     Dl = np.sum(A, 0)
# #     num_node = A.shape[0]
# #     Dn = np.zeros((num_node, num_node))
# #     for i in range(num_node):
# #         if Dl[i] > 0:
# #             Dn[i, i] = Dl[i]**(-1)
# #     AD = np.dot(A, Dn)
# #     return AD
np.random.seed(2)
x = np.random.randint(1,5,size=(1,3))
# # A = np.random.randint(0,2,size=(1, 5, 3, 3))
# # importance = np.random.randint(2,3,size=(5,1,1))
A = np.zeros((3,3))
A[0, 0] = 1
A[1, 0] = 0.1
A[2, 0] = 0
A[0, 1] = 0.1
A[0, 2] = 0
A[1,1] = 1
A[2,2] = 1

x = torch.from_numpy(x.astype(np.float64))

# # print(importance)
print('x\n', x)
# x1 = torch.sum(x,dim=2).reshape(5,-1)
# # print('x\n', x[:, 0:1])
# # # x2 = x.permute(0,2,1,3).contiguous()
# # # print('x\n',x2)
# #
print('A\n',A)
print('A1\n', normalize_undigraph(A))

# # print('A2\n', normalize_digraph(A))
A1 = torch.from_numpy(normalize_undigraph(A))

x = torch.einsum('nw,vw->nv', (x, A1))
print('X\n', x)
# # print(torch.cuda.device_count())
# # print(torch.cuda.get_device_name(0))
# #
# # import os
# # # fname = './graph_model/checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'
# # path = 'U:/thesis_code/data/data/graph/train/features_context_pad_resize/vgg16_bn'
# # img_folder = os.listdir(path)
# # # print(img_folder)
# #
# # for img_f in img_folder:
# #     new_path = os.path.join(path, img_f)
# #     files = os.listdir(new_path)
# #     for file in files:
# #         for pick in os.listdir(os.path.join(new_path, file)):
# #             if (pick.split('.pkl')[0].split('_')[-1]) == '0':
# #                 del_path = os.path.join(new_path, file, pick)
# #                 os.remove(del_path)
#                 # print(del_path)
#
#
#
# # import os
# # # fname = './graph_model/checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'
# # path = 'D:/thesis_analysis'
# # img_folder = os.listdir(path)
# # print(img_folder)
# # for img_f in img_folder:
#     # if img_f == 'error_3' or img_f == 'error_2':
#
# # new_path = os.path.join(path, 'error_3')
# # files = os.listdir(new_path)
# # new_path = os.path.join(path, 'error')
# # files1 = os.listdir(new_path)
# # count = 0
# # for f in files:
# #     if f not in files1:
# #         print(f)
# #         count += 1
# #
# # print(count)
#             # for pick in os.listdir(os.path.join(new_path, file)):
#             #     if (pick.split('.pkl')[0].split('_')[-1]) == '0':
#             #         del_path = os.path.join(new_path, file, pick)
#             #         os.remove(del_path)
#             #         print(del_path)

# coding:utf-8
# 3个嵌入向量的整合

'''
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.regularizers import l1, l2, l1l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from CXevaluate import evaluate_model
from CXDataset import Dataset
from time import time
import sys
import SoGMF_Ciao, SoMLP_Ciao
import argparse


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run CXNeuMF1_3_1_1.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=0,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='Pretrain/ml-1m_CXGMF1_30.01.h5',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='Pretrain/ml-1m_CXMLP1_30.04.h5',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()


def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)


def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)

    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input_i = Input(shape=(1,), dtype='int32', name='user_input_i')
    user_input_j = Input(shape=(1,), dtype='int32', name='user_input_j')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # Embedding layer
    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
                                  init=init_normal, W_regularizer=l2(reg_mf), input_length=1)
    # MF_Embedding_User_social = Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user_social',
    #                                      init=init_normal, W_regularizer=l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
                                  init=init_normal, W_regularizer=l2(reg_mf), input_length=1)

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0] / 2, name="mlp_embedding_user",
                                   init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
    # MLP_Embedding_User_social = Embedding(input_dim=num_users, output_dim=layers[0] / 2,
    #                                       name="mlp_embedding_user_social",init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0] / 2, name='mlp_embedding_item',
                                   init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)

    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input_i))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_user_social_latent = Flatten()(MF_Embedding_User(user_input_j))
    mf_vector = merge([mf_user_latent, mf_item_latent], mode='mul')  # element-wise multiply
    mf_vector_social = merge([mf_user_latent, mf_user_social_latent], mode='mul')

    # MLP part
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input_i))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_user_social_latent = Flatten()(MLP_Embedding_User(user_input_j))
    mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode='concat')
    mlp_vector_social = merge([mlp_user_latent, mlp_user_social_latent], mode='concat')

    for idx in range(1, num_layer):
        layer = Dense(layers[idx], W_regularizer=l2(reg_layers[idx]), activation='relu', name="layer%d" % idx)
        mlp_vector = layer(mlp_vector)

    for idx1 in range(1, num_layer):
        layer = Dense(layers[idx1], W_regularizer=l2(reg_layers[idx1]), activation='relu', name="layer_social%d" % idx1)
        mlp_vector_social = layer(mlp_vector_social)

    # Concatenate MF and MLP parts
    # mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    # mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    predict_vector = merge([mf_vector, mlp_vector], mode='concat')
    predict_vector_social = merge([mf_vector_social, mlp_vector_social], mode='concat')

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name="prediction")(predict_vector)
    prediction_social = Dense(1, activation='sigmoid', init='lecun_uniform', name="prediction_social")(
        predict_vector_social)
    model = Model(input=[user_input_i, user_input_j, item_input],
                  output=[prediction, prediction_social])

    return model


def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('x_user_embedding').get_weights()
    # gmf_user_embeddings_social = gmf_model.get_layer('x_user_embedding_social').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('x_item_embedding').get_weights()

    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    # model.get_layer('mf_embedding_user_social').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)

    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    # mlp_user_social_embeddings = mlp_model.get_layer('user_social_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()

    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    # model.get_layer('mlp_embedding_user_social').set_weights(mlp_user_social_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)

    # MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' % i).get_weights()
        model.get_layer('layer%d' % i).set_weights(mlp_layer_weights)

    # MLP layers
    for j in range(1, num_layers):
        mlp_layer_social_weights = mlp_model.get_layer('social_layer%d' % j).get_weights()
        model.get_layer('layer_social%d' % j).set_weights(mlp_layer_social_weights)

    # Prediction weights
    gmf_prediction = gmf_model.get_layer('x_prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5 * new_weights, 0.5 * new_b])

    # Prediction weights
    gmf_prediction_social = gmf_model.get_layer('x_prediction_social').get_weights()
    mlp_prediction_social = mlp_model.get_layer('prediction_social').get_weights()
    new_weights_social = np.concatenate((gmf_prediction_social[0], mlp_prediction_social[0]), axis=0)
    new_b_social = gmf_prediction_social[1] + mlp_prediction_social[1]
    model.get_layer('prediction_social').set_weights([0.5 * new_weights_social, 0.5 * new_b_social])

    return model


def get_train_instances(train, num_negatives, social_train, social_dict):
    user_input_i, user_input_j, item_input, labels, labels_social = [], [], [], [], []
    num_users_social = 9801
    leng = len(train)
    for m in range(0, leng, 5):
        u = train[m][0]
        if u in social_dict.keys():
            b = social_dict[u]
            if len(b) == 0:
                continue
            else:
                # i= 0
                for i in range(5):
                    user_input_i.append(train[m + i][0])
                    item_input.append(train[m + i][1])
                    labels.append(train[m + i][2])

                for i in range(5):
                    user_input_i.append(train[m + i][0])
                    item_input.append(train[m + i][1])
                    labels.append(train[m + i][2])

                for n in range(2):
                    user_input_j.append(b[0])
                    labels_social.append(1)
                    b.append(b[0])
                    del b[0]
                    social_dict[u] = b
                    # negative instances
                    for t1 in range(num_negatives):
                        j1 = np.random.randint(num_users_social)
                        while (u, j1) in social_train:
                            j1 = np.random.randint(num_users_social)
                        user_input_j.append(j1)
                        labels_social.append(0)

        else:
            continue

    return user_input_i, user_input_j, item_input, labels, labels_social


if __name__ == '__main__':
    c = [1, 0.2]
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain

    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("CXNeuMF1_3_1_1 arguments: %s " % (args))
    # model_out_file = 'Pretrain/%s_BXNeuMF1_3_1_%d_%s_%d.h5' % (args.dataset, mf_dim, args.layers, time())

    # Loading data
    t1 = time()
    dataset1 = Dataset(args.path + args.dataset)
    social_dict, social_train, train, testRatings, testNegatives, valRatings, valNegatives = dataset1.socialDict, dataset1.socialMatrix, dataset1.trainMatrix, dataset1.testRatings, dataset1.testNegatives, dataset1.valRatings, dataset1.valNegatives
    num_users, num_items = 9801,7569
    print("Load data done [%.1f s]. #user=%d, #item=%d,#test=%d"
          % (time() - t1, num_users, num_items, len(testRatings)))
    # Build model
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy', loss_weights=c)
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy', loss_weights=c)
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', loss_weights=c)
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy', loss_weights=c)

    # Load pretrain model
    if mf_pretrain != '' and mlp_pretrain != '':
        gmf_model = SoGMF_Ciao.get_model(num_users, num_items, mf_dim)
        gmf_model.load_weights(mf_pretrain)
        mlp_model = SoMLP_Ciao.get_model(num_users, num_items, layers, reg_layers)
        mlp_model.load_weights(mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
        print("Load pretrained CXGMF1_3 (%s) and CXMLP1_3 (%s) models done. " % (mf_pretrain, mlp_pretrain))
    # Init performance
    (hits, ndcgs) = evaluate_model(model, valRatings, valNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: val HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    # if args.out > 0:
    #     model.save_weights(model_out_file, overwrite=True)
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: test HR = %.4f, NDCG = %.4f' % (hr, ndcg))

    # Training model
    for epoch in range(num_epochs):
        t1 = time()
        # Generate training instances
        user_input_i, user_input_j, item_input, labels, labels_social = get_train_instances(train, num_negatives,
                                                                                            social_train, social_dict)

        # Training
        hist = model.fit([np.array(user_input_i), np.array(user_input_j), np.array(item_input)],  # input
                         [np.array(labels), np.array(labels_social)],  # labels
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, valRatings, valNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            (hits2, ndcgs2) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr2, ndcg2, loss2 = np.array(hits2).mean(), np.array(ndcgs2).mean(), hist.history['loss'][0]
            print(
                'Iteration %d [%.1f s]: val HR = %.4f, NDCG = %.4f, loss = %.4f , test HR = %.4f, NDCG = %.4f, loss = %.4f[%.1f s]'
                % (epoch, t2 - t1, hr, ndcg, loss, hr2, ndcg2, loss2, time() - t2))

            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                # if args.out > 0:
                #     model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    for i in range(1, 20, 2):
        (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, i, evaluation_threads)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print('top ' + str(i) + ': HR=' + str(hr) + ', NDCG=' + str(ndcg))
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, 10, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('top ' + str(10) + ': HR=' + str(hr) + ', NDCG=' + str(ndcg))

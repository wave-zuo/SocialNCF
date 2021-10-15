#coding:utf-8

#按老师最新提的思路改进,社交关系为双向，用户物品与社交关系比为1:2
#作为初始化参数


'''
Created on Aug 9, 2016

Keras Implementation of Generalized Matrix Factorization (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
import keras
import random
from keras import backend as K
from keras import initializations
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from CXDataset import Dataset
from CXevaluate import evaluate_model
from time import time
import os
import tensorflow as tf
import sys
import math
import argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def parse_args():
    parser = argparse.ArgumentParser(description="Run CXGMF1_3.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    # parser.add_argument('--mf_pretrain', nargs='?', default='Pretrain/ml-1m_CXGMF1_30.01.h5',
    #                     help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    return parser.parse_args()


def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)


def get_model(num_users, num_items, latent_dim, regs=[0, 0]):
    # Input variables
    user_input_i = Input(shape=(1,), dtype='int32', name='user_input_i')
    user_input_j = Input(shape=(1,), dtype='int32', name='user_input_j')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=latent_dim, name='x_user_embedding',
                                  init=init_normal, W_regularizer=l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=latent_dim, name='x_item_embedding',
                                  init=init_normal, W_regularizer=l2(regs[1]), input_length=1)
    # MF_Embedding_User_social = Embedding(input_dim=num_users, output_dim=latent_dim, name='x_user_embedding_social',
    #                                      init=init_normal, W_regularizer=l2(regs[0]), input_length=1)
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input_i))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    user_latent_social = Flatten()(MF_Embedding_User(user_input_j))

    # Element-wise product of user and item embeddings
    predict_vector = merge([user_latent, item_latent], mode='mul')
    predict_vector_social = merge([user_latent, user_latent_social], mode='mul')

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='x_prediction')(predict_vector)
    prediction_social = Dense(1, activation='sigmoid', init='lecun_uniform', name='x_prediction_social')(
        predict_vector_social)
    model = Model(input=[user_input_i, user_input_j, item_input],
                  output=[prediction, prediction_social])

    return model


def load_pretrain_model(model, gmf_model):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('x_user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('x_item_embedding').get_weights()
    model.get_layer('x_user_embedding').set_weights(gmf_user_embeddings)
    model.get_layer('x_item_embedding').set_weights(gmf_item_embeddings)

    # Prediction weights
    gmf_prediction = gmf_model.get_layer('x_prediction').get_weights()
    new_weights = gmf_prediction[0]
    new_b = gmf_prediction[1]
    model.get_layer('x_prediction').set_weights([0.5 * new_weights, 0.5 * new_b])

    # Social Prediction weights
    gmf_prediction_social = gmf_model.get_layer('x_prediction_social').get_weights()
    new_weights_social = gmf_prediction_social[0]
    new_b_social = gmf_prediction_social[1]
    model.get_layer('x_prediction_social').set_weights([0.5 * new_weights_social, 0.5 * new_b_social])
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
    aa=[[1,0.01]]
    # aa = [[1, 0.02],[1,0.04],[1,0.06],[1,0.08],[1,0.10],[1,0.12],[1,0.14],[1,0.16],[1,0.18],[1,0.20]]
    seed_value = 1623
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.set_random_seed(seed_value)

    for c in aa:
        args = parse_args()
        num_factors = args.num_factors
        regs = eval(args.regs)
        num_negatives = args.num_neg
        learner = args.learner
        learning_rate = args.lr
        epochs = args.epochs
        batch_size = args.batch_size
        verbose = args.verbose
        mf_pretrain = args.mf_pretrain

        topK = 10
        evaluation_threads = 1  # mp.cpu_count()
        print("CXGMF1_3 arguments: %s" % (args))
        model_out_file = 'Pretrain/%s_CXGMF1_3%.2f.h5' % (args.dataset, c[1])

        # Loading data
        t1 = time()
        dataset = Dataset(args.path + args.dataset)
        social_dict, social_train, train, testRatings, testNegatives, valRatings, valNegatives = dataset.socialDict, dataset.socialMatrix, dataset.trainMatrix, dataset.testRatings, dataset.testNegatives, dataset.valRatings, dataset.valNegatives

        num_users, num_items = 9801,7569

        print("Load data done [%.1f s]. #user=%d, #item=%d,#test=%d"
              % (time() - t1, num_users, num_items,len(testRatings)))
        # Build model
        model = get_model(num_users, num_items, num_factors, regs)
        if learner.lower() == "adagrad":
            model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy', loss_weights=c)
        elif learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy', loss_weights=c)
        elif learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', loss_weights=c)
        else:
            model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy', loss_weights=c)

        # Load pretrain model
        if mf_pretrain != '':
            gmf_model = get_model(num_users, num_items, num_factors)
            gmf_model.load_weights(mf_pretrain)
            model = load_pretrain_model(model, gmf_model)
            print("Load pretrained GMF1_3 (%s) models done. " % (mf_pretrain))

        # Init performance
        t1 = time()
        (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()

        print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time() - t1))
        # Train model
        best_hr, best_ndcg, best_iter = hr, ndcg, -1
        for epoch in range(epochs):
            t1 = time()
            # Generate training instances

            user_input_i, user_input_j, item_input, labels, labels_social = get_train_instances(train, num_negatives,social_train,social_dict)

            hist = model.fit([np.array(user_input_i), np.array(user_input_j), np.array(item_input)],  # input
                             [np.array(labels), np.array(labels_social)],  # labels
                             nb_epoch=1, verbose=0, shuffle=True)
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
                    if args.out > 0:
                        model.save_weights(model_out_file, overwrite=True)

        print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
        for i in range(1, 20, 2):
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, i, evaluation_threads)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print('top ' + str(i) + ': HR=' + str(hr) + ', NDCG=' + str(ndcg))



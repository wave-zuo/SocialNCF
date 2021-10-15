# coding:utf-8
#与原始的XMLP相比，少一个嵌入向量,1:0.19
# 社交关系为双向，用户物品与社交关系比为1:2
#作为初始化参数

import numpy as np
import keras
from keras import backend as K
from keras import initializations
from keras.regularizers import l2, activity_l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from CXevaluate import evaluate_model
from CXDataset import Dataset
from time import time
import random
import tensorflow as tf
import argparse


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run CXMLP1_31.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[1e-4,1e-4,1e-4,1e-4]',
                        help="Regularization for each layer")
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
    # parser.add_argument('--mlp_pretrain', nargs='?', default='Pretrain/ml-1m_CXMLP1_310.04.h5',
    #                     help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()


def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)


def get_model(num_users, num_items, layers=[20, 10], reg_layers=[0, 0]):
    assert len(layers) == len(reg_layers)

    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input_i = Input(shape=(1,), dtype='int32', name='user_input_i')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    user_input_j = Input(shape=(1,), dtype='int32', name='user_input_j')

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0] / 2, name='user_embedding',
                                   init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0] / 2, name='item_embedding',
                                   init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input_i))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    user_latent_social = Flatten()(MLP_Embedding_User(user_input_j))



    # The 0-th layer is the concatenation of embedding layers
    vector = merge([user_latent, item_latent], mode='concat')
    vector_social = merge([user_latent,user_latent_social], mode='concat')

    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], W_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' % idx)
        vector = layer(vector)

    for idx1 in range(1, num_layer):
        layer1 = Dense(layers[idx1], W_regularizer=l2(reg_layers[idx1]), activation='relu', name='social_layer%d' % idx1)
        vector_social= layer1(vector_social)

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(vector)
    prediction_social = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction_social')(vector_social)

    model = Model(input=[user_input_i,user_input_j,item_input],
                  output=[prediction,prediction_social])

    return model


def load_pretrain_model(model,mlp_model, num_layers):
    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('user_embedding').set_weights(mlp_user_embeddings)
    model.get_layer('item_embedding').set_weights(mlp_item_embeddings)
    # MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' % i).get_weights()
        model.get_layer('layer%d' % i).set_weights(mlp_layer_weights)

    for j in range(1, num_layers):
        mlp_layer_social_weights = mlp_model.get_layer('social_layer%d' %j).get_weights()
        model.get_layer('social_layer%d' %j).set_weights(mlp_layer_social_weights)

    # Prediction weights
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = mlp_prediction[0]
    new_b = mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5 * new_weights, 0.5 * new_b])


    mlp_prediction_social = mlp_model.get_layer('prediction_social').get_weights()
    new_weights_social =mlp_prediction_social[0]
    new_b_social =mlp_prediction_social[1]
    model.get_layer('prediction_social').set_weights([0.5 * new_weights_social, 0.5 * new_b_social])



    return model

# 训练时正负样本比为1:4
def get_train_instances(train, num_negatives,social_train,social_dict):
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
    aa = [[1, 0.04]]
    # aa = [[1, 0.02],[1,0.04],[1,0.06],[1,0.08],[1,0.10],[1,0.12],[1,0.14],[1,0.16],[1,0.18],[1,0.20]]

    for c in aa:

        seed_value = 1623
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.set_random_seed(seed_value)

        args = parse_args()
        path = args.path
        layers = eval(args.layers)
        reg_layers = eval(args.reg_layers)
        num_negatives = args.num_neg
        learner = args.learner
        learning_rate = args.lr
        batch_size = args.batch_size
        epochs = args.epochs
        verbose = args.verbose
        mlp_pretrain = args.mlp_pretrain
        topK = 10
        evaluation_threads = 1  # mp.cpu_count()
        print("CXMLP1_31 arguments: %s " % (args))
        model_out_file = 'Pretrain/%s_CXMLP1_3%.2f.h5' % (args.dataset, c[1])

        # Loading data
        t1 = time()
        dataset = Dataset(args.path + args.dataset)
        social_dict, social_train, train, testRatings, testNegatives, valRatings, valNegatives = dataset.socialDict, dataset.socialMatrix, dataset.trainMatrix, dataset.testRatings, dataset.testNegatives, dataset.valRatings, dataset.valNegatives
        num_users, num_items = 9801,7569
        print("Load data done [%.1f s]. #user=%d, #item=%d, #test=%d"
              % (time() - t1, num_users, num_items,len(testRatings)))

        # Build model
        model = get_model(num_users, num_items, layers, reg_layers)
        if learner.lower() == "adagrad":
            model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy', loss_weights=c)
        elif learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy', loss_weights=c)
        elif learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', loss_weights=c)
        else:
            model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy', loss_weights=c)

        # Load pretrain model
        if mlp_pretrain != '':
            mlp_model = get_model(num_users, num_items, layers, reg_layers)
            mlp_model.load_weights(mlp_pretrain)
            model = load_pretrain_model(model, mlp_model, len(layers))
            print("Load pretrained MLP1 (%s) models done. " % (mlp_pretrain))

        # Check Init performance
        t1 = time()
        (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))

        # Train model
        best_hr, best_ndcg, best_iter = hr, ndcg, -1
        for epoch in range(epochs):
            t1 = time()
            # Generate training instances
            user_input_i, user_input_j, item_input, labels, labels_social = get_train_instances(train, num_negatives,
                                                                                                social_train,
                                                                                                social_dict)

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
                    if args.out > 0:
                        model.save_weights(model_out_file, overwrite=True)

        print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
        for i in range(1, 20, 2):
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, i, evaluation_threads)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print('top ' + str(i) + ': HR=' + str(hr) + ', NDCG=' + str(ndcg))
        # (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, 10, evaluation_threads)
        # hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        # print('top ' + str(10) + ': HR=' + str(hr) + ', NDCG=' + str(ndcg))

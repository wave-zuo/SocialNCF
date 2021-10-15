# coding:utf-8
'''
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np


class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.socialDict, self.socialMatrix = self.load_rating_file_as_socialmatrix(path + ".Bedges2.txt")
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".Btrain1.txt")
        self.testRatings, self.testNegatives = self.load_rating_file_as_list(path + ".Btest1.txt")
        self.valRatings, self.valNegatives = self.load_rating_file_as_list(path + ".Bval1.txt")

        self.num_users, self.num_items = 11163, 5019

    def load_rating_file_as_list(self, filename):
        ratingList = []
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                if int(arr[2]) == 1:
                    user, item = int(arr[0]), int(arr[1])
                    ratingList.append([user, item])
                    negatives = []
                else:
                    negatives.append(int(arr[1]))
                if len(negatives) == 99:
                    negativeList.append(negatives)
                line = f.readline()
        return ratingList, negativeList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        mat = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                u, i, j = int(arr[0]), int(arr[1]), int(arr[2])
                ma = []
                ma.append(u)
                ma.append(i)
                ma.append(j)
                mat.append(ma)
                line = f.readline()

        return mat

    def load_rating_file_as_socialmatrix(self, filename):
        # Construct matrix
        mat = sp.dok_matrix((11163, 11163), dtype=np.float32)
        d = dict()
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, item = int(arr[0]), int(arr[1])
                if user not in d.keys():
                    d[user] = list()
                d[user].append(item)
                mat[user, item] = 1.0
                line = f.readline()
        return d, mat

    # def NoFriend(self):
    #     sa=[]
    #     for (u, i) in self.trainMatrix.keys():
    #         if u in self.socialDict.keys():
    #             b = self.socialDict[u]
    #             if len(b) == 0:
    #                 sa.append(u)
    #         else:
    #             sa.append(u)
    #     return sa






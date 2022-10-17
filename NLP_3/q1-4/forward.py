import re
import math
import colorsys
import numpy as np
from collections import Counter
import argparse
import os
import time
from data import *


def get_transition_prob(train_sents,labels,max_seq=20):
    num_labels = len(labels)
    prob = np.zeros([num_labels,num_labels,max_seq])
    bin_counter = Counter()
    uni_counter = Counter()
    obeservations = []
    for line in train_sents:
        obs = [labels[x[1]] for x in [[None,"*"]]+line]
        bin_counter.update([x for x in zip(obs,obs[1:],range(len(obs)))])
        uni_counter.update([x for x in zip(obs,range(len(obs)))])
    
    for i in range(num_labels):
        for j in range(num_labels):
            for k in range(max_seq):
                if uni_counter[i,k]>0:
                    prob[j][i][k]=bin_counter[i,j,k]/uni_counter[i,k]
    return prob



def sumprod(seq_length, labels, prob):
    num_labels = len(labels)
    alpha = np.zeros([seq_length,num_labels])
    beta = np.zeros([seq_length,num_labels])
    for y in range(num_labels):
        alpha[0][y]=prob[y][labels["*"]][0]
    ### YOUR CODE HERE
    for j in range(1, seq_length):
        for y in range(num_labels):
            alpha_j_y = 0
            for y_tag in range(num_labels):
                 alpha_j_y += alpha[j-1][y_tag] * prob[y][y_tag][j]
            alpha[j][y] = alpha_j_y
    beta[-1] = np.ones([num_labels])
    for j in range(seq_length - 2, 0, -1):
        for y in range(num_labels):
            beta_j_y = 0
            for y_tag in range(num_labels):
                beta_j_y += beta[j+1][y_tag] * prob[y_tag][y][j]
            beta[j][y] = beta_j_y
    Z = np.sum(alpha[-1])
    ### END YOUR HERE
    mu = (alpha*beta)/Z
    return mu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int,default=14)
    parser.add_argument("--label", type=str,default="LOC")
    parser.add_argument("--position", type=int,default=8)
    args = parser.parse_args()
    
    train_sents = read_conll_ner_file("data/train.conll")
    labels_list = ["*"]+NER_LBLS
    labels_dict = {k:v for k,v in  zip(labels_list,range(len(labels_list)))}
    prob = get_transition_prob(train_sents,labels_dict,args.seq_len)
    
    mu = sumprod(args.seq_len,labels_dict,prob)
    print(f"P(x_{args.position} = '{args.label}') = {mu[args.position][labels_dict[args.label]]}")


        

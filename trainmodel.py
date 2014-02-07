#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Original parts taken from gensim:
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
# Added parts by Franziska Horn <cod3licious@gmail.com>
# Not yet licensed under anything. Feel free to use for whatever except military,
# NSA, and related stuff

from copy import deepcopy
from numpy import random
import numpy as np


def train_cbowHSM(model, sentence, alpha):
    """
    Update cbow hierarchical softmax model by training on a single sentence.

    The sentence is a list of Vocab objects (or None, where the corresponding
    word is not in the vocabulary. Called internally from `Word2Vec.train()`.
    """
    for pos, word in enumerate(sentence):
        if not word or (word.prob and word.prob < random.rand()):
            continue  # OOV word in the input sentence or subsampling => skip
        reduced_window = random.randint(model.window) # how much is SUBSTRACTED from the original window
        # get sum of representation from all words in the (reduced) window (if in vocab and not the `word` itself)
        start = max(0, pos - model.window + reduced_window)
        word2_indices = [word2.index for pos2, word2 in enumerate(sentence[start:pos+model.window+1-reduced_window], start) if (word2 and not (pos2 == pos))]
        l1 = np.sum(model.syn0[word2_indices],axis=0) # 1xlayer1_size
        # work on the entire tree at once --> 2d matrix, codelen x layer1_size
        l2 = deepcopy(model.syn1[word.point])
        # propagate hidden -> output
        f = 1. / (1. + np.exp(-np.dot(l1, l2.T)))
        # vector of error gradients multiplied by the learning rate
        g = (1. - word.code - f) * alpha
        # learn hidden -> output
        model.syn1[word.point] += np.outer(g, l1)
        # learn input -> hidden, here for all words in the window separately
        model.syn0[word2_indices] += np.dot(g, l2)
    return len([word for word in sentence if word])

def train_cbowNEG(model, sentence, alpha, k=13):
    """
    Update cbow negative sampling model by training on a single sentence.

    The sentence is a list of Vocab objects (or None, where the corresponding
    word is not in the vocabulary. Called internally from `Word2Vec.train()`.

    k = number of "noise words"
    """
    # precompute k labels
    labels = np.zeros(k+1)
    labels[0] = 1.
    for pos, word in enumerate(sentence):
        if not word or (word.prob and word.prob < random.rand()):
            continue  # OOV word in the input sentence or subsampling => skip
        reduced_window = random.randint(model.window) # how much is SUBSTRACTED from the original window
        # get sum of representation from all words in the (reduced) window (if in vocab and not the `word` itself)
        start = max(0, pos - model.window + reduced_window)
        word2_indices = [word2.index for pos2, word2 in enumerate(sentence[start:pos+model.window+1-reduced_window], start) if (word2 and not (pos2 == pos))]
        l1 = np.sum(model.syn0[word2_indices],axis=0) # 1xlayer1_size
        # use this word (label = 1) + k other random words not from this sentence (label = 0)
        word_indices = [word.index]
        while len(word_indices) < k+1:
            w = model.table[random.randint(model.table.shape[0])]
            if not (w == word.index or w in word2_indices):
                word_indices.append(w)
        # 2d matrix, k+1 x layer1_size
        l2 = deepcopy(model.syn1[word_indices])
        # propagate hidden -> output
        f = 1. / (1. + np.exp(-np.dot(l1, l2.T)))
        # vector of error gradients multiplied by the learning rate
        g = (labels - f) * alpha
        # learn hidden -> output
        model.syn1[word_indices] += np.outer(g, l1)
        # learn input -> hidden, here for all words in the window separately
        model.syn0[word2_indices] += np.dot(g, l2)
    return len([word for word in sentence if word])

def train_skipgramHSM(model, sentence, alpha):
    """
    Update skip-gram hierarchical softmax model by training on a single sentence.

    The sentence is a list of Vocab objects (or None, where the corresponding
    word is not in the vocabulary. Called internally from `Word2Vec.train()`.
    """
    for pos, word in enumerate(sentence):
        if not word or (word.prob and word.prob < random.rand()):
            continue  # OOV word in the input sentence or subsampling => skip
        reduced_window = random.randint(model.window)
        # now go over all words from the (reduced) window, predicting each one in turn
        start = max(0, pos - model.window + reduced_window)
        for pos2, word2 in enumerate(sentence[start : pos + model.window + 1 - reduced_window], start):
            # don't train on OOV words and on the `word` itself
            if word2 and not (pos2 == pos):
                l1 = model.syn0[word2.index]
                # work on the entire tree at once --> 2d matrix, codelen x layer1_size
                l2 = deepcopy(model.syn1[word.point])
                # propagate hidden -> output
                f = 1. / (1. + np.exp(-np.dot(l1, l2.T)))
                # vector of error gradients multiplied by the learning rate
                g = (1. - word.code - f) * alpha 
                # learn hidden -> output
                model.syn1[word.point] += np.outer(g, l1)
                # learn input -> hidden
                model.syn0[word2.index] += np.dot(g, l2)  
    return len([word for word in sentence if word])

def train_skipgramNEG(model, sentence, alpha, k=13):
    """
    Update skip-gram negative sampling model by training on a single sentence.

    The sentence is a list of Vocab objects (or None, where the corresponding
    word is not in the vocabulary. Called internally from `Word2Vec.train()`.

    k = number of "noise words"
    """
    # precompute k labels
    labels = np.zeros(k+1)
    labels[0] = 1.
    for pos, word in enumerate(sentence):
        if not word or (word.prob and word.prob < random.rand()):
            continue  # OOV word in the input sentence or subsampling => skip
        reduced_window = random.randint(model.window)
        # now go over all words from the (reduced) window, predicting each one in turn
        start = max(0, pos - model.window + reduced_window)
        for pos2, word2 in enumerate(sentence[start : pos + model.window + 1 - reduced_window], start):
            # don't train on OOV words and on the `word` itself
            if word2 and not (pos2 == pos):
                l1 = model.syn0[word2.index]
                # use this word (label = 1) + k other random words not from this sentence (label = 0)
                word_indices = [word.index]
                while len(word_indices) < k+1:
                    w = model.table[random.randint(model.table.shape[0])]
                    if not (w == word.index or w == word2.index):
                        word_indices.append(w)
                # 2d matrix, k+1 x layer1_size
                l2 = deepcopy(model.syn1[word_indices])
                # propagate hidden -> output
                f = 1. / (1. + np.exp(-np.dot(l1, l2.T))) 
                # vector of error gradients multiplied by the learning rate
                g = (labels - f) * alpha
                # learn hidden -> output
                model.syn1[word_indices] += np.outer(g, l1)
                # learn input -> hidden 
                model.syn0[word2.index] += np.dot(g, l2)  
    return len([word for word in sentence if word])

def train_bskipgramHSM(model, sentence, alpha):
    """
    Update skip-gram hierarchical softmax model by training on a single sentence.

    The sentence is a list of Vocab objects (or None, where the corresponding
    word is not in the vocabulary. Called internally from `Word2Vec.train()`.
    """
    for pos, word in enumerate(sentence):
        if not word or (word.prob and word.prob < random.rand()):
            continue  # OOV word in the input sentence or subsampling => skip
        reduced_window = random.randint(model.window)
        # now go over all words from the (reduced) window (at once), predicting each one in turn
        start = max(0, pos - model.window + reduced_window)
        word2_indices = [word2.index for pos2, word2 in enumerate(sentence[start:pos+model.window+1-reduced_window], start) if (word2 and not (pos2 == pos))]
        l1 = model.syn0[word2_indices] # len(word2_indices) x layer1_size
        # work on the entire tree at once --> 2d matrix, codelen x layer1_size
        l2 = deepcopy(model.syn1[word.point])
        # propagate hidden -> output (len(word2_indices) x codelen)
        f = 1. / (1. + np.exp(-np.dot(l1, l2.T)))
        # vector of error gradients multiplied by the learning rate
        g = (1. - np.tile(word.code,(len(word2_indices),1)) - f) * alpha 
        # learn hidden -> output (codelen x layer1_size) batch update
        model.syn1[word.point] += np.dot(g.T, l1)
        # learn input -> hidden
        model.syn0[word2_indices] += np.dot(g, l2)
    return len([word for word in sentence if word])

def train_bskipgramNEG(model, sentence, alpha, k=13):
    """
    Update skip-gram negative sampling model by training on a single sentence.

    The sentence is a list of Vocab objects (or None, where the corresponding
    word is not in the vocabulary. Called internally from `Word2Vec.train()`.

    k = number of "noise words"
    """
    # precompute k labels
    labels = np.zeros(k+1)
    labels[0] = 1.
    for pos, word in enumerate(sentence):
        if not word or (word.prob and word.prob < random.rand()):
            continue  # OOV word in the input sentence or subsampling => skip
        reduced_window = random.randint(model.window)
        # now go over all words from the (reduced) window (at once), predicting each one in turn
        start = max(0, pos - model.window + reduced_window)
        word2_indices = [word2.index for pos2, word2 in enumerate(sentence[start:pos+model.window+1-reduced_window], start) if (word2 and not (pos2 == pos))]
        l1 = deepcopy(model.syn0[word2_indices]) # len(word2_indices) x layer1_size
        # use this word (label = 1) + k other random words not from this sentence (label = 0)
        word_indices = [word.index]
        while len(word_indices) < k+1:
            w = model.table[random.randint(model.table.shape[0])]
            if not (w == word.index or w in word2_indices):
                word_indices.append(w)
        # 2d matrix, k+1 x layer1_size
        l2 = model.syn1[word_indices]
        # propagate hidden -> output
        f = 1. / (1. + np.exp(-np.dot(l1, l2.T))) 
        # vector of error gradients multiplied by the learning rate
        g = (np.tile(labels,(len(word2_indices),1)) - f) * alpha
        # learn hidden -> output (batch update)
        model.syn1[word_indices] += np.dot(g.T, l1)
        # learn input -> hidden 
        model.syn0[word2_indices] += np.dot(g, l2)  
    return len([word for word in sentence if word])

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 21:11:48 2017

@author: dhingratul
"""
from __future__ import print_function
import numpy as np
import cv2
from operator import itemgetter 
import pandas as pd


def getdistanceM(images, eq=False):
    """
    Generates distance Matrix based on number of matches from FLANN and SIFT
    Input: {images, eq}: {images = List of images with their entire path, 
    eq = {True, False}: Use CLAHE equalizer with True --default is False}
    Output: {distM} : distance Matrix based on number of matches from FLANN, 
    higher the number, better the match
    """
    distM = np.zeros([len(images), len(images)])
    im_data = []
    if eq is True:
        clahe = cv2.createCLAHE()  # Hist Eq
    for i in range(len(images)):
        img = cv2.imread(images[i], 0)
        if eq is True:
            cl = clahe.apply(img)
            im_data.append(cl)
        else:
            im_data.append(img)
    # Initiate SIFT detector
    sift = cv2.SIFT()
    
    # find the keypoints and descriptors with SIFT
    KP = []
    DES = []
    for i in range(len(im_data)):
        kp1, des1 = sift.detectAndCompute(im_data[i],None)
        KP.append(kp1)
        DES.append(des1)
        
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    for i in range(len(images)):
        for j in range(len(images)):
            matches = flann.knnMatch(DES[i], DES[j], k=2)
        # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
            distM[i,j] = len(good)
    return distM


def isSimilar(distM, thresh):
    """
    Generates a dictionary with most similar images for each image based on 
    pairwise distances and a user defined threshold
    Input: {distM, thresh}: {distM = distance Matrix from getDistanceM, 
    thresh = user provide threshold -- default is 5}
    Output: {dic}: Dictionary with most similar images based on pairwise 
    distances from getDistanceM
    """
    dic = {}
    # Make symmetric
    distM = np.maximum( distM, distM.transpose() )
    for i in range(distM.shape[0]):
        vals = distM[i,:]
        vals[i] = 0
        v_min = np.min(vals[np.nonzero(vals)])
        v_max = np.max(vals)
        v_avg = np.mean(vals)
        if (v_min + v_max/2) > 2 * v_avg:
            idx = np.argwhere(vals > v_avg + thresh)
            dic[i] = idx
        else:
            idx = np.argmax(vals)
            dic[i] = idx
    return dic
    

def tieBreak(sets, l, i, j, distM):
    """
    Tie-breaking operation betwen two cometing sets for inclusion of new image
    Input: {sets, l, i, j, distM}: {sets = collection of all different sets
    (clusters) as a list, l = current images to be assigned clusters,
    i,j = index of clusters competing for the new image, distM = see getDistM}
    Output: {sets}: Return sets by filling in tie-breaks in place
    """
    if len(sets[i].intersection(l)) > len(sets[j].intersection(l)):
        sets[i] = sets[i].union(l)
        sets[j] = sets[j].difference(l)
    elif len(sets[i].intersection(l)) < len(sets[j].intersection(l)):
        sets[j] = sets[j].union(l)
        sets[i] = sets[i].difference(l)
    elif len(sets[i].intersection(l)) == len(sets[j].intersection(l)):
        # Tie break based on average distance
        temp_l = list(l)
        temp_i = list(sets[i].difference(l))
        temp_j = list(sets[j].difference(l))
        for a in temp_l:
            d = 0
            for b in temp_i:
                d += distM[a][b]
            x = d/len(temp_i)
            d = 0
            for c in temp_j:
                d += distM[a][c]
            y = d/len(temp_j)
            l = set()
            l.add(a)
            if x > y:
                sets[i] = sets[i].union(l)
                sets[j] = sets[j].difference(l)
            else:
                sets[j] = sets[j].union(l)
                sets[i] = sets[i].difference(l)

    return sets


def cluster(dic, images, distM):
    """
    Clusters the output 
    Input: {dic, images, distM}: {dic = see isSimilar, 
    images = same as getDistanceM, distM = see getDistanceM}
    Output: {dic_out}: Returns a dictionary with different cluster assignments
    """
    dic_out = {}
    n = 3
    sets = [set() for _ in range(n)]
    for i in dic:
        l = set(dic[i].flatten())
        l.add(i)
        if len(sets[0]) == 0:
            sets[0] = sets[0].union(l)

        elif sets[0] & l and sets[1] & l:
            sets = tieBreak(sets, l, 0, 1, distM)
        
        elif sets[0] & l and sets[2] & l:
            sets = tieBreak(sets, l, 0, 2, distM)
                
        elif sets[1] & l and sets[2] & l:
            sets = tieBreak(sets, l, 1, 2, distM)  
            
        elif sets[0] & l:
            sets[0] = sets[0].union(l)
        
        elif len(sets[1]) == 0:
            sets[1] = sets[1].union(l)
                            
        elif sets[1] & l:
            sets[1] = sets[1].union(l)
        else:
            sets[2] = sets[2].union(l)
            
    L = list(sets)
    n = 0
    for i in L:
        if i:
            n = n + 1
    df = pd.DataFrame(images)
    names = df[0].str.split('/')
    names = list(names.apply(lambda x: x[-1]).as_matrix())
    for i in range(n):
        dic_out[i] = itemgetter(*list(L[i]))(names)
    return dic_out
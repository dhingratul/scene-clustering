from __future__ import print_function
import numpy as np
import glob
import json
import os
import utils


# Folder Directory
mdir = '../data/'
folders = os.listdir(mdir)
np.random.seed(seed=0)
for folder in folders:
    print(folder + '/' + str(len(folders)))
    fdir = folder + '/'
    j_out = '../out/'
    json_out = j_out + folder + '.json'
    images = glob.glob(mdir + fdir + '*.jpg')
    # Algorithm
    distM = utils.getdistanceM(images, eq=True)
    dic = utils.isSimilar(distM, thresh=5)
    dic_out = utils.cluster(dic, images, distM)
    with open(json_out, 'w') as fp:
        json.dump(dic_out.values(), fp)
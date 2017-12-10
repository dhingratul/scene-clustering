# scene-clustering
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Clustering scenes based on images

# Requirements
1. opencv 2.4
2. numpy
3. pandas
4. python 2.7

# Run
1. Clone the repository
2. cd into src/ directory
2. `python run.py` 

# Note:
1. The 'distance' measure betwen the pairwise images is computed based on number of good matches obtained from SIFT features and FLANN matcher(`utils.getDistanceM`), the higher the measure, more closely the image is related
2. An equalizer option is also provided(deafult= False) in `utils.getDistanceM`, which utilizes CLAHE (Contrast Limited Adaptive Histogram Equalization) algorithm for histogram equalization. This has a very overhead in terms of time execution, for example with 3 images, the time is increased from 7.9s to 123.5s for the execution of `utils.getDistanceM`
3. The produced distanceM is fed into `utils.isSimilar` to generate a dictionary with images similar based on the average distance and a user provided threshold(default = 5)
4. Once the dictionary is obtained, a clustering operation is executed in `utils.cluster` which clusters it into different clusters and outputs a dictionary with different cluster assignments. 
5. A tiebreaking operation is also executed in `utils.tieBreak` in the case an image belongs to two different clusters. This is done based on average distance from the image to all the other images in the cluster, the bigger value wins the tie.
6. The output is dumped as a .json file in src/

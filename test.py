import os
import math
import numpy as np
import tensorflow as tf
from datasets import dataset as dataset
from models.nn import AlexNet as ConvNet
from learning.evaluators import AccuracyEvaluator as Evaluator

from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors

category = ['others','bird','insect','worm','butfly','wheel','struct','piano','plane','bottle','glass','woman','flower']

 # print predict result
def print_predict_labels(labels) :
    i = labels.argmax()
    print( category[i] )

# print the image's real label
def print_real_labels(labels) :
    for i, lable in enumerate(labels) :
        if int(lable) == 1 :
            break
    print( category[i] )

""" 1. Load dataset """
root_dir = os.path.join('..', 'image')    # FIXME ..\image
test_dir = os.path.join(root_dir, 'test')
image_dir = os.path.join(root_dir, 'image')

# Load test set
x_test, y_test, z_test = dataset.read_dataset_csv(test_dir, one_hot=True)
test_set = dataset.DataSet(x_test, y_test, z_test)


# Sanity check
print('Test set stats:')
print(test_set.images.shape)
print(test_set.images.min(), test_set.images.max())
print((test_set.labels[:, 1] == 0).sum(), (test_set.labels[:, 1] == 1).sum())


""" 2. Set test hyperparameters """
hp_d = dict()
image_mean = np.load('/tmp/mean.npy')    # load mean image
hp_d['image_mean'] = image_mean

# FIXME: Test hyperparameters
hp_d['batch_size'] = 256
hp_d['augment_pred'] = True


""" 3. Build graph, load weights, initialize a session and start test """
# Initialize
graph = tf.get_default_graph()
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

model = ConvNet([227, 227, 3], 13, **hp_d) # num_classes = 13
evaluator = Evaluator()
saver = tf.train.Saver()

sess = tf.Session(graph=graph, config=config)
saver.restore(sess, '/tmp/model.ckpt')    # restore learned weights C:\ybchoi\tensorflow\dog_cat_master\train_result

test_y_pred = model.predict(sess, test_set, **hp_d)


# write predict value into pred_value.csv
# np.savetxt("pred_value_names.csv", test_set.names, delimiter=",", fmt='%s')
# np.savetxt("pred_value.csv", test_y_pred, delimiter=",")

# read predict value from pred_value.csv
pred_list = np.genfromtxt('pred_value.csv', delimiter=',')
pred_name_list = np.genfromtxt('pred_value_names.csv', delimiter=',', dtype=str)


# knn search
knn = NearestNeighbors(n_neighbors=10)
knn.fit(pred_list)
knn_predict = knn.kneighbors(test_y_pred.reshape(test_y_pred.shape[0],-1), return_distance=False)
print(knn_predict)


# search for multiple images
results = np.empty(test_set.images.shape[0], dtype=object)
for idx in range(0, test_set.images.shape[0]) :
    target_image = test_set.images[idx]
    target_image_name = test_set.names[idx]
    target_pred_value = test_y_pred[idx]

    result = np.empty(10, dtype=object)
    for i, closest in enumerate(knn_predict[idx]) :
        result[i] = pred_name_list[closest]

    results[idx] = result
    print(target_image_name)
    print(result)










# # search for one image
# # set target image
# index = np.where(test_set.names == 'n04515003_31174.JPEG')[0][0]
# target_image = test_set.images[index]
# target_image_name = test_set.names[index]
# target_pred_value = test_y_pred[index]

# result = np.empty(10, dtype=object)
# for i, closest in enumerate(knn_predict[0]) :
#     result[i] = pred_name_list[closest]
#     file_path = os.path.join(image_dir, result[i])
#     img = mpimg.imread(file_path)
#     plt.subplot(2 ,6, i+2)
#     plt.imshow(img)

# print('------------------------------------------')
# test_score = evaluator.score(test_set.labels, test_y_pred)
# print('Test accuracy: {}'.format(test_score))
# print('------------------------------------------')

# i = 0
# for ls in test_y_pred :
#     print(test_set.names[i])
#     print_predict_labels(ls)
#     i += 1

# plt.subplot(2, 6, 1)
# plt.imshow(target_image)
# plt.show()


# save results as text file
text_file = open("results.txt", "w")
for idx in range(0, results.shape[0]) :
    text_file.write("%s:" % test_set.names[idx])
    for i in range(0, 9) :
        text_file.write("%s," % results[idx][i])
    text_file.write("%s\n" % results[idx][9])    # result has 10 images per a test image
text_file.close()


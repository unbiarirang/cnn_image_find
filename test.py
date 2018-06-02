import os
import math
import numpy as np
import tensorflow as tf
from datasets import asirra as dataset
from models.nn import AlexNet as ConvNet
from learning.evaluators import AccuracyEvaluator as Evaluator

from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

category = ['others','bird','insect','worm','butfly','wheel','struct','piano','plane','bottle','glass','woman','flower']

def print_labels_by_category(labels) :
    for i, lable in enumerate(labels) :
        if int(lable) == 1 :
            break
    print( category[i] )

""" 1. Load dataset """
root_dir = os.path.join('..', 'image')    # FIXME ..\image
test_dir = os.path.join(root_dir, 'test')

# Load test set
x_test, y_test, z_test = dataset.read_asirra_subset_csv(test_dir, one_hot=True)
test_set = dataset.DataSet(x_test, y_test, z_test)

# Sanity check
print('Test set stats:')
print(test_set.images.shape)
print(test_set.images.min(), test_set.images.max())
print((test_set.labels[:, 1] == 0).sum(), (test_set.labels[:, 1] == 1).sum())


""" 2. Set test hyperparameters """
hp_d = dict()
image_mean = np.load('/tmp/asirra_mean.npy')    # load mean image
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
np.savetxt("pred_value_names.csv", test_set.names, delimiter=",", fmt='%s')
np.savetxt("pred_value.csv", test_y_pred, delimiter=",")

# read predict value from pred_value.csv
pred_list = np.genfromtxt('pred_value.csv', delimiter=',')
pred_name_list = np.genfromtxt('pred_value_names.csv', delimiter=',', dtype=str)

# set target image
index = np.where(test_set.names == 'but.jpg')[0][0]
target_image = test_set.images[index]
target_image_name = test_set.names[index]
target_pred_value = test_y_pred[index]

distances = np.empty(pred_list.shape[0])
for i, pred_values in enumerate(pred_list) :
    dist = 0
    for j, x in enumerate(np.nditer(pred_values)) :
        dist += (x - target_pred_value[j]) * (x - target_pred_value[j])
    dist = math.sqrt(dist)
    distances[i] = dist

# find results
results = np.empty(3, dtype=object)
for i in range(0,3) :
    closest = distances.argmin()
    distances[closest] = 13 # max distance value
    results[i] = pred_name_list[closest]
    index = np.where(test_set.names == results[i])[0][0]
    plt.subplot(2 ,6, i+2)
    plt.imshow(test_set.images[index])


test_score = evaluator.score(test_set.labels, test_y_pred)
print('Test accuracy: {}'.format(test_score))

print('------------------------------------------')

# for ls in test_set.labels :
#     print_labels_by_category(ls)

print(results)
print(target_image_name)
plt.subplot(2, 6, 1)
plt.imshow(target_image)
plt.show()

print('------------------------------------------')

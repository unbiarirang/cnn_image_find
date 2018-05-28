import os
import numpy as np
import tensorflow as tf
from datasets import asirra as dataset
from models.nn import AlexNet as ConvNet
from learning.evaluators import AccuracyEvaluator as Evaluator

from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

category = ['기타','새','곤충','애벌레','나비','물레방아','건축물','피아노','비행기','술병','술잔','여자','꽃']

def print_lables_by_category(lables) :
    for i, lable in enumerate(lables) :
        if int(lable) == 1 :
            break
    print( category[i] )

""" 1. Load dataset """
root_dir = os.path.join('..', 'image')    # FIXME ..\image
test_dir = os.path.join(root_dir, 'test')

# Load test set
X_test, y_test = dataset.read_asirra_subset_csv(test_dir, one_hot=True)
test_set = dataset.DataSet(X_test, y_test)

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

test_score = evaluator.score(test_set.labels, test_y_pred)
print('Test accuracy: {}'.format(test_score))

print('------------------------------------------')

for ls in test_set.labels :
    print_lables_by_category(ls)

print('------------------------------------------')

# -*- coding: utf-8 -*-
# @Time    : 2018\12\20 0020 14:31
# @Author  : 凯
# @File    : tmp.py
import tensorflow as tf
with tf.device(device_type): # <= This is optional
    n_input  = 784
    n_output = 10
    weights  = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),
        'wd1': tf.Variable(tf.random_normal([14*14*64, n_output], stddev=0.1))
    }
    biases   = {
        'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
        'bd1': tf.Variable(tf.random_normal([n_output], stddev=0.1))
    }
    def conv_simple(_input, _w, _b):
        # Reshape input
        _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
        # Convolution
        _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
        # Add-bias
        _conv2 = tf.nn.bias_add(_conv1, _b['bc1'])
        # Pass ReLu
        _conv3 = tf.nn.relu(_conv2)
        # Max-pooling
        _pool  = tf.nn.max_pool(_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # Vectorize
        _dense = tf.reshape(_pool, [-1, _w['wd1'].get_shape().as_list()[0]])
        # Fully-connected layer
        _out = tf.add(tf.matmul(_dense, _w['wd1']), _b['bd1'])
        # Return everything
        out = {
            'input_r': _input_r, 'conv1': _conv1, 'conv2': _conv2, 'conv3': _conv3
            , 'pool': _pool, 'dense': _dense, 'out': _out
        }
        return out
print ("CNN ready")

import scipy.io
import numpy as np
import os
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf
cwd           = os.getcwd()
VGG_PATH      = cwd + "/data/imagenet-vgg-verydeep-19.mat"
CONTENT_PATH  = cwd + "/images/zly1.jpg"
CONTENT_LAYER = 'relu2_2'
STYLE_LAYERS  = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
# STYLE_LAYERS  = ('relu1_1', 'relu2_1')
raw_content = scipy.misc.imread(CONTENT_PATH)


content_image = raw_content.astype(np.float)
content_shape = (1,) + content_image.shape # (h, w, nch) =>  (1, h, w, nch)
print ("Packages loaded")

image = tf.placeholder('float', shape=content_shape)
nets, content_mean_pixel, _ = net(VGG_PATH, image)
data_path = VGG_PATH
input_image = image
layers = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)
layer = data['layers']
print("layer.shape:", layer.shape)
# print(layer)输出是(1, 1),只有一个元素
print("layer[0].shape:", layer[0].shape)
# layer[0][0].shape: (1,),说明只有一个元素
print("layer[0][0].shape:", layer[0][0].shape)

# layer[0][0][0].shape: (1,),说明只有一个元素
print("layer[0][0][0].shape:", layer[0][0][0].shape)
# len(layer[0][0]):5，即weight(含有bias), pad(填充元素,无用), type, name, stride信息
print("len(layer[0][0][0]):", len(layer[0][0][0]))
# 所以应该能按照如下方式拿到信息，比如说name，输出为['conv1_1']
print("name:", layer[0][0][0][3])
# 查看一下weights的权重，输出(1,2),再次说明第一维是虚的,weights中包含了weight和bias
print("layer[0][0][0][0].shape", layer[0][0][0][0].shape)
print("layer[0][0][0][0].len", len(layer[0][0][0][0]))

# weights[0].shape: (2,),weights[0].len: 2说明两个元素就是weight和bias
print("layer[0][0][0][0][0].shape:", layer[0][0][0][0][0].shape)
print("layer[0][0][0][0].len:", len(layer[0][0][0][0][0]))

weights = layer[0][0][0][0][0]
# 解析出weight和bias
weight, bias = weights
# weight.shape: (3, 3, 3, 64)
print("weight.shape:", weight.shape)
# bias.shape: (1, 64)


data = scipy.io.loadmat(data_path)
mean = data['normalization'][0][0][0]
mean_pixel = np.mean(mean, axis=(0, 1))
weights = data['layers'][0]
net = {}
current = input_image
for i, name in enumerate(layers):
    kind = name[:4]
    if kind == 'conv':
        kernels, bias = weights[i][0][0][0][0]
        # matconvnet: weights are [width, height, in_channels, out_channels]
        # tensorflow: weights are [height, width, in_channels, out_channels]
        kernels = np.transpose(kernels, (1, 0, 2, 3))
        bias = bias.reshape(-1)
        current = _conv_layer(current, kernels, bias)
    elif kind == 'relu':
        current = tf.nn.relu(current)
    elif kind == 'pool':
        current = _pool_layer(current)
    net[name] = current
assert len(net) == len(layers)
return net, mean_pixel, layers
def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)
def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')
def preprocess(image, mean_pixel):
    return image - mean_pixel
def unprocess(image, mean_pixel):
    return image + mean_pixel
def imread(path):
    return scipy.misc.imread(path).astype(np.float)
def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)
print ("Network for VGG ready")
cwd           = os.getcwd()
VGG_PATH      = cwd + "/data/imagenet-vgg-verydeep-19.mat"
CONTENT_PATH  = cwd + "/images/zly1.jpg"
CONTENT_LAYER = 'relu2_2'
STYLE_LAYERS  = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
# STYLE_LAYERS  = ('relu1_1', 'relu2_1')

raw_content = scipy.misc.imread(CONTENT_PATH)
plt.figure(0, figsize=(10, 5))
plt.imshow(raw_content)
plt.title("Original content image")
plt.show()

content_image = raw_content.astype(np.float)
content_shape = (1,) + content_image.shape # (h, w, nch) =>  (1, h, w, nch)
with tf.Graph().as_default(), tf.Session() as sess:
    image = tf.placeholder('float', shape=content_shape)
    nets, content_mean_pixel, _ = net(VGG_PATH, image)
    content_image_pre = np.array([preprocess(content_image, content_mean_pixel)])
    content_features = nets[CONTENT_LAYER].eval(feed_dict={image: content_image_pre})
    print (" Type of 'features' is ", type(content_features))
    print (" Shape of 'features' is %s" % (content_features.shape,))
    # Plot response
    for i in range(5):
        plt.figure(i, figsize=(10, 5))
        plt.matshow(content_features[0, :, :, i], cmap=plt.cm.gray, fignum=i)
        plt.title("%d-layer content feature" % (i))
        plt.colorbar()
        plt.show()
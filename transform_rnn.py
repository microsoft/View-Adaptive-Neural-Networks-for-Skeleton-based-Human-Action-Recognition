# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import theano.tensor as T
import numpy as np
import theano
from keras import backend as K
from keras.engine.topology import Layer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
floatX = theano.config.floatX


def _transform_trans(theta,input):
    batch1, step1, dim1 = input.shape
    input = K.reshape(input,(batch1,step1,dim1//3,3))
    input = K.reshape(input,(batch1*step1,dim1//3,3))
    input = K.permute_dimensions(input,[0,2,1])
    add = T.ones((batch1*step1,1,dim1//3))
    input= K.concatenate([input,add],axis=1)

    output = K.batch_dot(theta,input)
    output = K.permute_dimensions(output,[0,2,1])
    output = K.reshape(output,(output.shape[0],dim1))
    output = K.reshape(output,(batch1,step1,output.shape[1]))

    return output

def _trans(theta):
    tx = theta[:,3:4]
    ty = theta[:,4:5]
    tz = theta[:,5:6]
    zero = K.zeros_like(tx)
    one = K.ones_like(tx)
    first = K.reshape(K.concatenate([one,zero,zero,tx],axis=1),(-1,1,4))
    second = K.reshape(K.concatenate([zero,one,zero,ty],axis=1),(-1,1,4))
    third = K.reshape(K.concatenate([zero,zero,one,tz],axis=1),(-1,1,4))
    trans = K.concatenate([first,second,third],axis=1)
    trans = trans.reshape((trans.shape[0],3,4))

    return trans

class VA(Layer):
    """The layer for transforming the skeleton to the observed viewpoints"""
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(VA, self).__init__()

    def compute_output_shape(self,input_shapes):
        shp = input_shapes[0]
        return tuple((shp[0],shp[1],shp[2]))

    def compute_mask(self, input, mask):
        return mask[1]
    
    def call(self,x,mask=None):
        conv_input,theta = x
        s = theta.shape
        theta = T.reshape(theta,[-1,s[2]])
        m = K.not_equal(conv_input,0.)

        #### For translation
        trans = _trans(theta)
        output = _transform_trans(trans, conv_input)
        output = output * K.cast(m,K.floatx())

        ### For rotation
        M = _fusion(theta)
        output = _transform_rot(M,output)

        return output

def _rotation_y(theta):
    r1 = K.cos(theta[:,0:1])
    r2 = K.sin(theta[:,0:1])
    zero = K.zeros_like(r1)
    one = K.ones_like(r1)
    first = K.reshape(K.concatenate([r1,zero,r2,zero],axis=1),(-1,1,4))
    second = K.reshape(K.concatenate([zero,one,zero,zero],axis=1),(-1,1,4))
    third = K.reshape(K.concatenate([-r2,zero,r1,zero],axis=1),(-1,1,4))
    fourth = K.reshape(K.concatenate([zero,zero,zero,one],axis=1),(-1,1,4))
    rotation_y = K.concatenate([first,second,third,fourth],axis=1)
    rotation_y = T.reshape(rotation_y,[-1,4,4])
    return rotation_y

def _rotation_x(theta):
    r1 = K.cos(theta[:,1:2])
    r2 = K.sin(theta[:,1:2])
    zero = K.zeros_like(r1)
    one = K.ones_like(r1)
    first = K.reshape(K.concatenate([one,zero,zero,zero],axis=1),(-1,1,4))
    second = K.reshape(K.concatenate([zero,r1,-r2,zero],axis=1),(-1,1,4))
    third = K.reshape(K.concatenate([zero,r2,r1,zero],axis=1),(-1,1,4))
    fourth = K.reshape(K.concatenate([zero,zero,zero,one],axis=1),(-1,1,4))
    rotation_x = K.concatenate([first,second,third,fourth],axis=1)
    rotation_x = T.reshape(rotation_x,[-1,4,4])
    return rotation_x

def _rotation_z(theta):
    r1 = K.cos(theta[:,2:3])
    r2 = K.sin(theta[:,2:3])
    zero = K.zeros_like(r1)
    one = K.ones_like(r1)
    first = K.reshape(K.concatenate([r1,-r2,zero,zero],axis=1),(-1,1,4))
    second = K.reshape(K.concatenate([r2,r1,zero,zero],axis=1),(-1,1,4))
    third = K.reshape(K.concatenate([zero,zero,one,zero],axis=1),(-1,1,4))
    fourth = K.reshape(K.concatenate([zero,zero,zero,one],axis=1),(-1,1,4))
    rotation_z = K.concatenate([first,second,third,fourth],axis=1)
    rotation_z = T.reshape(rotation_z,[-1,4,4])
    return rotation_z

def _trans_rot_new(theta):
    tx = theta[:,3:4]
    ty = theta[:,4:5]
    tz = theta[:,5:6]
    zero = K.zeros_like(tx)
    one = K.ones_like(tx)
    first = K.reshape(K.concatenate([one,zero,zero,tx],axis=1),(-1,1,4))
    second = K.reshape(K.concatenate([zero,one,zero,ty],axis=1),(-1,1,4))
    third = K.reshape(K.concatenate([zero,zero,one,tz],axis=1),(-1,1,4))
    fourth = K.reshape(K.concatenate([zero,zero,zero,one],axis=1),(-1,1,4))
    trans = K.concatenate([first,second,third,fourth],axis=1)

    trans = T.reshape(trans,[-1,4,4])
    return trans

def _fusion(theta):

    rotation_x = _rotation_x(theta)
    rotation_y = _rotation_y(theta)
    rotation_z = _rotation_z(theta)

    rot = K.batch_dot(rotation_z,rotation_y)
    rot = K.batch_dot(rot,rotation_x)

    return rot

def _transform_rot(theta,input):

    batch1, step1, dim1 = input.shape
    input = T.reshape(input,[-1,dim1//3,3])
    input = K.permute_dimensions(input,[0,2,1])
    add = T.ones((batch1*step1,1,dim1//3))
    input= K.concatenate([input,add],axis=1)

    output = K.batch_dot(theta,input)
    output = K.permute_dimensions(output,[0,2,1])
    output = output[:,:,0:3]
    output = T.reshape(output,[-1,dim1])
    output = T.reshape(output,[-1,step1,dim1])

    return output

class augmentaion(Layer):
    def __init__(self,**kwargs):
        self.supports_masking = True
        self.uses_learning_phase = True
        super(augmentaion, self).__init__(**kwargs)

    def compute_mask(self, input, mask):
        return mask
    def call(self,x,training=None):
        deta1 = 0.3
        deta2 = 0.3
        deta3 = 0.3
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
        theta1 = rng.uniform(size=(x.shape[0],1),low=-deta1,high=deta1,dtype='float32')
        theta2 = rng.uniform(size=(x.shape[0],1),low=-deta2,high=deta2,dtype='float32')
        theta3 = rng.uniform(size=(x.shape[0],1),low=-deta3,high=deta3,dtype='float32')
        theta = K.concatenate([theta1,theta2,theta3],axis=-1)
        theta = K.tile(theta,x.shape[1])
        theta = theta.reshape((x.shape[0], x.shape[1], 3))

        theta = theta.reshape((theta.shape[0]*theta.shape[1], theta.shape[2]))
        M = _fusion(theta)
        output = _transform_rot(M, x)

        return K.in_train_phase(output,x,training = training)


class Noise(Layer):
    """Add Guassian Noise"""

    def __init__(self, sigma, **kwargs):
        self.supports_masking = True
        self.sigma = sigma
        self.uses_learning_phase = True
        super(Noise, self).__init__(**kwargs)

    def compute_mask(self, input, mask):
        return mask

    def call(self, x, mask=None, training=None):
        m = K.not_equal(x, 0.)
        noise_x = x + K.random_normal(shape=K.shape(x),
                                      mean=0.,
                                      stddev=self.sigma)
        noise_x = noise_x * K.cast(m, K.floatx())

        return K.in_train_phase(noise_x, x, training=training)

    def get_config(self):
        config = {'sigma': self.sigma}
        base_config = super(Noise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MeanOverTime(Layer):
    """Average the score of every step"""
    def __init__(self, **kwargs):
        self.supports_masking = True

        super(MeanOverTime, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        x = K.sum(inputs, axis=1)
        mask = K.cast(mask, K.floatx())
        mask = K.sum(mask, axis=1, keepdims=True)
        output = x / mask
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {}
        base_config = super(MeanOverTime, self).get_config()
        return dict(list(base_config.items()))
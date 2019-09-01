# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
parser = argparse.ArgumentParser(description='View adaptive')
parser.add_argument('--dataset', type=str, default='NTU',
                    help='type of dataset')
parser.add_argument('--model', type=str, default='VA',
                    help='type of recurrent net (VA, basline)')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=0.005,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch size')
parser.add_argument('--case', type=int, default=0,
                    help='select dataset')
parser.add_argument('--norm', type=float, default=0.001,
                    help='LSMT recurrent initializer')
parser.add_argument('--aug', type=int, default=1,
                    help='data augmentation')
parser.add_argument('--save', type=int, default=1,
                    help='save results')
parser.add_argument('--gpu', type=int, default=0,
                    help='which gpu is used to train')
parser.add_argument('--train', type=int, default=1,
                    help='train or test')
args = parser.parse_args()


import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
from keras import initializers
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dropout, LSTM, Dense,Activation,TimeDistributed,Masking,Concatenate
from keras.callbacks import EarlyStopping,CSVLogger,ReduceLROnPlateau, ModelCheckpoint
from transform_rnn import VA, Noise,MeanOverTime, augmentaion
from data_rnn import  get_data, get_cases, get_activation


def creat_model(input_shape, num_class):

    init = initializers.Orthogonal(gain=args.norm)
    sequence_input =Input(shape=input_shape)
    mask = Masking(mask_value=0.)(sequence_input)
    if args.aug:
        mask = augmentaion()(mask)
    X = Noise(0.075)(mask)
    if args.model[0:2]=='VA':
        # VA
        trans = LSTM(args.nhid,recurrent_activation='sigmoid',return_sequences=True,implementation=2,recurrent_initializer=init)(X)
        trans = Dropout(0.5)(trans)
        trans = TimeDistributed(Dense(3,kernel_initializer='zeros'))(trans)
        rot = LSTM(args.nhid,recurrent_activation='sigmoid',return_sequences=True,implementation=2,recurrent_initializer=init)(X)
        rot = Dropout(0.5)(rot)
        rot = TimeDistributed(Dense(3,kernel_initializer='zeros'))(rot)
        transform = Concatenate()([rot,trans])
        X = VA()([mask,transform])

    X = LSTM(args.nhid,recurrent_activation='sigmoid',return_sequences=True,implementation=2,recurrent_initializer=init)(X)
    X = Dropout(0.5)(X)
    X = LSTM(args.nhid,recurrent_activation='sigmoid',return_sequences=True,implementation=2,recurrent_initializer=init)(X)
    X = Dropout(0.5)(X)
    X = LSTM(args.nhid,recurrent_activation='sigmoid',return_sequences=True,implementation=2,recurrent_initializer=init)(X)
    X = Dropout(0.5)(X)
    X = TimeDistributed(Dense(num_class))(X)
    X = MeanOverTime()(X)
    X = Activation('softmax')(X)

    model=Model(sequence_input,X)
    return model

def main(rootdir, case, results):
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_data(args.dataset, case)

    input_shape = (train_x.shape[1], train_x.shape[2])
    num_class = train_y.shape[1]
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)
    filepath = os.path.join(rootdir, str(case) + '.hdf5')
    saveto = os.path.join(rootdir, str(case) + '.csv')
    optimizer = Adam(lr=args.lr, clipnorm=args.clip)
    pred_dir = os.path.join(rootdir, str(case) + '_pred.txt')

    if args.train:
        model = creat_model(input_shape, num_class)
        early_stop = EarlyStopping(monitor='val_acc', patience=15, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5, mode='auto', cooldown=3., verbose=1)
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
        csv_logger = CSVLogger(saveto)
        if args.dataset=='NTU' or args.dataset == 'PKU':
            callbacks_list = [csv_logger, checkpoint, early_stop, reduce_lr]
        else:
            callbacks_list = [csv_logger, checkpoint]

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.fit(train_x, train_y, validation_data=[valid_x, valid_y], epochs=args.epochs,
                  batch_size=args.batch_size, callbacks=callbacks_list, verbose=2)

    # test
    model = creat_model(input_shape, num_class)
    model.load_weights(filepath)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    scores = get_activation(model, test_x, test_y, pred_dir, VA=10, par=9)

    results.append(round(scores, 2))


if __name__ == '__main__':
    results = list()
    rootdir = os.path.join('./results/VA-RNN', args.dataset, args.model)
    cases = get_cases(args.dataset)

    for case in range(cases):
        args.case = case
        main(rootdir, args.case, results)
    np.savetxt(rootdir + '/resuult.txt', results, fmt = '%f')

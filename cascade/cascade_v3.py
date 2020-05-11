import sys
sys.path.append(".")
import os
print(os.getcwd())
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.core import Dense, Flatten, RepeatVector, Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.recurrent import GRU
from keras.layers.normalization import BatchNormalization
from chemvae import hyperparameters
from chemvae.train_vae import vectorize_data
from keras.models import Sequential
import copy
import json
from keras.callbacks import CSVLogger, LambdaCallback
# from chemvae.models import encoder_model, decoder_model

def layer_model(params):
    totallayer=(params['conv_depth']*2+4+params['middle_layer']*3*2+params['gru_depth'])
    print('totalalyer',totallayer)
    print('-'*20)
    layer=[None]*totallayer
    layer[0] = keras.layers.Input(shape=(params['MAX_LEN'], params['NCHARS']), name='input_molecule_smi')
    layer[1] = Convolution1D(int(params['conv_dim_depth'] *
                          params['conv_d_growth_factor']),
                      int(params['conv_dim_width'] *
                          params['conv_w_growth_factor']),
                      activation='tanh',
                      name="encoder_conv0")
    if params['batchnorm_conv']:
        layer[2] = BatchNormalization(axis=-1, name="encoder_norm0")

    for j in range(1, params['conv_depth']):
        layer[1+j*2] = Convolution1D(int(params['conv_dim_depth'] *
                              params['conv_d_growth_factor'] ** (j)),
                          int(params['conv_dim_width'] *
                              params['conv_w_growth_factor'] ** (j)),
                          activation='tanh',
                          name="encoder_conv{}".format(j))
        if params['batchnorm_conv']:
            layer[2+j*2] = BatchNormalization(axis=-1,
                                   name="encoder_norm{}".format(j))
    n=params['conv_depth']*2+1

    layer[n]=Flatten(name='encoder_flatten')

    if params['middle_layer'] > 0:
        layer[n+1]=Dense(int(params['hidden_dim'] *
                           params['hg_growth_factor'] ** (params['middle_layer'] - 1)),
                          activation=params['activation'], name='encoder_dense0')
        
        if params['dropout_rate_mid'] > 0:
            layer[n+2] = Dropout(params['dropout_rate_mid'],name='encoder_drop_0')
        if params['batchnorm_mid']:
            layer[n+3] = BatchNormalization(axis=-1, name='encoder_norm_0')

        for i in range(0, params['middle_layer']):
            layer[n+4+i*3] = Dense(int(params['hidden_dim'] *
                               params['hg_growth_factor'] ** (params['middle_layer'] - i)),
                           activation=params['activation'], name='encoder_dense{}'.format(i+1))
            if params['dropout_rate_mid'] > 0:
                layer[n+5+i*3] = Dropout(params['dropout_rate_mid'],name='encoder_drop_{}'.format(i+1))
            if params['batchnorm_mid']:
                layer[n+6+i*3] = BatchNormalization(axis=-1,
                                            name='encoder_norm_{}'.format(i+1))
    n=params['middle_layer']*3+n+1

    layer[n] = Dense(int(params['hidden_dim']),
              activation=params['activation'],
              name="decoder_dense0")
  
    if params['dropout_rate_mid'] > 0:
        layer[n+1] = Dropout(params['dropout_rate_mid'])
    if params['batchnorm_mid']:
        layer[n+2] = BatchNormalization(axis=-1, name="decoder_norm_0")

    for i in range(1, params['middle_layer']+1):
        layer[n+i*3] = Dense(int(params['hidden_dim'] *
                      params['hg_growth_factor'] ** (i)),
                  activation=params['activation'],
                  name="decoder_dense{}".format(i))
        if params['dropout_rate_mid'] > 0:
            layer[n+1+i*3] = Dropout(params['dropout_rate_mid'])
        if params['batchnorm_mid']:
            layer[n+2+i*3] = BatchNormalization(axis=-1,
                                   name="decoder_norm_{}".format(i))

    # Necessary for using GRU vectors
    n = n + params['middle_layer']*3+1

    layer[n] = RepeatVector(params['MAX_LEN'],name='decoder_repeat')

    if params['gru_depth'] > 1:
        layer[n+1] = GRU(params['recurrent_dim'],
                    return_sequences=True, activation='tanh',
                    name="decoder_gru0")

        for k in range(params['gru_depth'] - 2):
            layer[n+2+k] = GRU(params['recurrent_dim'],
                        return_sequences=True, activation='tanh',
                        name="decoder_gru{}".format(k + 1))

        if params['do_tgru']:
            layer[n+k+3] = TerminalGRU(params['NCHARS'],
                                rnd_seed=params['RAND_SEED'],
                                recurrent_dropout=params['tgru_dropout'],
                                return_sequences=True,
                                activation='softmax',
                                temperature=0.01,
                                name='decoder_tgru',
                                implementation=params['terminal_GRU_implementation'])([x_dec, true_seq_in])
        else:
            layer[n+k+3]= GRU(params['NCHARS'],
                        return_sequences=True, activation='softmax',
                        name='decoder_gru_final')

    else:
        
        layer[n+1] = GRU(params['NCHARS'],
                    return_sequences=True, activation='softmax',
                    name='decoder_gru_final'
                    )

    return layer
   
def Cascade(params, training_layer, X_train, Y_train, X_test, Y_test,epochs=20,loss='categorical_crossentropy',
                        optimizer='sgd',initialLr=0.01,weightDecay=10e-4,patience=10,
                        windowSize=5,batch_size=128,outNeurons=64,nb_classes=10,index=0,fast=True,gradient=False):
    nextModelToTrain = list() 
    saveEncoderLayersIndexes = list() 
    saveDecoderLayersIndexes = list()
    saveFCEncoderLayersIndexes = list() 
    saveFCDecoderLayersIndexes = list()
    history = dict()
    # totallayer=(params['conv_depth']*2+4+params['middle_layer']*3*2+params['gru_depth'])
    # encoder_depth=len([v for v,k in layers.items() if k.startswith('conv')])
    
    i = 0
    for currentLayer in training_layer: 
        if (currentLayer.name[0:12] == 'encoder_conv'):
            saveEncoderLayersIndexes.append(i)
        if (currentLayer.name[0:15] == 'encoder_flatten'):
            saveFCEncoderLayersIndexes.append(i)
        if (currentLayer.name[0:13] == 'encoder_dense'):
            saveFCEncoderLayersIndexes.append(i)
        
        if (currentLayer.name[0:13] == 'decoder_dense'):
            saveFCDecoderLayersIndexes.append(i)
        if (currentLayer.name[0:14] == 'decoder_repeat'):
            saveFCDecoderLayersIndexes.append(i)
        if (currentLayer.name[0:11] == 'decoder_gru'):
            saveDecoderLayersIndexes.append(i)
        i += 1
    print('encoder',saveEncoderLayersIndexes)
    print('decoder',saveDecoderLayersIndexes)
    print('encoderfc',saveFCEncoderLayersIndexes)
    print('decoderfc',saveFCDecoderLayersIndexes)
    print('layers',training_layer)
    print('-'*30)
  
    for i in range(0,len(saveEncoderLayersIndexes)): 
        if ('iter' + str(i) not in history.keys()): 
            history['iter' + str(i)] = dict() 

        print('layer is',i)

        if(i == 0):
            for j in training_layer[0:saveEncoderLayersIndexes[1]]: #FOR CORRESPONDING LAYERS FOR CURRENT RUN IN MODEL
                nextModelToTrain.append(j)
            for j in training_layer[saveFCEncoderLayersIndexes[0]:saveFCDecoderLayersIndexes[0]]: #FOR CORRESPONDING LAYERS FOR CURRENT RUN IN MODEL
                nextModelToTrain.append(j)
       
            for j in training_layer[saveFCDecoderLayersIndexes[0]:saveDecoderLayersIndexes[0]]: #FOR CORRESPONDING LAYERS FOR CURRENT RUN IN MODEL
                nextModelToTrain.append(j)
            for j in training_layer[saveDecoderLayersIndexes[-1]:saveDecoderLayersIndexes[-2]:-1]: #FOR CORRESPONDING LAYERS FOR CURRENT RUN IN MODEL
                nextModelToTrain.append(j)

        else:

            for k in training_layer[saveEncoderLayersIndexes[i]+1:saveEncoderLayersIndexes[i]-1:-1]: #GET THE LAYERS OF NEXT MODEL TO TRAIN
                    print('KKKKKKK',k)
                    nextModelToTrain.insert(1+i*2,k)
            print('LLLLLLL',saveDecoderLayersIndexes[-i-1],saveDecoderLayersIndexes[-i])
            j = training_layer[saveDecoderLayersIndexes[-i-1]] #GET THE LAYERS OF NEXT MODEL TO TRAIN
            print('JJJJJJ',j)
            nextModelToTrain.insert(-i,j)

        nextModelToTrainInputs = Input(shape=(params['MAX_LEN'], params['NCHARS']), name='input_smi')
        print('model',nextModelToTrain)
        

        ModelToTrain=Sequential()

        for currentlayerIndex in range(len(nextModelToTrain[1:])):
            currentlayer = nextModelToTrain[1+currentlayerIndex]
            if currentlayerIndex==1+i*2+1 or currentlayerIndex==2+i*2+1 or currentlayerIndex==len(nextModelToTrain[1:])-i:
                ModelToTrain.add(currentlayer)
            else:
                _layer=copy.deepcopy(currentlayer)
                ModelToTrain.add(_layer)
        
        '''
        x = nextModelToTrain[1](nextModelToTrainInputs)
        for current_layer_index in range(2,len(nextModelToTrain)):
            print('currentlayer',nextModelToTrain[current_layer_index])
            x = nextModelToTrain[current_layer_index](x)
            print('current shape',x.shape)
        

        ModelToTrain = Model(inputs=nextModelToTrainInputs, outputs=x)
        '''
        
        for layer in ModelToTrain.layers[0:1+i*2]:
            layer.trainable=False
        for layer in ModelToTrain.layers[i*2+2:-i]:
            layer.trainable=False
        for layer in ModelToTrain.layers[-i+1:]:
            layer.trainable=False
        print('TTTTTTTTT', 1+i*2, i*2+2, -i+1)
        # K.set_value(optimizer.lr,initialLr) #SET INITIAL LEARNING RATE (IT MIGHT HAVE BEEN CHANGED BY PREVIOUS ITERATIONS)
        ModelToTrain.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
        model_train_targets = {'x_pred':X_train,
                              'z_mean_log_var':np.ones((np.shape(X_train)[0], params['hidden_dim'] * 2))}
        model_test_targets = {'x_pred':X_test,
                              'z_mean_log_var':np.ones((np.shape(X_test)[0], params['hidden_dim'] * 2))}
        
        print('PPPPPPPP',X_test.shape, X_train.shape)
        # json_log = open('weights_log.json', mode='a', buffering=1)
        # print_weights = LambdaCallback(on_train_end=lambda logs: json_log.write(json.dumps(ModelToTrain.get_weights())))
        # print_weights = LambdaCallback(on_train_end=lambda logs:print(ModelToTrain.get_weights()))
        csv_logger = CSVLogger('training.csv',append=True)
        tmpHistory=ModelToTrain.fit(X_train, X_train,
                            batch_size=params['batch_size'],
                            epochs=params['epochs'],
                            initial_epoch=params['prev_epochs'],
                            callbacks=[csv_logger],
                            verbose=params['verbose_print'],
                            validation_data=[X_test, X_test])
        print('Weigths are', '-'*20, ModelToTrain.get_weights())
        history['iter'+str(i)]['lossTraining'] = tmpHistory.history['loss']
        history['iter'+str(i)]['accuracyTraining'] = tmpHistory.history['acc']
       
        print('iter',str(i), history['iter'+str(i)])
        
    return ModelToTrain, history


def main_run(params):
    X_train, X_test, Y_train, Y_test = vectorize_data(params)
    auto_model = layer_model(params)
    print('Model is',auto_model)
    model, history = Cascade(params, auto_model, X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_file',
                        help='experiment file', default='exp.json')
    parser.add_argument('-d', '--directory',
                        help='exp directory', default=None)
    args = vars(parser.parse_args())

    if args['directory'] is not None:
        args['exp_file'] = os.path.join(args['directory'], args['exp_file'])
    params = hyperparameters.load_params(args['exp_file'])

    main_run(params)
    
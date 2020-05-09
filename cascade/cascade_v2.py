import sys
sys.path.append(".")
import os
print(os.getcwd())
import argparse
import numpy as np
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
# from chemvae.models import encoder_model, decoder_model

def final_model(params):
    x_in = Input(shape=(params['MAX_LEN'], params[
        'NCHARS']), name='input_molecule_smi')

    # Convolution layers
    x = Convolution1D(int(params['conv_dim_depth'] *
                          params['conv_d_growth_factor']),
                      int(params['conv_dim_width'] *
                          params['conv_w_growth_factor']),
                      activation='tanh',
                      name="encoder_conv0")(x_in)
    if params['batchnorm_conv']:
        x = BatchNormalization(axis=-1, name="encoder_norm_0")(x)

    for j in range(1, params['conv_depth'] - 1):
        x = Convolution1D(int(params['conv_dim_depth'] *
                              params['conv_d_growth_factor'] ** (j)),
                          int(params['conv_dim_width'] *
                              params['conv_w_growth_factor'] ** (j)),
                          activation='tanh',
                          name="encoder_conv{}".format(j))(x)
        if params['batchnorm_conv']:
            x = BatchNormalization(axis=-1,
                                   name="encoder_norm_{}".format(j))(x)
    print('SSSSSSS',x.shape)
    x = Flatten(name="encoder_flatten0")(x)
    print('SSSSSSS',x.shape)
    print('FFFFFFFFF', int(params['hidden_dim'] *
                           params['hg_growth_factor'] ** (params['middle_layer'] - 1)))
    # Middle layers
    if params['middle_layer'] > 0:
        middle = Dense(int(params['hidden_dim'] *
                           params['hg_growth_factor'] ** (params['middle_layer'] - 1)),
                          activation=params['activation'], name='encoder_dense0')(x)

        
        if params['dropout_rate_mid'] > 0:
            middle = Dropout(params['dropout_rate_mid'])(middle)
        if params['batchnorm_mid']:
            middle = BatchNormalization(axis=-1, name='encoder_norm0')(middle)

        for i in range(2, params['middle_layer'] + 1):
            middle = Dense(int(params['hidden_dim'] *
                               params['hg_growth_factor'] ** (params['middle_layer'] - i)),
                           activation=params['activation'], name='encoder_dense{}'.format(i))(middle)
            if params['dropout_rate_mid'] > 0:
                middle = Dropout(params['dropout_rate_mid'])(middle)
            if params['batchnorm_mid']:
                middle = BatchNormalization(axis=-1,
                                            name='encoder_norm{}'.format(i))(middle)
    else:
        middle = x

    print('MMMMMMM',middle.shape)
    print(params['hidden_dim'])

    z = Dense(int(params['hidden_dim']),
              activation=params['activation'],
              name="decoder_dense0"
              )(middle)
    print('PPPPPPPP',z.shape)
    if params['dropout_rate_mid'] > 0:
        z = Dropout(params['dropout_rate_mid'])(z)
    if params['batchnorm_mid']:
        z = BatchNormalization(axis=-1, name="decoder_norm_0")(z)

    for i in range(1, params['middle_layer']):
        z = Dense(int(params['hidden_dim'] *
                      params['hg_growth_factor'] ** (i)),
                  activation=params['activation'],
                  name="decoder_dense{}".format(i))(z)
        if params['dropout_rate_mid'] > 0:
            z = Dropout(params['dropout_rate_mid'])(z)
        if params['batchnorm_mid']:
            z = BatchNormalization(axis=-1,
                                   name="decoder_norm_{}".format(i))(z)

    # Necessary for using GRU vectors
    z_reps = RepeatVector(params['MAX_LEN'])(z)

    # Encoder parts using GRUs
    if params['gru_depth'] > 1:
        x_dec = GRU(params['recurrent_dim'],
                    return_sequences=True, activation='tanh',
                    name="decoder_gru0"
                    )(z_reps)

        for k in range(params['gru_depth'] - 2):
            x_dec = GRU(params['recurrent_dim'],
                        return_sequences=True, activation='tanh',
                        name="decoder_gru{}".format(k + 1)
                        )(x_dec)

        if params['do_tgru']:
            x_out = TerminalGRU(params['NCHARS'],
                                rnd_seed=params['RAND_SEED'],
                                recurrent_dropout=params['tgru_dropout'],
                                return_sequences=True,
                                activation='softmax',
                                temperature=0.01,
                                name='decoder_tgru',
                                implementation=params['terminal_GRU_implementation'])([x_dec, true_seq_in])
        else:
            x_out = GRU(params['NCHARS'],
                        return_sequences=True, activation='softmax',
                        name='decoder_gru_final')(x_dec)

    else:
        if params['do_tgru']:
            x_out = TerminalGRU(params['NCHARS'],
                                rnd_seed=params['RAND_SEED'],
                                recurrent_dropout=params['tgru_dropout'],
                                return_sequences=True,
                                activation='softmax',
                                temperature=0.01,
                                name='decoder_tgru',
                                implementation=params['terminal_GRU_implementation'])([z_reps, true_seq_in])
        else:
            x_out = GRU(params['NCHARS'],
                        return_sequences=True, activation='softmax',
                        name='decoder_gru_final'
                        )(z_reps)
    return Model(x_in, x_out, name="final_model")





def CascadeTraining(params,model,X_train,Y_train,X_test,Y_test,stringOfHistory=None,dataAugmentation=None,
                        X_val=None,Y_val=None,epochs=20,loss='categorical_crossentropy',
                        optimizer='sgd',initialLr=0.01,weightDecay=10e-4,patience=10,
                        windowSize=5,batch_size=128,outNeurons=64,nb_classes=10,index=0,fast=True,gradient=False):
    nextModelToTrain = list() 
    saveEncoderLayersIndexes = list() 
    saveDecoderLayersIndexes = list()
    saveFCEncoderLayersIndexes = list() 
    saveFCDecoderLayersIndexes = list()
    history = dict()
    nextModelToPredict = None

    i = 0
    for currentLayer in model.layers: 
        if (currentLayer.get_config()['name'][0:12] == 'encoder_conv'):
            saveEncoderLayersIndexes.append(i)
        if (currentLayer.get_config()['name'][0:15] == 'encoder_flatten'):
            saveFCEncoderLayersIndexes.append(i)
        if (currentLayer.get_config()['name'][0:13] == 'encoder_dense'):
            saveFCEncoderLayersIndexes.append(i)
    
        if (currentLayer.get_config()['name'][0:13] == 'decoder_dense'):
            saveFCDecoderLayersIndexes.append(i)
        if (currentLayer.get_config()['name'][0:11] == 'decoder_gru'):
            saveDecoderLayersIndexes.append(i)
        i += 1
    
    for i in range(len(saveEncoderLayersIndexes)): 
        if ('iter' + str(i) not in history.keys()): 
            history['iter' + str(i)] = dict() 
           
            if(i == 0):
                for j in model.layers[0:saveFCEncoderLayersIndexes[0]]: #FOR CORRESPONDING LAYERS FOR CURRENT RUN IN MODEL
                    nextModelToTrain.append(j)
                for j in model.layers[saveFCEncoderLayersIndexes[0]:saveFCDecoderLayersIndexes[0]]: #FOR CORRESPONDING LAYERS FOR CURRENT RUN IN MODEL
                    nextModelToTrain.append(j)
           
                for j in model.layers[saveFCDecoderLayersIndexes[0]:saveDecoderLayersIndexes[0]]: #FOR CORRESPONDING LAYERS FOR CURRENT RUN IN MODEL
                    nextModelToTrain.append(j)
                for j in model.layers[saveDecoderLayersIndexes[0]:saveDecoderLayersIndexes[1]]: #FOR CORRESPONDING LAYERS FOR CURRENT RUN IN MODEL
                    nextModelToTrain.append(j)

            else:
                for k in model.layers[saveEncoderLayersIndexes[i]:saveEncoderLayersIndexes[i+1]]: #GET THE LAYERS OF NEXT MODEL TO TRAIN
                        nextModelToTrain.insert(i,k)
           
                for j in model.layers[saveDecoderLayersIndexes[i]:saveDecoderLayersIndexes[i+1]]: #GET THE LAYERS OF NEXT MODEL TO TRAIN
                        nextModelToTrain.insert(i+(params['middle_layer'])*2,j)

            nextModelToTrainInputs = Input(shape=(params['MAX_LEN'], params['NCHARS']), name='input_smi')
            print('model',nextModelToTrain)
            
            x = nextModelToTrain[0](nextModelToTrainInputs)
            for current_layer_index in range(len(nextModelToTrain)-1):
                print(nextModelToTrain[current_layer_index+1])
                x = nextModelToTrain[current_layer_index+1](x)
            

            nextModelToTrain = Model(inputs=nextModelToTrainInputs, outputs=x)

            for layer in nextModelToTrain.layers[0:i]:
                layer.trainable=False
            for layer in nextModelToTrain.layers[i+1:i+(params['middle_layer'])*2]:
                layer.trainable=False
            for layer in nextModelToTrain.layers[i+(params['middle_layer'])*2+1:]:
                layer.trainable=False

            # K.set_value(optimizer.lr,initialLr) #SET INITIAL LEARNING RATE (IT MIGHT HAVE BEEN CHANGED BY PREVIOUS ITERATIONS)
            nextModelToTrain.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
            model_train_targets = {'x_pred':X_train,
                                  'z_mean_log_var':np.ones((np.shape(X_train)[0], params['hidden_dim'] * 2))}
            model_test_targets = {'x_pred':X_test,
                                  'z_mean_log_var':np.ones((np.shape(X_test)[0], params['hidden_dim'] * 2))}
            tmpHistory=nextModelToTrain.fit(X_train, model_train_targets,
                                batch_size=params['batch_size'],
                                epochs=params['epochs'],
                                initial_epoch=params['prev_epochs'],
                                verbose=params['verbose_print'])
                                # validation_data=[X_test, model_test_targets])
        
            history['iter'+str(i)]['lossTraining'] = tmpHistory.history['loss']
            history['iter'+str(i)]['accuracyTraining'] = tmpHistory.history['acc']
            print(history['iter'+str(i)])
            exit()
    return nextModelToTrain, history

def main_run(params):
    X_train, X_test, Y_train, Y_test = vectorize_data(params)
    auto_model = final_model(params)
    print('Model is',auto_model)
    model, history = CascadeTraining(params, auto_model, X_train, Y_train, X_test, Y_test)

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
    

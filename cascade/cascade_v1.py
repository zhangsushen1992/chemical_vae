import sys
sys.path.append(".")
import os
print(os.getcwd())
import argparse
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
from chemvae.models import encoder_model, decoder_model

'''
def encoder_model(params):
    x_in = Input(shape=(params['MAX_LEN'], params['NCHARS']), name='input_molecule_smi')
    # Convolution layers
    x = Convolution1D(int(params['conv_dim_depth'] *
                          params['conv_d_growth_factor']),
                      int(params['conv_dim_width'] *
                          params['conv_w_growth_factor']),
                      activation='tanh',
                      name="encoder_conv0")(x_in)
    if params['batchnorm_conv']:
        x = BatchNormalization(axis=-1, name="encoder_norm0")(x)

    for j in range(1, params['conv_depth'] - 1):
        x = Convolution1D(int(params['conv_dim_depth'] *
                              params['conv_d_growth_factor'] ** (j)),
                          int(params['conv_dim_width'] *
                              params['conv_w_growth_factor'] ** (j)),
                          activation='tanh',
                          name="encoder_conv{}".format(j))(x)
        if params['batchnorm_conv']:
            x = BatchNormalization(axis=-1,
                                   name="encoder_norm{}".format(j))(x)
    x = Flatten()(x)
    return Model(x_in, x, name="encoder")

def decoder_model(params):
    z_in = Input(shape=(int(params['hidden_dim'] * params['hg_growth_factor'] ** params['middle_layer']),), name='decoder_input')
    print('z_in',z_in.shape)
    z_reps = RepeatVector(params['MAX_LEN'])(z_in)
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

        x_out = GRU(params['NCHARS'],
                        return_sequences=True, activation='softmax',
                        name='decoder_gru_final')(x_dec)
    return Model(z_in, x_out, name="decoder")
'''


def CascadeTraining(params,encoder_model,decoder_model,X_train,Y_train,X_test,Y_test,stringOfHistory=None,dataAugmentation=None,
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
    for currentLayer in encoder_model.layers: 
        if (currentLayer.get_config()['name'][0:12] == 'encoder_conv'):
            saveEncoderLayersIndexes.append(i)
        if (currentLayer.get_config()['name'][0:15] == 'encoder_flatten'):
            saveFCEncoderLayersIndexes.append(i)
        if (currentLayer.get_config()['name'][0:13] == 'encoder_dense'):
            saveFCEncoderLayersIndexes.append(i)
        i+=1
    j=0
    for currentLayer in decoder_model.layers:
        if (currentLayer.get_config()['name'][0:13] == 'decoder_dense'):
            saveFCDecoderLayersIndexes.append(j)
        if (currentLayer.get_config()['name'][0:11] == 'decoder_gru'):
            saveDecoderLayersIndexes.append(j)
        j += 1
    
    for i in range(len(saveEncoderLayersIndexes)): 
        if ('iter' + str(i) not in history.keys()): 
            history['iter' + str(i)] = dict() 
           
            if(i == 0):
                for j in encoder_model.layers[0:saveEncoderLayersIndexes[1]]: #FOR CORRESPONDING LAYERS FOR CURRENT RUN IN MODEL
                    nextModelToTrain.append(j)
                for j in encoder_model.layers[saveFCEncoderLayersIndexes[0]:saveDecoderLayersIndexes[-1]]: #FOR CORRESPONDING LAYERS FOR CURRENT RUN IN MODEL
                    nextModelToTrain.append(j)
                '''
                if params['middle_layer'] > 0:
                    for k in range(0, params['middle_layer'] + 1):
                        nextModelToTrain.append(Dense(int(params['hidden_dim'] *
                                                params['hg_growth_factor'] ** (params['middle_layer'] - k)),
                                                activation=params['activation'], name='encoder_dense{}'.format(k)))
                        if params['dropout_rate_mid'] > 0:    
                            nextModelToTrain.append(Dropout(params['dropout_rate_mid']))
                        if params['batchnorm_mid']:
                            nextModelToTrain.append(BatchNormalization(axis=-1,
                                                    name='encoder_dense{}_norm'.format(i)))
                
                if params['middle_layer']>0:
                    for k in range(0, params['middle_layer']+1):
                        nextModelToTrain.append(Dense(int(params['hidden_dim'] *
                                                        params['hg_growth_factor'] ** (k)),
                                                        activation=params['activation'],
                                                        name="decoder_dense{}".format(k)))
                        if params['dropout_rate_mid'] > 0:
                            nextModelToTrain.append(Dropout(params['dropout_rate_mid']))
                        if params['batchnorm_mid']:
                            nextModelToTrain.append(BatchNormalization(axis=-1, name="decoder_dense{}_norm".format(k)))
                '''             
                for j in decoder_model.layers[0:saveFCDecoderLayersIndexes[-1]]: #FOR CORRESPONDING LAYERS FOR CURRENT RUN IN MODEL
                    nextModelToTrain.append(j)
                for j in decoder_model.layers[:saveDecoderLayersIndexes[1]]: #FOR CORRESPONDING LAYERS FOR CURRENT RUN IN MODEL
                    nextModelToTrain.append(j)

            else:
                for k in encoder_model.layers[saveEncoderLayersIndexes[i]:saveEncoderLayersIndexes[i+1]]: #GET THE LAYERS OF NEXT MODEL TO TRAIN
                        nextModelToTrain.insert(i,k)
           
                for j in decoder_model.layers[saveDecoderLayersIndexes[i]:saveDecoderLayersIndexes[i+1]]: #GET THE LAYERS OF NEXT MODEL TO TRAIN
                        nextModelToTrain.insert(i+(params['middle_layer'])*2,j)

            nextModelToTrainInputs = Input(shape=(params['MAX_LEN'], params['NCHARS']), name='input_molecule_smi')
            print('model',nextModelToTrain)
            
            x = nextModelToTrain[0](nextModelToTrainInputs)
            for current_layer_index in range(len(nextModelToTrain)-1):
                print(nextModelToTrain[current_layer_index+1])
                x = nextModelToTrain[current_layer_index+1](x)
            exit()

            nextModelToTrain = Model(inputs=nextModelToTrainInputs, outputs=x)

            for layer in nextModelToTrain.layers[0:i]:
                layer.trainable=False
            for layer in nextModelToTrain.layers[i+1:i+(params['middle_layer'])*2]:
                layer.trainable=False
            for layer in nextModelToTrain.layers[i+(params['middle_layer'])*2+1:]:
                layer.trainable=False

            K.set_value(optimizer.lr,initialLr) #SET INITIAL LEARNING RATE (IT MIGHT HAVE BEEN CHANGED BY PREVIOUS ITERATIONS)
            nextModelToTrain.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
            tmpHistory=nextModelToTrain.fit(X_train, model_train_targets,
                                batch_size=params['batch_size'],
                                epochs=params['epochs'],
                                initial_epoch=params['prev_epochs'],
                                verbose=keras_verbose,
                                validation_data=[X_test, model_test_targets])
        
            history['iter'+str(i)]['lossTraining'] = tmpHistory.history['loss']
            history['iter'+str(i)]['accuracyTraining'] = tmpHistory.history['acc']
            print(history['iter'+str(i)])
    return nextModelToTrain, history

def main_run(params):
    X_train, X_test, Y_train, Y_test = vectorize_data(params)
    encoder = encoder_model(params)
    decoder = decoder_model(params)
    model, history = CascadeTraining(params, encoder, decoder, X_train, Y_train, X_test, Y_test)

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
    

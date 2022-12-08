import candle
import os
import numpy as np
import pandas as pd
import _pickle as cp
from Modeling_Functions import CNN2D_Regressor, CNN2D_Classifier
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras import backend
from keras.models import load_model
from tensorflow.keras import backend as K



# This should be set outside as a user environment variable
file_path = os.path.dirname(os.path.realpath(__file__))


additional_definitions = [
    {'name': 'cancer_col_name',
     'type': str,
     'help': 'Column name of cancer ID'
     },
    {'name': 'drug_col_name',
     'type': str,
     'help': 'Column name of drug ID'
     },
    {'name': 'res_col_name',
     'type': str,
     'help': 'Column name of response'
     },
    {'name': 'rlr_factor',
     'type': float,
     'help': 'Learning rate reduction factor'
     },
    {'name': 'rlr_min_delta',
     'type': float,
     'help': 'Learning rate reduction minimum delta'
     },
    {'name': 'rlr_cooldown',
     'type': int,
     'help': 'Learning rate reduction cooldown'
     },
    {'name': 'rlr_min_lr',
     'type': float,
     'help': 'Learning rate reduction minimum learning rate'
     },
    {'name': 'rlr_patience',
     'type': int,
     'help': 'Learning rate reduction patience'
     },
    {'name': 'es_patience',
     'type': int,
     'help': 'Early stop patience'
     },
    {'name': 'es_min_delta',
     'type': float,
     'help': 'Early stop minimum delta'
     },
    {'name': 'classification_task',
     'type': bool,
     'help': 'Is the task classification or not'
     },
    {'name': 'cnn_activation',
     'type': str,
     'help': 'Activation function for convolution layers'
     },
    {'name': 'train_task',
     'type': str,
     'help': 'Name of training task'
     }
]

required = ['output_dir', 'conv', 'dropout', 'epochs', 'pool', 'dense', 'activation', 'loss',
            'optimizer', 'verbose', 'batch_size', 'early_stop', 'train_data', 'val_data',
            'data_url']


class igtd(candle.Benchmark):

    def set_locals(self):
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def initialize_parameters():
    igtd_bmk = igtd(file_path, 'Default_Params.txt', 'keras', prog='igtd', desc='Candle compliant IGTD')
    gParameters = candle.finalize_parameters(igtd_bmk)

    return gParameters



def run(params):
# fetch data
# preprocess data
# save preprocessed data
# define callbacks
# build / compile model
# train model
# infer using model
# etc

    params['verbose'] = 2

    candle.get_file(fname=params['train_data'],
                origin=os.path.join(params['data_url'], params['train_data']),
                unpack=False, md5_hash=None,
                datadir=None,
                cache_subdir='')
    pkl_file = open(os.path.join(os.environ['CANDLE_DATA_DIR'], params['train_data']), 'rb')
    temp_data = cp.load(pkl_file)
    pkl_file.close()
    trainData = temp_data['data']
    trainLabel = temp_data['label']
    trainSample = temp_data['sample']

    if 'val_data' in params.keys():
        candle.get_file(fname=params['val_data'],
                        origin=os.path.join(params['data_url'], params['val_data']),
                        unpack=False, md5_hash=None,
                        datadir=None,
                        cache_subdir='')
        pkl_file = open(os.path.join(os.environ['CANDLE_DATA_DIR'], params['val_data']), 'rb')
        temp_data = cp.load(pkl_file)
        pkl_file.close()
        valData = temp_data['data']
        valLabel = temp_data['label']
        valSample = temp_data['sample']
        monitor = 'val_loss'
    else:
        valData = None
        valLabel = None
        valSample = None
        monitor = 'loss'

    batch_size = params['batch_size']

    if isinstance(trainData, list):
        input_data_dim = []
        for i in range(len(trainData)):
            input_data_dim.append([trainData[i].shape[1], trainData[i].shape[2]])
    else:
        input_data_dim = [[trainData.shape[1], trainData.shape[2]]]

    train_logger = CSVLogger(params['output_dir'] + '/log.csv')
    model_saver = ModelCheckpoint(params['output_dir'] + '/model.h5',
                                  monitor=monitor, save_best_only=True, save_weights_only=False)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=params['rlr_factor'], patience=params['rlr_patience'],
                                  verbose=1, mode='auto', min_delta=params['rlr_min_delta'],
                                  cooldown=params['rlr_cooldown'], min_lr=params['rlr_min_lr'])
    early_stop = EarlyStopping(monitor=monitor, patience=params['es_patience'], min_delta=params['es_min_delta'],
                               verbose=1)
    callbacks = [model_saver, train_logger, reduce_lr, early_stop]

    if params['classification_task']:
        num_class = int(len(np.unique(trainLabel)))
        temp = CNN2D_Classifier(params, input_data_dim, num_class, params['dropout'])
        weight = len(trainLabel) / (num_class * np.bincount(trainLabel))
        class_weight = {}
        for i in range(num_class):
            class_weight[i] = weight[i]
    else:
        temp = CNN2D_Regressor(params, input_data_dim, params['dropout'])

    if valData is None:
        if params['classification_task']:
            history = temp.model.fit(x=trainData, y=trainLabel, batch_size=batch_size, epochs=params['epochs'],
                                     verbose=params['verbose'], callbacks=callbacks, validation_data=None,
                                     class_weight=class_weight, shuffle=True)
        else:
            history = temp.model.fit(x=trainData, y=trainLabel, batch_size=batch_size, epochs=params['epochs'],
                                     verbose=params['verbose'], callbacks=callbacks, validation_data=None,
                                     shuffle=True)
    else:
        if params['classification_task']:
            history = temp.model.fit(x=trainData, y=trainLabel, batch_size=batch_size, epochs=params['epochs'],
                                     verbose=params['verbose'], callbacks=callbacks, validation_data=(valData, valLabel),
                                     class_weight=class_weight, shuffle=True)
        else:
            history = temp.model.fit(x=trainData, y=trainLabel, batch_size=batch_size, epochs=params['epochs'],
                                     verbose=params['verbose'], callbacks=callbacks, validation_data=(valData, valLabel),
                                     shuffle=True)
    backend.clear_session()
    model = load_model(params['output_dir'] + '/model.h5')
    predResult = {}

    if params['classification_task']:
        trainPredResult = np.argmax(a=model.predict(trainData).values, axis=1)
    else:
        trainPredResult = model.predict(trainData)[:, 0]
    predResult['train'] = pd.DataFrame({params['cancer_col_name']: [i.split('|')[0] for i in trainSample],
                                        params['drug_col_name']: [i.split('|')[1] for i in trainSample],
                                        params['res_col_name']: trainLabel,
                                        'Prediction': trainPredResult}, index=trainSample)
    predResult['train'].to_csv(params['output_dir'] + '/Prediction_Result_Train_' + params['train_task'] + '.txt', header=True,
                               index=False, sep='\t', line_terminator='\r\n')

    if valData is not None:
        if params['classification_task']:
            valPredResult = np.argmax(a=model.predict(valData).values, axis=1)
        else:
            valPredResult = model.predict(valData)[:, 0]
        predResult['val'] = pd.DataFrame({params['cancer_col_name']: [i.split('|')[0] for i in valSample],
                                          params['drug_col_name']: [i.split('|')[1] for i in valSample],
                                          params['res_col_name']: valLabel,
                                          'Prediction': valPredResult}, index=valSample)
        predResult['val'].to_csv(params['output_dir'] + '/Prediction_Result_Val_' + params['train_task'] + '.txt', header=True,
                                 index=False, sep='\t', line_terminator='\r\n')

    backend.clear_session()

    pcc = predResult['val'].corr(method='pearson').loc['AUC', 'Prediction']
    scc = predResult['val'].corr(method='spearman').loc['AUC', 'Prediction']
    rmse = ((predResult['val']['AUC'] - predResult['val']['Prediction']) ** 2).mean() ** .5
    val_loss = history.history['val_loss'][-1]

    scores = {'val_loss':val_loss, 'pcc':pcc, 'scc':scc, 'rmse':rmse}
    
    print("\nIMPROVE_RESULT val_loss:\t{}\n".format(scores["val_loss"]))
    with open(Path(params.output_dir) / "scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    
    return history



def main():
    params = initialize_parameters()
    run(params)



if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()

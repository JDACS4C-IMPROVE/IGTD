import os
import numpy as np
import pandas as pd
import _pickle as cp
from pathlib import Path
from Modeling_Functions import CNN2D_Regressor, CNN2D_Classifier
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras import backend
from keras.models import load_model
from tensorflow.keras import backend as K

from improve import framework as frm

filepath = Path(__file__).resolve().parent

app_train_params = [
    {"name": "canc_col_name",
     "default": "improve_sample_id",
     "type": str,
     "help": "Column name that contains the cancer sample ids.",
     },
    {"name": "drug_col_name",
     "default": "improve_chem_id",
     "type": str,
     "help": "Column name that contains the drug ids.",
     }
]

model_train_params = [
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

train_params = model_train_params + app_train_params

req_train_args = [ll["name"] for ll in train_params]

req_train_args.extend(['model_outdir', 'y_col_name', 'train_ml_data_dir', 'val_ml_data_dir', 'model_file_name',
                       'model_file_format', 'data_format', 'learning_rate', 'conv', 'dropout', 'epochs', 'pool', 'dense',
                       'activation', 'loss', 'optimizer', 'verbose', 'batch_size', 'early_stop'])

metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]



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

    # ------------------------------------------------------
    # [Req] Create output dir for the model.
    # ------------------------------------------------------
    frm.create_outdir(outdir=params["model_outdir"])
    modelpath = str(frm.build_model_path(params, model_dir=params["model_outdir"]))

    # ------------------------------------------------------
    # [Req] Create data names for train and val
    # ------------------------------------------------------
    train_data_fname = frm.build_ml_data_name(params, stage="train")
    val_data_fname = frm.build_ml_data_name(params, stage="val")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    pkl_file = open(os.path.join(params['train_ml_data_dir'], train_data_fname), 'rb')
    temp_data = cp.load(pkl_file)
    pkl_file.close()
    trainData = temp_data['data']
    trainLabel = temp_data['label']
    trainSample = temp_data['sample']

    if 'val_ml_data_dir' in params.keys():
        pkl_file = open(os.path.join(params['val_ml_data_dir'], val_data_fname), 'rb')
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

    train_logger = CSVLogger(filename=params['model_outdir'] + '/log.csv')
    # model_saver = ModelCheckpoint(params['model_outdir'] + '/model.h5',
    model_saver = ModelCheckpoint(filepath=modelpath, monitor=monitor, save_best_only=True, save_weights_only=False)
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
    model = load_model(modelpath)
    predResult = {}

    if params['classification_task']:
        trainPredResult = np.argmax(a=model.predict(trainData).values, axis=1)
    else:
        trainPredResult = model.predict(trainData)[:, 0]
    predResult['train'] = pd.DataFrame({params['canc_col_name']: [i.split('|')[0] for i in trainSample],
                                        params['drug_col_name']: [i.split('|')[1] for i in trainSample],
                                        params['y_col_name']: trainLabel,
                                        'Prediction': trainPredResult}, index=trainSample)
    predResult['train'].to_csv(params['model_outdir'] + '/Prediction_Result_Train_' + params['train_task'] + '.txt', header=True,
                               index=False, sep='\t', line_terminator='\r\n')

    if valData is not None:
        if params['classification_task']:
            valPredResult = np.argmax(a=model.predict(valData).values, axis=1)
        else:
            valPredResult = model.predict(valData)[:, 0]
        predResult['val'] = pd.DataFrame({params['canc_col_name']: [i.split('|')[0] for i in valSample],
                                          params['drug_col_name']: [i.split('|')[1] for i in valSample],
                                          params['y_col_name']: valLabel,
                                          'Prediction': valPredResult}, index=valSample)
        predResult['val'].to_csv(params['model_outdir'] + '/Prediction_Result_Val_' + params['train_task'] + '.txt', header=True,
                                 index=False, sep='\t', line_terminator='\r\n')

    backend.clear_session()

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=valLabel, y_pred=valPredResult, stage="val",
        outdir=params["model_outdir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performace_scores(
        params,
        y_true=valLabel, y_pred=valPredResult, stage="val",
        outdir=params["model_outdir"], metrics=metrics_list
    )

    return val_scores



def main():
    # [Req]
    params = frm.initialize_parameters(
        filepath,
        default_model="IGTD_params.txt",
        additional_definitions=train_params,
        required=req_train_args
    )
    val_scores = run(params)
    print("\nFinished training GraphDRP model.")


if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()

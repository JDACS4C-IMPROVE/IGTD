import candle
import os
import numpy as np
import pandas as pd
import _pickle as cp
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
    {'name': 'classification_task',
     'type': bool,
     'help': 'Is the task classification or not'
     },
]

required = ['output_dir', 'test_data', 'data_url', 'candle_data_dir']


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
    os.environ['CANDLE_DATA_DIR'] = params['candle_data_dir']

    if 'test_data' in params.keys():
        candle.get_file(fname=params['test_data'],
                        origin=os.path.join(params['data_url'], params['test_data']),
                        unpack=False, md5_hash=None,
                        datadir=None,
                        cache_subdir='')
        pkl_file = open(os.path.join(params['candle_data_dir'], params['test_data']), 'rb')
        temp_data = cp.load(pkl_file)
        pkl_file.close()
        testData = temp_data['data']
        testLabel = temp_data['label']
        testSample = temp_data['sample']
    else:
        testData = None
        testLabel = None
        testSample = None

    model = load_model(params['output_dir'] + '/model.h5')
    predResult = {}
    if testData is not None:
        predResult['test'] = {}
        for s in testData.keys():
            if params['classification_task']:
                testPredResult = np.argmax(a=model.predict(testData[s]).values, axis=1)
            else:
                testPredResult = model.predict(testData[s])[:, 0]
            if testLabel[s] is not None:
                predResult['test'][s] = pd.DataFrame({params['cancer_col_name']: [i.split('|')[0] for i in testSample[s]],
                                                   params['drug_col_name']: [i.split('|')[1] for i in testSample[s]],
                                                   params['res_col_name']: testLabel[s],
                                                   'Prediction': testPredResult}, index=testSample[s])
            else:
                predResult['test'][s] = pd.DataFrame({params['cancer_col_name']: [i.split('|')[0] for i in testSample[s]],
                                                   params['drug_col_name']: [i.split('|')[1] for i in testSample[s]],
                                                   'Prediction': testPredResult}, index=testSample[s])
            predResult['test'][s].to_csv(params['output_dir'] + '/' + 'Test_' + s + '_Prediction_Result.txt',
                                         header=True, index=False, sep='\t', line_terminator='\r\n')

    backend.clear_session()

def main():
    params = initialize_parameters()
    run(params)



if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()




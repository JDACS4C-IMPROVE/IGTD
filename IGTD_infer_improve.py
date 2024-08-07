import os
import numpy as np
import pandas as pd
import _pickle as cp
from keras import backend
from keras.models import load_model
from tensorflow.keras import backend as K
# [Req] IMPROVE/CANDLE imports
from improve import framework as frm



# This should be set outside as a user environment variable
filepath = os.path.dirname(os.path.realpath(__file__))



app_infer_params = [
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

model_infer_params = [
    {'name': 'classification_task',
     'type': bool,
     'help': 'Is the task classification or not'
     },
    {'name': 'inference_task',
     'type': str,
     'help': 'Name of inference task'
     }
]

infer_params = model_infer_params + app_infer_params

req_infer_args = [ll["name"] for ll in infer_params]

req_infer_args.extend(['model_dir', 'y_col_name', 'test_ml_data_dir', 'model_file_name', 'model_file_format',
                       'data_format', 'infer_outdir'])

metrics_list = ["mse", "pcc", "scc", "r2"]



def run(params):

    # ------------------------------------------------------
    # [Req] Create output dir
    # ------------------------------------------------------
    frm.create_outdir(outdir=params["infer_outdir"])

    # ------------------------------------------------------
    # [Req] Create data name for test set
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_name(params, stage="test")

    pkl_file = open(os.path.join(params['test_ml_data_dir'], test_data_fname), 'rb')
    temp_data = cp.load(pkl_file)
    pkl_file.close()
    testData = temp_data['data']
    testLabel = temp_data['label']
    testSample = temp_data['sample']

    # [Req] Build model path
    modelpath = str(frm.build_model_path(params, model_dir=params["model_dir"]))
    model = load_model(modelpath)

    predResult = {}
    if params['classification_task']:
        testPredResult = np.argmax(a=model.predict(testData).values, axis=1)
    else:
        testPredResult = model.predict(testData)[:, 0]
    if testLabel is not None:
        predResult['test'] = pd.DataFrame({params['canc_col_name']: [i.split('|')[0] for i in testSample],
                                           params['drug_col_name']: [i.split('|')[1] for i in testSample],
                                           params['y_col_name']: testLabel,
                                           'Prediction': testPredResult}, index=testSample)
    else:
        predResult['test'] = pd.DataFrame({params['canc_col_name']: [i.split('|')[0] for i in testSample],
                                           params['drug_col_name']: [i.split('|')[1] for i in testSample],
                                           'Prediction': testPredResult}, index=testSample)
    predResult['test'].to_csv(params['infer_outdir'] + '/Prediction_Result_Test_' + params['inference_task'] + '.txt',
                              header=True, index=False, sep='\t', line_terminator='\r\n')

    backend.clear_session()

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=testLabel, y_pred=testPredResult, stage="test",
        outdir=params["infer_outdir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    test_scores = frm.compute_performace_scores(
        params,
        y_true=testLabel, y_pred=testPredResult, stage="test",
        outdir=params["infer_outdir"], metrics=metrics_list
    )

    return test_scores


def main():
    # [Req]
    params = frm.initialize_parameters(
        filepath,
        default_model="IGTD_params.txt",
        additional_definitions=infer_params,
        required=req_infer_args
    )
    test_scores = run(params)
    print("\nFinished model inference.")



if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()

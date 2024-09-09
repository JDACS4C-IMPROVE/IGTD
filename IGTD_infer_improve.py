import os
import numpy as np
import pandas as pd
import _pickle as cp
from keras import backend
from keras.models import load_model
from tensorflow.keras import backend as K
# [Req] IMPROVE/CANDLE imports
#from improve import framework as frm
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
import improvelib.utils as frm
from model_params_def import infer_params



# This should be set outside as a user environment variable
filepath = os.path.dirname(os.path.realpath(__file__))



def run(params):

    # ------------------------------------------------------
    # [Req] Create output dir
    # ------------------------------------------------------
    frm.create_outdir(outdir=params["output_dir"])

    # ------------------------------------------------------
    # [Req] Create data name for test set
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")

    pkl_file = open(os.path.join(params['input_data_dir'], test_data_fname), 'rb')
    temp_data = cp.load(pkl_file)
    pkl_file.close()
    testData = temp_data['data']
    testLabel = temp_data['label']
    testSample = temp_data['sample']

    # [Req] Build model path
    modelpath = str(frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["input_model_dir"]))
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
    predResult['test'].to_csv(params['output_dir'] + '/Prediction_Result_Test_' + params['inference_task'] + '.txt',
                              header=True, index=False, sep='\t', line_terminator='\r\n')

    backend.clear_session()

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=testLabel,
        y_pred=testPredResult,
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"])

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
            y_true=testLabel,
            y_pred=testPredResult,
            stage="test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )

    return test_scores


def main():
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="IGTD_params.txt",
        additional_definitions=infer_params)
    test_scores = run(params)
    print("\nFinished model inference.")



if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()
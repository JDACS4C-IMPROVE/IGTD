import pandas as pd
import numpy as np
import os
from pathlib import Path
import _pickle as cp
from Table2Image_Functions import min_max_transform, table_to_image, select_features_by_variation, \
    generate_unique_id_mapping, load_data
#from improve import framework as frm
#from improve import drug_resp_pred as drp
import multiprocessing

from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
import improvelib.utils as frm
import improvelib.applications.drug_response_prediction.drug_utils as drugs_utils
import improvelib.applications.drug_response_prediction.omics_utils as omics_utils
import improvelib.applications.drug_response_prediction.drp_utils as drp
from model_params_def import preprocess_params

filepath = Path(__file__).resolve().parent


def run(params):

    # [Req] Build paths and create output dir
    processed_outdir = frm.create_outdir(outdir=params["output_dir"])

    print("\nLoading omics data...")
    oo = omics_utils.OmicsLoader(params)
    print(oo)
    ge = oo.dfs['cancer_gene_expression.tsv']  # get the needed canc x data

    ############## temporary code ##############
    ge.index = ge.iloc[:, 0]
    ge = ge.iloc[:, 1:]
    #####################################

    print("\nLoading drugs data...")
    dd = drugs_utils.DrugsLoader(params)
    print(dd)
    md = dd.dfs['drug_mordred.tsv']  # get the needed drug x data

    rr_train = drp.DrugResponseLoader(params, split_file=params["train_split_file"], verbose=True)
    rr_val = drp.DrugResponseLoader(params, split_file=params["val_split_file"], verbose=True)
    rr_test = drp.DrugResponseLoader(params, split_file=params["test_split_file"], verbose=True)
    df_response = pd.concat((rr_train.dfs["response.tsv"], rr_val.dfs["response.tsv"], rr_test.dfs["response.tsv"]),
                            axis=0)

    if not os.path.exists(processed_outdir):
        os.makedirs(processed_outdir, exist_ok=True)

    # ################# temporary code ###############
    # ge = ge.iloc[:, :3000]
    # ############################################

    ge = ge.loc[np.unique(df_response.improve_sample_id), :]
    id = np.where(np.std(ge, axis=0) > 0)[0]
    ge = ge.iloc[:, id]
    ge_map, ge_unique_data = generate_unique_id_mapping(ge)
    ge_map.columns = ['CancID', 'Unique_CancID']
    ge_map.to_csv(os.path.join(processed_outdir, 'CancID_Mapping.txt'), header=True, index=False, sep='\t',
                  line_terminator='\r\n')
    ge_unique_data.to_csv(os.path.join(processed_outdir, 'Unique_CancID_Data.txt'), header=True, index=True, sep='\t',
                          line_terminator='\r\n')

    md = md.loc[np.unique(df_response.improve_chem_id), :]
    id = np.where(np.std(md, axis=0) > 0)[0]
    md = md.iloc[:, id]
    md_map, md_unique_data = generate_unique_id_mapping(md)
    md_map.columns = ['DrugID', 'Unique_DrugID']
    md_map.to_csv(os.path.join(processed_outdir, 'DrugID_Mapping.txt'), header=True, index=False, sep='\t',
                  line_terminator='\r\n')
    md_unique_data.to_csv(os.path.join(processed_outdir, 'Unique_DrugID_Data.txt'), header=True, index=True, sep='\t',
                          line_terminator='\r\n')

    # Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.
    data = pd.read_csv(os.path.join(processed_outdir, 'Unique_CancID_Data.txt'), low_memory=False, sep='\t', engine='c',
                       na_values=['na', '-', ''], header=0, index_col=0)
    if params['num_row'] * params['num_col'] > data.shape[1]:
        params['num_row'] = int(np.sqrt(data.shape[1]))
        params['num_col'] = int(np.sqrt(data.shape[1]))
    num = params['num_row'] * params['num_col']  # Number of features to be included for analysis, which is also the total number of pixels in image representation
    save_image_size = 1 + 16 / 10000 * num  # Size of pictures (in inches) saved during the execution of IGTD algorithm.
    fid = select_features_by_variation(data, variation_measure='var', threshold=None, num=num)
    data = data.iloc[:, fid]
    norm_data = min_max_transform(data.values)
    norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

    result_dir = os.path.join(processed_outdir, 'Image_Data', 'Cancer')
    os.makedirs(name=result_dir, exist_ok=True)
    # table_to_image(norm_data, [params['num_row'], params['num_col']], params['fea_dist_method'],
    #                params['image_dist_method'], save_image_size,
    #                params['max_step'], params['val_step'], result_dir, params['error'], min_gain=0.000001)
    proc1 = multiprocessing.Process(target=table_to_image, args=(norm_data, [params['num_row'], params['num_col']],
        params['fea_dist_method'], params['image_dist_method'], save_image_size, params['max_step'],
        params['val_step'], result_dir, params['error'], 0, 0.000001))
    proc1.start()

    # Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.
    data = pd.read_csv(os.path.join(processed_outdir, 'Unique_DrugID_Data.txt'), low_memory=False, sep='\t', engine='c',
                       na_values=['na', '-', ''], header=0, index_col=0)
    if params['num_row'] * params['num_col'] > data.shape[1]:
        params['num_row'] = int(np.sqrt(data.shape[1]))
        params['num_col'] = int(np.sqrt(data.shape[1]))
    num = params['num_row'] * params['num_col']  # Number of features to be included for analysis, which is also the total number of pixels in image representation
    save_image_size = 1 + 16 / 10000 * num  # Size of pictures (in inches) saved during the execution of IGTD algorithm.
    fid = select_features_by_variation(data, variation_measure='var', threshold=None, num=num)
    data = data.iloc[:, fid]
    norm_data = min_max_transform(data.values)
    norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

    result_dir = os.path.join(processed_outdir, 'Image_Data', 'Drug')
    os.makedirs(name=result_dir, exist_ok=True)
    # table_to_image(norm_data, [params['num_row'], params['num_col']], params['fea_dist_method'],
    #                params['image_dist_method'], save_image_size,
    #                params['max_step'], params['val_step'], result_dir, params['error'], min_gain=0.000001)
    proc2 = multiprocessing.Process(target=table_to_image, args=(norm_data, [params['num_row'], params['num_col']],
        params['fea_dist_method'], params['image_dist_method'], save_image_size, params['max_step'],
        params['val_step'], result_dir, params['error'], 0, 0.000001))
    proc2.start()
    proc1.join()
    proc2.join()

    cancer_image_data_filepath = os.path.join(processed_outdir, 'Image_Data', 'Cancer', 'Results.pkl')
    drug_image_data_filepath = os.path.join(processed_outdir, 'Image_Data', 'Drug', 'Results.pkl')
    cancer_id_mapping_filepath = os.path.join(processed_outdir, 'CancID_Mapping.txt')
    drug_id_mapping_filepath = os.path.join(processed_outdir, 'DrugID_Mapping.txt')
    stages = {"train": rr_train.dfs["response.tsv"],
              "val": rr_val.dfs["response.tsv"],
              "test": rr_test.dfs["response.tsv"]}
    for stage, res in stages.items():
        data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage=stage)
        data = load_data(res, cancer_image_data_filepath, drug_image_data_filepath, cancer_id_mapping_filepath,
                         drug_id_mapping_filepath, params['canc_col_name'], params['drug_col_name'], params['y_col_name'])
        output = open(os.path.join(processed_outdir, data_fname), 'wb')
        cp.dump(data, output, protocol=4)
        output.close()

        # [Req] Save y dataframe for the current stage
        res_to_save = pd.DataFrame({params['canc_col_name']: [i.split('|')[0] for i in data['sample']],
                                    params['drug_col_name']: [i.split('|')[1] for i in data['sample']],
                                    params['y_col_name']: data['label']}, index=None)
        frm.save_stage_ydf(ydf=res_to_save, stage=stage, output_dir=params["output_dir"])

    return processed_outdir



def main():
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="IGTD_params.txt",
        additional_definitions=preprocess_params)
    processed_outdir = run(params)
    print("\nFinished data preprocessing.")



# [Req]
if __name__ == "__main__":
    main()

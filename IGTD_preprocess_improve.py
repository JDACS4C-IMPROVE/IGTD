import pandas as pd
import sys
import numpy as np
import os
from pathlib import Path
# import candle
import _pickle as cp
from Table2Image_Functions import min_max_transform, table_to_image, select_features_by_variation, \
    generate_unique_id_mapping, load_data

from improve import framework as frm
from improve import drug_resp_pred as drp

filepath = Path(__file__).resolve().parent



# Model-specific params (Model: IGTD)
model_preproc_params = [
    {"name": "num_row",
     "type": int,
     "default": 50,
     "help": "Number of pixel rows in generated image.",
    },
    {"name": "num_col",
     "type": int,
     "default": 50,
     "help": "Number of pixel columns in generated image.",
    },
    {"name": "max_step",
     "type": int,
     "default": 30000,
     "help": "The maximum number of iterations to run the IGTD algorithm, if it does not converge.",
    },
    {"name": "val_step",
     "type": int,
     "default": 500,
     "help": "The number of iterations for determining algorithm convergence. If the error reduction rate.",
     },
    {"name": "fea_dist_method",
     "type": str,
     "choice": ["Pearson", "Spearman", "set"],
     "default": "Euclidean",
     "help": "Method used for calculating the pairwise distances between features.",
     },
    {"name": "image_dist_method",
     "type": str,
     "choice": ["Euclidean", "Manhattan"],
     "default": "Euclidean",
     "help": "Method used for calculating the distances between pixels in image.",
     },
    {"name": "error",
     "type": str,
     "choice": ["abs", "squared"],
     "default": "abs",
     "help": "Function for evaluating the difference between feature distance ranking and pixel distance ranking.",
     }
]



# App-specific params (App: drug response prediction)
# TODO: consider moving this list to drug_resp_pred.py module
drp_preproc_params = [
    {"name": "x_data_canc_files",  # app;
     # "nargs": "+",
     "type": str,
     "help": "List of feature files.",
    },
    {"name": "x_data_drug_files",  # app;
     # "nargs": "+",
     "type": str,
     "help": "List of feature files.",
    },
    {"name": "y_data_files",  # imp;
     # "nargs": "+",
     "type": str,
     "help": "List of output files.",
    },
    {"name": "canc_col_name",  # app;
     "default": "improve_sample_id",
     "type": str,
     "help": "Column name that contains the cancer sample ids.",
    },
    {"name": "drug_col_name",  # app;
     "default": "improve_chem_id",
     "type": str,
     "help": "Column name that contains the drug ids.",
    },
]

# gdrp_data_conf = []  # replaced with model_conf_params + drp_conf_params
# preprocess_params = model_conf_params + drp_conf_params
preprocess_params = model_preproc_params + drp_preproc_params

req_preprocess_args = [ll["name"] for ll in preprocess_params]  # TODO: it seems that all args specifiied to be 'req'. Why?

req_preprocess_args.extend(["y_col_name", "model_outdir"])  # TODO: Does 'req' mean no defaults are specified?

def run(params):

    params = frm.build_paths(params)  # paths to raw data
    processed_outdir = frm.create_ml_data_outdir(params)

    print("\nLoading omics data...")
    oo = drp.OmicsLoader(params)
    print(oo)
    ge = oo.dfs['cancer_gene_expression.tsv']  # get the needed canc x data

    print("\nLoading drugs data...")
    dd = drp.DrugsLoader(params)
    print(dd)
    md = dd.dfs['drug_mordred.tsv']  # get the needed drug x data

    rr_train = drp.DrugResponseLoader(params, split_file=params["train_split_file"], verbose=True)
    rr_val = drp.DrugResponseLoader(params, split_file=params["val_split_file"], verbose=True)
    rr_test = drp.DrugResponseLoader(params, split_file=params["test_split_file"], verbose=True)
    df_response = pd.concat((rr_train.dfs["response.tsv"], rr_val.dfs["response.tsv"], rr_test.dfs["response.tsv"]),
                            axis=0)

#    fdir = os.path.dirname(os.path.realpath(__file__))

    study = list(np.unique(df_response.source))

#    cancer_col_name = 'CancID'
#    drug_col_name = 'DrugID'
#    res_col_name = 'AUC'

    # input_data_path = os.path.join(fdir, 'Raw_Data')
    # output_data_dir = os.path.join(fdir, 'Processed_Data')

    # candle.get_file(fname='CSA_Data_July2020.tar.gz',
    #                 origin='https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/IGTD/CSA_Data_July2020.tar.gz',
    #                 unpack=True, md5_hash=None,
    #                 datadir=input_data_path,
    #                 cache_subdir='')
    #
    # input_data_path = os.path.join(input_data_path, 'CSA_Data_July2020')

    if not os.path.exists(processed_outdir):
        os.makedirs(processed_outdir, exist_ok=True)

    # # Load all data
    # data = {}
    # for s in study:
    #     data_dir = os.path.join(input_data_path, 'data.' + s)
    #     data[s] = {}
    #     data[s]['res'] = pd.read_csv(os.path.join(data_dir, 'rsp_' + s + '.csv'))  # Drug response
    #     data[s]['ge'] = pd.read_csv(os.path.join(data_dir, 'ge_' + s + '.csv'))  # Gene expressions
    #     data[s]['ge'].index = data[s]['ge'].iloc[:, 0]
    #     data[s]['ge'] = data[s]['ge'].iloc[:, 1:]
    #     data[s]['md'] = pd.read_csv(os.path.join(data_dir, 'mordred_' + s + '.csv'))  # Mordred descriptors
    #     data[s]['md'].index = data[s]['md'].iloc[:, 0]
    #     data[s]['md'] = data[s]['md'].iloc[:, 1:]
    #
    # # Combine all gene expression data and all drug data
    # ge = data[study[0]]['ge']
    # md = data[study[0]]['md']
    # for s in study[1:]:
    #     if np.sum(ge.columns != data[s]['ge'].columns) != 0:
    #         sys.exit('Column names of gene expressions do not match')
    #     if np.sum(md.columns != data[s]['md'].columns) != 0:
    #         sys.exit('Column names of drug descriptors do not match')
    #     ge = pd.concat((ge, data[s]['ge']), axis=0)
    #     md = pd.concat((md, data[s]['md']), axis=0)

    # Generate mappings to unique IDs for cancer case IDs and drug IDs
    # ge_map, ge_unique_data = generate_unique_id_mapping(ge.iloc[:, :1000]) # For generating example data
    ge_map, ge_unique_data = generate_unique_id_mapping(ge)
    ge_map.columns = ['CancID', 'Unique_CancID']
    ge_map.to_csv(os.path.join(processed_outdir, 'CancID_Mapping.txt'), header=True, index=False, sep='\t',
                  line_terminator='\r\n')
    ge_unique_data.to_csv(os.path.join(processed_outdir, 'Unique_CancID_Data.txt'), header=True, index=True, sep='\t',
                          line_terminator='\r\n')
    # md_map, md_unique_data = generate_unique_id_mapping(md.iloc[:, :1000]) # For generating example data
    md_map, md_unique_data = generate_unique_id_mapping(md)
    md_map.columns = ['DrugID', 'Unique_DrugID']
    md_map.to_csv(os.path.join(processed_outdir, 'DrugID_Mapping.txt'), header=True, index=False, sep='\t',
                  line_terminator='\r\n')
    md_unique_data.to_csv(os.path.join(processed_outdir, 'Unique_DrugID_Data.txt'), header=True, index=True, sep='\t',
                          line_terminator='\r\n')

    # Generate image data of gene expressions with unique IDs
    # num_row = 20    # Number of pixel rows in image representation # For generating example data
    # num_col = 20    # Number of pixel columns in image representation # For generating example data
    # num_row = 50  # Number of pixel rows in image representation
    # num_col = 50  # Number of pixel columns in image representation
    # max_step = 30000  # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
    # val_step = 500  # The number of iterations for determining algorithm convergence. If the error reduction rate
    # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

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

    # Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
    # distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
    # the pixel distance ranking matrix. Save the result in Test_1 folder.
    # fea_dist_method = 'Euclidean'
    # image_dist_method = 'Euclidean'
    # error = 'abs'
    result_dir = processed_outdir + '/Image_Data/Cancer/'
    os.makedirs(name=result_dir, exist_ok=True)
    table_to_image(norm_data, [params['num_row'], params['num_col']], params['fea_dist_method'],
                   params['image_dist_method'], save_image_size,
                   params['max_step'], params['val_step'], result_dir, params['error'], min_gain=0.000001)

    # Generate image data of drug descriptors with unique IDs
    # num_row = 20    # Number of pixel rows in image representation # For generating example data
    # num_col = 20    # Number of pixel columns in image representation # For generating example data
    # num_row = 37  # Number of pixel rows in image representation
    # num_col = 37  # Number of pixel columns in image representation
    # num = num_row * num_col  # Number of features to be included for analysis, which is also the total number of pixels in image representation
    # save_image_size = 1 + 16 / 10000 * num  # Size of pictures (in inches) saved during the execution of IGTD algorithm.
    # max_step = 30000  # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
    # val_step = 500  # The number of iterations for determining algorithm convergence. If the error reduction rate
    # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

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

    # Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
    # distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
    # the pixel distance ranking matrix. Save the result in Test_1 folder.
    # fea_dist_method = 'Euclidean'
    # image_dist_method = 'Euclidean'
    # error = 'abs'
    result_dir = processed_outdir + '/Image_Data/Drug/'
    os.makedirs(name=result_dir, exist_ok=True)
    table_to_image(norm_data, [params['num_row'], params['num_col']], params['fea_dist_method'],
                   params['image_dist_method'], save_image_size,
                   params['max_step'], params['val_step'], result_dir, params['error'], min_gain=0.000001)

    # To generate matched gene expression, drug descriptor, and response data
    for s in study:
        cancer_table_data_filepath = os.path.join(input_data_path, 'data.' + s, 'ge_' + s + '.csv')
        drug_table_data_filepath = os.path.join(input_data_path, 'data.' + s, 'mordred_' + s + '.csv')
        response_data_filepath = os.path.join(input_data_path, 'data.' + s, 'rsp_' + s + '.csv')
        cancer_image_data_filepath = os.path.join(output_data_dir, 'Image_Data', 'Cancer', 'Results.pkl')
        drug_image_data_filepath = os.path.join(output_data_dir, 'Image_Data', 'Drug', 'Results.pkl')
        cancer_id_mapping_filepath = os.path.join(output_data_dir, 'CancID_Mapping.txt')
        drug_id_mapping_filepath = os.path.join(output_data_dir, 'DrugID_Mapping.txt')
        res = pd.read_csv(response_data_filepath)

        data = load_data(cancer_table_data_filepath, drug_table_data_filepath,
                         cancer_image_data_filepath, drug_image_data_filepath, cancer_id_mapping_filepath,
                         drug_id_mapping_filepath, cancer_col_name, drug_col_name, res_col_name,
                         response_data_filepath)
        output = open(os.path.join(output_data_dir, 'data_test_' + s + '.pkl'), 'wb')
        cp.dump(data, output, protocol=4)
        output.close()

        for f in range(10):
            train_id = pd.read_csv(
                os.path.join(input_data_path, 'data.' + s, 'splits', 'split_' + str(f) + '_tr_id')).values[:, 0]
            num_train = len(train_id)
            val_id = np.random.permutation(num_train)[:int(np.floor(num_train / 9))]
            val_id = train_id[val_id]
            train_id = np.setdiff1d(train_id, val_id)
            test_id = pd.read_csv(
                os.path.join(input_data_path, 'data.' + s, 'splits', 'split_' + str(f) + '_te_id')).values[:, 0]

            # Generate matched data of training set
            train_response_data_filepath = os.path.join(output_data_dir,
                                                        'response_train_' + s + '_split_' + str(f) + '.csv')
            res.iloc[train_id, :].to_csv(train_response_data_filepath, header=True, index=False, sep=',',
                                         line_terminator='\r\n')
            data = load_data(cancer_table_data_filepath, drug_table_data_filepath, cancer_image_data_filepath,
                             drug_image_data_filepath, cancer_id_mapping_filepath, drug_id_mapping_filepath,
                             cancer_col_name, drug_col_name, res_col_name, train_response_data_filepath)
            output = open(os.path.join(output_data_dir, 'data_train_' + s + '_split_' + str(f) + '.pkl'), 'wb')
            cp.dump(data, output, protocol=4)
            output.close()

            # Generate matched data of validation set
            val_response_data_filepath = os.path.join(output_data_dir,
                                                      'response_val_' + s + '_split_' + str(f) + '.csv')
            res.iloc[val_id, :].to_csv(val_response_data_filepath, header=True, index=False, sep=',',
                                       line_terminator='\r\n')
            data = load_data(cancer_table_data_filepath, drug_table_data_filepath, cancer_image_data_filepath,
                             drug_image_data_filepath, cancer_id_mapping_filepath, drug_id_mapping_filepath,
                             cancer_col_name, drug_col_name, res_col_name, val_response_data_filepath)
            output = open(os.path.join(output_data_dir, 'data_val_' + s + '_split_' + str(f) + '.pkl'), 'wb')
            cp.dump(data, output, protocol=4)
            output.close()

            # Generate matched data of testing set
            test_response_data_filepath = os.path.join(output_data_dir,
                                                       'response_test_' + s + '_split_' + str(f) + '.csv')
            res.iloc[test_id, :].to_csv(test_response_data_filepath, header=True, index=False, sep=',',
                                        line_terminator='\r\n')
            data = load_data(cancer_table_data_filepath, drug_table_data_filepath, cancer_image_data_filepath,
                             drug_image_data_filepath, cancer_id_mapping_filepath, drug_id_mapping_filepath,
                             cancer_col_name, drug_col_name, res_col_name, test_response_data_filepath)
            output = open(os.path.join(output_data_dir, 'data_test_' + s + '_split_' + str(f) + '.pkl'), 'wb')
            cp.dump(data, output, protocol=4)
            output.close()

    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    scaler = None
    for stage, split_file in stages.items():

        # ---------------------------------
        # [Req] Load response data
        # ------------------------
        rr = drp.DrugResponseLoader(params, split_file=split_file, verbose=True)
        # print(rr)
        df_response = rr.dfs["response.tsv"]
        # ------------------------

        # Retain (canc, drug) response samples for which omic data is available
        df_y, df_canc = drp.get_common_samples(df1=df_response, df2=ge,
                                               ref_col=params["canc_col_name"])
        print(df_y[[params["canc_col_name"], params["drug_col_name"]]].nunique())

        # Scale features using training data
        if stage == "train":
            # Scale data
            df_canc, scaler = scale_df(df_canc, scaler_name=params["scaling"])
            # Store scaler object
            if params["scaling"] is not None and params["scaling"] != "none":
                scaler_fpath = processed_outdir / params["scaler_fname"]
                joblib.dump(scaler, scaler_fpath)
                print("Scaler object created and stored in: ", scaler_fpath)
        else:
            # Use passed scikit scaler object
            df_canc, _ = scale_df(df_canc, scaler=scaler)

        # Sub-select desired response column (y_col_name)
        # And reduce response dataframe to 3 columns: drug_id, cell_id and selected drug_response
        df_y = df_y[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]]
        # Further prepare data (model-specific)
        xd, xc, y = compose_data_arrays(df_y, smi, df_canc, params["drug_col_name"], params["canc_col_name"])
        print(stage.upper(), "data --> xd ", xd.shape, "xc ", xc.shape, "y ", y.shape)

        # -----------------------
        # [Req] Create data and save in processed_outdir/stage + "_" + params["data_suffix"]
        # The execution of this step, depends the model.
        # -----------------------
        # Save the processed data as PyTorch dataset. Note:
        # 1. We ignore data_format because TestbedDataset() appends the file
        #    name automatically with .pt
        # 2. TestbedDataset() which inherits from torch_geometric.data.InMemoryDataset
        #    automatically creates dir called "processed" inside root and saves the file
        #    inside. This results in: [root]/processed/[data_fname],
        #    e.g., ml_data/processed/train_data.pt
        # import ipdb; ipdb.set_trace()
        data_fname = frm.build_ml_data_name(params, stage, data_format=None)  # e.g., data_fname = train_data.pt
        TestbedDataset(root=processed_outdir,
                       dataset=data_fname,
                       xd=xd,
                       xt=xc,
                       y=y,
                       smile_graph=smiles_graphs)

        # [Req] Save y data dataframe for the current stage
        # rs_tr.to_csv(Path(root)/"train_response.csv", index=False) # That's what we originally used
        y_data_fname = f"{stage}_{params['y_data_suffix']}.csv"
        df_y.to_csv(processed_outdir / y_data_fname, index=False)

    return processed_outdir


def main():
    params = frm.initialize_parameters(
        filepath,
        default_model="graphdrp_default_model.txt",
        additional_definitions=preprocess_params,
        required=req_preprocess_args,
    )
    processed_outdir = run(params)
    print("\nFinished GraphDRP pre-processing (transformed raw DRP data to model input ML data).")


if __name__ == "__main__":
    main()























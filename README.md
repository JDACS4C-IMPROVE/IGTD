# IGTD

This repository demonstrates how to use the [IMPROVE library v0.1.0-2024-09-27](https://jdacs4c-improve.github.io/docs/v0.1.0-alpha/) for building a drug response prediction (DRP) model using IGTD, and provides examples with the benchmark [cross-study analysis (CSA) dataset](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

This version, tagged as `v0.1.0-2024-09-27`, introduces a new API which is designed to encourage broader adoption of IMPROVE and its curated models by the research community.



## Dependencies
Installation instuctions are detialed below in [Step-by-step instructions](#step-by-step-instructions).

Conda `yml` file [environment.yml](./environment.yml)

ML framework:
+ [TensorFlow](https://www.tensorflow.org/) -- deep learning framework for building the prediction model

IMPROVE dependencies:
+ [IMPROVE v0.1.0-2024-09-27](https://jdacs4c-improve.github.io/docs/v0.1.0-alpha/)



## Dataset
Benchmark data for cross-study analysis (CSA) can be downloaded from this [site](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/).

The data tree is shown below:
```
csa_data/raw_data/
├── splits
│   ├── CCLE_all.txt
│   ├── CCLE_split_0_test.txt
│   ├── CCLE_split_0_train.txt
│   ├── CCLE_split_0_val.txt
│   ├── CCLE_split_1_test.txt
│   ├── CCLE_split_1_train.txt
│   ├── CCLE_split_1_val.txt
│   ├── ...
│   ├── GDSCv2_split_9_test.txt
│   ├── GDSCv2_split_9_train.txt
│   └── GDSCv2_split_9_val.txt
├── x_data
│   ├── cancer_copy_number.tsv
│   ├── cancer_discretized_copy_number.tsv
│   ├── cancer_DNA_methylation.tsv
│   ├── cancer_gene_expression.tsv
│   ├── cancer_miRNA_expression.tsv
│   ├── cancer_mutation_count.tsv
│   ├── cancer_mutation_long_format.tsv
│   ├── cancer_mutation.parquet
│   ├── cancer_RPPA.tsv
│   ├── drug_ecfp4_nbits512.tsv
│   ├── drug_info.tsv
│   ├── drug_mordred_descriptor.tsv
│   └── drug_SMILES.tsv
└── y_data
    └── response.tsv
```




## Model scripts and parameter file
+ `IGTD_preprocess_improve.py` - takes benchmark data files and transforms into files for training and inference
+ `IGTD_train_improve.py` - trains the IGTD model
+ `IGTD_infer_improve.py` - runs inference with the trained IGTD model
+ `model_params_def.py` - definitions of parameters that are specific to the model
+ `IGTD_params.txt` - default parameter file (parameter values specified in this file override the defaults)



# Step-by-step instructions

### 1. Clone the model repository
```bash
git clone https://github.com/JDACS4C-IMPROVE/IGTD
cd IGTD
git checkout v0.1.0-2024-09-27
```


### 2. Set computational environment
Create conda env using `yml`
```bash
conda env create -f environment.yml -n IGTD
conda activate IGTD
```



### 3. Run `setup_improve.sh`.
```bash
source setup_improve.sh
```

This will:
1. Download cross-study analysis (CSA) benchmark data into `./csa_data/`.
2. Clone IMPROVE repo (checkout `develop`) outside the IGTD model repo
3. Set up `PYTHONPATH` (adds IMPROVE repo).


### 4. Preprocess CSA benchmark data (_raw data_) to construct model input data (_ML data_)
```bash
python IGTD_preprocess_improve.py --input_dir ./csa_data/raw_data --output_dir exp_result
```

Preprocesses the CSA data and creates train, validation (val), and test datasets.

Generates:
* three model input data files: `train_data.pkl`, `val_data.pkl`, `test_data.pkl`
* three tabular data files, each containing the drug response values (i.e. AUC) and corresponding metadata: `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`

```
ml_data
└── GDSCv1-GDSCv1
    └── split_4
        ├── Image_Data
        ├── CancID_Mapping.txt
        ├── Unique_CancID_Data.txt
        ├── Unique_DrugID_Data.txt  
        ├── DrugID_Mapping.txt 
        ├── test_y_data.csv  
        ├── train_y_data.csv
        ├── val_y_data.csv        
        ├── test_data.pkl  
        ├── train_data.pkl     
        └── val_data.pkl
```


### 5. Train IGTD model
```bash
python IGTD_train_improve.py --input_dir exp_result --output_dir exp_result
```

Trains IGTD using the model input data: `train_data.pkl` (training), `val_data.pkl` (for early stopping).

Generates:
* trained model: `model.pt`
* predictions on val data (tabular data): `val_y_data_predicted.csv`
* prediction performance scores on val data: `val_scores.json`
```
out_models
└── GDSCv1
    └── split_4
        ├── model.h5
        ├── log.csv
        ├── Prediction_Result_Train_.txt  
        ├── Prediction_Result_Val_.txt
        ├── val_scores.json
        └── val_y_data_predicted.csv
```


### 6. Run inference on test data with the trained model
```bash
python IGTD_infer_improve.py --input_data_dir exp_result --input_model_dir exp_result --output_dir exp_result --calc_infer_score true
```

Evaluates the performance on a test dataset with the trained model.

Generates:
* predictions on test data (tabular data): `test_y_data_predicted.csv`
* prediction performance scores on test data: `test_scores.json`
```
out_infer
└── GDSCv1-GDSCv1
    └── split_4
        ├── Prediction_Result_Test_.txt
        ├── test_scores.json
        └── test_y_data_predicted.csv
```
# Raw data
The raw data are located at https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/IGTD/CSA_Data_July2020.tar.gz
There are four types of raw data, including whole transcriptome data of cell lines, numeric descriptors of drug calculated using the Mordred package, the area under the dose response curve (AUC), and data partition files for cross-study analysis. The data files are organized into five folder, each of which is for one of the five drug screening studies including CCLE, CTRP, GDSC1, GDSC2, and GCSI. Take CCLE study as an example. The following data files and subfolder exist in the folder.
1) `ge_ccle.csv` is the whole transcriptome data of cancer cell lines.
2) `mordred_ccle.csv` is the drug descriptor data file.
3) `rsp_ccle.csv` is the drug response data file
4) `splits` subfolder includes index files for data partition of 10 cross-validation trials.


# ML data
The processed ML data are located at https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/IGTD/Processed_Data/ The script to process raw data and generate ML data is at https://github.com/JDACS4C-IMPROVE/IGTD/blob/develop/Preprocessing_for_CSA.py
The script `Preprocessing_for_CSA.py` first download raw data from the FTP site. It combines cell line gene expression data and drug descriptor data across studies to identify the unique cell lines and drug descriptors. It then converts gene expression profiles of unique cell lines and drug descriptor profiles of unique drugs into their respective image representations. Finally, it generates ML data according to the data partition files using image representations. The script will generate all data files required for the whole cross-study analysis. Three data files for one cross-validation trial have been uploaded to https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/IGTD/Processed_Data/ to demo the utility of drug response modeling scripts. The three data files are
data_train_ccle_split_0.pkl: includes gene expressions, drug descriptors, response values, and names of cell lines and drugs, for samples in the testing set.
data_val_ccle_split_0.pkl: includes gene expressions, drug descriptors, response values, and names of cell lines and drugs, for samples in the validation set. 
data_test_ccle_split_0.pkl: includes gene expressions, drug descriptors, response values, and names of cell lines and drugs, for samples in the testing set.  


# Using your own data
To apply IGTD for your own data, two steps are needed. First, convert genomic profiles of cancer models and drug feature profiles into respective image representations. Second, use the CNN model functions/scripts to perform response modeling and prediction.

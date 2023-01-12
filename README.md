# SANER2023_CPDPwithFL

## Abstract
This replication package includes the datasets, scripts, and additional_results.

## Additional results
./additinal_results This directory includes additional results.
/additional_results/RQ2/ This directory includes the results of four datasets (AEEEM, METRICSREPO, RELINK and SOFTLAB), which are the RQ2 results plus the SOFTLAB dataset.

## Dataset
./datasets This directory includes the dataset.

## Script
./scripts This directory includes scripts that run each of the RQ1 models.

### Required tools and packages
- pandas==1.3.4
- pyper==1.1.2
- scikit-learn==1.0.2
- tensorflow==2.5.0
- tensorflow-federated==0.20.0
- lace==2.1.2

### Config
The script can be configured with several parameters. These can be changed by editing the **scripts/config.py** file.

Option | Description
------------ | -------------
BATCH_SIZE  | Batch size of data set.
SHUFFLE_BUFFER | Width to shuffle the data set.
NUM_MODELS | Number of models to evaluate.
NUM_EPOCHS  | Number of epochs in learning
NUM_ROUNDS | Number of client-server interactions in the FLR model
SERVER_LEARNING_RATE | The learning rate of weights to the Server model.
CLIENT_LEARNING_RATE | The learning rate of weights to the Client model.

### Execute
Execute the following:
```bash
python cpdp_FLR-SL-LACE2.py
python cpdp_UL.py
```
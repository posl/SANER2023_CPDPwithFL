# Towards Privacy Preserving Cross Project Defect Prediction with Federated Learning - SANER2023

Hiroki Yamamoto, Dong Wang, Gopi Krishnan Rajbahadur, Masanari Kondo, Yasutaka Kamei and Naoyasu Ubayashi

Kyushu University, Fukuoka, Japan

Huawei Technologies Canada Co., Ltd., Canada

## Abstract
Defect prediction models can predict defects in software projects, and many researchers study defect prediction models to assist debugging efforts in software development. In recent years, there has been growing interest in Cross Project Defect Prediction (CPDP), which predicts defects in a project using a defect prediction model learned from other projects’ data when there is insufficient data to construct a defect prediction model. Since CPDP uses other projects’ data, data privacy preservation is one of the most significant issues. However, prior CPDP studies still require data sharing among projects to train models, and do not fully consider protecting project confidentiality. To address this, we propose a CPDP model FLR employing federated learning, a distributed machine learning approach that does not require data sharing. We evaluate FLR, using 25 projects, to investigate its effectiveness and feature interpretation. Our key results show that first, FLR outperforms the existing privacy-preserving methods (i.e., LACE2). Meanwhile, the performance is relatively comparable to the conventional methods (e.g., supervised and unsupervised learning). Second, the results of the interpretation analysis show that scale-related features have a common effect on the prediction performance of the FLR. In addition, further insights demonstrate that parameters of federated learning (e.g., learning rates and the number of clients) also play a role in the performance. This study is served as a first step to confirm the feasibility of the employment of federated learning in CPDP to ensure privacy preservation and lays the groundwork for future research on applying other machine learning models to federated learning.

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

## Citing the Work
```bibtex
@inproceedings{Yamamoto2023saner,
  author = {Hiroki Yamamoto, Dong Wang, Gopi Krishnan Rajbahadur, Masanari Kondo,
Yasutaka Kamei, Naoyasu Ubayashi},
  title = {Towards Privacy Preserving Cross Project Defect Prediction with Federated Learning},
  booktitle = {2023 IEEE International Conference on Software Analysis, Evolution and Reengineering},
  year = {2023},
  pages = {xxx-xxx}
}
```
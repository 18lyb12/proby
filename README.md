# proby
![logo](static/images/proby_logo.png)

## Installation

First clone the repo and cd into the directory:
```shell
git clone https://github.com/18lyb12/proby.git
cd proby
```

Then install dependencies.
```shell
pip install -r requirements.txt
```

## Model Download

The proby models can be accessed from [some link](https://some link). Put the downloaded model files into `proby/models/model_1`, `proby/models/model_1.5`, and `proby/models/model_2` respectively.

## Evaluation

You can reproduce the metrics in the paper.

### Data Download

The test data used in the paper can be accessed from [some link](https://some link). Put the downloaded test data files into `proby/data/processed_data/`.

### Command
- Make sure you are under folder `proby`.
- Run the command `python -m proby.evaluation.model_1` to evaluate Model 1. The predication data, ROC curve and PR curve will be stored in `proby/data/processed_data`, the metrics will be printed out.
- Run the command `python -m proby.evaluation.model_15` to evaluate Model 1.5. The predication data, ROC curve and PR curve will be stored in `proby/data/processed_data`, the metrics will be printed out.
- Run the command `python -m proby.evaluation.model_2` to evaluate Model 2. The predication data and parity plot will be stored in `proby/data/processed_data`, the metrics will be printed out.

## Local Application
We also build a simple application.

### Command
- Make sure you are under folder `proby`.
- Run the command `python app.py`.
- Open your browser and go to http://127.0.0.1:5000/ to see the local web page.

### Description
There will be 3 pages:
- Page 1: Predict Smiles from Files

  You have option to select Method 1 or Method 2, and upload multiple ".xlsx" or ".csv" files with `SMILES` column (case-insensitive). Find the sample data in `proby/data/app_sample_data`. This process will end up generating the prediction data.
    - Method 1
        - Step 1: Generate Model 1 prediction data input. `SMILES` column is given from input data. We cross join the input data with N most common (absorption, emission) pairs which is derived by `chemfluo的数据集` and `下载数据+人工整理.xlsx`.
        - Step 2. Run Model 1 to get prediction data.
        - Step 3: Group by SMILES. Each SMILES will have N predictions on different (absorption, emission) pairs, we pick the maximum prediction score as the prediction score for the certain SMILES. Select the SMILES whose scores are above the threshold (default = 0.95).
        - Step 4: Run Model 2 to get prediction data.
        - Step 5: Once the process is done. You can find the prediction data in 2 ways:
          - Refresh the page and download the prediction data  from the web page.
          - Find the data in `proby/data/prediction_data/method1_output`.
    - Method 2
        - Step 1: Run Model 1.5 to get prediction data.
        - Step 2: Select the SMILES whose scores are above the threshold (default = 0.95).
        - Step 3: Run Model 2 to get prediction data.
        - Step 3: Once the process is done. You can find the prediction data in 2 ways:
          - Refresh the page and download the prediction data  from the web page.
          - Find the data in `proby/data/prediction_data/method2_output`.


- Page 2: Identify Substructures

  We leverage `chemprop`'s interpret method to predict the substructure.


- Page 3: Display SMILES

  Display multiple SMILES.
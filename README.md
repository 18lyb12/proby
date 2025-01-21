# Proby
![logo](static/images/zhuzhu.png)

## Model Description
We have 3 models in this repo.
- **Model 1** is a classification model. Inputs are SMILES, absorption, emission. The output is a boolean value indicating whether the SMILES is active or inactive under the specified absorption and emission.
- **Model 1.5** is a classification model. Input are SMILES. The output is a boolean value indicating whether the SMILES is active or inactive under any absorption and emission.
- **Model 2** is a regression model. Input are SMILES given the SMILES are active under any absorption and emission. Outputs are properties including Absorption max (nm), Emission max (nm), Lifetime (ns), Quantum yield, log(e/mol-1 dm3 cm-1), abs FWHM (cm-1), emi FWHM (cm-1), abs FWHM (nm), emi FWHM (nm).

## Data Preprocessing
According to your input data, you need to implement the data preprocessing code in a way that best suits your input data.
In this repo, we have the following three data sources:
- chemfluo的数据集 

  This data folder contains both positive and negative samples. However, each CSV file in this folder comes from a different laboratory, so the definition of "active" and "inactive" (column `category`) might vary. Therefore, we need to label the data using a unified standard. In the data, the column `score` represents an objective value. Based on the values in column `score`, we re-labeled the data using 20 as the threshold. Note that the threshold of 20 was chosen to align as closely as possible with the original `category` column in the data.

  This data will be used for Model 1 and Model 1.5.

  - **Model 1**. Data preprocessing is trivial.
  - **Model 1.5**. Group it by SMILES. Our goal is to label each SMILES while being given the properties of each SMILES for specific (absorption, emission) pairs. Our logic is as follows: for a SMILES, if it is positive for any (absorption, emission) pair, we label it as positive. Otherwise, we label it as negative only when the absorption values of all its data points appear in at least three out of the four intervals: 300–400, 400–500, 500–600, and 600–700, or when the emission values appear in at least three of these intervals.

    The reasoning behind this approach is that if the absorption or emission values do not span a sufficiently wide range (e.g., both absorption and emission values are only within the 300–400 interval), we cannot determine whether the SMILES would be positive in a different range, such as 600–700. Therefore, in such cases, we cannot label the SMILES accurately, and we need to discard it.

- 下载数据+人工整理.xlsx

  This dataset includes active SMILES that were downloaded and manually curated, along with their properties, including Absorption max (nm), Emission max (nm), Lifetime (ns), Quantum yield, log(e/mol-1 dm3 cm-1), abs FWHM (cm-1), emi FWHM (cm-1), abs FWHM (nm), emi FWHM (nm).

  In Model 1 and Model 1.5, it will be used as positive samples. It will also be used for Model 2.
  - **Model 1**. Data preprocessing is trivial. They are all positive data.
  - **Model 1.5**. Group it by SMILES. They are all positive data.
  - **Model 2**. These target properties have different scales. We have option to standardize them and ask Model 2 to train the scaled values.
    
### Run the code
- Make sure you are under folder `proby`.
- Run the command `python -m proby.data_preprocessing.model_1_15` to generate Model 1 and Model 1.5 data. The data will be stored in `data/processed_data`.
- Run the command `python -m proby.data_preprocessing.model_2` to generate Model 2 data. The data will be stored in `data/processed_data`.

## Training
Leverage `chemprop` library.

### Run the code
- Make sure you are under folder `proby`.
- Run the command `python -m proby.training.model_1` to train Model 1. The model will be stored in `model_candidates` and the model folder name will be `model_1_{timestamp}`.
- Run the command `python -m proby.training.model_15` to train Model 1.5. The model will be stored in `model_candidates` and the model folder name will be `model_1.5_{timestamp}`.
- Run the command `python -m proby.training.model_2` to train Model 2. The model will be stored in `model_candidates` and the model folder name will be `model_2_{timestamp}`.


## Evaluation
We run the evaluation on the test data generated in Data Preprocessing step.
- Model 1 and Model 1.5. We calculate
  - Accuracy
  - Classification Report including precision, recall, f1-score
  - Confusion Matrix
  - Receiver Operating Characteristic (ROC) AUC and plot the ROC curve
  - Precision-Recall (PR) AUC and PR curve
  - Top K precision
- Model 2. We calculate
  - Parity plot including MAE, RMSE, R^2 score

### Run the code
- Make sure you are under folder `proby`.
- Run the command `python -m proby.evaluation.model_1` to evaluate Model 1. The predication data will be stored in `data/processed_data`, the metrics will be printed out, the ROC curve and PR curve will be stored in `data/intermediate`.
- Run the command `python -m proby.evaluation.model_15` to evaluate Model 1.5. The predication data will be stored in `data/processed_data`, the metrics will be printed out, the ROC curve and PR curve will be stored in `data/intermediate`.
- Run the command `python -m proby.evaluation.model_2` to evaluate Model 2. The predication data will be stored in `data/processed_data`, the metrics will be printed out, the parity plot will be stored in `data/intermediate`.

## Local Deploy
We built a simple app. There will be 3 pages:
- Page 1: Predict Smiles from Files
  
  You have option to select Method 1 or Method 2, and upload multiple files with `SMILES` column (case-insensitive). This process will end up generating the prediction data.
  - Method 1
    - Step 1: Generate Model 1 prediction data input. `SMILES` column is given from input data. We cross join the input data with N most common (absorption, emission) pairs which is derived by `chemfluo的数据集` and `下载数据+人工整理.xlsx`.
    - Step 2. Run Model 1 to get prediction data.
    - Step 3: Group by SMILES. Each SMILES will have N predictions on different (absorption, emission) pairs, we pick the maximum prediction score as the prediction score for the certain SMILES. Select the SMILES whose scores are above the threshold.
    - Step 4: Run Model 2 to get prediction data.
  - Method 2
    - Step 1: Run Model 1.5 to get prediction data.
    - Step 2: Select the SMILES whose scores are above the threshold.
    - Step 3: Run Model 2 to get prediction data.
    
- Page 2: Identify Substructures

  We leverage `chemprop`'s interpret method to predict the substructure.
  
- Page 3: Display SMILES
  
  Display multiple SMILES.
  
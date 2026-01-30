# AI-Driven Acceleration of Fluorescence Probe Discovery

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

[Journal Link](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202515604)

## ABSTRACT
Fluorescence imaging probes are indispensable tools for clinical navigation and preclinical research. However, the discovery of target-specific probes is hampered by the scarcity of targetable fluorophore scaffolds, making the development process slow, costly, and heavily reliant on trial-and-error design. Here, we present a hybrid strategy that integrates AI with bioassays to accelerate the development of target-specific fluorescent probes. We developed an AI model (PROBY) based on over one million molecule entries from nine datasets, capable of identifying fluorescent molecules and predicting seven key photophysical properties. Applying PROBY to a library of 26,416 target-validated molecules, we identified thousands of candidates with both target affinity and favorable optical characteristics. Focusing on three clinically relevant targets (tau, BCL-2, and TDP-43), we validated AI-identified candidates and discovered PE859, obatoclax, and B3, which supported applications in spectral analysis, drug screening, pathological labeling, cell imaging, and ex vivo tumor imaging. Guided by PROBY, we chemically modify PE859, yielding two optimized derivatives (859-1 and 859-2). With improved photophysical properties, 859-2 enabled in vivo two-photon imaging of tau pathology in transgenic mice. This hybrid AI-bioassay strategy substantially broadens the accessible scaffold landscape for designing target-specific fluorescence probes and provides a scalable, efficient, and cost-effective framework for next-generation probe discovery.

## Installation

First clone the repo and cd into the directory:
```shell

git clone https://github.com/18lyb12/proby.git
cd proby
conda create -n proby python=3.8
conda activate proby
pip install -r requirements.txt

```



## Model Download

The proby models can be accessed from [weights](https://drive.google.com/drive/folders/1oEL6XBQZXhrMlU0YYn5407uNH8383KhW?usp=sharing). Put the downloaded model files into `proby/models/model_1` and `proby/models/model_2` respectively.

### Dataset Download

The test data used in the paper can be accessed from [Dataset](https://drive.google.com/drive/folders/1C8iZuA4S3rJsC5EsoYugVHmr6aYoIahx?usp=sharing). Put the downloaded test data files into `proby/data/processed_data/`.


## Evaluation

#### Reproducibility

Ensure you are in the `proby` directory.

#### Evaluate Model 1
```shell
python -m proby.evaluation.model_1
```

The predication data, ROC curve and PR curve will be stored in `proby/data/processed_data`, the metrics will be printed out.


#### Evaluate Model 2
```shell
python -m proby.evaluation.model_2
```

The predication data and parity plot will be stored in `proby/data/processed_data`, the metrics will be printed out.

## Inference

Prepare one or more `.xlsx` or `.csv` files containing a `SMILES` column (case-insensitive). Sample data can be found in `proby/data/app_sample_data`. This process will generate prediction results.

- Ensure you are in the `proby` directory.
- There are three ways to run the command, depending on the number and location of your input files:

  - **All files in the `input` folder:**

    Simply run:
    ```shell
    python -m proby.app.pipeline
    ```

  - **A single file located elsewhere:**

    Specify the file path:
    ```shell
    python -m proby.app.pipeline --file your_file_path
    ```
    (Ensure there are no spaces in the file path.)

  - **All files in a custom folder:**
    
    Specify the folder path:
    ```shell
    python -m proby.app.pipeline --folder your_folder_path
    ```
    (Ensure there are no spaces in the folder path.)

- The prediction results will be saved in the `output` folder.


## Local Application
We also build a simple application.

### Command
- Ensure you are in the `proby` directory.
- Run the command `python app.py`.
- You should see a message similar to:
  ```
  * Serving Flask app 'app'
  * Debug mode: on
  * Running on http://127.0.0.1:5000
  ```
- Click the link (usually http://127.0.0.1:5000), or copy and paste it into your preferred browser. You will see the local web page in your browser.

### Description

There will be 3 pages:

#### Page 1: Predict Smiles from Files

  You can upload one or more `.xlsx` or `.csv` files containing a `SMILES` column (case-insensitive). Sample data can be found in `proby/data/app_sample_data`. This process will generate prediction results.
  - Step 1: Generate Model 1 prediction data input. `SMILES` column is given from input data. We cross join the input data with N most common (absorption, emission) pairs which is derived by PubChem dataset, Sci. Data dataset, and manually collected dataset.
  - Step 2. Run Model 1 to get prediction data.
  - Step 3: Group by SMILES. Each SMILES will have N predictions on different (absorption, emission) pairs, we pick the maximum prediction score as the prediction score for the certain SMILES. Select the SMILES whose scores are above the threshold (default = 0.95).
  - Step 4: Run Model 2 to get prediction data.
  - Step 5: Once the process is done. You can find the prediction data in 2 ways:
    - Refresh the page and download the prediction data  from the web page.
    - Find the data in `proby/data/prediction_data/output`.

#### Page 2: Identify Substructures

  This page informs users about our capability to identify molecular substructures using [**chemprop**](https://github.com/chemprop/chemprop-v1-old-branches)'s interpret method. Direct predictions are currently disabled in the UI; please contact the listed emails for assistance.

#### Page 3: Display SMILES

  Display multiple SMILES.

## Acknowledgements

The project was built on many amazing open-source repositories: [chemprop](https://github.com/chemprop/chemprop) . We thank the authors and developers for their contributions.



## Issues
- Please open new threads or address all questions to biyuezhu@hospital.cqmu.edu.cn or  xiyue.wang.scu@gmail.com. 

## License
proby is made available under the GPLv3 License and is available for non-commercial academic purposes. 

### Citation
Please use below to cite this [paper](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202515604) if you find our work useful in your research.
```

@article{jiang2025ai,
  title={AI-Driven Acceleration of Fluorescence Probe Discovery},
  author={Jiang, Xuefeng and Li, Yanbo and Tian, Xue and Yang, Sen and Luo, Ruina and Zhou, Cenxing and Liu, Yuxuan and Hu, Jingying and Feng, Sen and Gan, Lu and others},
  journal={Advanced Science},
  pages={e15604},
  year={2025},
  publisher={Wiley Online Library}
}

```

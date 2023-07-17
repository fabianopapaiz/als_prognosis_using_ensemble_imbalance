# Ensemble-Imbalance-based classification for amyotrophic lateral sclerosis prognostic prediction: identifying short-survival patients at diagnosis


This code uses ensemble and imbalance learning approaches to improve identifying short-survival amyotrophic lateral sclerosis patients at diagnosis time. Furthermore, we utilized the SHAP framework to explain how the best model performed the patient classifications.  
The results of this work have been published in the research article "Ensemble-Imbalance-based classification for amyotrophic lateral sclerosis prognostic prediction: identifying short-survival patients at diagnosis" (Papaiz et al., 2023).

[LICENSE](LICENSE.md)

---
For those wanting to try it out, this is what you need:
1) A working version of Python (version 3.9+) and jupyter-notebook.
2) Install the following Python packages:
    - numpy (1.23.5)
    - pandas (1.5.3)
    - matplotlib (3.7.0)
    - seaborn (0.12.2)
    - scikit-learn (1.2.1)
    - imbalanced-learn (0.10.1)
    - shap (0.41.0) 
3) Download the patient data analyzed from the Pooled Resource Open-Access ALS Clinical Trials (PRO-ACT) website (https://ncri1.partners.org/ProACT)
    - Register and log in to the website
    - Access the `Data` menu and download the `ALL FORMS` dataset
    - Extract the zipped data file into the `01_raw_data` folder
    - The `01_raw_data` folder will contain the following CSV files
      
      ![raw_data_folder](https://github.com/fabianopapaiz/als_prognosis_using_ensemble_imbalance/assets/16102250/dc9c533d-8152-44f0-b0f4-5b9112f34e04)
      
4) Perform the Extract-Load-Transform (ETL) step:    
    - Start the `jupyter-notebook` environment 
    - Open and execute all code of the `02.01 - Preprocess raw data.ipynb` file, which is inside the `02_ETL` folder
    - After execution, the preprocessed data will be saved in the `03_preprocessed_data` and `04_data_to_analyze` folders

    ![preprocessed data](https://github.com/fabianopapaiz/als_prognosis_using_ensemble_imbalance/assets/16102250/b86b4ecd-1f3d-44b4-aa54-ceb1b8860f3f)

5) Perform the Machine Learning (ML) pipeline:
    - Open and execute, in order, all code (`*.ipynb`) in the `05_Train_Validate_Models` folder
    - These steps will:
        - Split the dataset into _Training_ and _Validation_ subsets 
        - Train and validate the ML models for both scenarios (_Single-Model_ and _Ensemble-Imbalance_)
           - NOTE: It can take a long time to accomplish (even days) because more than 30,000 models will be executed.
        - Save the performance results into CSV files in the `05_Train_Validate_Models/exec_results` folder

6) Execute the SHAP explanations over the model that reached the best performance for the _Ensemble-Imbalance_ scenario

---
Finally, please let us know if you have any comments or suggestions, or if you have questions about the code or the procedure (correspondence e-mail: `fabianopapaiz at gmail.com`). 

** If you use this code for your research please cite this paper: **

1) Papaiz F, Dourado MET, Valentim RAdM, de Morais AHF, Arrais JP. Ensemble-Imbalance-based classification for amyotrophic lateral sclerosis prognostic prediction: identifying short-survival patients at diagnosis. npj Digital Medicine. 2023. 

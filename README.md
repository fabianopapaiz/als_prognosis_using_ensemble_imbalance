# Ensemble-Imbalance-based classification for amyotrophic lateral sclerosis prognostic: identifying short-survival patients at diagnosis


This code uses ensemble and imbalance learning approaches to improve identifying short-survival amyotrophic lateral sclerosis patients at diagnosis time. Furthermore, we utilized the SHAP framework to explain how the best model performed the patient classifications.  
The results of this work have been published in the research article "XXXXXX XXX XXXXX" (Papaiz et al., 2023).


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

      
5) The data: Unfortunately the files (specially the trained network weights) are too large to be hosted here. So we have put them here: 1-  [trained network weights](https://stanford.box.com/s/7wtkx1fr77156uec8h8apqm9my0aevpi) 2-  [a sample of lensing images to demonstrate the tool](https://stanford.box.com/s/tb2lpk824kee22ah3gz5b50trbp30vyx) 3-  [and a few cosmic ray and artifact maps](https://stanford.box.com/s/hn6l82pkmhm65xsls6g7tcjq63blj8v7)


Please download these, untar them (e.g., tar xvfz CosmicRays.tar.gz), and place them inside the "data/" folder. So at the end inside your "data" folder you should have a folder called "CosmicRays", another called "SAURON_TEST", and a third called "trained_weights" (in addition to the two files that are already there). 

With these you're good to go. You can run through the ipython notebook and model your first lenses with neural nets! Congratulation! 


 


If you'd like to get your hands a bit more dirty:
1) The data we have provided here is just for fun. You can produce your own simulated data using this script: "src/Lensing_TrainingImage_Generator.m". This is a matlab script that contains all the required functions to generate simulated lenses. The only thing needed for this is a sample of unlensed images (e.g., from the GalaxyZoo, or GREAT3 datasets). 

2) You can use "init.py" to setup the models. Then with "single_model_predictions.py" you can get the predictions of a single network. Alternatively, after running "init.py". you can run "combo_prediction.py", to combine the 4 models (see the paper referenced above). If you'd like to train your own model, use "train.py". You can train the existing models (there're about 11 models defined in "ensai_model.py"), or just throw a bunch of layers together yourself and see if you come up with something magical. 

Slightly more documentation can be found here: http://ensai.readthedocs.io/en/latest/

Finally: We'd love to hear from you. Please let us know if you have any comments or suggestions, or if you have questions about the code or the procedure. 
Also, if you'd like to contribute to this project please let us know.

** If you use this code for your research please cite these two papers: **

1) Hezaveh, Perreault Levasseur, & Marshall 2017 
2) Perreault Levasseur, Hezaveh, & Wechsler, 2017 

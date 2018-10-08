README

The following is the structure of the directory:

1. nbarooah3-analysis.pdf: My Analysis/Report
2. code/
      - seeds/
            - contains all code pertaining to Seeds dataset including EM, ICA, Kurtosis, KM, KM w/ ICA/PCA/RP/IG, PCA and RP.
      - wdbc/
            - - contains all code pertaining to WDBC dataset including BIC (for EM), EM, Kurtosis, KM, KM w/ ICA/PCA/RP, NN w/ clustering + DR, PCA and RP.
3. data/
      - seeds/
            - contains the Seeds dataset.
            - all folders with WEKA results for each type of DR and clustering.
      - wdbc/
            - contains the WDBC dataset
            - all folders with WEKA results for each type of DR and clustering.
4. results/
         - contains all NN results (test accuracy, train time, train accuracy) with DR, clustering, DR + clustering. Also contains an Excel sheet that I used to clculate MSE, SSE, Variance in assessing the reconstruction of the projected data.
         - figures/
                 - each folder corresponds to diagrams relating to it (NN, clustering, DR).
                 - weka/
                      - contains figures obtained from WEKA experiments.


INSTRUCTIONS
1. To recreate the first 2 parts of the assignment, use the seeds/ and wdbc/ folders within data/ to access WEKA experiments/results.
2. To recreate the third part of the assignment, use the seeds/ and wdbc/ folders within data/ to access WEKA experiments/results. Also use the python code present in seeds/ and wdbc/ within code/.
3.  To recreate the fourth and fifth part of the assignment, go to code/wdbc/nn.py and run the method that you are interested in. There are methods to apply all 4 DR algorithms to WDBC and clustering + DR to WDBC as well.

Tools Used:
1. Weka
2. R
3. Python (Scikit-learn)
4. MATLAB
5. Google Sheets
6. Student Filters package for WEKA

Datasets:
1. Breast Cancer Wisconsin Diagnostic Dataset (WDBC) - https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
2. Seeds Dataset - https://archive.ics.uci.edu/ml/datasets/seeds
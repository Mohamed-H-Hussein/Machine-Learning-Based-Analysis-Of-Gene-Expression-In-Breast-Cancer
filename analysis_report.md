
# Machine Learning project : Detailed Analysis Report

This document summarizes each analytical step performed to process, normalize, select features, build, and optimize machine learning models on the breast cancer gene expression dataset [GSE10810](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE10810). The workflow spans data preprocessing, feature selection, model training, evaluation, and hyperparameter tuning, providing a reproducible pipeline for classification and biomarker discovery.

---

# Machine Learning-Based Analysis of Gene Expression Profiles in Breast Cancer

*Analysis Overview and Notebook Descriptions*   
**Prepared by:** Mohamed Hussein.   
**Date:** 2025-10-08

---

### 📘 Notebook 0: Load Required Libraries


* **Notebook File Name:** [Notebook_0_Load_Libraries.ipynb](scripts/Notebook_0_Load_Libraries.ipynb)
* **HTML Output:** [Notebook_0_Load_Libraries.html](https://mohamed-h-hussein.github.io/Machine-Learning-Based-Analysis-Of-Gene-Expression-In-Breast-Cancer/Notebook_0_Load_Libraries.html)
* **Input:** None (Python libraries loaded from environment)
* **Output:** Confirmation of library versions and Python/Jupyter environment setup
* **Figures:** None

**Description:**
This notebook initializes the Python environment by loading all libraries required for the full machine learning pipeline, including data handling, preprocessing, feature selection, classification modeling, evaluation, dimensionality reduction, and model saving/loading. This setup ensures a consistent computational environment for all downstream analyses.

**Loaded Library Categories:**

* **Data Handling:** pandas, numpy
* **Data Visualization:** matplotlib, seaborn
* **Preprocessing:** LabelEncoder, MinMaxScaler
* **Feature Selection:** SelectKBest, mutual_info_classif
* **Classification Models:** RandomForestClassifier, SVC
* **Performance Metrics:** classification_report, confusion_matrix, accuracy_score
* **Model Saving/Loading:** pickle, joblib
* **Train/Test Splitting & Cross-Validation:** train_test_split, StratifiedKFold, GridSearchCV
* **Dimensionality Reduction:** PCA

**Environment Confirmation:**

* **Python version:** 3.12.3
* **Jupyter Notebook version:** 7.0.8
* **pandas version:** 2.2.2
* **numpy version:** 1.26.4
* **scikit-learn version:** 1.4.2
* **matplotlib version:** 3.8.4
* **seaborn version:** 0.13.2

**Processing Steps:**

1. Import all required Python libraries for the analysis pipeline.
2. Print versions of each library to ensure reproducibility.
3. Confirm Python and Jupyter Notebook environment details.

> **Key Notes:**
>
> * Ensures all required libraries are loaded before data analysis.
> * Version confirmation prevents compatibility issues.
> * No figures generated in this step.


---


## 📘 Notebook 1: Data Exploration & Cleaning

**Jupyter Notebook file name and location:**
[Notebook_1_Data_Exploration_and_Cleaning.ipynb](scripts/Notebook_1_Data_Exploration_and_Cleaning.ipynb)

**HTML Output file:**
[Notebook_1_Data_Exploration_and_Cleaning.html](https://mohamed-h-hussein.github.io/Machine-Learning-Based-Analysis-Of-Gene-Expression-In-Breast-Cancer/Notebook_1_Data_Exploration_and_Cleaning.html)

---

### 🔹 Input

* **Data file:** [GSE10810_Expression_Matrix_cleaned.xls](data/GSE10810_Expression_Matrix_cleaned.xls)
* **Description:** Pre-cleaned gene expression matrix containing breast cancer and normal tissue samples.

---

### 🔹 Output

* **Cleaned data summary tables**
* **Updated expression matrix (ready for normalization)**
* **Visualizations:**

  * [boxplot_comparison.png](figures/boxplot_comparison.png)
  * [histogram_distribution.png](figures/histogram_distribution.png)
  * [correlation_heatmap.png](figures/correlation_heatmap.png)
  * [pca_plot.png](figures/pca_plot.png)
  * [class_distribution_pie.png](figures/class_distribution_pie.png)

---

### 🔹 Figures

| Figure                                                           | Description                                                                          |
| :--------------------------------------------------------------- | :----------------------------------------------------------------------------------- |
| [boxplot_comparison.png](figures/boxplot_comparison.png)         | Boxplots showing expression value distribution across samples before normalization.  |
| [histogram_distribution.png](figures/histogram_distribution.png) | Histograms of gene expression values revealing data spread and potential skewness.   |
| [correlation_heatmap.png](figures/correlation_heatmap.png)       | Heatmap visualizing pairwise correlations among selected genes.                      |
| [pca_plot.png](figures/pca_plot.png)                             | Principal Component Analysis plot showing sample grouping patterns.                  |
| [class_distribution_pie.png](figures/class_distribution_pie.png) | Pie chart confirming balanced class distribution between cancer and control samples. |

---

### 🔹 Description

This notebook performs a comprehensive exploration of the **GSE10810 cleaned expression dataset**, including statistical summaries, visualization of distribution patterns, and inspection of sample balance.
Data were read into Pandas, verified for missing values, and visualized through boxplots, histograms, and correlation heatmaps.
PCA analysis was used to identify clustering trends between cancer and normal samples, ensuring the dataset was ready for normalization and feature extraction in subsequent steps.

---

### 🔹 Processing Steps

1. Loaded the cleaned dataset [GSE10810_Expression_Matrix_cleaned.xls](data/GSE10810_Expression_Matrix_cleaned.xls) into a Pandas DataFrame.
2. Inspected data structure, sample counts, and class distribution.
3. Verified the absence of missing values and uniform feature dimensions.
4. Generated boxplots and histograms to assess data spread and potential outliers.
5. Computed the correlation matrix and visualized it as a heatmap.
6. Applied PCA for dimensionality reduction and visual assessment of sample grouping.
7. Exported summary statistics and figures to the respective [figures](figures/) and [results](results/) directories.

---

### 🧩 Notes

* Analysis started with the cleaned expression matrix file [GSE10810_Expression_Matrix_cleaned.xls](data/GSE10810_Expression_Matrix_cleaned.xls).
* Outlier patterns and low-variance genes were visually inspected using boxplots.
* PCA and heatmap indicated a clear separation trend between cancer and normal samples.
* **Takeaway:** Dataset successfully cleaned, validated, and prepared for normalization in [Notebook 2](scripts/Notebook_2_Preprocessing_Normalization.ipynb).

---


## 📘 Notebook 2: Preprocessing & Normalization

**Jupyter Notebook file name:**
[Notebook_2_Preprocessing_Normalization.ipynb](scripts/Notebook_2_Preprocessing_Normalization.ipynb)

**HTML Output file:**
[Notebook_2_Preprocessing_Normalization.html](https://mohamed-h-hussein.github.io/Machine-Learning-Based-Analysis-Of-Gene-Expression-In-Breast-Cancer/Notebook_2_Preprocessing_Normalization.html)

---

### 🔹 Input

* **Data file:** [GSE10810_Expression_Matrix_cleaned.csv](data/GSE10810_Expression_Matrix_cleaned.csv)
* **Description:** Gene expression matrix (58 samples × 20,825 genes) containing 27 normal and 31 cancer breast tissue samples.
* **Purpose:** Prepare dataset for machine learning by filtering low-variance genes and normalizing expression values.

---

### 🔹 Output


* **Filtered high-variance gene matrix (IQR-based):** [data_filtered_iqr.xls](results/data_filtered_iqr.xls)
* **MinMax normalized dataset:** [data_normalized_minmax.xls](results/data_normalized_minmax.xls)
* **PCA-transformed coordinates:** [pca_normalized_coordinates.xls](results/pca_normalized_coordinates.xls)
* **Visualizations:**

  * [histogram_distribution_normalized.png](figures/histogram_distribution_normalized.png)
  * [pca_plot_normalized.png](figures/pca_plot_normalized.png)

---

### 🔹 Figures

| Figure                                                                                 | Description                                                                                                           |
| :------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| [histogram_distribution_normalized.png](figures/histogram_distribution_normalized.png) | Histogram displaying the distribution of normalized gene expression values, confirming range scaling between 0 and 1. |
| [pca_plot_normalized.png](figures/pca_plot_normalized.png)                             | PCA scatter plot (2 components) showing sample separation between cancer and normal tissue after normalization.       |

---

### 🔹 Description

This notebook focuses on **data preprocessing and normalization** of the GSE10810 breast cancer expression dataset.
After verifying sample labels and dimensions, low-variance genes were removed using the **Interquartile Range (IQR)** method to retain the most informative features.
Subsequently, gene expression values were scaled using **MinMaxScaler** to a [0,1] range, ensuring comparability across features.
Dimensionality reduction via **PCA** was performed to visualize separation patterns between cancer and control samples, confirming the normalization’s effectiveness.

---

### 🔹 **Processing Steps**

1. Loaded the cleaned dataset `GSE10810_Expression_Matrix_cleaned.csv` and transposed it (samples as rows, genes as columns).
2. Added class labels — 27 Normal and 31 Cancer samples.
3. Separated input features (`X`) and target labels (`y`).
4. Applied IQR filtering to remove low-variance genes below the 25th percentile, retaining 15,619 genes.
5. Normalized all features using MinMaxScaler to the [0,1] range.
6. Generated visualizations (histogram and PCA) to assess normalization effects.
7. Saved outputs:

   * [data_normalized_minmax.xls](results/data_normalized_minmax.xls)
   * [pca_normalized_coordinates.xls](results/pca_normalized_coordinates.xls)

---

### 🧩 **Notes**

* PCA confirmed distinct clustering between cancer and normal samples.
* IQR filtering effectively reduced dimensionality while retaining biological signal.
* Normalization ensured consistent feature scaling for downstream analysis.
* **Takeaway:** Dataset successfully preprocessed and normalized — ready for **Feature Selection (Notebook 3).**


---

## 📘 Notebook 3: Feature Selection

**Jupyter Notebook file name:**
[Notebook_3_Feature_Selection.ipynb](scripts/Notebook_3_Feature_Selection.ipynb)

**HTML Output file:**
[Notebook_3_Feature_Selection.html](https://mohamed-h-hussein.github.io/Machine-Learning-Based-Analysis-Of-Gene-Expression-In-Breast-Cancer/Notebook_3_Feature_Selection.html)

---

### 🔹 Input

* **Data file:** [data_normalized_minmax.xls](results/data_normalized_minmax.xls)
* **Description:** Normalized breast cancer expression dataset generated from Notebook 2 (58 samples × 15,619 genes).
* **Purpose:** Select the most informative 50 genes using Mutual Information for machine learning classification.

---

### 🔹 Output

* Selected top 50 genes [selected_genes_top50.xls](results/selected_genes_top50.xls)
* Training and testing subsets (pickled files)
   * [X_train_selected.pkl](results/X_train_selected.pkl)
   * [X_test_selected.pkl](results/X_test_selected.pkl)
   * [y_train.pkl](results/y_train.pkl)
   * [y_test.pkl](results/y_test.pkl)
   * [selected_gene_names.pkl](results/selected_gene_names.pkl)
* Feature importance visualization
* **Visualizations:**

  * [feature_scores_barplot.png](figures/feature_scores_barplot.png)

---

### 🔹 Figures

| Figure                                                           | Description                                                                                                                                        |
| :--------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- |
| [feature_scores_barplot.png](figures/feature_scores_barplot.png) | Bar chart displaying Mutual Information scores across all genes in the training dataset, highlighting overall variability and selection threshold. |

---

### 🔹 Description

This notebook identifies the most discriminative genes from the **normalized GSE10810 dataset** using **Mutual Information (MI)**-based feature selection.
The approach retains genes that maximize dependency between expression values and class labels (Normal vs. Cancer).
A total of **50 top-ranking genes** were selected and saved for model development in the next notebook.
The process ensures reduced dimensionality while preserving biologically relevant signals useful for classification.

---

### 🔹 **Processing Steps**

1. Loaded normalized dataset generated from Notebook 2.
2. Split dataset into training (70%) and testing (30%) subsets using stratified sampling.
3. Applied `SelectKBest` with `mutual_info_classif` to select top 50 informative genes.
4. Extracted gene names and corresponding feature scores.
5. Visualized feature importance across all genes.
6. Exported results as:

   * [selected_genes_top50.xls](results/selected_genes_top50.xls)
   * [X_train_selected.pkl](results/X_train_selected.pkl)
   * [X_test_selected.pkl](results/X_test_selected.pkl)
   * [y_train.pkl](results/y_train.pkl)
   * [y_test.pkl](results/y_test.pkl)
   * [selected_gene_names.pkl](results/selected_gene_names.pkl)

---

### 🧩 **Notes**

* Feature selection reduced gene count from **15,619 → 50**, optimizing computational efficiency.
* Selected genes (e.g., *CD300LG, ANLN, CRTAP, MOCS1, PTCH1*) are known to have roles in cancer biology.
* The Mutual Information approach effectively captured nonlinear dependencies between features and labels.
* **Takeaway:** Dataset successfully filtered to the 50 most informative genes — ready for [Notebook_4_Model_Building_and_Evaluation.ipynb](scripts/Notebook_4_Model_Building_and_Evaluation.ipynb)


---

## 📘 Notebook 4: Model Building & Evaluation

**Jupyter Notebook file name:** [Notebook_4_Model_Building_Evaluation.ipynb](scripts/Notebook_4_Model_Building_Evaluation.ipynb)
**HTML Output file:** [Notebook_4_Model_Building_Evaluation.html](https://mohamed-h-hussein.github.io/Machine-Learning-Based-Analysis-Of-Gene-Expression-In-Breast-Cancer/Notebook_4_Model_Building_Evaluation.html) 

---

### 🔹 Input

**Data files:**

* [X_train_selected.pkl](results/X_train_selected.pkl)
* [X_test_selected.pkl](results/X_test_selected.pkl)
* [y_train.pkl](results/y_train.pkl)
* [y_test.pkl](results/y_test.pkl)
* [selected_gene_names.pkl](results/selected_gene_names.pkl)

**Description:**
Feature-selected datasets generated from **Notebook 3 (Feature Selection)** using the top 50 informative genes.
These inputs are used to train and evaluate classification models distinguishing breast cancer from normal samples.

---

### 🔹 Output

**Generated Files:**

* [confusion_matrices.png](figures/confusion_matrices.png)
* [top_20_marker_genes.xls](results/top_20_marker_genes.xls)
* [top_20_marker_genes.png](figures/top_20_marker_genes.png)

**Description:**
Outputs include model evaluation metrics (accuracy, confusion matrices, classification reports) and visualization of the top 20 most informative marker genes identified by Random Forest feature importance.

---

### 🔹 Figures

| Figure                                                     | Description                                                                                                      |
| :--------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- |
| [confusion_matrices.png](figures/confusion_matrices.png)   | Heatmaps showing confusion matrices for Random Forest and SVM classifiers, illustrating classification accuracy. |
| [top_20_marker_genes.png](figures/top_20_marker_genes.png) | Bar plot showing the top 20 genes ranked by Random Forest feature importance.                                    |

---

### 🔹 Description

This notebook focuses on training and evaluating **machine learning models** — specifically **Random Forest** and **Support Vector Machine (SVM)** — using the top 50 genes identified in **Notebook 3**.
Both models were trained with **random_state=42** to ensure reproducibility.
The training aimed to classify samples as either **Cancer** or **Normal**, and performance was evaluated using **accuracy**, **confusion matrices**, and **classification reports**, followed by identifying the **top 20 marker genes** via Random Forest feature importance.

---

### 🔹 Model Evaluation Results

| Model                                              | Accuracy | Cancer Precision | Cancer Recall | Normal Precision | Normal Recall |
| :------------------------------------------------- | :------- | :--------------- | :------------ | :--------------- | :------------ |
| Random Forest (n_estimators=1000, random_state=42) | **0.94** | 1.00             | 0.90          | 0.89             | 1.00          |
| SVM (kernel='linear', random_state=42)             | **0.94** | 1.00             | 0.90          | 0.89             | 1.00          |

**Key Observation:**
Both classifiers achieved **94% accuracy**, demonstrating consistent and balanced performance between cancer and normal classes.

---

### 🔹 Top Marker Genes (Feature Importance)

The **Random Forest** model identified the following top marker genes contributing most to classification accuracy:
**CD300LG**, **ANLN**, **CRTAP**, **MOCS1**, **PTCH1**, among others.

Results were exported as:

* [top_20_marker_genes.xls](results/top_20_marker_genes.xls)
* [top_20_marker_genes.png](figures/top_20_marker_genes.png)

---

### 🔹 Processing Steps

1. Loaded all selected feature and label files from **Notebook 3 outputs**.
2. Trained **Random Forest** (n_estimators=1000, random_state=42) and **SVM (linear kernel, random_state=42)** models using the top 50 selected genes.
3. Evaluated models with **accuracy**, **confusion matrices**, and **classification reports**.
4. Visualized confusion matrices using Seaborn heatmaps.
5. Extracted **top 20 marker genes** from Random Forest feature importances.
6. Exported evaluation plots and ranked gene lists to `/figures` and `/results` directories.

---

### 🧩 Notes

* Both classifiers achieved **0.94 accuracy** and showed identical confusion matrices.
* **Reproducibility** was ensured with `random_state=42` across all steps.
* **Top marker genes** reflect biologically relevant pathways in breast cancer progression.
* **Takeaway:** Successful model training and evaluation — the dataset is ready for **Notebook 5: Cross-Validation and Hyperparameter Tuning**.
 
---

## Notebook 5 – Cross-Validation & Hyperparameter Tuning

**Jupyter Notebook file:** [Notebook_5_CrossValidation_Hyperparameter_Tuning.ipynb](./Notebook_5_CrossValidation_Hyperparameter_Tuning.ipynb)
**HTML Output:** [Notebook_5_CrossValidation_Hyperparameter_Tuning.html](https://mohamed-h-hussein.github.io/Machine-Learning-Based-Analysis-Of-Gene-Expression-In-Breast-Cancer/Notebook_5_CrossValidation_Hyperparameter_Tuning.html) 

### 🔹 Input

* [X_train_selected.pkl](results/X_train_selected.pkl)
* [X_test_selected.pkl](results/X_test_selected.pkl)
* [y_train.pkl](results/y_train.pkl)
* [y_test.pkl](results/y_test.pkl)
* [selected_gene_names.pkl](results/selected_gene_names.pkl)


**Description:** Feature-selected gene-expression data (50 genes, 58 samples) generated from Notebook 3.
**Purpose:** Optimize model hyperparameters using 5-fold cross-validation for Random Forest and SVM classifiers.

---

### 🔹 Output

د
* [best_RF_model.pkl](results/best_RF_model.pkl)
* [best_SVM_model.pkl](results/best_SVM_model.pkl)
* [cv_model_performance_summary.xls](results/cv_model_performance_summary.xls)
* [model_comparison_cv.png](figures/model_comparison_cv.png)


---

### 🔹 Figures

| Figure                  | Description                                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------------- |
| [model_comparison_cv.png](figures/model_comparison_cv.png) | Bar plot comparing cross-validation accuracy between Random Forest and SVM after hyperparameter tuning. |

---

### 🔹 Description

This notebook fine-tunes the Random Forest and SVM models through grid search and 5-fold cross-validation to achieve optimal classification performance on the breast cancer gene expression dataset.
The best parameter configurations were determined for both models, and the optimized versions were saved for final evaluation and deployment.

---

### 🔹 Processing Steps

1. Loaded feature-selected datasets (from Notebook 3).
2. Defined cross-validation strategy (`StratifiedKFold`, 5 splits, `random_state = 42`).
3. Set parameter grids for Random Forest and SVM models (`n_estimators = 100, 200, 300` etc.).
4. Performed grid search using `GridSearchCV` for each model.
5. Evaluated model accuracy through cross-validation and selected optimal parameters.
6. Compared Random Forest and SVM performances visually.
7. Saved optimized models and summary results for final reporting.

---

### 🔹 Model Evaluation Results

| Model         | Best Parameters                                                             | Best CV Accuracy |
| ------------- | --------------------------------------------------------------------------- | ---------------- |
| Random Forest | `max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100` | **1.0**          |
| SVM           | `C=0.1, gamma='scale', kernel='linear'`                                     | **1.0**          |

---

### 🧩 Notes

* Cross-validation performed with `random_state = 42` for reproducibility.
* Both models achieved perfect accuracy (1.0), indicating a highly separable dataset with strong class-specific gene expression patterns.
* However, such perfect accuracy may also suggest **possible overfitting**, especially given the small sample size (58 samples).
* Future work should include:

  * **Independent validation** on an external dataset (e.g., another breast cancer expression cohort).
  * **Feature reduction** or **regularization** to confirm robustness of selected genes.
  * **Biological interpretation** of top-ranked genes to ensure model interpretability and translational relevance.
* Optimized models (`best_RF_model.pkl`, `best_SVM_model.pkl`) are ready for external validation or deployment.
* Final model choice will consider both statistical performance and biological interpretability.

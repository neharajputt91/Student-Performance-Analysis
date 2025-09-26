# Student-Performance-Analysis
# Student Performance Analysis - Machine Learning

## Project Overview
This repository contains the code, dataset, and report for a Machine Learning project analyzing student performance. The project aims to predict academic outcomes, identify key drivers of success, and provide insights for educators and policymakers. It uses a range of algorithms to model student data across demographic and academic factors.

**Team Members**:
- Vinayak Kumar Singh (23MCA1030)
- Neha Singh (23MCA1049)

**Subject**: Machine Learning Lab (PMCA507P/PMCA507L)  
**Guided By**: Dr. B. Saleena

## Abstract
This study analyzes student performance to develop reliable ML models that predict outcomes and highlight factors influencing success. By leveraging classical statistical methods, ensemble techniques, and deep learning, it provides actionable insights to address educational challenges in diverse populations.

## Problem Statement
Analyze a student performance dataset to build predictive models forecasting academic outcomes. The dataset includes demographics (gender, race, etc.) and scores (math, reading, writing). Models should predict performance and identify contributing factors for targeted interventions.

## Dataset
The dataset is `StudentsPerformance.csv` (1000 entries, 8 columns):
- **gender**: female/male
- **race**: group A/B/C/D/E
- **parental_education**: e.g., bachelor's degree, master's degree
- **lunch**: standard/free/reduced
- **test_preparation_course**: none/completed
- **math_score**: 0-100
- **reading_score**: 0-100
- **writing_score**: 0-100

Source: Included in this repo. No missing values; numerical scores are correlated (e.g., reading-writing ~0.95).

## Tools and Libraries
- Python
- Pandas (data manipulation)
- NumPy (numerical operations)
- Scikit-learn (ML algorithms, metrics)
- Matplotlib & Seaborn (visualizations)
- CatBoost (gradient boosting)

## Algorithms Used
### Classification
1. Logistic Regression
2. CatBoost Classifier
3. Decision Tree
4. K-Nearest Neighbors (KNN)

### Clustering
5. K-Means
6. Hierarchical Clustering
7. Mean Shift
8. DBSCAN

### Ensemble/Deep Learning
9. Random Forest
10. Artificial Neural Network (ANN)

## Methodology
1. **Data Preprocessing**: Handle missing values, label encode categoricals, select features.
2. **Model Training**:
   - Classification: Predict gender from scores.
   - Clustering: Group students by performance patterns.
   - Ensemble/Deep: Forecast performance across subjects.
3. **Evaluation**: Accuracy, Precision, Recall, F1-Score for classification; Silhouette Score for clustering.
4. **Visualization**: Confusion matrices, feature importance plots, cluster visualizations.

## Inferences
- Dataset has 1000 entries; females excel in reading/writing, males in math.
- K-Means clusters students into 3 groups based on scores.
- KNN, ANN, and Random Forest achieve high accuracy (87-91%) for gender prediction.
- Math score is the most important feature.

## Novelty
- Applies a wide range of ML techniques to one dataset.
- Comprehensive analysis including gender differences and clustering.
- Uses ensemble and deep learning for enhanced performance.
- Visualizations provide insights; code is structured for reusability.

## Results
### Classification Algorithms
| Algorithm            | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|----------------------|--------------|---------------|------------|--------------|
| Logistic Regression | 89.5        | 92.71        | 86.41     | 89.45       |
| CatBoost            | 88.5        | 91.67        | 85.44     | 88.44       |
| Decision Tree       | 84.0        | 85.71        | 80.41     | 82.98       |
| K-Nearest Neighbors | 91.0        | 88.35        | 93.81     | 91.00       |

### Clustering Algorithms
| Algorithm     | Silhouette Score |
|---------------|------------------|
| K-Means      | 40.54           |
| Hierarchical | 35.24           |
| Mean Shift   | 47.71           |
| DBSCAN       | 54.07           |

### Ensemble/Deep Learning
| Technique            | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|----------------------|--------------|---------------|------------|--------------|
| Random Forest       | 87.0        | 91.40        | 82.52     | 86.73       |
| Artificial Neural Network | 90.0   | 93.68        | 86.41     | 89.90       |

## Conclusion
The project demonstrates effective use of ML for student performance analysis. ANN and Logistic Regression perform best for gender classification; KNN is top overall (91% accuracy). Insights can guide educational strategies. Future work: Feature engineering, hyperparameter optimization, and additional data (e.g., attendance).

## Installation and Usage
1. Clone the repo: `git clone https://github.com/[YourUsername]/Student-Performance-Analysis-Machine-Learning.git`
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn catboost`
3. Run the notebook: Open `Student_Performance.ipynb` in Jupyter/Colab.
4. View report: Open `Student Performance Analysis Report.pdf`.

Colab Link: [https://colab.research.google.com/drive/1nCRQSJOYZ0d1_mSibGnYfW1q_MtWSv8q?usp=sharing](https://colab.research.google.com/drive/1nCRQSJOYZ0d1_mSibGnYfW1q_MtWSv8q?usp=sharing)

## License
MIT License - Feel free to use and modify.

## Contact
For questions: [Your Email] or open an issue.

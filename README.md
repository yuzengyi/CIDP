# Exploring Explainable Machine Learning Models for Corporate Investment Decision Prediction

### Project Overview
This project focuses on using explainable machine learning models to predict corporate investment decisions, specifically targeting the debt-to-asset ratio (D/A ratio) as a critical metric. The dataset contains 57,522 corporate data points from the CSMAR database between 2000 and 2022, including a subset of 5,154 data points from the computer and communication industries. The project employs multi-class decision trees, K-means clustering for data preprocessing, and a random forest model for feature selection. This study aims to provide interpretable models that help companies make informed investment decisions and offer a valuable educational framework for teaching data-driven decision-making.

### Files in Repository
- `data_first.xlsx`: The original dataset containing corporate data for analysis.
- `newtree.py`: The main script for training and evaluating decision tree models.
- `new_data_with_clusters_sorted.xlsx`: The dataset after clustering and adding labels, with balanced classes.
- `README.md`: This readme file.

### Installation and Setup
1. Clone the repository to your local machine.
   ```sh
   git clone <repository_link>
   cd jingji_analysis
   ```
2. Create and activate a virtual environment.
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install dependencies from `requirements.txt`.
   ```sh
   pip install -r requirements.txt
   ```

### Dataset Description
- **Source**: The dataset was collected from the CSMAR database (2000-2022).
- **Key Features**:
  - `SPIC`, `SPWS`, `LSSSE`, `Insolvent`, `LIC`, `BPI`, `Size`, `EM1`, `DER`: These features are used for model training.
  - **Target**: `Y` - Represents corporate investment decision categories based on the D/A ratio.

### Model Pipeline
1. **Data Preprocessing**
   - The dataset is balanced using upsampling techniques to address class imbalance.
   - K-means clustering is used to preprocess and group the debt-to-asset ratio data into meaningful categories.

2. **Model Training and Pruning**
   - A **Decision Tree Classifier** is trained with pruning (`ccp_alpha=0.01`) to prevent overfitting and enhance model interpretability.
   - **Random Forest** is used to identify and select the most relevant features impacting the D/A ratio.

3. **Model Evaluation**
   - **Train-Test Split**: Data is split in a 70:30 ratio for training and testing.
   - **Cross-Validation**: 5-fold cross-validation is used to assess model stability and performance.
   - **Metrics**: Accuracy, precision, recall, F1-score, and cross-validation scores are calculated to evaluate the model.

### Results and Performance
- **Accuracy on Test Set**: 99.93%
- **Cross-Validation Scores**: Mean cross-validation score is 99.95%, showing consistent performance across different data splits.
- **Interpretability**: The decision tree model visualized provides insight into the impact of different features, such as Debt-to-Equity Ratio and insolvency status, on investment decisions.

### Running the Code
- To execute the decision tree analysis, run the following command:
  ```sh
  python newtree.py
  ```
- The script will output model evaluation metrics, including accuracy, precision, recall, F1-score, and a graphical representation of the decision tree.

### Dependencies
- `pandas` for data manipulation
- `numpy` for numerical operations
- `scikit-learn` for machine learning models and evaluation
- `matplotlib` for visualization

### Insights and Conclusion
This project demonstrates how machine learning models can be used to predict corporate investment decisions effectively. The decision tree classifier, combined with clustering and feature selection, provides a comprehensive and interpretable solution. The use of explainable models like decision trees enables better insight into the critical factors affecting corporate investment, offering value not only for practical business decisions but also for educational purposes in teaching data analytics and AI collaboration.

### Future Work
- **Time Series Analysis**: Explore time series models to capture the temporal patterns in financial data.
- **Hybrid Models**: Combine machine learning with econometric methods to enhance robustness and generalizability.

### Contact
- **Author**: Yiwen Chen
- **Email**: yiwen_chen2024@163.com
- **Affiliation**: Nanjing Normal University, Nanjing, China


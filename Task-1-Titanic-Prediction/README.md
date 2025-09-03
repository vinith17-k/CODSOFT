# Task 1: Titanic Survival Prediction 🚢

## 📜 Project Description
This project involves building a complete machine learning pipeline to predict the survival of passengers on the Titanic. The goal is to use passenger data (e.g., age, class, sex) to train a model that can accurately determine whether a passenger would have survived or not.

## 🚀 Workflow
The project follows a comprehensive data science workflow:
* **Exploratory Data Analysis (EDA)**: Investigated the dataset to understand the relationships between different features and survival.
* **Advanced Feature Engineering**: Created new, more informative features like `FamilySize`, `IsAlone`, and extracted passenger `Title` from their names. Binned `Age` and `Fare` into discrete categories.
* **Data Preprocessing Pipeline**: Built a robust `scikit-learn` pipeline to handle missing values, scale numerical features, and one-hot encode categorical features.
* **Model Training**: Utilized a state-of-the-art `XGBoost Classifier` for the prediction task.
* **Hyperparameter Tuning**: Used `GridSearchCV` to find the optimal hyperparameters for the XGBoost model, maximizing its performance.
* **Model Evaluation & Interpretation**: Evaluated the final model using a classification report and a confusion matrix, and interpreted its decisions by visualizing the feature importances.

## 📂 Files in this Folder
* `predict.py`: The main Python script containing the entire data processing, training, and evaluation pipeline.
* `titanic.csv`: The dataset used for training and testing the model.

## ⚙️ How to Run
1.  Ensure you have Python installed.
2.  Install the required libraries by running the following command in your terminal:
    ```bash
    pip install pandas matplotlib seaborn scikit-learn xgboost
    ```
3.  Run the script from your terminal:
    ```bash
    python predict.py
    ```

## 📈 Results and Findings
* **Final Model**: A tuned `XGBoost Classifier`.
* **Accuracy**: The final model achieved a cross-validated accuracy of approximately **84%** on the training set and a similar accuracy on the unseen test set.
* **Key Findings**: The feature importance plot revealed that the model's predictions were most heavily influenced by:
    1.  A passenger's title (specifically `Title: Mr.`)
    2.  Socioeconomic status (`Class: 3rd`, `Fare`)
    3.  `Family Size`

This project demonstrates a complete, end-to-end machine learning process, from data cleaning and feature engineering to model tuning and interpretation.

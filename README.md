# Calories Burnt Prediction Using Machine Learning

## Project Overview
This project predicts the number of calories burnt during exercise based on user input features such as gender, age, height, weight, duration of exercise, heart rate, and body temperature. The project uses machine learning models for prediction and provides a simple GUI for user interaction.

## Features
- **Exploratory Data Analysis (EDA)**: Correlation heatmaps and pairplots to analyze relationships in the data.
- **Machine Learning Models**: Linear Regression, Random Forest, and XGBoost are implemented and compared.
- **Hyperparameter Tuning**: Grid search is used to find the best model parameters.
- **GUI Application**: A user-friendly graphical interface for predicting calories burnt.

## Project Structure
The project consists of the following files:

1. **`calorie_burnt_prediction.ipynb`**: Jupyter Notebook containing the complete code for data analysis, preprocessing, model training, evaluation, and GUI creation.
2. **`calories.csv`**: Dataset containing user information and the calories burnt.
3. **`exercise.csv`**: Dataset containing additional user exercise details.

## Dataset Description
The merged dataset includes the following columns:

| Column      | Description                              |
|-------------|------------------------------------------|
| `User_ID`   | Unique identifier for each user          |
| `Calories`  | Calories burnt during exercise           |
| `Gender`    | Gender of the user (male/female)         |
| `Age`       | Age of the user                          |
| `Height`    | Height of the user (in cm)               |
| `Weight`    | Weight of the user (in kg)               |
| `Duration`  | Duration of the exercise (in minutes)    |
| `Heart_Rate`| Heart rate during exercise               |
| `Body_Temp` | Body temperature during exercise         |

## Steps to Run the Project

### 1. Install Dependencies
Ensure you have the following Python libraries installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `tkinter`

You can install them using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 2. Run the Jupyter Notebook
Open the Jupyter Notebook file `calorie_burnt_prediction.ipynb` to:
- Explore the data
- Train machine learning models
- Evaluate model performance
- Save the best model as a pickle file (`pipeline.pkl`)

### 3. Launch the GUI Application
The GUI allows users to input details and predict calories burnt:
1. Ensure `pipeline.pkl` is created and saved by running the notebook.
2. Run the Python script containing the GUI code (available in the notebook).

### 4. Interact with the GUI
The GUI prompts users to:
- Select gender.
- Enter age, height, weight, exercise duration, heart rate, and body temperature.
- Click "Predict" to see the estimated calories burnt.

## Model Evaluation
The project evaluates three models: Linear Regression, Random Forest, and XGBoost. Model performance is compared using the following metrics:
- **R2 Score**: Explains the variance in the target variable (higher is better).
- **MAE**: Measures average error (lower is better).
- **RMSE**: Penalizes larger errors (lower is better).
- **MAPE**: Shows average percentage error (lower is better).

### Model Performance
| Model               | R2 Score | MAE   | RMSE  | MAPE   |
|---------------------|----------|-------|-------|--------|
| Linear Regression   | 0.967    | 8.44  | 11.49 | 0.292  |
| Random Forest       | 0.998    | 1.68  | 2.64  | 0.026  |
| XGBoost             | 0.999    | 1.19  | 1.66  | 0.025  |

## Results
The **XGBoost model** outperformed others and is saved as the final model in `pipeline.pkl`.

## Updates from Existing project
The updates to the models have led to notable improvements in performance. The XGBoost model saw a 0.056% increase in R2 score (from 0.99875 to 0.99931) and a 21.71% reduction in MAE (from 1.52 to 1.19). The Random Forest model showed a slight improvement in R2 (0.002%) and a small decrease in MAE by 0.60%. Additionally, RMSE and MAPE were introduced for a more comprehensive evaluation, with XGBoost achieving an impressive MAPE of 0.0249, demonstrating its superior accuracy for calorie predictions.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Tkinter Documentation](https://docs.python.org/3/library/tkinter.html)


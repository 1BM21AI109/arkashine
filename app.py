import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.api import OLS, add_constant
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Function to evaluate the model
def evaluate_model(X, y):
    metrics = {'R2_train': [], 'R2_test': [], 'RMSE_test': [], 'RPD': []}
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = OLS(y_train, add_constant(X_train)).fit()
        y_train_pred = model.predict(add_constant(X_train))
        y_test_pred = model.predict(add_constant(X_test))

        R2_train = r2_score(y_train, y_train_pred)
        R2_test = r2_score(y_test, y_test_pred)
        RMSE_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        RPD = y_test.std() / RMSE_test

        metrics['R2_train'].append(R2_train)
        metrics['R2_test'].append(R2_test)
        metrics['RMSE_test'].append(RMSE_test)
        metrics['RPD'].append(RPD)
    
    return metrics

# Function to select the best variables using an iterative process
def select_variables(X, y):
    best_vars = []
    remaining_vars = list(X.columns)
    best_R2 = -np.inf

    while remaining_vars:
        scores = []
        for var in remaining_vars:
            vars_to_test = best_vars + [var]
            X_subset = X[vars_to_test]
            model = OLS(y, add_constant(X_subset)).fit()
            scores.append((model.rsquared, var))
        
        scores.sort(reverse=True)
        if scores[0][0] > best_R2:
            best_R2 = scores[0][0]
            best_vars.append(scores[0][1])
            remaining_vars.remove(scores[0][1])
        else:
            break
    
    return best_vars

# Pearson correlation for variable selection in strategy 2
def get_highly_correlated_vars(data, target_column, n_vars):
    correlations = {}
    for col in reflectance_columns:
        correlations[col] = pearsonr(data[col], data[target_column])[0]
    sorted_vars = sorted(correlations, key=correlations.get, reverse=True)
    return sorted_vars[:n_vars]

# Main function to run the strategies
def run_strategies(X, y):
    # Set up KFold
    n_folds = 10
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Strategy 1: Iterative variable selection
    best_vars_strategy_1 = select_variables(X, y)
    X_selected_strategy_1 = X[best_vars_strategy_1]
    metrics_strategy_1 = evaluate_model(X_selected_strategy_1, y)

    # Strategy 2: Using the highest Pearson correlation variables
    best_vars_strategy_2 = get_highly_correlated_vars(data, Presin_column, len(best_vars_strategy_1))
    X_selected_strategy_2 = X[best_vars_strategy_2]
    metrics_strategy_2 = evaluate_model(X_selected_strategy_2, y)

    return best_vars_strategy_1, metrics_strategy_1, best_vars_strategy_2, metrics_strategy_2

# Main Streamlit app
def main():
    st.title('Soil Sampling Analysis')

    # Load data
    data = pd.read_csv('SoilSampling-AB - SahanaYuvraj (1).csv')
    reflectance_columns = [f'reflectance_{i+1}' for i in range(18)]
    Presin_column = 'Presin'
    data[Presin_column] = np.nan

    # Assuming we have some reference Presin values to start with (we'll create some synthetic values for the example)
    np.random.seed(42)
    data[Presin_column] = np.random.rand(len(data)) * 100  # Replace this with actual Presin values if available

    # Display the data
    st.subheader('Data')
    st.write(data)

    # Evaluate the models
    best_vars_strategy_1, metrics_strategy_1, best_vars_strategy_2, metrics_strategy_2 = run_strategies(data[reflectance_columns], data[Presin_column])

    # Display the results
    st.subheader('Results for Strategy 1 (Iterative Variable Selection):')
    st.write(f"Selected variables: {best_vars_strategy_1}")
    st.write(f"Average R2_train: {np.mean(metrics_strategy_1['R2_train'])}")
    st.write(f"Average R2_test: {np.mean(metrics_strategy_1['R2_test'])}")
    st.write(f"Average RMSE_test: {np.mean(metrics_strategy_1['RMSE_test'])}")
    st.write(f"Average RPD: {np.mean(metrics_strategy_1['RPD'])}")

    st.subheader('Results for Strategy 2 (Highest Pearson Correlation):')
    st.write(f"Selected variables: {best_vars_strategy_2}")
    st.write(f"Average R2_train: {np.mean(metrics_strategy_2['R2_train'])}")
    st.write(f"Average R2_test: {np.mean(metrics_strategy_2['R2_test'])}")
    st.write(f"Average RMSE_test: {np.mean(metrics_strategy_2['RMSE_test'])}")
    st.write(f"Average RPD: {np.mean(metrics_strategy_2['RPD'])}")

    # Plotting the results
    st.subheader('Metrics Plot')
    for key, metrics in zip(['Strategy 1', 'Strategy 2'], [metrics_strategy_1, metrics_strategy_2]):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Metrics for {key}')

        axs[0, 0].boxplot(metrics['R2_train'])
        axs[0, 0].set_title('R2_train')

        axs[0, 1].boxplot(metrics['R2_test'])
        axs[0, 1].set_title('R2_test')

        axs[1, 0].boxplot(metrics['RMSE_test'])
        axs[1, 0].set_title('RMSE_test')

        axs[1, 1].boxplot(metrics['RPD'])
        axs[1, 1].set_title('RPD')

        st.pyplot(fig)

    # Once the best variables are identified, the final model can be trained on the entire dataset
    best_vars = best_vars_strategy_1  # or best_vars_strategy_2 based on the results
    X_final = data[best_vars]
    model_final = OLS(data[Presin_column], add_constant(X_final)).fit()

    # Predicting Presin values for the entire dataset
    data['Predicted_Presin'] = model_final.predict(add_constant(X_final))

    # Save the predictions to a new CSV file
    data.to_csv('reflectance_data_with_predictions.csv', index=False)

    # Plotting actual vs. predicted Presin values
    st.subheader('Actual vs Predicted Presin')
    plt.figure(figsize=(10, 6))
    plt.scatter(data[Presin_column], data['Predicted_Presin'], color='blue', alpha=0.5)
    plt.plot([data[Presin_column].min(), data[Presin_column].max()], 
            [data['Predicted_Presin'].min(), data['Predicted_Presin'].max()], 
            color='red', lw=2)
    plt.xlabel('Actual Presin')
    plt.ylabel('Predicted Presin')
    plt.title('Actual vs Predicted Presin')
    st.pyplot()

    # Plotting actual vs. predicted Presin values with index
    st.subheader('Actual vs Predicted Presin with Index')
    plt.figure(figsize=(10, 6))
    plt.scatter(data.index, data[Presin_column], color='blue', alpha=0.5, label='Actual Presin')
    plt.scatter(data.index, data['Predicted_Presin'], color='red', alpha=0.5, label='Predicted Presin')
    plt.xlabel('Index')
    plt.ylabel('Presin')
    plt.title('Actual vs Predicted Presin')
    plt.legend()
    plt.grid(True)
    st.pyplot()

    # Calculate accuracy of the model
    st.subheader('Model Accuracy')
    y_actual = data[Presin_column]
    y_predicted = data['Predicted_Presin']

    # Mean Squared Error
    mse = mean_squared_error(y_actual, y_predicted)
    st.write(f"Mean Squared Error (MSE): {mse}")

    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")

    # Mean Absolute Error
    mae = mean

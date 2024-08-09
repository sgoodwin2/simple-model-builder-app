from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

df = pd.read_csv('data/dataset.csv')

@app.route('/')
def index():
    # Get column names
    columns = df.columns.tolist()
    return render_template('index.html', columns=columns)

@app.route('/result', methods=['POST'])
def result():
    # Get selected columns from the form
    selected_columns = request.form.getlist('columns')
    
    # Check if 'Temperature (C)' is in the selected columns
    if 'Temperature (C)' not in selected_columns:
        return render_template('result.html', error="Please select 'Temperature (C)' to predict.")

    # Ensure 'Temperature (C)' is included in the columns
    selected_columns.append('Temperature (C)')
    
    # Prepare data for modeling
    data = df[selected_columns].dropna()
    X = data.drop('Temperature (C)', axis=1)
    y = data['Temperature (C)']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy (mean squared error as a metric)
    mse = mean_squared_error(y_test, y_pred)
    
    return render_template('result.html', mse=mse)

if __name__ == '__main__':
    app.run(debug=True)

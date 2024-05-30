from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)

# Load dataset
file_path = 'bank_loan_data.xlsx'
df = pd.read_excel(file_path)

# Preprocess the dataset
df = df.dropna()
df = df.replace(to_replace=('#', '-'), value='O')
df = df.replace({'Gender': {'M': 0, 'F': 1, 'O': 2}})
df = df.replace({'Home Ownership': {'Home Mortage': 0, 'Home Owner': 1, 'Rent': 3}})
df = df.replace({'Personal Loan': {' ': 0}})

# Split data
X = df.drop(columns=['ID', 'Personal Loan', 'ZIP Code'], axis=1)
Y = df['Personal Loan']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the SVM model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Train using Gradient Boost Classifier model
gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, random_state=2)
gradient_boosting_model.fit(X_train, Y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        features = np.array([[
            data['Age'],
            data['Gender'],
            data['Experience'],
            data['Income'],
            data['Family'],
            data['CCAvg'],
            data['Education'],
            data['Home Ownership'],
            data['Mortgage'],
            data['Securities Account'],
            data['CD Account'],
            data['Online'],
            data['CreditCard']
        ]])
        
        prediction = gradient_boosting_model.predict(features)
        result = "Personal loan is Accepted" if prediction[0] == 1 else "Personal Loan is Denied"
        
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

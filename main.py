from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get values from the HTML form
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        # Create a DataFrame for the new user input
        new_data = pd.DataFrame({
            'age': [age], 
            'sex': [sex], 
            'bmi': [bmi], 
            'children': [children], 
            'smoker': [smoker], 
            'region': [region]
        })

        # Load original dataset to ensure consistent feature encoding (get_dummies)
        df_orig = pd.read_csv('insurance.csv')
        
        # Combine original data and new input, drop target column 'expenses'
        df_combined = pd.concat([df_orig.drop('expenses', axis=1), new_data], ignore_index=True)
        
        # Apply get_dummies to match training conditions
        df_encoded = pd.get_dummies(df_combined, drop_first=True, dtype=int)
        
        # Extract the last row (which represents our new user input)
        final_input = df_encoded.iloc[[-1]]

        # Make prediction
        prediction = model.predict(final_input)
        output = round(prediction[0], 2)

        return render_template('index.html', pred=f"Estimated Medical Insurance Cost: ${output}")
        
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

rfr = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
kmeans = joblib.load('kmeans.pkl')
feature_columns = joblib.load('feature_columns.pkl') 

data = pd.read_csv('House Price India.csv')

def preprocess_data_for_cluster(df):
    df['age_of_house'] = 2025 - df['Built Year']
    df['renovation_age'] = df.apply(lambda row: 2025 - row['Renovation Year'] if row['Renovation Year'] != 0 else 2025 - row['Built Year'], axis=1)
    df['total_area'] = df['living area']  
    X = df[['number of bedrooms', 'number of bathrooms', 'total_area', 'lot area', 'number of floors',
            'waterfront present', 'condition of the house', 'grade of the house', 'age_of_house',
            'renovation_age', 'Number of schools nearby', 'Distance from the airport', 'Postal Code']]
    postal_encoded = encoder.transform(X[['Postal Code']])
    postal_df = pd.DataFrame(postal_encoded, columns=encoder.get_feature_names_out(['Postal Code']))
    X_numeric = X.drop('Postal Code', axis=1)
    X_combined = pd.concat([X_numeric, postal_df], axis=1)
    X_combined = X_combined.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(X_combined)
    return X_scaled

data_cluster_input = preprocess_data_for_cluster(data)
data['cluster'] = kmeans.predict(data_cluster_input)

app = Flask(__name__)

def preprocess_input(user_input_df):
    user_input_df['age_of_house'] = 2025 - user_input_df['built_year']
    user_input_df['renovation_age'] = user_input_df.apply(
        lambda row: 2025 - row['renovation_year'] if row['renovation_year'] != 0 else 2025 - row['built_year'],
        axis=1)
    user_input_df['total_area'] = user_input_df['living_area']

    features = ['number of bedrooms', 'number of bathrooms', 'total_area', 'lot area', 'number of floors',
                'waterfront present', 'condition of the house', 'grade of the house', 'age_of_house',
                'renovation_age', 'Number of schools nearby', 'Distance from the airport', 'Postal Code']
    
    X = user_input_df[features]
    postal_encoded = encoder.transform(X[['Postal Code']])
    postal_df = pd.DataFrame(postal_encoded, columns=encoder.get_feature_names_out(['Postal Code']))
    X_numeric = X.drop('Postal Code', axis=1)
    X_combined = pd.concat([X_numeric, postal_df], axis=1)
    X_combined = X_combined.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(X_combined)
    return X_scaled

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = {
            'number of bedrooms': int(request.form['bedrooms']),
            'number of bathrooms': float(request.form['bathrooms']),
            'living_area': float(request.form['living_area']),
            'lot area': float(request.form['lot_area']),
            'number of floors': int(request.form['floors']),
            'waterfront present': int(request.form['metro']),
            'condition of the house': int(request.form['condition']),
            'grade of the house': int(request.form['grade']),
            'built_year': int(request.form['built_year']),
            'renovation_year': int(request.form['renovation_year']),
            'Number of schools nearby': int(request.form['schools']),
            'Distance from the airport': float(request.form['airport_distance']),
            'Postal Code': request.form['postal_code']
        }

        input_df = pd.DataFrame([user_input])

        X_input_scaled = preprocess_input(input_df)
        price = rfr.predict(X_input_scaled)[0]
        cluster = kmeans.predict(X_input_scaled)[0]

        
        similar_houses = data[data['cluster'] == cluster][
            ['number of bedrooms', 'number of bathrooms', 'living area', 'Price', 'Postal Code']
        ].to_dict('records')

        return render_template('result.html', price=price, similar_houses=similar_houses)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

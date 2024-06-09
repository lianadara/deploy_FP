import streamlit as st
import pandas as pd
import joblib
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the preprocessing function
def preprocess_data(df):
    df['date_out'] = pd.to_datetime(df['date_out'])

    df['month'] = df['date_out'].dt.month
    df['year'] = df['date_out'].dt.year

    with open('numeric_transformer.pkl', 'rb') as f:
        loaded_numeric_transformer = pickle.load(f)

    # Memetakan nilai-nilai review ke skala yang diinginkan
    review_mapping = {
        'Sangat Tidak Puas': 1,
        'Tidak Puas': 2,
        'Cukup Puas': 3,
        'Puas': 4,
        'Sangat Puas': 5
    }
    df['review_value'] = df['review_name'].map(review_mapping)
    df['name_gender_age'] = df['patient_name'] + '_' + df['gender'] + '_' + df['age'].astype(str)
    # print(df.columns.to_list())

    # Mengelompokkan dan menghitung total_pasien, revenue, dan profit
    new_df = df.groupby(['branch_name', 'month', 'year']).agg(
        jumlah_pasien=('name_gender_age', 'count'),
        avg_review=('review_value', 'mean'),
        cogs=('cogs', 'sum'),
        total_revenue=('revenue', 'sum'),
        total_profit=('profit', 'sum')
    ).reset_index()

    numeric_features = ['month', 'year', 'jumlah_pasien', 'avg_review', 'cogs', 'total_revenue', 'total_profit']
    new_data_scaled = pd.DataFrame(loaded_numeric_transformer.transform(new_df[numeric_features]),
                               columns=numeric_features)
    
    # new_data_scaled = new_data_scaled[['branch_name','jumlah_pasien','avg_review', 'total_revenue','total_profit']]
    new_data_scaled_with_branch = pd.concat([new_df[['branch_name']], new_data_scaled], axis=1)

    # print(new_data_scaled_with_branch)
    return new_data_scaled_with_branch

def predict_score(data):
    # Load the pipeline from .pkl file
    loaded_pipeline = joblib.load('model_linear.pkl')

    # Preprocess the input data
    processed_data = preprocess_data(data)

    # Extract branch names
    branch_names = processed_data['branch_name']

    # Drop branch_name from the data to be used for prediction
    prediction_data = processed_data.drop(columns=['branch_name','month','year','cogs'])
    
    # Make predictions using the loaded pipeline
    predictions = loaded_pipeline.predict(prediction_data)

    # Combine the branch names with the predictions
    result_df = pd.DataFrame({
        'branch_name': branch_names,
        'prediction': predictions[:,0]
    })
    return result_df

def run():
    st.title('Hospital Performance Index Score Prediction')
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)
        
        # Show the uploaded data
        st.write('Uploaded Data:')
        st.write(data)
        
        # Make predictions
        predictions = predict_score(data)
        
        # Show predictions
        st.write('Predictions:')
        st.write(predictions)

if __name__ == '__main__':
    run()

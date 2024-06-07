import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the preprocessing function
def preprocess_data(df):
    # Konversi kolom date_out menjadi tipe data datetime
    df['date_out'] = pd.to_datetime(df['date_out'])

    # Menambahkan kolom bulan dan tahun
    df['month'] = df['date_out'].dt.month
    df['year'] = df['date_out'].dt.year

    # Memetakan nilai-nilai review ke skala yang diinginkan
    review_mapping = {
        'Sangat Tidak Puas': 1,
        'Tidak Puas': 2,
        'Cukup Puas': 3,
        'Puas': 4,
        'Sangat Puas': 5
    }
    df['review_value'] = df['review_name'].map(review_mapping)

    # Mengelompokkan dan menghitung total_pasien, total_cost, dan total_revenue
    new_df = df.groupby(['branch_name', 'month', 'year']).agg(
        jumlah_pasien=('id_trx', 'count'),
        avg_review=('review_value', 'mean'),
        cogs=('cogs', 'sum'),
        total_cost=('total_cost', 'sum'),
        total_revenue=('revenue', 'sum')
    ).reset_index()

    # Normalisasi kolom jumlah_pasien, total_revenue, dan avg_review
    scaler = MinMaxScaler()
    new_df[['jumlah_pasien', 'total_revenue','avg_review']] = scaler.fit_transform(new_df[['jumlah_pasien', 'total_revenue','avg_review']])
    # new_df['avg_review'] = new_df['avg_review'] / 5.0  # Mengingatkan bahwa nilai-nilai review telah dipetakan ke skala 1-5

    return new_df

def predict_score(data):
    # Load the pipeline from .pkl file
    loaded_pipeline = joblib.load('model_pipeline.pkl')

    # Preprocess the input data
    processed_data = preprocess_data(data)

    # Make predictions using the loaded pipeline
    predictions = loaded_pipeline.predict(processed_data)

    return predictions

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

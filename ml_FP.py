import os

import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.callbacks import BaseCallback
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser


class StreamlitCallback(BaseCallback):
    def __init__(self, container) -> None:
        """Initialize callback handler."""
        self.container = container

    def on_code(self, response: str):
        self.container.code(response)


class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return

def load_data(path):
    df = pd.read_csv(path)
    return df

def get_column_descriptions():
    descriptions = """
    This dataset contains monthly data for Mulia Hospital across its different branches. 
    Each row represents data for a specific branch, recorded in a particular month and year. 
    The dataset includes various metrics related to patient numbers, reviews, financial performance, and a calculated performance score.

    Here are the descriptions of the columns in the dataset:
    - `branch_name`: The name of the hospital branch where the data was recorded. This helps identify which branch the data pertains to.
    - `month`: The month in which the data was collected. It is usually represented as a numeric value (e.g., 1 for January, 2 for February, etc.).
    - `year`: The year in which the data was recorded. This helps to identify the time period of the data.
    - `jumlah_pasien`: The total number of patients treated in the branch during the specified month and year.
    - `avg_review`: The average review score given by patients for their experience at the branch. This is typically on a scale (e.g., 1 to 5), where higher scores indicate better reviews.
    - `cogs`: Cost of Goods Sold. This refers to the direct costs attributable to the production of the services provided by the hospital, such as medical supplies and labor directly associated with patient care.
    - `total_revenue`: The total revenue generated by the branch during the specified month and year. This includes all income from patient services and other activities.
    - `total_profit`: The total profit made by the branch during the specified month and year. This is calculated as total revenue minus cogs.
    - `score`: This column represents a composite score calculated to evaluate the performance of the hospital branch. It is calculated using the following formula:
    score = (jumlah_pasien * 0.2) + (total_profit * 0.45) + (avg_review * 0.35)
    """
    return descriptions

def run():
    st.write("# Chat with Mulia Hospital Dataset 🏥")

    df = load_data('data_month.csv')

    with st.expander("🔎 Dataframe Preview"):
        st.write(df.tail(10))

    query = st.text_area("🧑‍⚕️ Ask me!")
    container = st.container()

    if query:
        llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"])
        descriptions = get_column_descriptions()
        prompt = f"{descriptions}\n\nQuery: {query}"
        # llm = OpenAI(
        #     api_token=os.environ["OPENAI_API_KEY"],
        #     model="gpt-3.5-turbo-16k"  # specify the model here
        # )
        query_engine = SmartDataframe(
            df,
            config={
                "llm": llm,
                "response_parser": StreamlitResponse,
                # "callback": StreamlitCallback(container),
            },
        )

        answer = query_engine.chat(prompt)
        if answer:
            st.write("### Response:")
            st.write(answer)
        else:
            st.write("Sorry, I can't answer the question yet")

if __name__ == '__main__':
    run()

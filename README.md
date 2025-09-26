# AI Financial Advisor

This is a Streamlit web application that acts as an AI-powered financial advisor. It analyzes your financial data, provides visualizations, and offers insights and recommendations to help you manage your finances better.

## Features

- **Upload Your Data:** Upload your financial data in CSV format.
- **Sample Data:** Use built-in sample data to see how the app works.
- **Interactive Visualizations:**
    - **Spending by Category:** A pie chart showing the distribution of your expenses across different categories.
    - **Daily Spending Trend:** A line chart illustrating your spending habits over time.
- **AI-Powered Insights:** Get financial advice from state-of-the-art language models.
    - **OpenAI (GPT-3.5-turbo):** Get insights from one of the most powerful language models.
    - **Google Gemini:** Leverage Google's latest generative AI model for financial advice.
- **Quick Statistics:** View key metrics like total spending, average transaction amount, and the number of transactions at a glance.
- **Downloadable Sample Data:** A sample CSV is provided to guide you on the required data format.

## Technologies Used

- **Python:** The core programming language for the application.
- **Streamlit:** For building the interactive web application.
- **Pandas:** For data manipulation and analysis.
- **Plotly Express:** For creating interactive charts and visualizations.
- **OpenAI API:** To get financial advice from GPT-3.5-turbo.
- **Google Generative AI (Gemini):** To get financial advice from the Gemini-pro model.

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Get your API keys:**
    -   You will need an API key from [OpenAI](https://platform.openai.com/signup) and/or [Google AI Studio](https://makersuite.google.com/u/0/app/home) for the AI features.

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

5.  **Use the app:**
    -   Open your web browser and go to the local URL provided by Streamlit (usually `http://localhost:8501`).
    -   Enter your API key in the sidebar, choose your preferred AI model, and start analyzing your finances.
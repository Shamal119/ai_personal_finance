import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import openai
import os
from openai import OpenAI, APIError
from google.api_core import exceptions

# --- Constants ---
GPT_MODEL = "gpt-3.5-turbo"
GEMINI_MODEL = "gemini-pro"

# --- Data Loading and Processing ---

@st.cache_data
def get_sample_data():
    """Generate and cache sample financial data."""
    data = {
        'Date': [
            '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
            '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10'
        ],
        'Description': [
            'Walmart', 'Netflix', 'Gas Station', 'Restaurant', 'Amazon',
            'Utilities', 'Grocery Store', 'Coffee Shop', 'Phone Bill', 'Target'
        ],
        'Category': [
            'Groceries', 'Entertainment', 'Transportation', 'Dining', 'Shopping',
            'Utilities', 'Groceries', 'Dining', 'Utilities', 'Shopping'
        ],
        'Amount': [
            120.50, 13.99, 45.00, 35.75, 89.99,
            150.00, 95.45, 4.25, 75.00, 65.30
        ]
    }
    return pd.DataFrame(data)

@st.cache_data
def process_dataframe(df):
    """Process and cache dataframe for visualization."""
    df_viz = df.copy()
    df_viz['Amount'] = df_viz['Amount'].abs()
    category_spending = df_viz.groupby('Category')['Amount'].sum().reset_index()
    return df_viz, category_spending

def convert_df_to_csv(df):
    """Convert dataframe to CSV format for download."""
    return df.to_csv(index=False).encode('utf-8')

# --- AI Insights ---

def get_financial_advice_prompt(df):
    """Generate the prompt for financial advice."""
    total_spent = df['Amount'].sum()
    by_category = df.groupby('Category')['Amount'].sum().to_dict()
    
    return f"""
    As a financial advisor, analyze this spending data:
    Total spent: ${total_spent:.2f}
    
    Spending by category:
    {by_category}
    
    Please provide:
    1. Key observations
    2. Areas of concern
    3. Specific recommendations for improvement
    """

@st.cache_data
def get_openai_advice(_df_hash, df, api_key):
    """Get financial advice using OpenAI, with caching."""
    client = OpenAI(api_key=api_key)
    prompt = get_financial_advice_prompt(df)
    
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a professional financial advisor."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

@st.cache_data
def get_gemini_advice(_df_hash, df, api_key):
    """Get financial advice using Gemini, with caching."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = get_financial_advice_prompt(df)
    
    response = model.generate_content(prompt)
    return response.text

# --- UI Components ---

def render_sidebar():
    """Render the sidebar with LLM configuration and data download."""
    with st.sidebar:
        st.header("LLM Configuration")
        st.session_state.llm_choice = st.radio(
            "Choose LLM Provider:",
            ["OpenAI", "Gemini"],
            key="llm_choice_radio"
        )
        
        api_key = st.text_input(
            f"Enter your {st.session_state.llm_choice} API Key:",
            type="password",
            key="api_key_input"
        )
        if api_key:
            st.session_state.api_key = api_key
        
        st.write("---")
        st.subheader("Sample Data")
        sample_df = get_sample_data()
        
        st.download_button(
            label="📥 Download Sample CSV",
            data=convert_df_to_csv(sample_df),
            file_name="sample_finance.csv",
            mime="text/csv",
            help="Click to download a sample CSV file with the correct format"
        )
        st.caption("Use this sample file as a template")

def display_visualizations(df_viz, category_spending):
    """Display the pie chart and line chart."""
    st.subheader("Transaction Overview")
    st.dataframe(df_viz.style.format({"Amount": "${:,.2f}"}))

    fig1 = px.pie(
        category_spending,
        values='Amount',
        names='Category',
        title='Spending by Category',
        hole=0.3
    )
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig1, use_container_width=True)

    df_viz['Date'] = pd.to_datetime(df_viz['Date'])
    daily_spending = df_viz.groupby('Date')['Amount'].sum().reset_index()

    fig2 = px.line(
        daily_spending,
        x='Date',
        y='Amount',
        title='Daily Spending Trend'
    )
    fig2.update_traces(mode='lines+markers')
    fig2.update_layout(yaxis_title='Amount ($)')
    st.plotly_chart(fig2, use_container_width=True)

def display_statistics(df, category_spending):
    """Display key financial statistics."""
    st.subheader("Quick Statistics")
    total_spent = df['Amount'].sum()
    avg_transaction = df['Amount'].mean()
    num_transactions = len(df)

    st.metric("Total Spent", f"${total_spent:,.2f}")
    st.metric("Average Transaction", f"${avg_transaction:,.2f}")
    st.metric("Number of Transactions", num_transactions)

    st.subheader("Top Spending Categories")
    top_categories = category_spending.nlargest(3, 'Amount')
    for _, row in top_categories.iterrows():
        st.metric(row['Category'], f"${row['Amount']:,.2f}")

def handle_ai_insights(df):
    """Handle the AI insights generation and display."""
    st.subheader("AI Insights")
    if st.button("Analyze Finances"):
        if not st.session_state.api_key:
            st.error("Please enter an API key to get AI insights.")
            return

        with st.spinner("Generating insights..."):
            try:
                # Pass a hash of the dataframe to the caching function
                df_hash = pd.util.hash_pandas_object(df).sum()
                if st.session_state.llm_choice == "OpenAI":
                    insights = get_openai_advice(df_hash, df, st.session_state.api_key)
                else:
                    insights = get_gemini_advice(df_hash, df, st.session_state.api_key)
                st.write(insights)
            except APIError as e:
                st.error(f"OpenAI API Error: {e.message}")
            except exceptions.GoogleAPICallError as e:
                st.error(f"Gemini API Error: {e.message}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

# --- Main Application ---
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="AI Financial Advisor",
        page_icon="💰",
        layout="wide"
    )

    st.title("AI Financial Advisor")

    # Initialize session state
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    if 'llm_choice' not in st.session_state:
        st.session_state.llm_choice = 'OpenAI'

    render_sidebar()

    st.write("Upload your financial data or use sample data to get AI-powered insights")

    data_source = st.radio(
        "Choose your data source:",
        ["Use Sample Data", "Upload Your Own CSV"],
        horizontal=True
    )

    df = None
    if data_source == "Upload Your Own CSV":
        uploaded_file = st.file_uploader("Upload your CSV financial statement", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                return
    else:
        df = get_sample_data()
        st.info("Using sample data for demonstration. Upload your own CSV for personal insights.")

    if df is not None:
        try:
            df_viz, category_spending = process_dataframe(df)
            
            col1, col2 = st.columns([2, 1])

            with col1:
                display_visualizations(df_viz, category_spending)

            with col2:
                handle_ai_insights(df)
                display_statistics(df, category_spending)
        except Exception as e:
            st.error(f"An error occurred during data processing: {e}")

if __name__ == "__main__":
    main()
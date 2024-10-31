import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import openai
import os

def get_sample_data():
    """Generate sample financial data"""
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

def get_openai_advice(df, api_key):
    """Get financial advice using OpenAI"""
    client = openai.OpenAI(api_key=api_key)
    
    total_spent = df['Amount'].sum()
    by_category = df.groupby('Category')['Amount'].sum().to_dict()
    
    prompt = f"""
    As a financial advisor, analyze this spending data:
    Total spent: ${total_spent:.2f}
    
    Spending by category:
    {by_category}
    
    Please provide:
    1. Key observations
    2. Areas of concern
    3. Specific recommendations for improvement
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional financial advisor."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def get_gemini_advice(df, api_key):
    """Get financial advice using Gemini"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    total_spent = df['Amount'].sum()
    by_category = df.groupby('Category')['Amount'].sum().to_dict()
    
    prompt = f"""
    As a financial advisor, analyze this spending data:
    Total spent: ${total_spent:.2f}
    
    Spending by category:
    {by_category}
    
    Please provide:
    1. Key observations
    2. Areas of concern
    3. Specific recommendations for improvement
    """
    
    response = model.generate_content(prompt)
    return response.text

def process_dataframe(df):
    """Process dataframe for visualization"""
    df_viz = df.copy()
    df_viz['Amount'] = df_viz['Amount'].abs()
    category_spending = df_viz.groupby('Category')['Amount'].sum().reset_index()
    return df_viz, category_spending

def convert_df_to_csv(df):
    """Convert dataframe to CSV format"""
    return df.to_csv(index=False).encode('utf-8')

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'llm_choice' not in st.session_state:
    st.session_state.llm_choice = 'OpenAI'

# Page configuration
st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="💰",
    layout="wide"
)

def main():
    st.title(" AI Financial Advisor")
    
    # Sidebar
    with st.sidebar:
        st.header("LLM Configuration")
        st.session_state.llm_choice = st.radio(
            "Choose LLM Provider:",
            ["OpenAI", "Gemini"]
        )
        
        api_key = st.text_input(
            f"Enter your {st.session_state.llm_choice} API Key:",
            type="password",
            value=st.session_state.api_key
        )
        if api_key:
            st.session_state.api_key = api_key
        
        # Sample data download
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

    # Main content
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
            df = pd.read_csv(uploaded_file)
    else:
        df = get_sample_data()
        st.info("Using sample data for demonstration. Upload your own CSV for personal insights.")

    if df is not None:
        df_viz, category_spending = process_dataframe(df)
        
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Transaction Overview")
            st.dataframe(df.style.format({"Amount": "${:,.2f}"}))

            # Category spending visualization
            fig1 = px.pie(
                category_spending, 
                values='Amount', 
                names='Category', 
                title='Spending by Category',
                hole=0.3
            )
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig1, use_container_width=True)

            # Spending trend visualization
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

        with col2:
            st.subheader("AI Insights")
            if st.button("Analyze Finances"):
                if not st.session_state.api_key:
                    st.error("Please enter an API key to get AI insights.")
                else:
                    with st.spinner("Generating insights..."):
                        try:
                            if st.session_state.llm_choice == "OpenAI":
                                insights = get_openai_advice(df, st.session_state.api_key)
                            else:
                                insights = get_gemini_advice(df, st.session_state.api_key)
                            st.write(insights)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            # Statistics
            st.subheader("Quick Statistics")
            total_spent = df['Amount'].sum()
            avg_transaction = df['Amount'].mean()
            num_transactions = len(df)
            
            st.metric("Total Spent", f"${total_spent:,.2f}")
            st.metric("Average Transaction", f"${avg_transaction:,.2f}")
            st.metric("Number of Transactions", num_transactions)
            
            # Top categories
            st.subheader("Top Spending Categories")
            top_categories = category_spending.nlargest(3, 'Amount')
            for _, row in top_categories.iterrows():
                st.metric(row['Category'], f"${row['Amount']:,.2f}")

if __name__ == "__main__":
    main()
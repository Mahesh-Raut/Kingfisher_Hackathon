import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Define ideal financial metric ranges for different industries
# This is a key part of the refined business logic
INDUSTRY_RANGES = {
    'Technology': {
        'pe_min': 20, 'pe_max': 40,
        'ps_min': 3, 'ps_max': 8,
        'debt_to_equity_max': 2.0,
        'pb_max': 5.0
    },
    'Finance': {
        'pe_min': 10, 'pe_max': 25,
        'ps_min': 2, 'ps_max': 5,
        'debt_to_equity_max': 3.0,
        'pb_max': 3.0
    },
    'Automotive': {
        'pe_min': 8, 'pe_max': 20,
        'ps_min': 0.5, 'ps_max': 2.0,
        'debt_to_equity_max': 2.5,
        'pb_max': 2.5
    },
    'Consumer Goods': {
        'pe_min': 20, 'pe_max': 35,
        'ps_min': 4, 'ps_max': 8,
        'debt_to_equity_max': 1.5,
        'pb_max': 4.0
    },
    'Conglomerate': {
        'pe_min': 15, 'pe_max': 30,
        'ps_min': 2, 'ps_max': 5,
        'debt_to_equity_max': 2.5,
        'pb_max': 3.5
    }
}

def app():
    """
    The main Streamlit application logic.
    """
    st.set_page_config(page_title="Investment Advisor", layout="wide")

    st.title('🤖 Kingfisher AI-Powered Investment Advisor')
    st.markdown("""
    This tool provides a fundamental analysis-based suggestion on whether a stock is investable.
    The logic is based on a simulated financial model trained on high-fidelity synthetic data.
    """)

    # Load data from the CSV file
    try:
        df = pd.read_csv("updated_synthetic_stock_data.csv")
    except FileNotFoundError:
        st.error("Error: The file 'updated_synthetic_stock_data.csv' was not found. Please ensure it is in the same directory.")
        return

    # --- User Input Section ---
    st.subheader("Enter Stock Details")

    with st.form("stock_input_form"):
        # Create a dropdown for company names
        company_names = df['company_name'].unique()
        selected_company_name = st.selectbox(
            "Select Company Name",
            options=company_names,
            index=0
        )

        # Pre-fill inputs with data from the selected company
        selected_row = df[df['company_name'] == selected_company_name].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            industry = st.selectbox(
                "Industry",
                options=list(INDUSTRY_RANGES.keys()),
                index=list(INDUSTRY_RANGES.keys()).index(selected_row['industry'])
            )
            current_price = st.number_input("Current Price", min_value=1.0, value=float(selected_row['current_price']), step=10.0)
            book_value = st.number_input("Book Value", min_value=1.0, value=float(selected_row['book_value']), step=10.0)
        
        with col2:
            pe_ratio = st.number_input("P/E Ratio", min_value=0.1, value=float(selected_row['pe_ratio']), step=0.1)
            ps_ratio = st.number_input("P/S Ratio", min_value=0.1, value=float(selected_row['ps_ratio']), step=0.1)
            debt_to_equity = st.number_input("Debt to Equity Ratio", min_value=0.0, value=float(selected_row['debt_to_equity']), step=0.1)
            revenue_growth = st.number_input("Revenue Growth (%)", min_value=-20.0, value=float(selected_row['revenue_growth']) * 100, step=0.1) / 100.0

        submit_button = st.form_submit_button("Get Investment Suggestion")

    # --- Analysis Logic and Output ---
    if submit_button:
        # Get industry-specific ideal ranges
        ideal_ranges = INDUSTRY_RANGES.get(industry, {})

        # List of reasons for the suggestion
        reasons = []
        investable = True

        # Rule 1: Evaluate P/E Ratio against industry average
        pe_min, pe_max = ideal_ranges.get('pe_min', 0), ideal_ranges.get('pe_max', 100)
        if pe_ratio < pe_min and pe_ratio > 0:
            reasons.append(f"P/E Ratio of {pe_ratio:.2f} is below the industry's typical range of {pe_min}-{pe_max}.")
        elif pe_ratio > pe_max:
            reasons.append(f"P/E Ratio of {pe_ratio:.2f} is high compared to the industry's typical range of {pe_min}-{pe_max}.")
            investable = False
        else:
            reasons.append(f"P/E Ratio of {pe_ratio:.2f} is within the industry's ideal range.")

        # Rule 2: Evaluate P/S Ratio against industry average
        ps_min, ps_max = ideal_ranges.get('ps_min', 0), ideal_ranges.get('ps_max', 20)
        if ps_ratio < ps_min and ps_ratio > 0:
            reasons.append(f"P/S Ratio of {ps_ratio:.2f} is below the industry's typical range of {ps_min}-{ps_max}.")
        elif ps_ratio > ps_max:
            reasons.append(f"P/S Ratio of {ps_ratio:.2f} is high compared to the industry's typical range of {ps_min}-{ps_max}.")
            investable = False
        else:
            reasons.append(f"P/S Ratio of {ps_ratio:.2f} is within the industry's ideal range.")

        # Rule 3: Evaluate Price-to-Book Value against industry max
        price_to_book = current_price / book_value if book_value > 0 else np.inf
        pb_max = ideal_ranges.get('pb_max', 3.0)
        if price_to_book > pb_max:
            reasons.append(f"Current Price-to-Book Value of {price_to_book:.2f} is high (above {pb_max}), suggesting the stock may be overvalued.")
            investable = False
        else:
            reasons.append(f"Current Price-to-Book Value of {price_to_book:.2f} is within a reasonable range (below {pb_max}).")

        # Rule 4: Evaluate Debt-to-Equity Ratio
        debt_to_equity_max = ideal_ranges.get('debt_to_equity_max', 2.0)
        if debt_to_equity > debt_to_equity_max:
            reasons.append(f"Debt-to-Equity ratio of {debt_to_equity:.2f} is high (above {debt_to_equity_max}), indicating higher financial risk.")
            investable = False
        else:
            reasons.append(f"Debt-to-Equity ratio of {debt_to_equity:.2f} is low, indicating a strong balance sheet.")

        # Rule 5: Evaluate Revenue Growth
        if revenue_growth < 0.05:
            reasons.append(f"Revenue Growth of {revenue_growth * 100:.2f}% is low, which may signal a lack of strong business growth.")
            if investable:
                investable = False
        else:
            reasons.append(f"Revenue Growth of {revenue_growth * 100:.2f}% is strong, which is a positive indicator.")
        
        # Display the final suggestion and detailed analysis
        st.markdown("---")
        st.subheader("Analysis Summary for " + selected_company_name)

        if investable:
            st.success(f"**Suggestion: Investable**")
        else:
            st.error(f"**Suggestion: Not Investable**")
        
        st.markdown("#### Key Metrics and Rationale:")
        st.markdown(f"- **P/E Ratio**: **{pe_ratio:.2f}**")
        st.markdown(f"- **P/S Ratio**: **{ps_ratio:.2f}**")
        st.markdown(f"- **Price-to-Book Value**: **{price_to_book:.2f}**")
        st.markdown(f"- **Debt-to-Equity**: **{debt_to_equity:.2f}**")
        st.markdown(f"- **Revenue Growth**: **{revenue_growth * 100:.2f}%**")
        
        st.markdown("#### Detailed Rationale:")
        for reason in reasons:
            st.markdown(f"- {reason}")

    st.markdown("---")
    st.subheader("Graphical Context for Key Ratios and Historic Price")
    
    # Filter data for the selected company and sort by date for the line graph
    df['trading_date'] = pd.to_datetime(df['trading_date'])
    end_date = df['trading_date'].max()
    start_date = end_date - pd.DateOffset(years=1)
    company_df = df[(df['company_name'] == selected_company_name) & (df['trading_date'] >= start_date)].sort_values(by='trading_date')
    
    # Visualization 1: Historic Price for the Last 1 Year
    st.markdown("#### Historic Price over the Last Year")
    fig_price, ax_price = plt.subplots(figsize=(10, 6))
    
    # Resample to get monthly data points
    monthly_data = company_df.resample('M', on='trading_date').agg({
        'current_price': 'last'
    }).reset_index()

    ax_price.plot(monthly_data['trading_date'], monthly_data['current_price'], marker='o', linestyle='-', color='purple')
    ax_price.set_title(f'Historic Price for {selected_company_name} (Last 1 Year)')
    ax_price.set_xlabel('Date')
    ax_price.set_ylabel('Current Price')
    ax_price.tick_params(axis='x', rotation=45)
    ax_price.grid(True)
    st.pyplot(fig_price)

    # Visualization 2: P/E Ratio
    st.markdown("#### P/E Ratio vs. Industry Standard")
    
    # Get the ideal P/E range for the selected industry
    pe_min = INDUSTRY_RANGES[industry]['pe_min']
    pe_max = INDUSTRY_RANGES[industry]['pe_max']
    
    fig_pe, ax_pe = plt.subplots(figsize=(10, 4))
    
    # Create a horizontal bar for the industry range
    ax_pe.barh('P/E Ratio', pe_max - pe_min, left=pe_min, color='lightgreen', alpha=0.6, label=f'Ideal Range for {industry}')
    
    # Add a marker for the user's input
    ax_pe.axvline(x=pe_ratio, color='red', linestyle='--', label=f'Your P/E: {pe_ratio:.2f}')
    
    ax_pe.set_title(f'P/E Ratio for {selected_company_name}')
    ax_pe.set_xlabel('P/E Ratio')
    ax_pe.set_yticks([])  # Hide y-axis ticks
    ax_pe.set_xlim(left=0, right=max(pe_max + 10, pe_ratio + 5))
    ax_pe.legend()
    st.pyplot(fig_pe)

    # Visualization 3: P/S Ratio
    st.markdown("#### P/S Ratio vs. Industry Standard")
    
    # Get the ideal P/S range for the selected industry
    ps_min = INDUSTRY_RANGES[industry]['ps_min']
    ps_max = INDUSTRY_RANGES[industry]['ps_max']
    
    fig_ps, ax_ps = plt.subplots(figsize=(10, 4))
    
    # Create a horizontal bar for the industry range
    ax_ps.barh('P/S Ratio', ps_max - ps_min, left=ps_min, color='lightblue', alpha=0.6, label=f'Ideal Range for {industry}')
    
    # Add a marker for the user's input
    ax_ps.axvline(x=ps_ratio, color='red', linestyle='--', label=f'Your P/S: {ps_ratio:.2f}')
    
    ax_ps.set_title(f'P/S Ratio for {selected_company_name}')
    ax_ps.set_xlabel('P/S Ratio')
    ax_ps.set_yticks([])  # Hide y-axis ticks
    ax_ps.set_xlim(left=0, right=max(ps_max + 5, ps_ratio + 2))
    ax_ps.legend()
    st.pyplot(fig_ps)

if __name__ == "__main__":
    app()


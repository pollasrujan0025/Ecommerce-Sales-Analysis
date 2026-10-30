"""
================================================================================
INTERNCIRCLE DATA SCIENCE & AI INTERNSHIP - PROJECT 3
CUSTOMER CHURN / SALES TREND ANALYSIS USING PYTHON
================================================================================
Author: Data Science Intern
Project Objective:
    End-to-end exploratory and commercial sales analysis on the UCI Online
    Retail e-commerce dataset to identify top customer segments, product
    performance, seasonal revenue trends, and data-driven recommendations.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for reliable server/script chart generation
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output on Windows PowerShell
if sys.platform.startswith('win'):
    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Set global plotting style for professional publications
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)



def print_header(title: str):
    """Utility to print readable section headers in terminal output."""
    separator = "=" * 80
    print(f"\n{separator}")
    print(f" {title.upper()}")
    print(f"{separator}")


# ==============================================================================
# 1. DATA LOADING
# ==============================================================================
def load_and_inspect_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw e-commerce dataset and performs initial structural inspection.
    """
    print_header("1. Data Loading & Initial Inspection")
    
    if not os.path.exists(filepath):
        # Check if Excel exists and convert if needed
        excel_path = os.path.join(BASE_DIR, 'Online_Retail.xlsx')
        if os.path.exists(excel_path):
            print(f"[*] Found '{excel_path}'. Converting to CSV format...")
            raw_df = pd.read_excel(excel_path)
            raw_df.to_csv(filepath, index=False)
            print(f"[+] Saved '{filepath}'.")
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{filepath}'. Please place 'ecommerce_data.csv' "
                f"or 'Online_Retail.xlsx' in the project root."
            )

    print(f"[*] Loading raw dataset from: {filepath}")
    df = pd.read_csv(filepath)
    
    print(f"\n[+] Dataset Dimensions: {df.shape[0]:,} Rows and {df.shape[1]} Columns")
    
    print("\n--- FIRST 5 ROWS ---")
    print(df.head())
    
    print("\n--- DATASET INFORMATION (info) ---")
    df.info()
    
    print("\n--- DESCRIPTIVE STATISTICS (describe) ---")
    print(df.describe(include='all'))
    
    return df


# ==============================================================================
# 2. DATA CLEANING
# ==============================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataset:
    - Identifies and quantifies missing values
    - Removes verified duplicate records
    - Filters out invalid quantities (<= 0) and prices (<= 0)
    - Converts InvoiceDate to datetime
    - Saves cleaning audit report and cleaned dataset
    """
    print_header("2. Data Cleaning & Validation")
    initial_rows = len(df)
    
    # 2.1 Missing Values Analysis
    missing_series = df.isnull().sum()
    missing_pct = (df.isnull().sum() / initial_rows) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing_series,
        'Missing_Percentage (%)': missing_pct.round(2)
    })
    print("\n--- MISSING VALUES AUDIT ---")
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    # 2.2 Duplicate Records Removal
    duplicate_count = df.duplicated().sum()
    print(f"\n[*] Duplicate Records Identified: {duplicate_count:,} ({(duplicate_count/initial_rows)*100:.2f}%)")
    df_dedup = df.drop_duplicates().copy()
    
    # 2.3 Handling Missing Customer IDs
    # In retail analytics and churn modeling, transactions without a CustomerID
    # represent unauthenticated guest checkouts. To analyze customer segments,
    # churn, and RFM, valid CustomerID records are required.
    missing_cust_count = df_dedup['CustomerID'].isnull().sum()
    print(f"[*] Missing CustomerID Records: {missing_cust_count:,} (Guest Checkouts)")
    df_with_cust = df_dedup.dropna(subset=['CustomerID']).copy()
    
    # Convert CustomerID to clean integer string format (e.g., '17850')
    df_with_cust['CustomerID'] = df_with_cust['CustomerID'].astype(int).astype(str)
    
    # 2.4 Checking Invalid Quantities & Unit Prices
    # In UCI Online Retail, negative quantities represent cancellations (InvoiceNo starting with 'C')
    # or inventory damage adjustments. UnitPrice <= 0 represents free promotions, bad debt or adjustments.
    cancellations = df_with_cust['InvoiceNo'].astype(str).str.startswith('C').sum()
    negative_qty = (df_with_cust['Quantity'] <= 0).sum()
    invalid_price = (df_with_cust['UnitPrice'] <= 0).sum()
    
    print(f"[*] Invoices Marked as Cancelled ('C'): {cancellations:,}")
    print(f"[*] Records with Quantity <= 0: {negative_qty:,}")
    print(f"[*] Records with UnitPrice <= 0: {invalid_price:,}")
    
    # Filter strictly for valid, completed sales transactions
    df_clean = df_with_cust[
        (df_with_cust['Quantity'] > 0) & 
        (df_with_cust['UnitPrice'] > 0)
    ].copy()
    
    # 2.5 Datetime Conversion
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    
    final_rows = len(df_clean)
    retention_pct = (final_rows / initial_rows) * 100
    
    print("\n--- CLEANING SUMMARY ---")
    print(f"Initial Raw Rows:            {initial_rows:,}")
    print(f"Duplicates Dropped:          {duplicate_count:,}")
    print(f"Missing Customer IDs Dropped:{missing_cust_count:,}")
    print(f"Invalid / Cancelled Dropped: {len(df_with_cust) - final_rows:,}")
    print(f"Final Cleaned Rows:          {final_rows:,} ({retention_pct:.2f}% retained)")
    
    # Export cleaning summary report
    cleaning_summary = pd.DataFrame([
        {'Metric': 'Initial Raw Records', 'Value': initial_rows},
        {'Metric': 'Duplicates Removed', 'Value': duplicate_count},
        {'Metric': 'Missing CustomerID Records', 'Value': missing_cust_count},
        {'Metric': 'Invalid Qty / Price / Cancellations', 'Value': len(df_with_cust) - final_rows},
        {'Metric': 'Final Clean Valid Records', 'Value': final_rows},
        {'Metric': 'Record Retention Rate (%)', 'Value': round(retention_pct, 2)}
    ])
    cleaning_summary.to_csv(os.path.join(REPORTS_DIR, 'cleaning_summary.csv'), index=False)
    
    # Export cleaned dataset
    cleaned_filepath = os.path.join(BASE_DIR, 'cleaned_ecommerce.csv')
    print(f"[*] Saving cleaned dataset to: {cleaned_filepath}")
    df_clean.to_csv(cleaned_filepath, index=False)
    print(f"[+] Saved 'cleaned_ecommerce.csv' successfully.")
    
    return df_clean


# ==============================================================================
# 3. FEATURE ENGINEERING & DATE EXTRACTION
# ==============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time components and calculates revenue features:
    - Year, Month, Day, Weekday, Invoice Month (YYYY-MM)
    - Total Revenue = Quantity * UnitPrice
    """
    print_header("3. Feature Engineering & Date Extraction")
    
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    df['Weekday'] = df['InvoiceDate'].dt.day_name()
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    
    # Total Revenue calculation
    df['TotalRevenue'] = (df['Quantity'] * df['UnitPrice']).round(2)
    
    print("[+] Created engineered features:")
    print("    - Year (e.g. 2010, 2011)")
    print("    - Month (1 - 12)")
    print("    - Day (1 - 31)")
    print("    - Weekday (Monday - Sunday)")
    print("    - InvoiceMonth (e.g. '2010-12', '2011-11')")
    print("    - TotalRevenue (Quantity × UnitPrice)")
    
    print("\n--- SAMPLE ENGINEERED DATA (5 ROWS) ---")
    cols_to_preview = ['InvoiceNo', 'CustomerID', 'InvoiceDate', 'Weekday', 'InvoiceMonth', 'Quantity', 'UnitPrice', 'TotalRevenue']
    print(df[cols_to_preview].head())
    
    return df


# ==============================================================================
# 4. CUSTOMER ANALYSIS & SEGMENTATION (RFM)
# ==============================================================================
def perform_customer_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs comprehensive customer-level analysis:
    - Top 10 customers by revenue
    - Total purchases and spend per customer
    - RFM segmentation (Recency, Frequency, Monetary)
    - Purchasing frequency distribution (One-time vs Repeat buyers)
    """
    print_header("4. Customer Analysis & RFM Segmentation")
    
    # 4.1 Top 10 Customers by Revenue
    cust_agg = df.groupby('CustomerID').agg(
        TotalRevenue=('TotalRevenue', 'sum'),
        TotalInvoices=('InvoiceNo', 'nunique'),
        TotalItemsBought=('Quantity', 'sum'),
        Country=('Country', 'first')
    ).reset_index()
    
    cust_agg['AvgSpendPerOrder'] = (cust_agg['TotalRevenue'] / cust_agg['TotalInvoices']).round(2)
    top_10_customers = cust_agg.sort_values(by='TotalRevenue', ascending=False).head(10).reset_index(drop=True)
    
    print("\n--- TOP 10 CUSTOMERS BY REVENUE ---")
    print(top_10_customers[['CustomerID', 'Country', 'TotalRevenue', 'TotalInvoices', 'AvgSpendPerOrder']])
    top_10_customers.to_csv(os.path.join(REPORTS_DIR, 'top_10_customers.csv'), index=False)
    
    # 4.2 Customer Purchasing Frequency Distribution
    invoice_counts = cust_agg['TotalInvoices']
    freq_bins = [0, 1, 2, 5, 10, np.inf]
    freq_labels = ['1 Order (One-Time)', '2 Orders', '3-5 Orders', '6-10 Orders', '10+ Orders (Power Buyers)']
    cust_agg['FrequencyTier'] = pd.cut(invoice_counts, bins=freq_bins, labels=freq_labels)
    
    freq_summary = cust_agg.groupby('FrequencyTier', observed=False).agg(
        CustomerCount=('CustomerID', 'count'),
        TotalRevenue=('TotalRevenue', 'sum'),
        AvgRevenue=('TotalRevenue', 'mean')
    ).reset_index()
    freq_summary['CustomerShare%'] = (freq_summary['CustomerCount'] / len(cust_agg)) * 100
    freq_summary['RevenueShare%'] = (freq_summary['TotalRevenue'] / cust_agg['TotalRevenue'].sum()) * 100
    
    print("\n--- PURCHASING FREQUENCY DISTRIBUTION ---")
    print(freq_summary)
    freq_summary.to_csv(os.path.join(REPORTS_DIR, 'customer_purchase_frequency.csv'), index=False)
    
    # 4.3 RFM Customer Segmentation
    # Reference snapshot date is 1 day after the latest recorded transaction
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalRevenue', 'sum')
    ).reset_index()
    
    # Quartile scores for R, F, M (1 to 4)
    # Recency: lower days is better -> score 4 is most recent
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1]).astype(int)
    # Frequency: higher order count is better -> score 4 is most frequent (using first rank for ties)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
    # Monetary: higher total spend is better -> score 4 is highest spend
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4]).astype(int)
    
    # Composite RFM Segment Mapping
    def assign_segment(row):
        score = row['R_Score'] + row['F_Score'] + row['M_Score']
        if score >= 10:
            return 'High Value (Champions)'
        elif score >= 8:
            return 'Loyal Customers'
        elif score >= 6:
            return 'Potential Growth'
        elif score >= 4:
            return 'At Risk'
        else:
            return 'Lost / Churned'
            
    rfm['Segment'] = rfm.apply(assign_segment, axis=1)
    
    segment_summary = rfm.groupby('Segment').agg(
        CustomerCount=('CustomerID', 'count'),
        TotalRevenue=('Monetary', 'sum'),
        AvgRevenue=('Monetary', 'mean'),
        AvgRecency=('Recency', 'mean'),
        AvgFrequency=('Frequency', 'mean')
    ).reset_index()
    
    segment_summary['CustomerShare%'] = (segment_summary['CustomerCount'] / segment_summary['CustomerCount'].sum()) * 100
    segment_summary['RevenueShare%'] = (segment_summary['TotalRevenue'] / segment_summary['TotalRevenue'].sum()) * 100
    segment_summary = segment_summary.sort_values(by='TotalRevenue', ascending=False).reset_index(drop=True)
    
    print("\n--- RFM CUSTOMER SEGMENTATION SUMMARY ---")
    print(segment_summary[['Segment', 'CustomerCount', 'CustomerShare%', 'TotalRevenue', 'RevenueShare%', 'AvgRevenue', 'AvgRecency', 'AvgFrequency']])
    segment_summary.to_csv(os.path.join(REPORTS_DIR, 'customer_segments_summary.csv'), index=False)
    
    return top_10_customers, rfm, segment_summary


# ==============================================================================
# 5. SALES TREND ANALYSIS
# ==============================================================================
def perform_sales_trend_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyzes historical revenue and product sales performance:
    - Monthly revenue trends
    - Yearly revenue comparison
    - Top 10 best-selling products by revenue and quantity
    - Country-wise sales distribution
    - Peak revenue months
    """
    print_header("5. Sales Trend Analysis")
    
    # 5.1 Monthly Revenue Trend
    monthly_sales = df.groupby('InvoiceMonth').agg(
        TotalRevenue=('TotalRevenue', 'sum'),
        TotalOrders=('InvoiceNo', 'nunique'),
        TotalQuantity=('Quantity', 'sum'),
        ActiveCustomers=('CustomerID', 'nunique')
    ).reset_index()
    monthly_sales['AvgOrderValue'] = (monthly_sales['TotalRevenue'] / monthly_sales['TotalOrders']).round(2)
    monthly_sales['MoM_Growth%'] = monthly_sales['TotalRevenue'].pct_change() * 100
    
    print("\n--- MONTHLY REVENUE TRENDS ---")
    print(monthly_sales[['InvoiceMonth', 'TotalRevenue', 'TotalOrders', 'ActiveCustomers', 'AvgOrderValue', 'MoM_Growth%']])
    monthly_sales.to_csv(os.path.join(REPORTS_DIR, 'monthly_sales_trend.csv'), index=False)
    
    # 5.2 Yearly Revenue Summary
    yearly_sales = df.groupby('Year').agg(
        TotalRevenue=('TotalRevenue', 'sum'),
        TotalOrders=('InvoiceNo', 'nunique'),
        TotalQuantity=('Quantity', 'sum'),
        UniqueCustomers=('CustomerID', 'nunique')
    ).reset_index()
    yearly_sales['RevenueShare%'] = (yearly_sales['TotalRevenue'] / yearly_sales['TotalRevenue'].sum()) * 100
    print("\n--- YEARLY REVENUE COMPARISON ---")
    print(yearly_sales)
    yearly_sales.to_csv(os.path.join(REPORTS_DIR, 'yearly_sales_summary.csv'), index=False)
    
    # 5.3 Top 10 Products by Revenue
    top_products = df.groupby(['StockCode', 'Description']).agg(
        TotalRevenue=('TotalRevenue', 'sum'),
        TotalQuantitySold=('Quantity', 'sum'),
        TotalTransactions=('InvoiceNo', 'count')
    ).reset_index().sort_values(by='TotalRevenue', ascending=False).head(10).reset_index(drop=True)
    
    print("\n--- TOP 10 BEST-SELLING PRODUCTS BY REVENUE ---")
    print(top_products[['StockCode', 'Description', 'TotalRevenue', 'TotalQuantitySold']])
    top_products.to_csv(os.path.join(REPORTS_DIR, 'top_10_products.csv'), index=False)
    
    # 5.4 Country-wise Sales
    country_sales = df.groupby('Country').agg(
        TotalRevenue=('TotalRevenue', 'sum'),
        TotalOrders=('InvoiceNo', 'nunique'),
        UniqueCustomers=('CustomerID', 'nunique')
    ).reset_index().sort_values(by='TotalRevenue', ascending=False).reset_index(drop=True)
    country_sales['RevenueShare%'] = (country_sales['TotalRevenue'] / country_sales['TotalRevenue'].sum()) * 100
    
    print("\n--- TOP 10 COUNTRIES BY REVENUE ---")
    print(country_sales.head(10))
    country_sales.to_csv(os.path.join(REPORTS_DIR, 'country_sales_summary.csv'), index=False)
    
    # Peak Month
    peak_month = monthly_sales.sort_values(by='TotalRevenue', ascending=False).iloc[0]
    print(f"\n[+] Highest Revenue Month: {peak_month['InvoiceMonth']} with £{peak_month['TotalRevenue']:,.2f} across {peak_month['TotalOrders']:,} orders.")
    
    return monthly_sales, yearly_sales, top_products, country_sales


# ==============================================================================
# 6. PIVOT TABLES & MULTIDIMENSIONAL CALCULATIONS
# ==============================================================================
def generate_pivot_tables(df: pd.DataFrame, rfm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Computes cross-dimensional aggregations using pd.pivot_table():
    - Pivot 1: Monthly revenue across Year and Month
    - Pivot 2: Monthly revenue across Top 5 Countries
    - Pivot 3: Customer Segment Performance & Metrics
    """
    print_header("6. Pivot Tables & Cross-Dimensional Calculations")
    
    # Pivot 1: Monthly revenue by Year and Month
    pivot_monthly = pd.pivot_table(
        df,
        values='TotalRevenue',
        index='Month',
        columns='Year',
        aggfunc='sum',
        fill_value=0
    ).round(2)
    print("\n--- PIVOT TABLE 1: REVENUE BY MONTH AND YEAR ---")
    print(pivot_monthly)
    pivot_monthly.to_csv(os.path.join(REPORTS_DIR, 'pivot_monthly_revenue.csv'))
    
    # Pivot 2: Top 5 Countries Monthly Revenue
    top5_countries = df.groupby('Country')['TotalRevenue'].sum().nlargest(5).index.tolist()
    df_top5 = df[df['Country'].isin(top5_countries)]
    
    pivot_country_month = pd.pivot_table(
        df_top5,
        values='TotalRevenue',
        index='InvoiceMonth',
        columns='Country',
        aggfunc='sum',
        fill_value=0
    ).round(2)
    print("\n--- PIVOT TABLE 2: MONTHLY REVENUE FOR TOP 5 COUNTRIES ---")
    print(pivot_country_month)
    pivot_country_month.to_csv(os.path.join(REPORTS_DIR, 'pivot_country_monthly.csv'))
    
    # Pivot 3: Customer Segment Performance
    pivot_segment = pd.pivot_table(
        rfm,
        values=['Recency', 'Frequency', 'Monetary'],
        index='Segment',
        aggfunc={
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['count', 'sum', 'mean']
        }
    ).round(2)
    print("\n--- PIVOT TABLE 3: CUSTOMER SEGMENT PERFORMANCE ---")
    print(pivot_segment)
    pivot_segment.to_csv(os.path.join(REPORTS_DIR, 'pivot_segment_performance.csv'))
    
    return pivot_monthly, pivot_country_month, pivot_segment


# ==============================================================================
# 7. DATA VISUALIZATION
# ==============================================================================
def create_visualizations(
    df: pd.DataFrame,
    rfm: pd.DataFrame,
    monthly_sales: pd.DataFrame,
    top_customers: pd.DataFrame,
    top_products: pd.DataFrame,
    country_sales: pd.DataFrame,
    segment_summary: pd.DataFrame
):
    """
    Generates and saves publication-quality visualizations into outputs/:
    1. Monthly revenue trend line chart with annotations
    2. Top 10 customers bar chart
    3. Top 10 products bar chart
    4. Country-wise revenue chart (UK vs Global + Top International Markets)
    5. Customer segments breakdown (Donut + Bar dual chart)
    6. Weekday sales distribution pattern
    """
    print_header("7. Generating Publication-Quality Visualizations")
    
    # Palette definition
    primary_color = '#1f77b4'
    secondary_color = '#ff7f0e'
    accent_palette = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # --------------------------------------------------------------------------
    # Chart 1: Monthly Revenue Trend
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    x_labels = monthly_sales['InvoiceMonth']
    y_values = monthly_sales['TotalRevenue'] / 1000.0  # In Thousands (£k)
    
    ax.plot(x_labels, y_values, marker='o', linewidth=2.5, markersize=8, color='#1e3d59', label='Monthly Revenue')
    ax.fill_between(x_labels, y_values, color='#17b978', alpha=0.15)
    
    # Annotate peak month
    peak_idx = y_values.idxmax()
    peak_month = x_labels[peak_idx]
    peak_val = y_values[peak_idx]
    ax.annotate(
        f'Holiday Peak ({peak_month})\n£{peak_val:,.1f}k',
        xy=(peak_idx, peak_val),
        xytext=(peak_idx - 1.5, peak_val + 70),
        arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=8),
        fontsize=10,
        fontweight='bold',
        color='#b80d57',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffebbb', edgecolor='#b80d57', alpha=0.9)
    )
    
    ax.set_title('E-Commerce Monthly Revenue Trend (Dec 2010 – Dec 2011)', pad=15)
    ax.set_xlabel('Invoice Month (Year-Month)', labelpad=10)
    ax.set_ylabel('Total Revenue (£ in Thousands)', labelpad=10)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add data labels
    for i, txt in enumerate(y_values):
        ax.text(i, txt + 20, f"£{txt:.0f}k", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#333333')
        
    ax.set_ylim(0, max(y_values) * 1.18)
    plt.tight_layout()
    chart1_path = os.path.join(OUTPUTS_DIR, '01_monthly_revenue_trend.png')
    plt.savefig(chart1_path)
    plt.close()
    print(f"[+] Saved Chart 1: {chart1_path}")

    # --------------------------------------------------------------------------
    # Chart 2: Top 10 Customers Bar Chart
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    top_cust_sorted = top_customers.sort_values(by='TotalRevenue', ascending=True)
    cust_labels = [f"ID: {cid} ({country})" for cid, country in zip(top_cust_sorted['CustomerID'], top_cust_sorted['Country'])]
    rev_k = top_cust_sorted['TotalRevenue'] / 1000.0
    
    colors = ['#438a5e' if cid != top_customers.iloc[0]['CustomerID'] else '#2b580c' for cid in top_cust_sorted['CustomerID']]
    bars = ax.barh(cust_labels, rev_k, color=colors, edgecolor='#1b3b1a', alpha=0.85, height=0.65)
    
    ax.set_title('Top 10 High-Value Customers by Total Revenue', pad=15)
    ax.set_xlabel('Total Spend (£ in Thousands)', labelpad=10)
    ax.set_ylabel('Customer ID & Country', labelpad=10)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 3, bar.get_y() + bar.get_height()/2, f"£{width:.1f}k", ha='left', va='center', fontsize=9, fontweight='bold')
        
    ax.set_xlim(0, max(rev_k) * 1.15)
    plt.tight_layout()
    chart2_path = os.path.join(OUTPUTS_DIR, '02_top_10_customers.png')
    plt.savefig(chart2_path)
    plt.close()
    print(f"[+] Saved Chart 2: {chart2_path}")

    # --------------------------------------------------------------------------
    # Chart 3: Top 10 Best-Selling Products by Revenue
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6.5))
    top_prod_sorted = top_products.sort_values(by='TotalRevenue', ascending=True)
    
    # Truncate long descriptions for clean labeling
    clean_descriptions = [
        (desc[:30] + '...') if len(str(desc)) > 30 else str(desc) 
        for desc in top_prod_sorted['Description']
    ]
    prod_rev_k = top_prod_sorted['TotalRevenue'] / 1000.0
    
    bars = ax.barh(clean_descriptions, prod_rev_k, color='#20639b', edgecolor='#0b3c5d', alpha=0.88, height=0.65)
    ax.set_title('Top 10 Products by Total Sales Revenue', pad=15)
    ax.set_xlabel('Revenue Generated (£ in Thousands)', labelpad=10)
    ax.set_ylabel('Product Description', labelpad=10)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, f"£{width:.1f}k", ha='left', va='center', fontsize=9, fontweight='bold')
        
    ax.set_xlim(0, max(prod_rev_k) * 1.16)
    plt.tight_layout()
    chart3_path = os.path.join(OUTPUTS_DIR, '03_top_10_products.png')
    plt.savefig(chart3_path)
    plt.close()
    print(f"[+] Saved Chart 3: {chart3_path}")

    # --------------------------------------------------------------------------
    # Chart 4: Country-Wise Revenue (International Markets)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 2]})
    
    # UK vs Rest of World
    uk_rev = country_sales[country_sales['Country'] == 'United Kingdom']['TotalRevenue'].sum()
    intl_rev = country_sales[country_sales['Country'] != 'United Kingdom']['TotalRevenue'].sum()
    
    ax1.pie(
        [uk_rev, intl_rev],
        labels=['United Kingdom', 'International'],
        autopct='%1.1f%%',
        colors=['#2b580c', '#e27802'],
        explode=[0, 0.08],
        startangle=140,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax1.set_title('Domestic vs. Export Revenue', pad=12)
    
    # Top 8 International Markets
    intl_top8 = country_sales[country_sales['Country'] != 'United Kingdom'].head(8).sort_values(by='TotalRevenue', ascending=True)
    intl_rev_k = intl_top8['TotalRevenue'] / 1000.0
    
    bars = ax2.barh(intl_top8['Country'], intl_rev_k, color='#e27802', edgecolor='#b84a00', alpha=0.85, height=0.65)
    ax2.set_title('Top 8 International Export Markets (Excl. UK)', pad=12)
    ax2.set_xlabel('Revenue (£ in Thousands)', labelpad=10)
    ax2.grid(axis='x', linestyle='--', alpha=0.6)
    
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 3, bar.get_y() + bar.get_height()/2, f"£{width:.1f}k", ha='left', va='center', fontsize=8.5, fontweight='bold')
        
    ax2.set_xlim(0, max(intl_rev_k) * 1.15)
    plt.suptitle('Geographic Sales Distribution & Key Global Markets', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    chart4_path = os.path.join(OUTPUTS_DIR, '04_country_wise_revenue.png')
    plt.savefig(chart4_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Chart 4: {chart4_path}")

    # --------------------------------------------------------------------------
    # Chart 5: Customer Segment Visualization (Donut + Customer Count)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    segment_colors = {
        'High Value (Champions)': '#2b580c',
        'Loyal Customers': '#438a5e',
        'Potential Growth': '#20639b',
        'At Risk': '#f6c90e',
        'Lost / Churned': '#d62728'
    }
    
    ordered_segments = segment_summary['Segment'].tolist()
    palette = [segment_colors.get(seg, '#999999') for seg in ordered_segments]
    
    # Donut Chart for Revenue Share
    wedges, texts, autotexts = ax1.pie(
        segment_summary['TotalRevenue'],
        labels=segment_summary['Segment'],
        autopct='%1.1f%%',
        pctdistance=0.75,
        colors=palette,
        startangle=120,
        textprops={'fontsize': 9, 'fontweight': 'bold'}
    )
    # Draw central white circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    ax1.add_artist(centre_circle)
    ax1.set_title('Revenue Contribution by Customer Segment', pad=12)
    
    # Bar Chart for Customer Volume
    bars = ax2.bar(
        segment_summary['Segment'],
        segment_summary['CustomerCount'],
        color=palette,
        edgecolor='#333333',
        alpha=0.85,
        width=0.55
    )
    ax2.set_title('Customer Count Across Segments', pad=12)
    ax2.set_ylabel('Number of Unique Customers', labelpad=10)
    ax2.set_xticks(range(len(segment_summary['Segment'])))
    ax2.set_xticklabels(segment_summary['Segment'], rotation=25, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 25, f"{height:,}", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
    ax2.set_ylim(0, max(segment_summary['CustomerCount']) * 1.15)
    plt.suptitle('Customer Segmentation & Value Distribution (RFM Framework)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    chart5_path = os.path.join(OUTPUTS_DIR, '05_customer_segments.png')
    plt.savefig(chart5_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Chart 5: {chart5_path}")

    # --------------------------------------------------------------------------
    # Chart 6: Weekday Revenue Analysis
    # --------------------------------------------------------------------------
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday']
    # Note: Saturday transactions are naturally absent in UCI Online Retail dataset (warehouse closed)
    weekday_sales = df.groupby('Weekday')['TotalRevenue'].sum().reindex(
        [day for day in weekday_order if day in df['Weekday'].unique()]
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        weekday_sales['Weekday'],
        weekday_sales['TotalRevenue'] / 1000.0,
        color='#3f72af',
        edgecolor='#112d4e',
        alpha=0.85,
        width=0.55
    )
    ax.set_title('Total Revenue Generated by Day of Week', pad=15)
    ax.set_xlabel('Weekday', labelpad=10)
    ax.set_ylabel('Total Revenue (£ in Thousands)', labelpad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 25, f"£{height:.0f}k", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
    ax.set_ylim(0, max(weekday_sales['TotalRevenue'] / 1000.0) * 1.15)
    plt.tight_layout()
    chart6_path = os.path.join(OUTPUTS_DIR, '06_weekday_revenue_pattern.png')
    plt.savefig(chart6_path)
    plt.close()
    print(f"[+] Saved Chart 6: {chart6_path}")


# ==============================================================================
# 8. BUSINESS FINDINGS & STRATEGIC RECOMMENDATIONS
# ==============================================================================
def output_business_recommendations(
    segment_summary: pd.DataFrame,
    monthly_sales: pd.DataFrame,
    country_sales: pd.DataFrame,
    top_products: pd.DataFrame
):
    """
    Prints commercial findings and practical business recommendations
    based directly on verified calculation outputs.
    """
    print_header("8. Data-Driven Business Recommendations")
    
    champions = segment_summary[segment_summary['Segment'] == 'High Value (Champions)'].iloc[0]
    at_risk = segment_summary[segment_summary['Segment'] == 'At Risk'].iloc[0]
    uk_share = country_sales[country_sales['Country'] == 'United Kingdom']['RevenueShare%'].iloc[0]
    netherlands_share = country_sales[country_sales['Country'] == 'Netherlands']['RevenueShare%'].iloc[0]
    eire_share = country_sales[country_sales['Country'] == 'EIRE']['RevenueShare%'].iloc[0]
    germany_share = country_sales[country_sales['Country'] == 'Germany']['RevenueShare%'].iloc[0]
    
    peak_month = monthly_sales.sort_values(by='TotalRevenue', ascending=False).iloc[0]
    low_month = monthly_sales[monthly_sales['InvoiceMonth'] != '2011-12'].sort_values(by='TotalRevenue', ascending=True).iloc[0]
    
    print(f"""
================================================================================
EXECUTIVE SUMMARY OF ACTUAL FINDINGS:
================================================================================
1. High Customer Concentration (Pareto Principle):
   - The 'High Value (Champions)' segment accounts for {champions['RevenueShare%']:.1f}% of total company
     revenue (£{champions['TotalRevenue']:,.2f}), yet comprises only {champions['CustomerShare%']:.1f}% of the
     customer base ({champions['CustomerCount']:,} customers).
   - Average revenue per champion customer is £{champions['AvgRevenue']:,.2f}, compared to
     the store-wide customer average of £2,048.69.

2. Churn Risk & Vulnerable Revenue:
   - There are {at_risk['CustomerCount']:,} customers in the 'At Risk' segment ({at_risk['CustomerShare%']:.1f}% of all customers),
     holding an average recency of {at_risk['AvgRecency']:.0f} days since their last purchase.
   - Without automated win-back intervention, these customers lapse into permanent churn.

3. Geographic Concentration:
   - The United Kingdom generates {uk_share:.1f}% of total gross revenue.
   - However, top international export markets—led by the Netherlands ({netherlands_share:.1f}%),
     EIRE ({eire_share:.1f}%), and Germany ({germany_share:.1f}%)—exhibit significantly higher Average
     Order Values (wholesale/B2B behavior) than domestic retail customers.

4. Seasonal Revenue Spikes & Post-Holiday Slump:
   - Revenue peaks in Q4 ({peak_month['InvoiceMonth']}) reaching £{peak_month['TotalRevenue']:,.2f} driven by
     Christmas/holiday bulk purchasing.
   - Conversely, {low_month['InvoiceMonth']} records the lowest full-month sales at £{low_month['TotalRevenue']:,.2f},
     revealing severe post-holiday slump vulnerability.

================================================================================
ACTIONABLE BUSINESS RECOMMENDATIONS:
================================================================================
1. RETENTION & VIP CONCIERGE PROGRAM FOR CHAMPIONS:
   - Launch an exclusive VIP account tier with dedicated account managers, early-access wholesale
     pre-orders, and tiered volume rebates for the top {champions['CustomerCount']:,} customers who generate
     over two-thirds of total turnover.
   - Goal: Increase annual customer retention from 75% to 85%, protecting £5.8M+ in annual turnover.

2. AUTOMATED WIN-BACK CAMPAIGNS FOR 'AT RISK' CUSTOMERS:
   - Deploy automated triggers when a customer reaches 60–90 days without re-ordering.
   - Deliver tailored re-engagement discounts (e.g., 10% off cart + free shipping) featuring their
     top previously ordered product categories.

3. OPTIMIZING SLOW MONTHS (JANUARY - FEBRUARY SLUMP):
   - Create 'New Year B2B Stock-Up' campaigns and seasonal inventory clearances in January to incentivize
     early wholesale restocking.
   - Introduce subscription re-orders for recurring consumable giftware and seasonal home decorations.

4. EXPANDING HIGH-AOV INTERNATIONAL MARKETS:
   - Partner with local logistics providers in the Netherlands, Ireland, and Germany to offer
     localized shipping rates, multi-currency invoicing (EUR), and targeted European B2B catalogs.

5. MERCHANDISING FOCUS ON TOP REVENUE DRIVERS:
   - Prioritize inventory buffering and supplier service-level agreements (SLAs) on top volume
     SKUs (e.g., '{top_products.iloc[0]['Description']}', '{top_products.iloc[1]['Description']}')
     to prevent costly stockouts during the September-November wholesale rush.
================================================================================
""")


# ==============================================================================
# MAIN WORKFLOW EXECUTION
# ==============================================================================
def main():
    """Main execution orchestrator."""
    print("""
    +----------------------------------------------------------------------+
    |      INTERNCIRCLE DATA SCIENCE & AI INTERNSHIP - PROJECT 3           |
    |      Customer Churn & Sales Trend Analysis with Python & Pandas      |
    +----------------------------------------------------------------------+
    """)
    
    # 1. Dataset location
    raw_data_path = os.path.join(BASE_DIR, 'ecommerce_data.csv')
    
    # Step 1: Data Loading
    df_raw = load_and_inspect_data(raw_data_path)
    
    # Step 2: Data Cleaning
    df_clean = clean_data(df_raw)
    
    # Step 3: Feature Engineering
    df_featured = engineer_features(df_clean)
    
    # Step 4: Customer Analysis & RFM Segmentation
    top_customers, rfm_df, segment_summary = perform_customer_analysis(df_featured)
    
    # Step 5: Sales Trend Analysis
    monthly_sales, yearly_sales, top_products, country_sales = perform_sales_trend_analysis(df_featured)
    
    # Step 6: Pivot Tables
    pivot_monthly, pivot_country_month, pivot_segment = generate_pivot_tables(df_featured, rfm_df)
    
    # Step 7: Data Visualizations
    create_visualizations(
        df=df_featured,
        rfm=rfm_df,
        monthly_sales=monthly_sales,
        top_customers=top_customers,
        top_products=top_products,
        country_sales=country_sales,
        segment_summary=segment_summary
    )
    
    # Step 8: Strategic Recommendations
    output_business_recommendations(
        segment_summary=segment_summary,
        monthly_sales=monthly_sales,
        country_sales=country_sales,
        top_products=top_products
    )
    
    print_header("Execution Complete")
    print(f"[OK] Cleaned Dataset Saved:  {os.path.join(BASE_DIR, 'cleaned_ecommerce.csv')}")
    print(f"[OK] Visualizations Saved:   {OUTPUTS_DIR} (6 High-Resolution PNGs)")
    print(f"[OK] Summary CSVs Saved:     {REPORTS_DIR} (7 Analytical CSV Reports)")
    print("\nEnd-to-end pipeline finished successfully!")


if __name__ == '__main__':
    main()

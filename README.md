# Customer Churn & Sales Trend Analysis Using Python

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12%2B-4c72b0.svg)](https://seaborn.pydata.org/)
[![InternCircle](https://img.shields.io/badge/InternCircle-Project%203-orange.svg)](https://interncircle.com)

**InternCircle Data Science & AI Internship — Project 3**  
**Project Title:** Customer Churn / Sales Trend Analysis Using Python  
**Author:** Data Science Intern  

---

## 1. Project Introduction

This project delivers an end-to-end commercial sales and customer behavior analysis for a transnational e-commerce retailer. Using transaction-level data, the project assesses revenue drivers, seasonal sales fluctuations, geographic expansion opportunities, and customer churn risks.

By applying exploratory data analysis (EDA), RFM (Recency, Frequency, Monetary) customer segmentation, and multidimensional pivot aggregations, the project uncovers empirical business patterns to guide customer retention, merchandising strategies, and operational planning.

---

## 2. Project Objectives

1. **Data Cleaning & Integrity Validation:** Identify and remediate missing records, duplicates, order cancellations, and invalid price/quantity entries.
2. **Feature Engineering & Time-Series Extraction:** Derive business attributes including transaction revenue, calendar components (Year, Month, Day, Weekday), and year-month cohorts.
3. **Customer Segmentation & Churn Profiling:** Segment customer cohorts via the RFM framework to identify high-value VIP buyers, regular loyalists, and at-risk churn candidates.
4. **Sales Trend & Seasonality Diagnostics:** Evaluate monthly and yearly revenue trajectories, peak demand periods, and day-of-week sales variations.
5. **Product & Geographic Performance Analysis:** Identify best-selling SKUs by quantity and revenue, and benchmark domestic UK sales against high-AOV European export markets.
6. **Commercial Strategy Formulation:** Formulate practical, calculated recommendations to improve customer retention, prevent revenue leakage, and optimize slow-season trade.

---

## 3. Dataset Description & Provenance

* **Dataset Name:** Online Retail Dataset
* **Repository:** [UC Irvine Machine Learning Repository (UCI MLR)](https://archive.ics.uci.edu/dataset/352/online+retail)
* **Dataset Scope:** 541,909 transnational transaction records spanning **December 1, 2010 to December 9, 2011**.
* **Business Profile:** A UK-based, registered non-store online retailer specializing in unique all-occasion giftware. A substantial portion of customers are wholesalers.

### Attribute Specification

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `InvoiceNo` | String / Object | 6-digit transaction identifier. Invoices starting with 'C' indicate cancellations. |
| `StockCode` | String / Object | Unique product identifier code. |
| `Description` | String / Object | Commercial description / title of the merchandise item. |
| `Quantity` | Integer | Units of merchandise ordered per invoice line item. |
| `InvoiceDate` | Datetime | Exact date and time timestamp when transaction occurred. |
| `UnitPrice` | Float | Unit price of product in Sterling (£ / GBP). |
| `CustomerID` | Float / String | Unique 5-digit identifier assigned to registered customers. |
| `Country` | String / Object | Country of residence / delivery for the customer. |

---

## 4. Technologies & Environment

* **Programming Language:** Python 3.10+ (tested on Python 3.14 on Windows PowerShell)
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **File & Excel Processing:** `openpyxl`
* **Interactive Exploration:** `jupyter`, `ipykernel`

---

## 5. Project Directory Structure

```text
Ecommerce_Sales_Analysis/
├── main.py                         # Complete end-to-end Python analysis pipeline
├── README.md                       # Comprehensive documentation & business recommendations
├── requirements.txt                # Required Python libraries and dependencies
├── ecommerce_data.csv              # Full raw dataset (~541,909 records)
├── cleaned_ecommerce.csv           # Cleaned and validated dataset (~392,692 records)
├── outputs/                        # High-resolution publication-grade visualization charts
│   ├── 01_monthly_revenue_trend.png
│   ├── 02_top_10_customers.png
│   ├── 03_top_10_products.png
│   ├── 04_country_wise_revenue.png
│   ├── 05_customer_segments.png
│   └── 06_weekday_revenue_pattern.png
└── reports/                        # Aggregated analytical metrics and pivot tables (CSV)
    ├── cleaning_summary.csv
    ├── country_sales_summary.csv
    ├── customer_purchase_frequency.csv
    ├── customer_segments_summary.csv
    ├── monthly_sales_trend.csv
    ├── yearly_sales_summary.csv
    ├── top_10_customers.csv
    ├── top_10_products.csv
    ├── pivot_monthly_revenue.csv
    ├── pivot_country_monthly.csv
    └── pivot_segment_performance.csv
```

---

## 6. Data Cleaning Methodology

An audit of the raw dataset revealed four data quality anomalies that required resolution:

1. **Missing Data Identification:**
   * `Description`: 1,454 missing records (0.27%).
   * `CustomerID`: 135,080 missing records (24.93%). In e-commerce analytics, null customer IDs represent anonymous guest checkouts. Because customer identification is vital for RFM segmentation, customer lifetime spend, and churn analysis, guest records cannot be assigned to an account entity and were dropped for customer cohort analysis.
2. **Duplicate Records Removal:**
   * 5,268 exact duplicate rows (0.97%) were identified and removed.
3. **Invalid Quantities & Cancellations:**
   * 8,872 records contained negative or zero quantities (starting with 'C' in `InvoiceNo` representing customer returns/cancellations, or warehouse write-offs).
   * 40 records exhibited negative or zero unit prices (representing bad debt write-offs, gift items, or system tests).
   * All negative and zero value records were filtered out to reflect valid, settled commercial transactions.
4. **Data Type Standardization:**
   * `CustomerID` was converted to clean, discrete string format.
   * `InvoiceDate` was parsed into a native `pandas.Timestamp`.

### Data Cleaning Audit Summary

| Stage / Metric | Count | % of Raw Records |
| :--- | :--- | :--- |
| **Initial Raw Records** | 541,909 | 100.00% |
| **Duplicate Rows Removed** | 5,268 | 0.97% |
| **Missing Customer IDs Dropped** | 135,037 | 24.92% |
| **Invalid Qty / UnitPrice / Cancellations Dropped** | 8,912 | 1.64% |
| **Final Cleaned Commercial Records** | **392,692** | **72.46%** |

---

## 7. Feature Engineering

To facilitate time-series forecasting, day-level scheduling, and cohort revenue modeling, several new features were engineered:

* **`TotalRevenue`:** `Quantity × UnitPrice` (represents gross turnover per line item).
* **`Year`:** Calendar year extracted from `InvoiceDate` (2010 vs 2011).
* **`Month`:** Numerical month of purchase (1–12).
* **`Day`:** Calendar day of the transaction (1–31).
* **`Weekday`:** Day of the week (Monday, Tuesday, Wednesday, Thursday, Friday, Sunday).
* **`InvoiceMonth`:** Monthly cohort period formatted as `YYYY-MM` (e.g. `2010-12` through `2011-12`).

---

## 8. Customer Analysis & RFM Segmentation

### Top 10 Customers by Revenue

The top 10 individual accounts represent an extraordinary concentration of enterprise purchasing power:

| Rank | Customer ID | Country | Total Revenue (£) | Orders Placed | Avg Spend / Order (£) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **14646** | Netherlands | **£280,206.02** | 73 | £3,838.44 |
| 2 | **18102** | United Kingdom | **£259,657.30** | 60 | £4,327.62 |
| 3 | **17450** | United Kingdom | **£194,390.79** | 46 | £4,225.89 |
| 4 | **16446** | United Kingdom | **£168,472.50** | 2 | £84,236.25 |
| 5 | **14911** | EIRE (Ireland) | **£143,711.17** | 201 | £714.98 |
| 6 | **12415** | Australia | **£124,914.53** | 21 | £5,948.31 |
| 7 | **14156** | EIRE (Ireland) | **£117,210.08** | 55 | £2,131.09 |
| 8 | **17511** | United Kingdom | **£91,062.38** | 31 | £2,937.50 |
| 9 | **16029** | United Kingdom | **£80,850.84** | 63 | £1,283.35 |
| 10 | **12346** | United Kingdom | **£77,183.60** | 1 | £77,183.60 |

### Customer Purchasing Frequency Distribution

| Frequency Tier | Customer Count | Customer Share (%) | Total Revenue (£) | Revenue Share (%) |
| :--- | :---: | :---: | :---: | :---: |
| **1 Order (One-Time Buyers)** | 1,493 | 34.42% | £614,089.47 | 6.91% |
| **2 Orders** | 835 | 19.25% | £737,466.86 | 8.30% |
| **3–5 Orders** | 1,138 | 26.23% | £1,641,757.29 | 18.47% |
| **6–10 Orders** | 535 | 12.33% | £1,509,997.35 | 16.99% |
| **10+ Orders (Power Buyers)** | 337 | 7.77% | £4,383,897.92 | **49.33%** |

> **Key Finding:** Only **7.77% of customers** (Power Buyers with >10 orders) generate **nearly half (49.33%)** of company revenue. Meanwhile, **34.42% of customers** buy only once and never return.

### RFM Segmentation Results

Customers were scored from 1 to 4 across three core metrics:
* **Recency (R):** Days elapsed since the customer's last order (relative to snapshot date: December 10, 2011).
* **Frequency (F):** Total distinct invoices completed.
* **Monetary (M):** Cumulative gross revenue spent.

| Segment | Customers | Customer % | Total Revenue (£) | Revenue % | Avg Revenue (£) | Avg Recency | Avg Frequency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **High Value (Champions)** | 1,267 | 29.21% | **£6,827,235.77** | **76.82%** | £5,388.50 | 20 days | 9.9 orders |
| **Loyal Customers** | 845 | 19.48% | **£1,051,599.98** | **11.83%** | £1,244.50 | 52 days | 3.2 orders |
| **Potential Growth** | 935 | 21.55% | **£650,529.15** | **7.32%** | £695.75 | 87 days | 1.9 orders |
| **At Risk** | 991 | 22.84% | **£309,485.76** | **3.48%** | £312.30 | 171 days | 1.2 orders |
| **Lost / Churned** | 300 | 6.92% | **£48,358.23** | **0.54%** | £161.19 | 268 days | 1.0 orders |

---

## 9. Sales Trend & Geographic Performance

### Monthly Revenue Performance

| Year-Month | Monthly Revenue (£) | Total Orders | Active Customers | MoM Growth (%) |
| :--- | :---: | :---: | :---: | :---: |
| **2010-12** | £570,422.73 | 1,400 | 885 | Baseline |
| **2011-01** | £568,101.31 | 987 | 741 | -0.41% |
| **2011-02** | £446,084.92 | 997 | 758 | **-21.48% (Trough)** |
| **2011-03** | £594,081.76 | 1,321 | 974 | +33.18% |
| **2011-04** | £468,374.33 | 1,149 | 856 | -21.16% |
| **2011-05** | £677,355.15 | 1,555 | 1,056 | +44.62% |
| **2011-06** | £660,046.05 | 1,393 | 991 | -2.56% |
| **2011-07** | £598,962.90 | 1,331 | 949 | -9.25% |
| **2011-08** | £644,051.04 | 1,280 | 935 | +7.53% |
| **2011-09** | £950,690.20 | 1,755 | 1,266 | **+47.61% (Q4 Ramp)** |
| **2011-10** | £1,035,642.45 | 1,929 | 1,364 | +8.94% |
| **2011-11** | **£1,156,205.61** | **2,657** | **1,665** | **+11.64% (Peak)** |
| **2011-12 (Partial)** | £517,190.44 | 778 | 617 | -55.27% (9 Days) |

### Top 10 Best-Selling Products by Revenue

| Rank | Stock Code | Product Description | Total Revenue (£) | Units Sold |
| :---: | :---: | :--- | :---: | :---: |
| 1 | `23843` | PAPER CRAFT , LITTLE BIRDIE | **£168,469.60** | 80,995 |
| 2 | `22423` | REGENCY CAKESTAND 3 TIER | **£142,592.95** | 12,374 |
| 3 | `85123A` | WHITE HANGING HEART T-LIGHT HOLDER | **£100,603.50** | 36,706 |
| 4 | `85099B` | JUMBO BAG RED RETROSPOT | **£85,220.78** | 46,078 |
| 5 | `23166` | MEDIUM CERAMIC TOP STORAGE JAR | **£81,416.73** | 77,916 |
| 6 | `POST` | POSTAGE | **£77,803.96** | 3,120 |
| 7 | `47566` | PARTY BUNTING | **£68,844.33** | 15,279 |
| 8 | `84879` | ASSORTED COLOUR BIRD ORNAMENT | **£56,580.34** | 35,263 |
| 9 | `M` | MANUAL | **£53,779.93** | 6,933 |
| 10 | `23084` | RABBIT NIGHT LIGHT | **£51,346.21** | 27,153 |

### Top 10 Countries by Revenue

| Rank | Country | Total Revenue (£) | Orders | Unique Accounts | Revenue Share (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **United Kingdom** | £7,285,024.64 | 16,646 | 3,920 | **81.97%** |
| 2 | **Netherlands** | £285,446.34 | 94 | 9 | **3.21%** |
| 3 | **EIRE (Ireland)** | £265,262.46 | 260 | 3 | **2.98%** |
| 4 | **Germany** | £228,678.40 | 457 | 94 | **2.57%** |
| 5 | **France** | £208,934.31 | 389 | 87 | **2.35%** |
| 6 | **Australia** | £138,453.81 | 57 | 9 | **1.56%** |
| 7 | **Spain** | £61,558.56 | 90 | 30 | **0.69%** |
| 8 | **Switzerland** | £56,443.95 | 51 | 21 | **0.64%** |
| 9 | **Belgium** | £41,196.34 | 98 | 25 | **0.46%** |
| 10 | **Sweden** | £38,367.83 | 36 | 8 | **0.43%** |

---

## 10. Pivot Tables Summary

### Monthly Revenue Comparison (Year vs Month)

```text
Year            2010 (£)        2011 (£)
Month                                  
January             0.00      568,101.31
February            0.00      446,084.92
March               0.00      594,081.76
April               0.00      468,374.33
May                 0.00      677,355.15
June                0.00      660,046.05
July                0.00      598,962.90
August              0.00      644,051.04
September           0.00      950,690.20
October             0.00    1,035,642.45
November            0.00    1,156,205.61
December      570,422.73      517,190.44  (Dec 2011 partial: 9 days)
```

### Top 5 International Markets Monthly Trend (£)

| InvoiceMonth | EIRE | France | Germany | Netherlands | United Kingdom |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2010-12** | £8,813.88 | £9,616.31 | £15,205.74 | £8,784.48 | £496,477.34 |
| **2011-01** | £21,904.19 | £17,740.12 | £16,880.84 | £26,611.16 | £440,876.33 |
| **2011-02** | £10,111.52 | £8,515.96 | £9,581.05 | £23,011.91 | £354,618.20 |
| **2011-03** | £21,674.36 | £14,584.95 | £14,392.69 | £22,416.49 | £465,784.19 |
| **2011-04** | £7,570.50 | £5,529.61 | £12,315.54 | £2,976.56 | £408,733.11 |
| **2011-05** | £15,977.15 | £17,606.48 | £25,734.70 | £29,185.88 | £550,359.35 |
| **2011-06** | £19,835.99 | £16,078.97 | £13,246.35 | £26,858.09 | £523,775.59 |
| **2011-07** | £40,854.15 | £10,000.19 | £16,440.98 | £26.02 | £484,545.59 |
| **2011-08** | £16,957.83 | £13,810.96 | £19,220.77 | £40,327.81 | £497,194.91 |
| **2011-09** | £40,832.64 | £23,428.04 | £18,079.47 | £26,937.26 | £794,806.69 |
| **2011-10** | £24,302.92 | £33,471.05 | £31,628.42 | £40,708.65 | £821,220.13 |
| **2011-11** | £29,448.41 | £31,274.75 | £27,967.68 | £25,874.01 | £975,251.39 |
| **2011-12** | £6,978.92 | £7,276.92 | £7,984.17 | £11,728.02 | £471,381.82 |

---

## 11. Visualizations Generated

All charts are exported to the `outputs/` folder at **300 DPI**:

1. **`01_monthly_revenue_trend.png`:** Line chart illustrating historical revenue trajectory, highlighting the November holiday peak (£1,156.2k) and monthly growth rates.
2. **`02_top_10_customers.png`:** Horizontal bar chart displaying the highest spenders (led by Dutch wholesale Customer 14646 with £280.2k).
3. **`03_top_10_products.png`:** Revenue ranking of best-selling products with trimmed descriptions and sales values in £k.
4. **`04_country_wise_revenue.png`:** Dual panel chart showing the domestic 82.0% UK share alongside a comparative breakdown of top European and international export destinations.
5. **`05_customer_segments.png`:** Two-panel figure consisting of a revenue share donut chart (showing 76.8% champion share) and customer volume distribution bar chart.
6. **`06_weekday_revenue_pattern.png`:** Day-of-week sales volume analysis highlighting peak order processing days (Thursday and Wednesday) and Sunday re-orders.

---

## 12. Actual Business Findings & Strategic Recommendations

### Empirical Findings

1. **Extreme Customer Revenue Concentration (Pareto Principle):**
   * The **High Value (Champions)** segment generates **76.82% of company revenue (£6,827,235.77)** while representing only **29.21% of accounts (1,267 customers)**.
   * Conversely, **1,493 customers (34.42%)** made only one order, accounting for merely 6.91% of total turnover.
2. **Imminent Churn Vulnerability:**
   * **991 customers (22.84%)** are currently in the **At Risk** segment with an average recency of **171 days** since their last purchase. If neglected, this segment represents £309k in lost revenue and customer attrition.
3. **Severe Seasonality Risk (Q1 Slump):**
   * Sales surge dramatically in September–November (peaking at £1.16M in November 2011), but experience a 61% plunge during January–February (£446k in Feb).
4. **High-Yield International B2B Markets:**
   * While the UK contributes 81.97% of sales, European export markets demonstrate far higher purchasing consolidation: only 9 customers in the Netherlands generated £285k (averaging **£31,716 per customer**), and 3 customers in Ireland generated £265k (averaging **£88,420 per customer**), signifying enterprise wholesale relationships.

---

### Actionable Business Recommendations

#### 1. Launch a VIP Concierge Retention Program
* **Target:** The 1,267 'High Value (Champions)' accounts.
* **Mechanism:** Introduce dedicated B2B account managers, early access to new seasonal stock catalogs, automated stock replenishment agreements, and volume-based loyalty rebates.
* **Impact:** Retaining an additional 10% of champions protects over **£680,000** in recurring annual gross revenue.

#### 2. Automated Win-Back Flows for 'At Risk' Customers
* **Target:** The 991 'At Risk' customers approaching 90–120 days of inactivity.
* **Mechanism:** Trigger personalized email journeys featuring re-order discounts (e.g. 10% discount + free freight) showcasing complementary products based on their past purchase history.
* **Impact:** Re-activating 15% of at-risk accounts recovers approximately **£46,000** in direct sales.

#### 3. Smoothing the Q1 Post-Holiday Revenue Slump
* **Target:** All customer tiers during January and February.
* **Mechanism:** Launch a "New Year Wholesale Restock" campaign in mid-January, combined with clearance bundles on slow-moving seasonal inventory. Introduce subscription replenishment for recurring consumable merchandise.
* **Impact:** Lifts baseline February turnover by an estimated 15–20% (£65k–£90k).

#### 4. International B2B Expansion & Localized Invoicing
* **Target:** Key European markets (Netherlands, Ireland, Germany, France).
* **Mechanism:** Establish localized pallet delivery agreements, European VAT-compliant invoicing, and multi-currency billing (EUR/GBP) to capture mid-sized continental wholesalers.
* **Impact:** Diversifies revenue beyond the UK domestic market and expands the international share from 18% to 25%.

#### 5. Inventory Protection on Core Cash-Cow Products
* **Target:** Top 10 revenue-generating SKUs (e.g., `22423 - REGENCY CAKESTAND 3 TIER`, `85123A - WHITE HANGING HEART T-LIGHT HOLDER`).
* **Mechanism:** Maintain minimum safety buffer stocks (at least 6 weeks of forecasted sales) and sign priority supply SLAs with manufacturers ahead of the September wholesale rush.
* **Impact:** Eliminates stockout-induced lost sales during peak Q4 demand.

---

## 13. How to Execute the Project

### Prerequisites
* Windows, macOS, or Linux
* Python 3.10 or higher installed

### Step-by-Step Execution on Windows PowerShell

1. **Clone or Navigate to the Project Directory:**
   ```powershell
   cd c:\Users\polla\OneDrive\Desktop\Ecommerce_Sales_Analysis
   ```

2. **(Optional) Create and Activate a Virtual Environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies:**
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. **Run the Analysis Pipeline:**
   ```powershell
   python main.py
   ```

5. **Verify Outputs:**
   * Cleaned dataset: `cleaned_ecommerce.csv`
   * Charts: View generated PNG files in `outputs/`
   * CSV Reports: View summary tables in `reports/`

---

## 14. GitHub & InternCircle Submission Guide

### Part A: Uploading the Project to GitHub

1. **Initialize Git Repository (if not already done):**
   ```powershell
   git init
   ```

2. **Add Files to Staging:**
   ```powershell
   git add main.py README.md requirements.txt outputs/ reports/
   ```
   *(Note: Large raw data files over 100MB should be managed using `.gitignore` or Git LFS if required).*

3. **Commit the Project:**
   ```powershell
   git commit -m "feat: complete end-to-end customer churn and sales trend analysis (InternCircle Project 3)"
   ```

4. **Connect to Your GitHub Repository:**
   ```powershell
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/Ecommerce_Sales_Analysis.git
   ```

5. **Push Code to GitHub:**
   ```powershell
   git push -u origin main
   ```

### Part B: Submitting to InternCircle

1. Log in to your **InternCircle Student/Intern Dashboard**.
2. Navigate to **Assignments / Projects** and select **Project 3: Customer Churn / Sales Trend Analysis**.
3. In the submission field, provide:
   * **GitHub Repository URL:** `https://github.com/<YOUR_GITHUB_USERNAME>/Ecommerce_Sales_Analysis`
   * **Project Title:** Customer Churn / Sales Trend Analysis Using Python
   * **Executive Summary:** Copy the executive findings and business recommendations from Section 12 of this README.
   * **Attachment / Deliverables:** Attach the zipped `outputs/` folder containing the 6 visualization PNGs and the `reports/` folder.
4. Click **Submit Project** for mentor review.

---

## 15. License & Academic Integrity

This project is submitted in completion of **Project 3** for the **InternCircle Data Science & AI Internship**. All code, analytical models, and strategic findings are developed for professional assessment and educational purposes.

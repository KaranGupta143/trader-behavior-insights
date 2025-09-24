# Trader Behavior Insights Project  

This project analyzes trader performance and behavior using historical trade data combined with market sentiment indicators.  
The goal is to understand **how traders behave**, **what factors impact profitability**, and **how sentiment correlates with trading outcomes**.  

---

## 📌 Features  

- Data preprocessing and cleaning of raw trading data  
- Aggregated performance metrics per **account** and per **day**  
- Profitability analysis (per trade, per account, per day)  
- Advanced analytics:
  - Sharpe ratio per account  
  - Max drawdown calculation (planned extension)  
  - Survival analysis of trading tenure (planned extension)  
  - Clustering traders based on behavioral features (future scope)  
- Integration of **Fear & Greed Index** sentiment data  
- Export of results to CSV for further reporting  

---

## 📂 Project Structure  
trader-sentiment-analysis/
│
├── data/                         # Raw input datasets
│   ├── fear_greed_index.csv
│   └── historical_data.csv
│
├── outputs/                      # Generated results
│   ├── account_daily_summary.csv
│   ├── account_pnl_stats.csv
│   ├── avg_pnl_by_fg_label.png
│   ├── feature_importances.csv
│   ├── profit_rate_vs_fg_score.png
│   └── rf_profitable_model.pkl
│
├── analysis.py                   # Main analysis script
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation

✅ Explanation for each file/folder:

data/ → Input datasets (Fear & Greed Index + Historical Trader Data).

outputs/ → Saare results (CSV, PNG, Model file).

analysis.py → Main Python script jisme saara data preprocessing, analysis, aur modeling ka code hoga.

requirements.txt → Dependencies list.

README.md → Documentation (project explanation, steps to run, results, insights).

## ⚙️ Requirements  

- Python 3.8+  
- Install dependencies:  

```bash
pip install -r requirements.txt
```

### Main Libraries Used:

- pandas
- numpy
- matplotlib (optional for charts)
🚀 How to Run

Clone this repository:

git clone https://github.com/KaranGupta143/trader-behavior-insights.git
cd trader-behavior-insights


Place your raw data files into the data/ folder.

Example: trades.csv, fear_greed.csv

Run the main analysis script:

python analysis.py


Check the outputs/ folder for generated reports.

📊 Example Outputs

account_daily_summary.csv: Aggregated daily PnL, win rates, number of trades, and sentiment scores

account_pnl_stats.csv: Per-account mean, std, total PnL, and Sharpe ratio

🔮 Future Enhancements

Add visualization dashboards for trader clusters

Apply machine learning to classify trader behavior

Perform Granger causality tests between sentiment and performance

🧑‍💻 Author

Developed by [Karan Gupta](https://github.com/KaranGupta143) as part of an academic assignment/project. 


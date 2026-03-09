# Football Betting: Profit Optimization through Machine Learning

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-179C3A?style=for-the-badge&logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

# Executive Summary
This end-to-end Machine Learning project focuses on predicting football match outcomes (Over/Under 2.5 goals). The **primary goal** of this repository is to quantify the **exact business value** of predictive modeling. 

By improving the model's accuracy from a baseline of **55.4%** to **57.9%**, the optimal betting volume increased from 67,419 to 77,911 bets per year. This risk reduction translates to a **increase in expected yearly profit by $6,707 USD** for a hypothetical bookmaker.

# Business Impact & Model Performance
The table below illustrates the direct correlation between machine learning accuracy and business profitability.

| Model | Accuracy | ROC AUC | Optimal Profit $\Pi^*$ | $\Delta\Pi$ (vs baseline) |
|:---|:---:|:---:|:---:|:---:|
| **Baseline ($A_0$)** | **0.554** | **0.571** | **$66,260** | — |
| **Extended Model ($A_1$)** | **0.579** | **0.624** | **$72,967** | **+$6,707** |


## Project Architecture & Notebooks
The project is structured into a logical, sequential pipeline:

* `00_data_preprocessing.ipynb` & `00b_data_preprocessing_additional_features.ipynb`
    * Data cleaning, normalization, and handling European date formats.
    * Feature engineering: Building leakage-safe rolling stats and aggregated odds.
* `01_data_exploration.ipynb`
    * Exploratory Data Analysis (EDA) of goal patterns across different countries (e.g., highest O2.5 rates in Germany/Netherlands, lowest in Spain/Greece).
* `02_baseline_model.ipynb`
    * Establishing the $A_0$ benchmark (Accuracy: 0.55) using basic pre-match identifiers and date-derived features.
* `03_advanced_model_features_only.ipynb`
    * Training the extended $A_1$ model (XGBoost).
    * **Key finding:** Hyperparameter tuning and feature selection (using XGBoost's Top 10 features) significantly outperformed raw 49-feature models.
* `04_Profit_calculation_for_A1.ipynb` & `04b_accuracy_profit_plot_updated.ipynb`
    * Applying the mathematical framework to convert ROC AUC / Accuracy into tangible financial metrics ($\Pi^*$).
* `05_model_overview.ipynb` & `05b_economic_summary.ipynb`
    * Final evaluation, executive summaries, and business interpretation.

## Key Technical Learnings
1.  **Feature Selection vs. Noise:** The initial 49-feature set introduced noise. Applying targeted feature selection drastically improved XGBoost performance from 0.5916 to 0.6032 (AUC).
2.  **Hyperparameter Tuning:** Default parameters were suboptimal. Tuning increased the XGBoost AUC from 0.6032 to 0.6241.
3.  **Algorithmic Business Optimization:** A model's value isn't just in its ROC curve. Small accuracy improvements allow a bookmaker to lower their optimal margins, safely attracting a higher volume of bets, which exponentially increases absolute profit.

## How to Run
1. Clone the repository.
2. Ensure you have the required libraries installed (`pip install pandas numpy scikit-learn xgboost matplotlib seaborn`).
3. Run the Jupyter notebooks sequentially from `00` to `05`. 
*(Note: Ensure your local `/data/raw/` directory structure matches the data loaders in notebook `00`).*

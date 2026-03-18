# Real-State-Finance
# King County House Price Analysis
> Predicting residential real estate prices in King County, WA (May 2014 – May 2015)
 
---
 
## Project Overview
 
This project builds a machine learning pipeline to predict house sale prices in King County, including Seattle. The dataset contains **21,613 transactions** spanning one year, with **21 features** covering property characteristics, location, and sale metadata.
 
**Primary objective:** Identify which features most significantly drive house prices, and build a model that accurately predicts them — with special attention to the **luxury segment (≥ $650K)**.
 
---
 
## Repository Structure
 
```
king-county-analysis/
│
├── king_ country_ houses_aa.csv   # Raw dataset
├── king_county_analysis.ipynb     # Main Jupyter notebook (full pipeline)
├── presentation.pptx              # Mini-presentation
└── README.md                      # This file
```
---
## ML Pipeline
 
### 1 - Exploratory Data Analysis (EDA): Understand the data.
 
| Check | Finding |
|-------|---------|
| Shape | 21,613 rows × 21 columns |
| Missing values | None — no imputation needed |
| Data types | Mix of int64, float64, object (date) |
| Price range | $78,000 – $7,700,000 |
| Median price | $450,000 · Mean: $540,198 |
 
---
 
### 2 - Data Cleaning
 
Two specific issues identified and resolved:
 
**Outlier removal:**
- `bedrooms = 33` — flagged as a likely data entry error. Cross-checking `sqft_living` confirms it doesn't make sense for a 33-bedroom house. Removed.
- `bedrooms = 0` or `bathrooms = 0` — a house with no rooms is likely a plot of land or a data error. Removed.
 
**Zeros that are valid (kept as-is):**
- `sqft_basement = 0` → normal, many houses have no basement
- `yr_renovated = 0` → normal, means never renovated
 
**Decision rationale:** Unlike a blanket IQR removal, these are targeted removals based on domain logic. Preserved all other extreme values (large lots, high prices) because they represent real market segments, especially relevant for posterior luxury analysis.
 
---
 
### 3 - EDA Key Findings
 
**Price distribution:**
- Right-skewed (skewness = 4.026). log-transform reduces to 0.430
- This makes linear model assumptions more appropriate
 
**Top correlations with price:**
 
| Feature | r |
|---------|---|
| sqft_living | 0.702 |
| grade | 0.668 |
| sqft_above | 0.605 |
| sqft_living15 | 0.585 |
| bathrooms | 0.526 |
 
**Waterfront premium:** Waterfront properties have a median price approximately 3× higher than non-waterfront.
 
**Density lens (sqft_living vs sqft_lot):** High-value properties (≥$650K) cluster toward high sqft_living regardless of lot size, confirming that interior space drives price more than land size in this market.
 
**Luxury vs Standard segment:**
- Luxury (≥$650K): **5,322 properties — 24.6%** of total
- Standard (<$650K): 16,274 properties — 75.4%
 
---
 
### 4 - Feature Engineering
 
7 new features derived from existing columns:
 
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `house_age` | `sale_year − yr_built` | Age is more intuitive than raw year built |
| `was_renovated` | `yr_renovated > 0` | Binary flag — renovated homes command a premium |
| `years_since_reno` | `sale_year − yr_renovated` | Recency of renovation matters |
| `price_per_sqft` | `price / sqft_living` | EDA only — not used as predictor (data leakage) |
| `has_basement` | `sqft_basement > 0` | Presence often more impactful than size |
| `living_vs_neighbors` | `sqft_living / sqft_living15` | Relative size vs neighborhood context |
| `sale_month` | from `date` | Seasonal price variation |
| `lot_utilization` | `sqft_living / sqft_lot` | EDA only — house-to-land ratio |
 
**Dropped from features:**
- `id` — no predictive value
- `lat`, `long` — too granular; location captured via `zipcode` neighborhood signals
- `price_per_sqft`, `lot_utilization` — computed from price/sqft_living, would cause data leakage if used as predictors
 
---
 
### 5 - Preprocessing
 
**Why log-transform the target?** Reduces skewness from 4.026 to 0.430, making the linear model's normality assumption more appropriate. All metrics are back-transformed via `np.expm1()` for interpretability in real dollar terms.
 
**Why fit scaler only on train?** Fitting on test data leaks test set statistics into the model, artificially inflating metrics. The scaler must only "see" training data.
 
---
 
### 6 - Model Selection & Comparison
 
Five models trained on `log(price)` and evaluated on back-transformed dollar predictions:
 
| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| **Gradient Boosting** | $184,198 | $119,308 | **0.7408** |
| Random Forest | $189,096 | $114,498 | 0.7268 |
| Lasso | $196,553 | $129,807 | 0.7048 |
| Ridge | $196,930 | $129,942 | 0.7037 |
| Linear Regression | $196,941 | $129,944 | 0.7037 |
 
**Why Gradient Boosting wins:**
Tree-based ensemble methods (RF and GB) outperform linear models because the price-feature relationships are non-linear, grade has diminishing returns at the high end, house_age interacts with grade, etc. Gradient Boosting specifically wins by sequentially correcting errors from previous trees.
 
**Note on GridSearchCV:** Hyperparameter tuning was evaluated but omitted for runtime. Gradient Boosting outperforms all models out-of-the-box. RandomizedSearchCV is recommended for production use.
 
---
 
### 7 - Results & Interpretation
 
**Final model — Gradient Boosting:**
 
| Metric | Overall | Luxury ≥$650K |
|--------|---------|----------------|
| R² | 0.7408 | 0.5631 |
| RMSE | $184,198 | $313,633 |
| N | 4,320 | 1,025 |
 
**Top feature importances (Gradient Boosting):**
 
| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `grade` | 0.409 |
| 2 | `sqft_living` | 0.334 |
| 3 | `house_age` | 0.099 |
| 4 | `sqft_living15` | 0.052 |
| 5 | `sqft_lot15` | 0.036 |
 
**Why grade > sqft_living in importance despite lower correlation?**
Correlation measures linear relationship with price in isolation. Feature importance measures contribution to prediction accuracy when all features interact simultaneously.
 
**Luxury segment (≥$650K):**
- R² drops from 0.74 → 0.56 and RMSE nearly doubles ($184K → $314K)
- **High-end pricing is more idiosyncratic — unique architectural features, views, and micro-location effects not captured in standard features**
- Suggests a dedicated model for the luxury segment would be beneficial
 
---
 
King County Real Estate Analysis Project
Dataset: [Kaggle — King County Houses](https://www.kaggle.com/datasets/minasameh55/king-country-houses-aa)

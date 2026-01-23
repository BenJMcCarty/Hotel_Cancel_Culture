# AI Coding Agent Instructions - Hotel Cancel Culture

## Project Overview
Data science project predicting hotel reservation cancellations using machine learning (Random Forest, Gradient Boosting classifiers) on historical booking data from two Portuguese hotels. Follows CRISP-DM framework with structured notebook-based workflow.

## Architecture & Data Flow

### Notebook Pipeline Structure (Sequential)
1. **1.0 Data Extraction** → Download ZIP from source URL, extract `H1.csv` and `H2.csv` to `data/raw/`
2. **1.2 Concatenate** → Add `HotelNumber` column ('H1'/'H2'), combine into `data/raw/combined.parquet` (zstd compression)
3. **2.1 Initial EDA** → Exploratory analysis, identifies features to drop (e.g., `company` - 95% missing)
4. **3.0 Feature Engineering** → Create temporal features (from `ArrivalDateYear/Month/Day`), occupancy calculations (adults+children+babies), outputs to `data/3.1_temporally_updated_data.parquet` and `data/3.2_data_with_occupancies.parquet`
5. **4.0 Advanced Modeling** → Final classification models consume engineered data

**Critical**: Each notebook consumes outputs from previous stages. Use relative paths from notebook location (e.g., `../../data/raw/combined.parquet`).

### Data Format Conventions
- **Raw data**: CSV → Parquet conversion (zstd or brotli compression via pyarrow)
- **Identifiers**: UUID generation via `db_utils.generate_uuid_list()` for each reservation
- **Target variable**: `is_canceled` (0=checkout, 1=cancellation including no-shows)

## Key Technical Patterns

### Data Processing (`src/data_preprocessing.py`)
```python
# Standard pattern for creating date columns
create_arrival_date_columns(df)  # Creates Arrival_Date, Departure_Date, Booking_Date
# Uses: ArrivalDateYear, ArrivalDateMonth, ArrivalDateDayOfMonth + LeadTime + Stays columns
```

### Database Utilities (`src/db_utils.py`)
- DuckDB integration with context managers for query operations
- `add_hotel_number_to_dataframe()` - Extracts hotel number from filename (e.g., "H1" from "H1.parquet"), adds UUID and HotelNumber columns

### Classification Modeling (`src/classification.py`)
- `model_scores()` returns: `(train_score, test_score, train_log_loss, test_log_loss)`
- Uses scikit-learn RandomForest/HistGradientBoosting with ColumnTransformer pipelines
- Visualization functions: `plot_comparison_hist()`, `plot_comparison_count()` for feature analysis

## Development Workflows

### Running Notebooks
Execute cells sequentially top-to-bottom. Import pattern:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
```

### Data Loading
- **From notebooks**: Use relative paths: `pd.read_parquet('../../data/3.2_data_with_occupancies.parquet')`
- **In src modules**: Use absolute paths or `config/config.yaml` paths (e.g., `raw_data_path__combined`)

### Feature Engineering Checklist
1. Drop post-arrival features (`reservation_status`, `reservation_status_date`, `assigned_room_type` - only known after checkout)
2. Handle categoricals with 100+ values by condensing to 4-5 key categories
3. Missing value strategy: Drop if >95% missing, else fill with mode/frequent value
4. See `data/Feature_Dictionary.md` for original variable definitions from source article

## Project-Specific Conventions

### Naming
- Hotels identified as 'H1' (resort) and 'H2' (city hotel)
- Notebook naming: `X.Y_Description.ipynb` where X = phase, Y = sequence
- Parquet outputs: `X.Y_descriptive_name.parquet` matching notebook version

### Important Features (from domain knowledge)
- **Lead time**: Days between booking and arrival - strong cancellation predictor
- **Deposit type**: No Deposit / Refundable / Non Refund - critical for modeling
- **Previous cancellations**: Historical customer behavior
- **Market segment**: TA (Travel Agents), TO (Tour Operators), Direct, etc.
- **ADR**: Average Daily Rate - calculated as total_lodging_cost / staying_nights

### Deprecated Patterns
- Old CSV concatenation approach (see 1.2 notebook warning) - now use Parquet throughout
- Time series modeling files (`time_series_modeling_old.py`) - focus on classification instead

## Common Gotchas
- `NULL` in categorical columns (Agent, Company) ≠ missing data, means "not applicable" (e.g., no travel agent)
- pandas display settings customized: `pd.set_option('display.max_columns', None)`
- Sweetviz library used for automated EDA reports in 2.1
- Config uses `pyarrow` engine with `brotli`/`snappy`/`zstd` compression - verify engine installed

## External Dependencies
- Data source: https://www.sciencedirect.com/science/article/pii/S2352340918315191
- Citation required in publications (see README.md)
- Original dataset: 40,060 observations (H1) + 79,330 (H2) from July 2015-Aug 2017

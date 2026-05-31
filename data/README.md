# Data Directory

This directory contains all data files for the Box Office Prediction ML Pipeline, organized following data science best practices.

## Directory Structure

```
data/
├── raw/                     # Original, immutable data dump
│   └── box_office_movies/   # Raw box office and movie data
├── external/                # Data from third-party sources
│   ├── general/             # General market and economic data
│   └── tmdb/                # The Movie Database (TMDb) API data
├── interim/                 # Intermediate data that has been transformed
├── processed/               # Final, analysis-ready datasets
└── README.md               # This file
```

## Data Categories

### Raw Data (`data/raw/`)
- **Purpose**: Original, immutable datasets as received from sources
- **Contents**: Box office revenue data, movie metadata, industry reports
- **Policy**: Never modify files in this directory; treat as read-only
- **Backup**: Should be backed up and versioned

### External Data (`data/external/`)
- **Purpose**: Data from third-party sources and APIs
- **Contents**: Economic indicators, social media metrics, industry benchmarks
- **Sources**: TMDb API, Box Office Mojo, economic data providers
- **Refresh**: May be updated periodically via automated pipelines

### Interim Data (`data/interim/`)
- **Purpose**: Intermediate transformations and data cleaning steps
- **Contents**: Partially processed datasets, feature engineering outputs
- **Usage**: Debugging, pipeline validation, exploratory analysis
- **Lifecycle**: Can be regenerated from raw data

### Processed Data (`data/processed/`)
- **Purpose**: Final, analysis-ready datasets for model training
- **Contents**: Training/validation splits, engineered features, model inputs
- **Format**: Optimized formats (Parquet, CSV) for ML consumption
- **Usage**: Direct input to ML models and analysis notebooks

## Data Governance

### File Naming Conventions
```
{source}_{type}_{version}_{date}.{extension}

Examples:
- box_office_movies_enriched_v7_20241201.csv
- tmdb_metadata_raw_20241201.json
- economic_indicators_processed_v2_20241201.parquet
```

### Version Control
- Raw data: Immutable, versioned by date received
- Processed data: Semantic versioning (v1.0, v1.1, etc.)
- Schema changes: Major version bump
- Feature additions: Minor version bump

### Data Quality
- All datasets include data quality reports
- Automated validation checks before processing
- Schema validation and type checking
- Missing value and outlier analysis

## Usage Examples

### Loading Data in Python
```python
import pandas as pd
from pathlib import Path

# Define data paths
DATA_ROOT = Path("data")
RAW_DATA = DATA_ROOT / "raw"
PROCESSED_DATA = DATA_ROOT / "processed"

# Load raw data
movies_raw = pd.read_csv(RAW_DATA / "box_office_movies" / "prod_movie_dataset_enriched_v8_imputed.csv")

# Load processed data
X_train = pd.read_csv(PROCESSED_DATA / "X_train_v1.csv")
y_train = pd.read_csv(PROCESSED_DATA / "y_train_v1.csv")
```

### Data Pipeline Integration
```python
from box_office.utils.data_loader import DataLoader

# Initialize data loader
loader = DataLoader(data_root="data")

# Load datasets for training
train_data = loader.load_training_data(version="v1")
validation_data = loader.load_validation_data(version="v1")
```

## Data Sources

### Primary Sources
1. **Box Office Data**: Revenue, theater counts, release patterns
2. **Movie Metadata**: Cast, crew, genres, production details
3. **Economic Data**: Market conditions, inflation, consumer confidence
4. **Social Media**: Twitter/X mentions, YouTube trailer views

### API Integrations
- TMDb API: Movie metadata and ratings
- Box Office Mojo: Historical revenue data
- Economic APIs: Federal Reserve, market indicators

## Security & Privacy

### Data Protection
- No personally identifiable information (PII) stored
- All data aggregated and anonymized
- Secure storage with encryption at rest
- Access controls and audit logging

### Compliance
- Follows TMDb API terms of service
- Complies with data usage policies
- Regular compliance audits
- Data retention policies enforced

## Maintenance

### Regular Tasks
- Weekly data quality checks
- Monthly external data refresh
- Quarterly data audit and cleanup
- Annual schema review and optimization

### Monitoring
- Data freshness monitoring
- Schema drift detection
- Quality metrics tracking
- Storage utilization monitoring

## Access Patterns

### Development
```bash
# Local development data access
export DATA_PATH="./data"
python box_office/ml/model_training.py --data-path $DATA_PATH
```

### Production
```bash
# Production data from S3
export DATA_PATH="s3://box-office-data-prod/datasets/"
python box_office/ml/model_training.py --data-path $DATA_PATH
```

### Analysis
```bash
# Jupyter notebook data exploration
jupyter notebook analysis/data_exploration.ipynb
```

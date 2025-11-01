# Data Source Directory

This directory contains datasets used for A/B testing validation.

## Structure

```
datasource/
└── data/
    └── sample_ab_test.csv  # Sample A/B test dataset
```

## Sample Dataset

The `sample_ab_test.csv` file contains synthetic A/B test data with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| user_id | int | Unique user identifier |
| group | string | Test group ('control' or 'treatment') |
| conversion | int | Whether user converted (1) or not (0) |
| revenue | float | Revenue generated from user |
| session_duration | int | Time spent in seconds |
| page_views | int | Number of pages viewed |
| signup_date | date | User signup date |

## Using Your Own Data

To use your own A/B test dataset:

1. Place your CSV file in the `datasource/data/` directory
2. Update the `dataset_path` in your `ABTestContext`
3. Ensure your dataset includes:
   - A group/variant column for test assignment
   - Columns for your success metrics
   - Sufficient sample size for your expected effect

## Data Requirements

For proper validation, your dataset should:
- Have clear group assignments (control vs treatment)
- Include all metrics referenced in success_metrics
- Have adequate sample size (the agent will perform power analysis)
- Be clean and properly formatted (CSV, JSON, or Parquet)

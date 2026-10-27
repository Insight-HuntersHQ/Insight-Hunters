# The Insight Hunters Streamlit Application

This directory contains the Streamlit dashboard and the validated tuberculosis inference pipeline.

## Required Model Files

The application requires two trained model files:

- `nih_tb_final_selected_compat.keras`
- `mobilenetv2_validator_best_compact.keras`

These files are not included in the GitHub repository because trained model checkpoints are stored separately.

Obtain the model files from the Insight Hunters team and place them directly inside the `app/` directory.

## Running the Application

From the repository's root directory, install dependencies:

```bash
pip install -r requirements.txt
```

Then launch the dashboard:

```bash
streamlit run app/tb_dash.py
```

## Important Notice

This application is an academic research prototype. It is not clinically validated and must not be used for medical diagnosis or treatment decisions.

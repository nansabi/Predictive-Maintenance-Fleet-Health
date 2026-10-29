# Predictive Maintenance Pipeline: RUL Forecasting on NASA C-MAPSS

An end-to-end machine learning project that predicts the remaining useful life (RUL) of aircraft turbofan engines using NASA's C-MAPSS dataset.

The project covers data processing, feature engineering, model training, experiment tracking, simulated drift monitoring, model retraining, local API serving with Docker, and AI-generated maintenance summaries.

## Project Overview

Aircraft engines gradually degrade as they operate. Predictive maintenance uses sensor readings to estimate how many operating cycles may remain before an engine reaches the end of its useful life.

This project builds a pipeline that:

- Processes engine sensor data using PySpark.
- Creates time-series features from sensor readings.
- Trains and evaluates regression models, including XGBoost.
- Tracks experiments and registers models with MLflow.
- Simulates data drift and evaluates whether retraining improves performance.
- Serves RUL predictions through a FastAPI application running in Docker.
- Generates maintenance summaries with Google Gemini and evaluates text overlap with ROUGE.

## Tech Stack

| Area | Tools |
|---|---|
| Data processing | PySpark, Parquet |
| Machine learning | Python, XGBoost, scikit-learn |
| Experiment tracking | MLflow |
| Drift monitoring | Population Stability Index (PSI), Kolmogorov-Smirnov (KS) test |
| API | FastAPI |
| Containerization | Docker |
| AI-generated summaries | Google Gemini |
| Summary evaluation | ROUGE and automated rule-based checks |
| Environment | Google Colab, local Docker |

## Dataset

**Dataset:** NASA Prognostics Center of Excellence, Turbofan Engine Degradation Simulation Data Set (C-MAPSS).

This project uses the **FD001** subset, which represents one operating condition and one fault mode.

The training dataset contains 20,631 sensor-reading records from 100 engines. The target is Remaining Useful Life (RUL), capped at 125 cycles.

Dataset resource: [NASA PCoE data repository](https://data.phmsociety.org/nasa/) (item 6, Turbofan Engine Degradation Simulation).

## Pipeline

1. **Data preparation:** Read the raw data, calculate RUL, and split engines into training and validation groups.
2. **Feature engineering:** Create rolling statistics, differences, and scaled sensor features.
3. **Model training:** Compare a mean baseline, Ridge regression, and XGBoost configurations.
4. **Experiment tracking:** Log experiments and register the selected model in MLflow.
5. **Drift monitoring:** Simulate data changes and evaluate retraining decisions.
6. **API serving:** Serve predictions through FastAPI in a local Docker container.
7. **AI summaries:** Generate short maintenance summaries and evaluate them against reference summaries.

## Feature Engineering

The pipeline creates **102 features** from 17 operating-condition and sensor columns.

- Original sensor and operating-condition values.
- Rolling means over windows of 5 and 10 cycles.
- Rolling standard deviations over windows of 5 and 10 cycles.
- Differences between consecutive cycles.
- Scaled features using parameters fitted on training data.

Rolling windows include the current cycle and available previous cycles. Scaling parameters are fitted on training data and reused for validation, testing, and API inference.

## Model Training and Results

Several models were evaluated. XGBoost configuration D was selected based on validation performance.

| Model | Validation RMSE |
|---|---:|
| Mean baseline | 41.53 |
| Ridge regression | 19.19 |
| XGBoost configuration D (v1) | **15.84** |

Lower RMSE indicates more accurate RUL predictions. RMSE is measured on capped RUL (125).

The v1 model was registered in MLflow under `cmapss_rul_model`. The project later simulated drift, retrained a challenger, and promoted v2 as the current champion.

### Test result for v1

| Metric | Result |
|---|---|
| Model evaluated | XGBoost configuration D (v1) |
| Test RMSE | 18.20 |
| Test engines | 100 |
| RUL target cap | 125 cycles |

The test evaluation uses the final available cycle for each of the 100 test engines. The test set was kept separate from model selection.

> **Model-version note:** The API serves the current champion, v2, which was retrained after simulated drift. The validation RMSE of 15.84 and test RMSE of 18.20 belong to **v1**. v2 has not been evaluated on the test set, so the v1 test result must not be read as the deployed model's test performance.

## Drift Monitoring and Retraining

Drift monitoring was tested using simulated changes in sensor data. PSI and KS tests were used to examine changes in feature distributions. Model performance was also monitored as part of the retraining decision process.

### Simulated drift experiments

| Experiment | Result | Decision |
|---|---|---|
| Batch 0: unchanged data | No drift trigger | No retraining |
| Batch 1: mild shift | Challenger RMSE 14.63; champion RMSE 13.90 | Challenger rejected |
| Batch 2: strong shift | Champion RMSE 25.67; challenger RMSE 15.48 | Challenger promoted |

The retraining process requires at least a 3% improvement on the same holdout engines before promoting a challenger.

These experiments demonstrate the decision process under simulated conditions. They do not establish how the system will behave under every real-world form of engine degradation or data drift.

**Limitation:** The effect of promoting v2 on unchanged data (Batch 0) has not yet been verified.

## FastAPI and Docker Serving

The project includes a FastAPI service that loads the model and generates RUL predictions from engine sensor readings. The API runs locally inside Docker; it has not been deployed to a public cloud service.

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check service health |
| POST | `/predict` | Predict RUL for one engine |
| POST | `/predict_batch` | Predict RUL for multiple engines |

Interactive API documentation is available at http://localhost:8000/docs while the service is running.

### Run locally

Prerequisite: Docker installed and running. From the repository root:

```bash
cd api
docker build -t cmapss-rul-api .
docker run --rm -p 8000:8000 cmapss-rul-api
```

Keep the terminal running. In a second terminal, from the `api` folder, send the sample request:

```powershell
curl.exe -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  --data-binary "@sample_request.json"
```

### Prediction output

The API returns an estimated RUL, a project-defined risk category, and a warning when fewer than 10 cycles of history are available.

| Predicted RUL | Risk category |
|---|---|
| Below 30 cycles | High |
| 30 to below 60 cycles | Medium |
| 60 cycles or more | Low |

Predicted RUL is clipped to the range 0 to 125 cycles. These categories are project-defined indicators, not certified maintenance decisions.

### Feature parity and API checks

The API feature builder was compared with the PySpark feature pipeline on 5 engines and 3 cycles per engine, covering 1,530 feature values. The maximum absolute difference was 6.87e-09.

Separately, a Docker API prediction was compared with the corresponding offline prediction for the sample request. Both produced approximately 123.83 cycles.

These results validate the tested examples; they do not guarantee correctness for every possible input.

## AI-Generated Maintenance Summaries

Google Gemini was used to generate short maintenance summaries. A set of 25 summaries was evaluated against reference summaries.

### ROUGE evaluation

| Metric | Template baseline | Gemini |
|---|---:|---:|
| ROUGE-1 | 0.4032 | 0.5769 |
| ROUGE-2 | 0.1703 | 0.3319 |
| ROUGE-L | 0.3214 | 0.4950 |

The Gemini summaries achieved higher ROUGE scores than the template baseline in this evaluation.

### Faithfulness checks

- Average automated rule-based checks passed per summary: 4.92 out of 5.
- Overall pass rate: 98.4%.
- Generated report: `fleet_report.csv`.

These checks measure consistency with the input facts according to the implemented rules. They do not establish complete factual correctness or guarantee that maintenance recommendations are safe.

**Evaluation limitations:** The evaluation covered 25 summaries. Reference summaries were AI-assisted and reviewed by one person. ROUGE measures overlap with reference text, while rule-based checks cover only the rules implemented. These results are exploratory rather than a comprehensive independent evaluation.

## Project Limitations

- **Dataset coverage:** Only FD001 is covered; FD002 to FD004 have not been evaluated.
- **Simulated drift:** Drift experiments use simulated changes rather than a live production data stream.
- **Model-version evaluation:** The reported test RMSE of 18.20 belongs to v1. The deployed champion is v2, which has not been evaluated on the test set.
- **Retraining safety:** The effect of promoting v2 on unchanged data has not yet been verified.
- **Serving:** The API runs locally in Docker and is not publicly cloud-deployed.
- **Security:** The API does not currently implement authentication.
- **Fleet-level analysis:** The project focuses on engine-level RUL predictions rather than a complete fleet maintenance or cost-optimization system.
- **Summary evaluation:** The summary evaluation is small and does not independently establish factual correctness.
- **Operational use:** Predictions are experimental estimates and should not be used as the sole basis for real aircraft maintenance decisions.

## Skills Demonstrated

- Time-series feature engineering
- PySpark data processing
- Regression model training and evaluation
- XGBoost model selection
- MLflow experiment tracking and model registry
- Data drift monitoring and retraining evaluation
- FastAPI REST API development
- Docker containerization
- API validation and parity testing
- LLM-generated text evaluation using ROUGE and rule-based checks

## Project Structure

```text
project/
├── notebooks/
├── api/
│   ├── app.py
│   ├── features.py
│   ├── feature_cols.json
│   ├── model.json
│   ├── model_info.json
│   ├── scaling_parameters.json
│   ├── sample_request.json
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README_api.md
├── data/
│   └── fleet_report.csv
├── docs/
└── README.md
```

## Future Improvements

- Evaluate the pipeline on additional C-MAPSS subsets.
- Evaluate v2 on the test set before reporting its test performance.
- Compare v1 and v2 on unchanged data to understand the effect of retraining.
- Add authentication and further production security controls to the API.
- Evaluate generated summaries for factual consistency and unsupported claims.
- Explore cloud deployment.
- Investigate fleet-level maintenance planning and cost-aware decisions.

## Dataset Reference

A. Saxena and K. Goebel, *Turbofan Engine Degradation Simulation Data Set (C-MAPSS)*, NASA Prognostics Center of Excellence.

---

**Project status:** End-to-end experimental pipeline built, with model evaluation, simulated drift experiments, a local Docker API, and an exploratory AI-summary evaluation. Real-world deployment and further validation remain future work.
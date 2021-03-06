# MorphL Model for Predicting Shopping Stage

## Installation

1. On a VM running the [MorphL Orchestrator](https://github.com/Morphl-AI/MorphL-Orchestrator), clone repository under `/opt/ga_epna`.

```
WHERE_GA_EPNA_IS='https://github.com/Morphl-AI/MorphL-Model-Ecommerce-Shopping-Stage'
git clone ${WHERE_GA_EPNA_IS} /opt/ga_epna
```

2. Create credentials files.

- Add service account for connecting to the Google Analytics API in the `/opt/secrets/ga_epna/service_account.json` file.
- Add Google Analytics view id in the `/opt/secrets/ga_epna/viewid.txt` file.

3. Create environment variables:

```
bash /opt/ga_epna/setup_environment.sh
```

4. Log out of `airflow` and back in again, and verify that your key file and view ID have been configured correctly:

```
env | grep GA_EPNA_KEY_FILE_LOCATION
env | grep GA_EPNA_VIEW_ID
```

5. Setup Cassandra tables:

```
bash /opt/ga_epna/install.sh
```

6. Load data from the Google Analytics API and start pipelines:

```
bash /opt/ga_epna/ingestion/load_historical_data/load_ga_epna_historical_data.sh
```

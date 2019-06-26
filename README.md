# MorphL Model for Predicting Shopping Stage

## Installation

1. On a VM running the [MorphL Orchestrator](https://github.com/Morphl-AI/MorphL-Orchestrator), clone repository under `/opt/ga_epna`.

```
WHERE_GA_EPNA_IS='https://github.com/Morphl-AI/MorphL-Model-Ecommerce-Shopping-Stage'
git clone ${WHERE_GA_EPNA_IS} /opt/ga_epna
```

2. Create environment variables:

```
bash /opt/ga_epna/setup_environment.sh
```

3. Log out of `airflow` and back in again, and verify that your key file and view ID have been configured correctly:

```
env | grep GA_EPNA_KEY_FILE_LOCATION
env | grep GA_EPNA_VIEW_ID
```

4. Setup Cassandra tables and Airflow pipelines:

```
bash /opt/ga_epna/install.sh
```

# TEMPFILE_A is the duration of the training interval in days
export TEMPFILE_A=$(mktemp)
# TEMPFILE_B is the duration of the predictions interval in days
export TEMPFILE_B=$(mktemp)
# TEMPFILE_C is the Python start date (today)
export TEMPFILE_C=$(mktemp)

python /opt/ga_epna/ingestion/load_historical_data/ga_epna_load_historical_data.py ${TEMPFILE_A} ${TEMPFILE_B} ${TEMPFILE_C}
rc=$?
if [ ${rc} -eq 0 ]; then
  echo 'Emptying the relevant Cassandra tables ...'
  echo
  cqlsh ${MORPHL_SERVER_IP_ADDRESS} -u morphl -p ${MORPHL_CASSANDRA_PASSWORD} -f /opt/ga_epna/ingestion/load_historical_data/ga_epna_truncate_tables_before_loading_historical_data.cql
  
  # Write configuration parameters in corresponding Cassandra table
  DAYS_TRAINING_INTERVAL=$(<${TEMPFILE_A})
  DAYS_PREDICTION_INTERVAL=$(<${TEMPFILE_B})

  echo ${DAYS_TRAINING_INTERVAL}
  echo ${DAYS_PREDICTION_INTERVAL}

  sed "s/DAYS_TRAINING_INTERVAL/${DAYS_TRAINING_INTERVAL}/g;s/DAYS_PREDICTION_INTERVAL/${DAYS_PREDICTION_INTERVAL}/g" /opt/ga_epna/ingestion/load_historical_data/insert_into_ga_epna_config_parameters.cql.template > /tmp/insert_into_config_parameters.cql
  cqlsh ${MORPHL_SERVER_IP_ADDRESS} -u morphl -p ${MORPHL_CASSANDRA_PASSWORD} -f /tmp/insert_into_config_parameters.cql

  # Reset Airflow and create dags
  echo 'Initiating the data load ...'
  echo
  stop_airflow.sh
  airflow delete 
  # airflow delete_dag ga_epna_training_pipeline
  airflow delete_dag ga_epna_prediction_pipeline
  python /opt/orchestrator/bootstrap/runasairflow/python/set_up_airflow_authentication.py
  
  # Create ingestion dag
  START_DATE_AS_PY_CODE=$(<${TEMPFILE_C})
  sed "s/START_DATE_AS_PY_CODE/${START_DATE_AS_PY_CODE}/g" /opt/ga_epna/ingestion/pipeline_setup/ga_epna_ingestion_airflow_dag.py.template > /home/airflow/airflow/dags/ga_epna_ingestion_pipeline.py

  # Create training dag and trigger pipeline
  # START_DATE_AS_PY_CODE=$(<${TEMPFILE_C})
  # sed "s/START_DATE_AS_PY_CODE/${START_DATE_AS_PY_CODE}/g;s/DAYS_TRAINING_INTERVAL/${DAYS_TRAINING_INTERVAL}/g;s/DAYS_PREDICTION_INTERVAL/${DAYS_PREDICTION_INTERVAL}/g" /opt/ga_epna/training/pipeline_setup/ga_epna_training_airflow_dag.py.template > /home/airflow/airflow/dags/ga_epna_training_pipeline.py
  # airflow trigger_dag ga_epna_training_pipeline
  
  # Create prediction dag
  # START_DATE_AS_PY_CODE=$(<${TEMPFILE_C})
  # sed "s/START_DATE_AS_PY_CODE/${START_DATE_AS_PY_CODE}/g;s/DAYS_PREDICTION_INTERVAL/${DAYS_PREDICTION_INTERVAL}/g" /opt/ga_epna/prediction/pipeline_setup/ga_epna_prediction_airflow_dag.py.template > /home/airflow/airflow/dags/ga_epna_prediction_pipeline.py
  
  start_airflow.sh
  echo 'The data load has been initiated.'
  echo
fi

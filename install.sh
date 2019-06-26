echo 'Installing the Shopping Stage Prediction module...'
echo

echo 'Initiating the Cassandra tables ...'
echo

cqlsh ${MORPHL_SERVER_IP_ADDRESS} -u morphl -p ${MORPHL_CASSANDRA_PASSWORD} -f /opt/ga_epna/cassandra_schema/ga_epna_cassandra_schema.cql

echo 'Setting up the pipeline...'
echo

stop_airflow.sh
# rm -rf /home/airflow/airflow/dags/*
# airflow resetdb -y &>/dev/null

# Write dynamic variables to the Airflow template file
# START_DATE=$(date +%Y-%m-%d)
# sed "s/START_DATE/${START_DATE}/g" /opt/ga_epna/pipeline/ga_epna_airflow_dag.py.template > /home/airflow/airflow/dags/ga_epna_airflow_dag.py

start_airflow.sh
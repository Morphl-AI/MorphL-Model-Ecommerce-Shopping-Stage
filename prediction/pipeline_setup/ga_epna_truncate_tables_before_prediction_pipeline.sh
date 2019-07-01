cqlsh ${MORPHL_SERVER_IP_ADDRESS} -u morphl -p ${MORPHL_CASSANDRA_PASSWORD} \
  -f /opt/ga_epna/prediction/pipeline_setup/ga_epna_truncate_tables_before_prediction_pipeline.cql

exit 0

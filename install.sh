echo 'Installing the Shopping Stage Prediction module...'
echo

echo 'Initiating the Cassandra tables ...'
echo

cqlsh ${MORPHL_SERVER_IP_ADDRESS} -u morphl -p ${MORPHL_CASSANDRA_PASSWORD} -f /opt/ga_epna/cassandra_schema/ga_epna_cassandra_schema.cql
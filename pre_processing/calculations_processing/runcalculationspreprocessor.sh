cp -r /opt/ga_epna /opt/code
cd /opt/code

spark-submit --driver-memory 4g --jars /opt/spark/jars/spark-cassandra-connector.jar,/opt/spark/jars/jsr166e.jar /opt/code/pre_processing/calculations_processing/ga_epna_calculations_preprocessor.py


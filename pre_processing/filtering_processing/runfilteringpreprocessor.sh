cp -r /opt/ga_epna /opt/code
cd /opt/code

spark-submit --jars /opt/spark/jars/spark-cassandra-connector.jar,/opt/spark/jars/jsr166e.jar /opt/code/pre_processing/filtering_processing/ga_epna_filtering_preprocessor.py


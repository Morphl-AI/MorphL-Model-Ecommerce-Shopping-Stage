from os import getenv
from pyspark.sql import functions as f, SparkSession


class CalculationsPreprocessor:

    def __init__(self):

        self.MASTER_URL = 'local[*]'
        self.APPLICATION_NAME = 'calculations_preprocessor'

        self.MORPHL_SERVER_IP_ADDRESS = getenv('MORPHL_SERVER_IP_ADDRESS')
        self.MORPHL_CASSANDRA_USERNAME = getenv('MORPHL_CASSANDRA_USERNAME')
        self.MORPHL_CASSANDRA_PASSWORD = getenv('MORPHL_CASSANDRA_PASSWORD')
        self.MORPHL_CASSANDRA_KEYSPACE = getenv('MORPHL_CASSANDRA_KEYSPACE')

    # Initialize the spark sessions and return it.
    def get_spark_session(self):

        spark_session = (
            SparkSession.builder
            .appName(self.APPLICATION_NAME)
            .master(self.MASTER_URL)
            .config('spark.cassandra.connection.host', self.MORPHL_SERVER_IP_ADDRESS)
            .config('spark.cassandra.auth.username', self.MORPHL_CASSANDRA_USERNAME)
            .config('spark.cassandra.auth.password', self.MORPHL_CASSANDRA_PASSWORD)
            .config('spark.sql.shuffle.partitions', 16)
            .config('parquet.enable.summary-metadata', 'true')
            .getOrCreate())

        log4j = spark_session.sparkContext._jvm.org.apache.log4j
        log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)

        return spark_session

    # Return a spark dataframe from a specified Cassandra table.
    def fetch_from_cassandra(self, c_table_name, spark_session):

        load_options = {
            'keyspace': self.MORPHL_CASSANDRA_KEYSPACE,
            'table': c_table_name,
            'spark.cassandra.input.fetch.size_in_rows': '150'}

        df = (spark_session.read.format('org.apache.spark.sql.cassandra')
                                .options(**load_options)
                                .load())

        return df

    def main(self):

        spark_session = self.get_spark_session()

        ga_epnau_features_filtered_df = self.fetch_from_cassandra(
            'ga_epnau_features_filtered', spark_session
        )

        ga_epnas_features_filtered_df = self.fetch_from_cassandra(
            'ga_epnas_features_filtered', spark_session
        )

        ga_epnah_features_filtered_df = self.fetch_from_cassandra(
            'ga_epnah_features_filtered', spark_session
        )

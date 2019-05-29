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

    def calculate_browser_device_features(self, users_df, sessions_df):
        users_sessions_df = (users_df
                             .join(
                                 sessions_df,
                                 'client_id',
                                 'inner'))

        transactions_by_device_df = (users_sessions_df
                                     .groupBy('mobile_device_branding')
                                     .agg(
                                         f.sum('transactions').alias(
                                             'transactions'),
                                         f.sum('transaction_revenue').alias(
                                             'transaction_revenue')
                                     ))

        transactions_by_device_df = (transactions_by_device_df
                                     .withColumn(
                                         'device_revenue_per_transaction',
                                         transactions_by_device_df.transaction_revenue /
                                         (transactions_by_device_df.transactions + 1e-5)
                                     ))

        transactions_by_browser_df = (users_sessions_df
                                      .groupBy('browser')
                                      .agg(
                                          f.sum('transactions').alias(
                                              'transactions'),
                                          f.sum('transaction_revenue').alias(
                                              'transaction_revenue')
                                      ))

        transactions_by_browser_df = (transactions_by_browser_df
                                      .withColumn(
                                          'browser_revenue_per_transaction',
                                          transactions_by_browser_df.transaction_revenue /
                                          (transactions_by_browser_df.transactions + 1e-5)
                                      ))

        return (users_sessions_df
                .join(
                    transactions_by_device_df,
                    'mobile_device_branding',
                    'inner'
                )
                .join(
                    transactions_by_browser_df,
                    'browser',
                    'inner')
                )

    def format_shopping_stages(self, df):
        format_stages_udf = f.udf(lambda stages: '|'.join(stages), 'string')

        return df.withColumn('shopping_stage', format_stages_udf('shopping_stage'))

    def aggregate_transactions(self, df):
        replace_stages_udf = f.udf(lambda stages: 'TRANSACTION' if stages.find(
            'TRANSACTION') != -1 else stages, 'string')

        return df.withColumn('shopping_stage', replace_stages_udf('shopping_stage'))

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

        users_df = self.calculate_browser_device_features(
            ga_epnau_features_filtered_df, ga_epnas_features_filtered_df)

        users_sessions_data = (users_df
                               .join(
                                   ga_epnas_features_filtered_df,
                                   'client_id',
                                   'inner'
                               ))

        final_data = (ga_epnah_features_filtered_df
                      .join(
                          users_sessions_data,
                          ['client_id', 'session_id'],
                          'inner'
                      ))

        final_data = self.format_shopping_stages(final_data)

        final_data = self.aggregate_transactions(final_data)

        final_data.show()

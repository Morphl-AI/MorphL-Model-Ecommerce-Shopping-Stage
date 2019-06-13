from os import getenv
from pyspark.sql import functions as f, SparkSession


class CalculationsPreprocessor:

    def __init__(self):

        HDFS_PORT = 9000
        PREDICTION_DAY_AS_STR = getenv('PREDICTION_DAY_AS_STR')
        UNIQUE_HASH = getenv('UNIQUE_HASH')

        self.MASTER_URL = 'local[*]'
        self.APPLICATION_NAME = 'calculations_preprocessor'

        self.MORPHL_SERVER_IP_ADDRESS = getenv('MORPHL_SERVER_IP_ADDRESS')
        self.MORPHL_CASSANDRA_USERNAME = getenv('MORPHL_CASSANDRA_USERNAME')
        self.MORPHL_CASSANDRA_PASSWORD = getenv('MORPHL_CASSANDRA_PASSWORD')
        self.MORPHL_CASSANDRA_KEYSPACE = getenv('MORPHL_CASSANDRA_KEYSPACE')

        self.HDFS_DIR_USER_FILTERED = f'hdfs://{self.MORPHL_SERVER_IP_ADDRESS}:{HDFS_PORT}/{PREDICTION_DAY_AS_STR}_{UNIQUE_HASH}_ga_epnau_filtered'
        self.HDFS_DIR_SESSION_FILTERED = f'hdfs://{self.MORPHL_SERVER_IP_ADDRESS}:{HDFS_PORT}/{PREDICTION_DAY_AS_STR}_{UNIQUE_HASH}_ga_epnas_filtered'
        self.HDFS_DIR_HIT_FILTERED = f'hdfs://{self.MORPHL_SERVER_IP_ADDRESS}:{HDFS_PORT}/{PREDICTION_DAY_AS_STR}_{UNIQUE_HASH}_ga_epnah_filtered'

        self.HDFS_DIR_CALCULATED_FEATURES = f'hdfs://{self.MORPHL_SERVER_IP_ADDRESS}:{HDFS_PORT}/{PREDICTION_DAY_AS_STR}_{UNIQUE_HASH}_ga_calculated_features'

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

    def calculate_browser_device_features(self, users_df, sessions_df):

        # Merge sessions and users dataframes
        users_sessions_df = (users_df
                             .join(
                                 sessions_df,
                                 'client_id',
                                 'inner'))

        # Cache df since it will be used to later
        users_sessions_df.cache()

        # Aggregate transactions and revenue by mobile device branding
        transactions_by_device_df = (users_sessions_df
                                     .groupBy('mobile_device_branding')
                                     .agg(
                                         f.sum('transactions').alias(
                                             'transactions'),
                                         f.sum('transaction_revenue').alias(
                                             'transaction_revenue')
                                     ))

        # Calculate device revenue per transaction column
        transactions_by_device_df = (transactions_by_device_df
                                     .withColumn(
                                         'device_revenue_per_transaction',
                                         transactions_by_device_df.transaction_revenue /
                                         (transactions_by_device_df.transactions + 1e-5)
                                     )
                                     .drop('transactions', 'transaction_revenue')
                                     )

        # Aggregate transactions and revenue by browser
        transactions_by_browser_df = (users_sessions_df
                                      .groupBy('browser')
                                      .agg(
                                          f.sum('transactions').alias(
                                              'transactions'),
                                          f.sum('transaction_revenue').alias(
                                              'transaction_revenue')
                                      ))

        # Calculate browser revenue per transaction column
        transactions_by_browser_df = (transactions_by_browser_df
                                      .withColumn(
                                          'browser_revenue_per_transaction',
                                          transactions_by_browser_df.transaction_revenue /
                                          (transactions_by_browser_df.transactions + 1e-5)
                                      )
                                      .drop('transactions', 'transaction_revenue')
                                      )

        # Merge new columns into main df and return them
        return (users_df
                .join(
                    transactions_by_device_df,
                    'mobile_device_branding',
                    'inner'
                )
                .join(
                    transactions_by_browser_df,
                    'browser',
                    'inner')
                ).repartition(32)

    # Format shopping stages and cast them to string
    def format_shopping_stages(self, df):
        format_stages_udf = f.udf(lambda stages: '|'.join(stages), 'string')

        return df.withColumn('shopping_stage', format_stages_udf('shopping_stage'))

    # Replace all shopping stages that contain the TRANSACTION stage with just TRANSACTION
    def aggregate_transactions(self, df):
        replace_stages_udf = f.udf(lambda stages: 'TRANSACTION' if stages.find(
            'TRANSACTION') != -1 else stages, 'string')

        return df.withColumn('shopping_stage', replace_stages_udf('shopping_stage'))

    # Remove outlying shopping stages
    def remove_outliers(self, df):

        # # Get the counts for all unique shopping stages
        # unique_stages_counts = (df
        #                         .groupBy('shopping_stage')
        #                         .agg(
        #                             f.countDistinct('client_id')
        #                             .alias('stages_count')
        #                         ))

        # unique_stages_counts.cache()

        # # Calculate the threshold for outliers
        # outlier_threshold = max(unique_stages_counts.agg(
        #     f.sum('stages_count')).collect()[0][0] / 1000, 20)

        # # Get a list of all shopping stages bellow the threshold
        # list_of_outlier_stages = (unique_stages_counts
        #                           .select('shopping_stage')
        #                           .where(unique_stages_counts.stages_count < outlier_threshold)
        #                           .rdd
        #                           .flatMap(lambda stage: stage)
        #                           .collect()
        #                           )

        stages_to_keep = ['ALL_VISITS', 'ALL_VISITS|PRODUCT_VIEW', 'ALL_VISITS|PRODUCT_VIEW|ADD_TO_CART',
                          'ALL_VISITS|PRODUCT_VIEW|ADD_TO_CART|CHECKOUT', 'ALL_VISITS|PRODUCT_VIEW|CHECKOUT', 'TRANSACTION']

        # Replace all outlying shopping stages with ALL_VISITS
        remove_outliers_udf = f.udf(
            lambda stages: stages if stages in stages_to_keep else 'ALL_VISITS', 'string')

        return df.withColumn('shopping_stage', remove_outliers_udf('shopping_stage'))

    # Replace single PRODUCT_VIEW shopping stages with ALL_VISITS|PRODUCT_VIEW
    def replace_single_product_views(self, df):

        replace_single_product_view_udf = f.udf(
            lambda stages: stages if stages != 'PRODUCT_VIEW' else 'ALL_VISITS|PRODUCT_VIEW')

        return df.withColumn('shopping_stage', replace_single_product_view_udf('shopping_stage'))

    def main(self):

        # Initialize spark session
        spark_session = self.get_spark_session()

        # Fetch dfs from hadoop
        ga_epnau_features_filtered_df = spark_session.read.parquet(
            self.HDFS_DIR_USER_FILTERED)

        ga_epnas_features_filtered_df = spark_session.read.parquet(
            self.HDFS_DIR_SESSION_FILTERED)

        ga_epnah_features_filtered_df = spark_session.read.parquet(
            self.HDFS_DIR_HIT_FILTERED)

        # Calculate revenue by device and revenue by browser columns
        users_df = self.calculate_browser_device_features(
            ga_epnau_features_filtered_df, ga_epnas_features_filtered_df)

        # Format shopping stages as strings
        hits_df = self.format_shopping_stages(ga_epnah_features_filtered_df)

        # Aggregate all transactions to a single output
        hits_df = self.aggregate_transactions(hits_df)

        # Replace single product views with all visits
        hits_data = self.replace_single_product_views(
            hits_data)

        # Remove shopping stage outliers
        hits_data = self.remove_outliers(hits_data).repartition(32)

        client_ids_session_counts = (ga_epnas_features_filtered_df
                                     .groupBy('client_id')
                                     .agg(
                                         f.first('client_id').alias(
                                             'client_id'),
                                         f.countDistinct('session_id').alias(
                                             'session_count')
                                     )
                                     )

        ga_epna_data_hits = (hits_data.
                             select(
                                 'client_id',
                                 'session_id',
                                 f.arrray(
                                     f.col('time_on_page'),
                                     f.col('product_list_clicks'),
                                     f.col('product_list_views'),
                                     f.col('product_detail_views')
                                 ).alias('hits_features').
                                 groupBy('session_id').
                                 agg(
                                     f.first('client_id').alias('client_id'),
                                     f.collect_list('hit_features').alias(
                                         'hit_features'),

                                 ).
                                 groupBy('client_Id'),
                                 agg(
                                     f.collect_list(
                                         'hit_features').alias('features')
                                 )
                             ).
                             join(
                                 client_ids_session_counts, 'client_id', 'left_outter'
                             ).repartition(32)
                             )


if __name__ == '__main__':
    preprocessor = CalculationsPreprocessor()
    preprocessor.main()

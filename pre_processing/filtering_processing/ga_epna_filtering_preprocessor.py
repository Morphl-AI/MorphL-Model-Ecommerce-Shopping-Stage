from os import getenv
from pyspark.sql import functions as f, SparkSession


class FilteringPreprocessor:

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

        self.HDFS_DIR_USER = f'hdfs://{MORPHL_SERVER_IP_ADDRESS}:{HDFS_PORT}/{PREDICTION_DAY_AS_STR}_{UNIQUE_HASH}_ga_epnau_filtered'
        self.HDFS_DIR_SESSION = f'hdfs://{MORPHL_SERVER_IP_ADDRESS}:{HDFS_PORT}/{PREDICTION_DAY_AS_STR}_{UNIQUE_HASH}_ga_epnas_filtered'
        self.HDFS_DIR_HIT = f'hdfs://{MORPHL_SERVER_IP_ADDRESS}:{HDFS_PORT}/{PREDICTION_DAY_AS_STR}_{UNIQUE_HASH}_ga_epnah_filtered'

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

    def filter_data(self, users_df, mobile_brand_df, sessions_df, shopping_stages_df, hits_df):

        # Get all the ids that have the required shopping stages.
        ids_with_stages = shopping_stages_df.select('client_id').distinct()

        # Cache this df since it will be used numerous times.
        ids_with_stages.cache()

        # Filter users by ids with shopping stages.
        filtered_users_df = (users_df.
                             drop('day_of_data_capture').
                             join(ids_with_stages, 'client_id', 'inner')
                             )

        filtered_users_df.repartition(32)

        # Filter mobile brand users by ids with shopping stages.
        filtered_mobile_brand_df = (mobile_brand_df.
                                    drop('day_of_data_capture', 'sessions').
                                    join(ids_with_stages, 'client_id', 'inner')
                                    )

        filtered_mobile_brand_df.repartition(32)

        # Filter hits by ids with shopping stages.
        filtered_hits_df = (hits_df.
                            drop('day_of_data_capture').
                            join(ids_with_stages, 'client_id', 'inner')
                            )

        filtered_hits_df.repartition(32)

        # Aggregate users data since it is spread out on
        # multiple days of data capture.
        aggregated_users_df = (filtered_users_df.
                               groupBy('client_id').
                               agg(
                                   f.first('device_category').alias(
                                       'device_category'),
                                   f.first('browser').alias('browser'),
                                   f.sum('bounces').alias('bounces'),
                                   f.sum('sessions').alias('sessions'),
                                   f.sum('revenue_per_user').alias(
                                       'revenue_per_user'),
                                   f.sum('transactions_per_user').alias(
                                       'transactions_per_user')
                               )
                               )

        aggregated_users_df.repartition(32)

        # Group shopping stages per session into a set.
        grouped_shopping_stages_df = (shopping_stages_df.
                                      groupBy('session_id').
                                      agg(f.collect_set('shopping_stage').alias(
                                          'shopping_stage'))
                                      )

        # Remove user duplicates and get their respective mobile device brand.
        grouped_mobile_brand_df = (filtered_mobile_brand_df.
                                   groupBy('client_id').
                                   agg(f.first('mobile_device_branding').alias(
                                       'mobile_device_branding'))
                                   )

        # Add the mobile data to uses that have a mobile device, otherwise
        # make the column equal to (not set).
        final_users_df = (aggregated_users_df.
                          join(grouped_mobile_brand_df,
                               'client_id', 'left_outer')
                          .na.fill({
                              'mobile_device_branding': '(not set)'})
                          )

        final_users_df.repartition(32)

        # Add the shopping stage data to hits by session id.
        final_hits_df = filtered_hits_df.join(
            grouped_shopping_stages_df, 'session_id', 'left_outer')

        final_hits_df.repartition(32)

        # Remove sessions that have a duration equal to 0.
        final_sessions_df = sessions_df.drop(
            'day_of_data_capture').filter('session_duration > 0')

        final_sessions_df.repartition(32)

        return {
            'user': final_users_df,
            'session': final_sessions_df,
            'hit': final_hits_df
        }

    def save_filtered_data(self, user_df, session_df, hit_df):

        user_df.cache()
        session_df.cache()
        hit_df.cache()

        user_df.write.parquet(HDFS_DIR_USER)
        session_df.write.parquet(HDFS_DIR_SESSION)
        hit_df.write.parquet(HDFS_DIR_HIT)

        save_options_ga_epnau_features_filtered = {
            'keyspace': self.MORPHL_CASSANDRA_KEYSPACE,
            'table': ('ga_epnau_features_filtered')
        }
        save_options_ga_epnas_features_filtered = {
            'keyspace': self.MORPHL_CASSANDRA_KEYSPACE,
            'table': ('ga_epnas_features_filtered')
        }
        save_options_ga_epnah_features_filtered = {
            'keyspace': self.MORPHL_CASSANDRA_KEYSPACE,
            'table': ('ga_epnah_features_filtered')
        }

        (user_df
         .write
         .format('org.apache.spark.sql.cassandra')
         .mode('append')
         .options(**save_options_ga_epnau_features_filtered)
         .save())

        (session_df
         .write
         .format('org.apache.spark.sql.cassandra')
         .mode('append')
         .options(**save_options_ga_epnas_features_filtered)
         .save())

        (hit_df
         .write
         .format('org.apache.spark.sql.cassandra')
         .mode('append')
         .options(**save_options_ga_epnah_features_filtered)
         .save())

    def main(self):

        spark_session = self.get_spark_session()

        ga_epnau_features_raw_df = self.fetch_from_cassandra(
            'ga_epnau_features_raw', spark_session
        )

        ga_epnas_features_filtered_df = self.fetch_from_cassandra(
            'ga_epnas_features_raw', spark_session
        )

        ga_epnah_features_raw = self.fetch_from_cassandra(
            'ga_epnah_features_raw', spark_session
        )

        mobile_brand_df = self.fetch_from_cassandra(
            'ga_epna_users_mobile_brand', spark_session)

        shopping_stages_df = self.fetch_from_cassandra(
            'ga_epna_sessions_shopping_stages', spark_session)

        # Get all the filtered dfs.
        filtered_data_dfs = (self.filter_data(
            ga_epnau_features_raw_df,
            mobile_brand_df,
            ga_epnas_features_filtered_df,
            shopping_stages_df,
            ga_epnah_features_raw
        ))

        self.save_filtered_data(
            filtered_data_dfs['user'], filtered_data_dfs['session'], filtered_data_dfs['hit'])


if __name__ == '__main__':
    preprocessor = FilteringPreprocessor()
    preprocessor.main()

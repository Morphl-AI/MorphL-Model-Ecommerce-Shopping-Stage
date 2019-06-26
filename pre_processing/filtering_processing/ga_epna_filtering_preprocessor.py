from os import getenv
from pyspark.sql import functions as f, SparkSession
from pyspark.sql.types import StringType


HDFS_PORT = 9000
PREDICTION_DAY_AS_STR = getenv('PREDICTION_DAY_AS_STR')
UNIQUE_HASH = getenv('UNIQUE_HASH')

MASTER_URL = 'local[*]'
APPLICATION_NAME = 'filtering_preprocessor'

MORPHL_SERVER_IP_ADDRESS = getenv('MORPHL_SERVER_IP_ADDRESS')
MORPHL_CASSANDRA_USERNAME = getenv('MORPHL_CASSANDRA_USERNAME')
MORPHL_CASSANDRA_PASSWORD = getenv('MORPHL_CASSANDRA_PASSWORD')
MORPHL_CASSANDRA_KEYSPACE = getenv('MORPHL_CASSANDRA_KEYSPACE')

HDFS_DIR_USER = f'hdfs://{MORPHL_SERVER_IP_ADDRESS}:{HDFS_PORT}/{PREDICTION_DAY_AS_STR}_{UNIQUE_HASH}_ga_epnau_filtered'
HDFS_DIR_SESSION = f'hdfs://{MORPHL_SERVER_IP_ADDRESS}:{HDFS_PORT}/{PREDICTION_DAY_AS_STR}_{UNIQUE_HASH}_ga_epnas_filtered'
HDFS_DIR_HIT = f'hdfs://{MORPHL_SERVER_IP_ADDRESS}:{HDFS_PORT}/{PREDICTION_DAY_AS_STR}_{UNIQUE_HASH}_ga_epnah_filtered'
HDFS_DIR_SHOPPING = f'hdfs://{MORPHL_SERVER_IP_ADDRESS}:{HDFS_PORT}/{PREDICTION_DAY_AS_STR}_{UNIQUE_HASH}_ga_epna_shopping_stages_filtered'

# Initialize the spark sessions and return it.


def get_spark_session():

    spark_session = (
        SparkSession.builder
        .appName(APPLICATION_NAME)
        .master(MASTER_URL)
        .config('spark.cassandra.connection.host', MORPHL_SERVER_IP_ADDRESS)
        .config('spark.cassandra.auth.username', MORPHL_CASSANDRA_USERNAME)
        .config('spark.cassandra.auth.password', MORPHL_CASSANDRA_PASSWORD)
        .config('spark.sql.shuffle.partitions', 16)
        .config('parquet.enable.summary-metadata', 'true')
        .getOrCreate())

    log4j = spark_session.sparkContext._jvm.org.apache.log4j
    log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)

    return spark_session

# Return a spark dataframe from a specified Cassandra table.


def fetch_from_cassandra(c_table_name, spark_session):

    load_options = {
        'keyspace': MORPHL_CASSANDRA_KEYSPACE,
        'table': c_table_name,
        'spark.cassandra.input.fetch.size_in_rows': '150'}

    df = (spark_session.read.format('org.apache.spark.sql.cassandra')
          .options(**load_options)
          .load())

    return df


def format_and_filter_shopping_stages(stages):
    stages_to_keep = ['ALL_VISITS', 'ALL_VISITS|PRODUCT_VIEW', 'ADD_TO_CART|ALL_VISITS|PRODUCT_VIEW',
                      'ADD_TO_CART|ALL_VISITS|CHECKOUT|PRODUCT_VIEW', 'ALL_VISITS|CHECKOUT|PRODUCT_VIEW', 'TRANSACTION']

    stages.sort()
    stages = '|'.join(stages)
    stages = 'TRANSACTION' if stages.find('TRANSACTION') != -1 else stages
    stages = stages if stages != 'PRODUCT_VIEW' else 'ALL_VISITS|PRODUCT_VIEW'
    stages = stages if stages in stages_to_keep else 'ALL_VISITS'

    return stages


def filter_data(users_df, mobile_brand_df, sessions_df, shopping_stages_df, hits_df):

    # Get all the ids that have the required shopping stages.
    client_ids_with_stages = shopping_stages_df.select(
        'client_id').distinct()
    session_ids_with_stages = shopping_stages_df.select(
        'session_id').distinct()

    # Get all distinct client ids for users, sessions and hits.
    client_ids_users = users_df.select('client_id').distinct()
    client_ids_sessions = sessions_df.select(
        'client_id').distinct()
    client_ids_hits = hits_df.select('client_id').distinct()

    # Get client_ids that exist in all dfs
    complete_client_ids = (client_ids_users
                           .intersect(
                               client_ids_sessions
                            )
                           .intersect(
                               client_ids_hits
                            )
                           .intersect(
                               client_ids_with_stages,
                           )
                           )

    # Cache these dfs since they will be used numerous times.
    session_ids_with_stages.cache()
    complete_client_ids.cache()

    # Filter users by client ids with shopping stages.
    filtered_users_df = (users_df.
                         drop('day_of_data_capture').
                         join(complete_client_ids, 'client_id', 'inner')
                         )

    filtered_users_df.repartition(32)

    # Filter mobile brand users by client ids with shopping stages.
    filtered_mobile_brand_df = (mobile_brand_df.
                                drop('day_of_data_capture', 'sessions').
                                join(complete_client_ids,
                                     'client_id', 'inner')
                                )

    filtered_mobile_brand_df.repartition(32)

    # Filter hits by session ids with shopping stages.
    filtered_hits_df = (hits_df.
                        drop('day_of_data_capture').
                        join(session_ids_with_stages,
                             'session_id', 'inner').
                        join(complete_client_ids,
                             'client_id', 'inner')
                        )

    filtered_hits_df.repartition(32)

    # Filter sessions by session ids with shopping stages.
    filtered_sessions_df = (sessions_df.
                            drop('day_of_data_capture').
                            join(session_ids_with_stages,
                                 'session_id', 'inner').
                            join(complete_client_ids,
                                 'client_id', 'inner')
                            )

    filtered_sessions_df.repartition(32)

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

    shopping_stage_formatter = f.udf(
        format_and_filter_shopping_stages, StringType())

    # Group shopping stages per session into a set.
    final_shopping_stages_df = (shopping_stages_df.
                                orderBy('session_id').
                                groupBy('session_id').
                                agg(f.first('client_id').alias('client_id'),
                                    f.collect_set('shopping_stage').alias(
                                    'shopping_stage')
                                    ).
                                withColumn('shopping_stage', shopping_stage_formatter(
                                    'shopping_Stage'))
                                )

    final_shopping_stages_df.repartition(32)

    return {
        'user': final_users_df,
        'session': filtered_sessions_df,
        'hit': filtered_hits_df,
        'shopping_stages': final_shopping_stages_df,
    }


def save_filtered_data(user_df, session_df, hit_df, shopping_stage_df):

    # Cache and save data to Hadoop
    user_df.cache()
    session_df.cache()
    hit_df.cache()
    shopping_stage_df.cache()

    user_df.write.parquet(HDFS_DIR_USER)
    session_df.write.parquet(HDFS_DIR_SESSION)
    hit_df.write.parquet(HDFS_DIR_HIT)
    shopping_stage_df.write.parquet(HDFS_DIR_SHOPPING)

    save_options_ga_epnau_features_filtered = {
        'keyspace': MORPHL_CASSANDRA_KEYSPACE,
        'table': ('ga_epnau_features_filtered')
    }
    save_options_ga_epnas_features_filtered = {
        'keyspace': MORPHL_CASSANDRA_KEYSPACE,
        'table': ('ga_epnas_features_filtered')
    }
    save_options_ga_epnah_features_filtered = {
        'keyspace': MORPHL_CASSANDRA_KEYSPACE,
        'table': ('ga_epnah_features_filtered')
    }
    save_options_ga_epna_shopping_stages_filtered = {
        'keyspace': MORPHL_CASSANDRA_KEYSPACE,
        'table': ('ga_epna_shopping_stages_filtered')
    }

    # Save data to Cassandra
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

    (shopping_stage_df
     .write
     .format('org.apache.spark.sql.cassandra')
     .mode('append')
     .options(**save_options_ga_epna_shopping_stages_filtered)
     .save())


def main():

    spark_session = get_spark_session()

    ga_epnau_features_raw_df = fetch_from_cassandra(
        'ga_epnau_features_raw', spark_session
    )

    ga_epnas_features_filtered_df = fetch_from_cassandra(
        'ga_epnas_features_raw', spark_session
    )

    ga_epnah_features_raw = fetch_from_cassandra(
        'ga_epnah_features_raw', spark_session
    )

    mobile_brand_df = fetch_from_cassandra(
        'ga_epna_users_mobile_brand', spark_session)

    shopping_stages_df = fetch_from_cassandra(
        'ga_epna_sessions_shopping_stages', spark_session)

    # Get all the filtered dfs.
    filtered_data_dfs = (filter_data(
        ga_epnau_features_raw_df,
        mobile_brand_df,
        ga_epnas_features_filtered_df,
        shopping_stages_df,
        ga_epnah_features_raw
    ))

    save_filtered_data(
        filtered_data_dfs['user'], filtered_data_dfs['session'], filtered_data_dfs['hit'])


if __name__ == '__main__':
    main()

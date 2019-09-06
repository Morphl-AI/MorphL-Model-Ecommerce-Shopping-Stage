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

# Filters the data and makes sure that the client_ids we make predictions on
# have data in all relevant tables.
def filter_data(users_df, mobile_brand_df, sessions_df, shopping_stages_df, hits_df, product_info_df, session_index_df):

    # Add product info to hits and replace missing values with 0.0
    hits_df = (hits_df
               .join(
                   product_info_df.drop('product_name'),
                   ['client_id',
                    'session_id',
                    'day_of_data_capture',
                    'date_hour_minute'
                    ],
                   'left_outer'
               )
               .fillna(
                   0.0,
                   [
                       'product_detail_views',
                       'item_quantity',
                   ]
               )
               .repartition(32)
               )

    # Get the number of sessions a user has
    user_session_counts = session_index_df.groupBy(
        'client_id').agg(f.max('session_index').alias('session_count'))

    # Add the session count column to the users dataframe
    users_df = users_df.join(
        user_session_counts, 'client_id', 'inner').repartition(32)

    # Add the session index to each session
    sessions_df = (sessions_df
                   .join(
                       session_index_df.drop('day_of_data_capture'),
                       ['client_id', 'session_id'],
                       'inner'
                   )
                   .repartition(32)
                   )

    # Get the session ids that are present in all tables.
    sessions_df_session_ids = sessions_df.select('session_id').distinct()
    hits_df_session_ids = hits_df.select('session_id').distinct()
    shopping_stages_df_session_ids = shopping_stages_df.select(
        'session_id').distinct()

    complete_session_ids = (sessions_df_session_ids
                            .intersect(
                                hits_df_session_ids
                            )
                            .intersect(
                                shopping_stages_df_session_ids
                            )
                            )

    # Will be reused multiple times so caching will improve performance.
    complete_session_ids.cache()

    # Only keep sessions that we have shopping stage and hit data for.
    sessions_filtered_by_session_id_df = (sessions_df.
                                          drop('day_of_data_capture').
                                          join(complete_session_ids,
                                               'session_id', 'inner')
                                          )

    # Only keep hits that we have stage and session data for.
    hits_filtered_by_session_id_df = (hits_df.
                                      drop('day_of_data_capture').
                                      join(complete_session_ids,
                                           'session_id', 'inner')
                                      )

    # Only keep shopping stage info that we have hit and session data on.
    shopping_stages_filtered_by_session_id_df = (shopping_stages_df.
                                                 drop('day_of_data_capture').
                                                 join(complete_session_ids,
                                                      'session_id', 'inner')
                                                 )

    # Get all distinct client ids for users, sessions, shopping stages and hits.
    client_ids_users = users_df.select('client_id').distinct()
    client_ids_sessions = sessions_filtered_by_session_id_df.select(
        'client_id').distinct()
    client_ids_hits = hits_filtered_by_session_id_df.select(
        'client_id').distinct()
    client_ids_with_stages = (shopping_stages_filtered_by_session_id_df.
                              select('client_id')
                              .distinct()
                              )

    # Get client_ids that exist in all dfs.
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

    # Will be reused so caching improves performance.
    complete_client_ids.cache()

    # Only keep users with complete data.
    filtered_users_df = (users_df.
                         drop('day_of_data_capture').
                         join(complete_client_ids, 'client_id', 'inner')
                         )

    filtered_users_df.repartition(32)

    filtered_mobile_brand_df = (mobile_brand_df.
                                drop('day_of_data_capture', 'sessions').
                                join(complete_client_ids,
                                     'client_id', 'inner')
                                )

    filtered_mobile_brand_df.repartition(32)

    # Only keep hits with complete data
    filtered_hits_df = (hits_filtered_by_session_id_df.
                        drop('day_of_data_capture').
                        join(complete_client_ids,
                             'client_id', 'inner')
                        )

    filtered_hits_df.repartition(32)

    # Only keep sessions with complete data
    filtered_sessions_df = (sessions_filtered_by_session_id_df.
                            drop('day_of_data_capture').
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
                               f.first('city').alias('city'),
                               f.first('session_count').alias('session_count')
                           )
                           )

    aggregated_users_df.repartition(32)

    # Remove user duplicates and get their respective mobile device brand.
    grouped_mobile_brand_df = (filtered_mobile_brand_df.
                               groupBy('client_id').
                               agg(f.first('mobile_device_branding').alias(
                                   'mobile_device_branding'))
                               )

    # Add the mobile data to users that have a mobile device, otherwise
    # make the column equal to (not set).
    final_users_df = (aggregated_users_df.
                      join(grouped_mobile_brand_df,
                           'client_id', 'left_outer')
                      .na.fill({
                          'mobile_device_branding': '(not set)'})
                      )

    final_users_df.repartition(32)

    return {
        'user': final_users_df,
        'session': filtered_sessions_df,
        'hit': filtered_hits_df,
    }


def save_filtered_data(user_df, session_df, hit_df):

    # # Cache and save data to Hadoop
    # user_df.cache()
    # session_df.cache()
    # hit_df.cache()
    # shopping_stage_df.cache()

    # user_df.write.parquet(HDFS_DIR_USER)
    # session_df.write.parquet(HDFS_DIR_SESSION)
    # hit_df.write.parquet(HDFS_DIR_HIT)
    # shopping_stage_df.write.parquet(HDFS_DIR_SHOPPING)

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

    ga_epnap_features_raw = fetch_from_cassandra(
        'ga_epnap_features_raw', spark_session)

    session_index_df = fetch_from_cassandra(
        'ga_epna_session_index', spark_session)

    # Get all the filtered dfs.
    filtered_data_dfs = (filter_data(
        ga_epnau_features_raw_df,
        mobile_brand_df,
        ga_epnas_features_filtered_df,
        shopping_stages_df,
        ga_epnah_features_raw,
        ga_epnap_features_raw,
        session_index_df
    ))

    save_filtered_data(
        filtered_data_dfs['user'], filtered_data_dfs['session'], filtered_data_dfs['hit'])


if __name__ == '__main__':
    main()

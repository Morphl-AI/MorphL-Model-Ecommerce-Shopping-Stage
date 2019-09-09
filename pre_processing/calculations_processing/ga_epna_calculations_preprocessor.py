from os import getenv
from pyspark.sql import functions as f, SparkSession, Window
from pyspark.sql.types import ArrayType, DoubleType


HDFS_PORT = 9000
PREDICTION_DAY_AS_STR = getenv('PREDICTION_DAY_AS_STR')
UNIQUE_HASH = getenv('UNIQUE_HASH')

MASTER_URL = 'local[*]'
APPLICATION_NAME = 'calculations_preprocessor'

MORPHL_SERVER_IP_ADDRESS = getenv('MORPHL_SERVER_IP_ADDRESS')
MORPHL_CASSANDRA_USERNAME = getenv('MORPHL_CASSANDRA_USERNAME')
MORPHL_CASSANDRA_PASSWORD = getenv('MORPHL_CASSANDRA_PASSWORD')
MORPHL_CASSANDRA_KEYSPACE = getenv('MORPHL_CASSANDRA_KEYSPACE')

HDFS_DIR_USER_FILTERED = f'hdfs://{MORPHL_SERVER_IP_ADDRESS}:{HDFS_PORT}/{PREDICTION_DAY_AS_STR}_{UNIQUE_HASH}_ga_epnau_filtered'
HDFS_DIR_SESSION_FILTERED = f'hdfs://{MORPHL_SERVER_IP_ADDRESS}:{HDFS_PORT}/{PREDICTION_DAY_AS_STR}_{UNIQUE_HASH}_ga_epnas_filtered'
HDFS_DIR_HIT_FILTERED = f'hdfs://{MORPHL_SERVER_IP_ADDRESS}:{HDFS_PORT}/{PREDICTION_DAY_AS_STR}_{UNIQUE_HASH}_ga_epnah_filtered'

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


def calculate_browser_device_features(users_df, sessions_df):

    # Merge sessions and users dataframes.
    users_sessions_df = (users_df
                         .join(
                             sessions_df,
                             'client_id',
                             'inner'))

    # Cache df since it will be used to later.
    users_sessions_df.cache()

    # Aggregate transactions and revenue by mobile device branding.
    transactions_by_device_df = (users_sessions_df
                                 .groupBy('mobile_device_branding')
                                 .agg(
                                     f.sum('transactions').alias(
                                         'transactions'),
                                     f.sum('transaction_revenue').alias(
                                         'transaction_revenue'),
                                     f.countDistinct(
                                         'client_id').alias('users')
                                 ))

    # Calculate device revenue per transaction and device transactions per user columns.
    transactions_by_device_df = (transactions_by_device_df
                                 .withColumn(
                                     'device_revenue_per_transaction',
                                     transactions_by_device_df.transaction_revenue /
                                     (transactions_by_device_df.transactions + 1e-5)
                                 )
                                 .withColumn(
                                     'device_transactions_per_user',
                                     transactions_by_device_df.transactions / transactions_by_device_df.users
                                 )
                                 .drop('transactions', 'transaction_revenue')
                                 )

    # Aggregate transactions and revenue by browser.
    transactions_by_browser_df = (users_sessions_df
                                  .groupBy('browser')
                                  .agg(
                                      f.sum('transactions').alias(
                                          'transactions'),
                                      f.sum('transaction_revenue').alias(
                                          'transaction_revenue'),
                                      f.countDistinct(
                                          'client_id').alias('users')
                                  ))

    # Calculate browser revenue per transaction and browser transactions per user columns.
    transactions_by_browser_df = (transactions_by_browser_df
                                  .withColumn(
                                      'browser_revenue_per_transaction',
                                      transactions_by_browser_df.transaction_revenue /
                                      (transactions_by_browser_df.transactions + 1e-5)
                                  )
                                  .withColumn(
                                      'browser_transactions_per_user',
                                      transactions_by_browser_df.transactions / transactions_by_browser_df.users
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


def calculate_time_on_page_features(user_features, hit_features):
    time_on_page_features_df = (hit_features
                                    .groupBy('client_id')
                                    .agg(
                                        f.sum('time_on_page').alias('total_time_on_page'), 
                                        f.mean('time_on_page').alias('avg_time_on_page')
                                    )
                                    .select('client_id', 'total_time_on_page', 'avg_time_on_page')
                                )
    
    return user_features.join(time_on_page_features_df, 'client_id', 'inner')


def calculate_search_features(user_features, session_features):
    search_features = (session_features
                    .drop_duplicates(subset=['client_id', 'session_id'])
                    .groupBy('client_id')
                    .agg(
                        f.count('session_id').alias('number_of_sessions'),
                        f.sum('search_result_views').alias('number_of_searches')
                    )
                )

    search_features = search_features.withColumn('searches_per_session', search_features['number_of_searches']/search_features['number_of_sessions'])

    search_features = search_features.select('client_id', 'searches_per_session')

    return user_features.join(search_features, 'client_id', 'inner')


# Sets values outside of [0.0, 1.0] to 0.0 or 1.0.
def clip(value):
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0

    return value


# Normalizes hit data.
def min_max_hits(hit_features):
    # Max and min used when training for each feature
    # ['time_on_page', 'product_detail_views', 'cart_to_detail_rate', 'item_quantity', 'item_revenue',
    #  'product_adds_to_cart', 'product_checkouts', 'quantity_added_to_cart'
    # ]
    min = [0.0] * 8
    max = [1451.0, 2.0, 100.0, 1.0, 482.0, 1.0, 1.0, 1.0]

    for i in range(0, 8):
        hit_features[i] = clip((hit_features[i] - min[i]) / (max[i] - min[i]))

    return hit_features

# Normalizes session data.


def min_max_sessions(session_features):
    # Max and min used when training for each feature.
    # ['session_duration', 'unique_pageviews', 'transactions', 'revenue', 'unique_purchases', 'days_since_last_session',
    #   'search_result_views', 'search_uniques', 'search_depth', 'search_refinements'
    # ]
    min = [8.0, 1.0] + [0.0] * 8
    max = [11196.0, 115.0, 1.0, 4477.0, 5.0, 149.0, 40.0, 22.0, 118.0, 25.0]

    for i in range(0, 10):
        session_features[i] = clip(
            (session_features[i] - min[i]) / (max[i] - min[i]))

    return session_features

# Normalizes user data.


def min_max_users(users_features):
    # Max and min values used when training for each feature.
    # [ 'session_count', 'device_transactions_per_user', 'device_revenue_per_transaction', 'browser_transactions_per_user',
    # 'browser_revenue_per_transaction'
    # ]
    min = [1.0, 0.007, 225.0, 0.015, 1540.256]
    max = [6305, 0.048, 2432.607, 0.061, 4209.32]

    for i in range(0, 5):
        users_features[i] = clip(
            (users_features[i] - min[i]) / (max[i] - min[i]))

    return users_features

def pad_with_zero(hits_features):
    max_hit_count = 0
    for session in hits_features:
        max_hit_count = max_hit_count if len(session) < max_hit_count else len(session)

    for session_count in range(len(hits_features)):
        hits_features[session_count] = hits_features[session_count] + \
            [[0.0] * 8] * \
            (max_hit_count - len(hits_features[session_count]))

    return hits_features


# Save array data to Cassandra.


def save_data(hits_data, hits_num_data, session_data, user_data):

    ga_epna_batch_inference_data = (hits_data
                                    .join(hits_num_data, 'client_id', 'inner')
                                    .join(session_data, 'client_id', 'inner')
                                    .join(user_data, 'client_id', 'inner')
                                    .repartition(32)
                                    )

    save_options_ga_epna_batch_inference_data = {
        'keyspace': MORPHL_CASSANDRA_KEYSPACE,
        'table': ('ga_epna_batch_inference_data')
    }

    (ga_epna_batch_inference_data
        .write
        .format('org.apache.spark.sql.cassandra')
        .mode('append')
        .options(**save_options_ga_epna_batch_inference_data)
        .save())


def main():

    # Initialize spark session
    spark_session = get_spark_session()

    # Fetch dfs from hadoop
    ga_epnau_features_filtered_df = spark_session.read.parquet(
        HDFS_DIR_USER_FILTERED)

    ga_epnas_features_filtered_df = spark_session.read.parquet(
        HDFS_DIR_SESSION_FILTERED)

    ga_epnah_features_filtered_df = spark_session.read.parquet(
        HDFS_DIR_HIT_FILTERED)

    # Calculate revenue by device and revenue by browser columns
    users_df = calculate_browser_device_features(
        ga_epnau_features_filtered_df, ga_epnas_features_filtered_df)

    users_df = calculate_time_on_page_features(ga_epnau_features_filtered_df, ga_epnah_features_filtered_df)
    users_df = calculate_search_features(ga_epnau_features_filtered_df, ga_epnas_features_filtered_df)

    # Initialize udfs
    min_maxer_hits = f.udf(min_max_hits, ArrayType(DoubleType()))
    min_maxer_sessions = f.udf(min_max_sessions, ArrayType(DoubleType()))
    min_maxer_users = f.udf(min_max_users, ArrayType(DoubleType()))
    zero_padder = f.udf(pad_with_zero, ArrayType(
        ArrayType(ArrayType(DoubleType()))))

    # Initialize windows used for future grouping of data while keeping order.
    hits_window_by_sessions = Window.partitionBy(
        'session_id').orderBy('date_hour_minute')
    hits_window_by_user = Window.partitionBy('client_id').orderBy('session_id')

    # Grab the hit features into an array column 'hits_features',
    # apply the normalizer to that column,
    # collect the the hits_features arrays into a list over the window paritioned by session_id,
    # group the data by session id and get the client_id and last collected list with all the data.
    # collect all the session lists with hit lists inside over the window partitioned by client_id,
    # group data by client id and get the client_id and last collected list.
    # Add the session counts and hit counts then pad the hits with zero according to the max hit count
    # of users with that session count.
    #
    # user[
    #     session[
    #         hit[1.0, 2.0, 3.0, 4.0, 0.5, 0.6, 0.7, 0.8],
    #         hit[5.0, 6.0, 7.0, 8.0, 0.9, 0.1, 0.2, 0.3]
    #     ],
    #     session[
    #         hit[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    #         hit[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     ]
    # ]
    ga_epna_data_hits = (ga_epnah_features_filtered_df
                         .select(
                             'client_id',
                             'session_id',
                             'date_hour_minute',
                             f.array(
                                 f.col('time_on_page'),
                                 f.col('product_detail_views'),
                                 f.col('cart_to_detail_rate'),
                                 f.col('item_quantity'),
                                 f.col('item_revenue'),
                                 f.col('product_adds_to_cart'),
                                 f.col('product_checkouts'),
                                 f.col('quantity_added_to_cart')
                             ).alias('hits_features')
                         ).
                         withColumn('hits_features', min_maxer_hits('hits_features')).
                         withColumn('hits_features', f.collect_list('hits_features').over(hits_window_by_sessions)).
                         groupBy('session_id').agg(
                             f.last('hits_features').alias('hits_features'),
                             f.first('client_id').alias('client_id')
                         ).
                         withColumn('hits_features', f.collect_list(
                             'hits_features').over(hits_window_by_user))
                         .groupBy('client_id').agg(
                             f.last('hits_features').alias('hits_features')
                         ).
                         withColumn('hits_features',
                                    zero_padder('hits_features'))
                         .repartition(32)
                         )

    # Get session arrays. Similiar process to hits but data is collect at session level
    # and we also one hot encode categorical data.
    #
    # user[
    #     session[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    #     session[13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
    # ]
    sessions_window_by_user = Window.partitionBy(
        'client_id').orderBy('session_id')

    ga_epna_data_sessions = (ga_epnas_features_filtered_df
                             .withColumn(
                                 'with_site_search',
                                 f.when(f.col('search_used') == 'Visits With Site Search', 1.0).otherwise(
                                     0.0)
                             )
                             .withColumn(
                                 'without_site_search',
                                 f.when(f.col('search_used') == 'Visits Without Site Search', 1.0).otherwise(
                                     0.0)
                             )
                             .withColumn('new_visitor',
                                f.when(f.col('session_index') == 1, 1.0).otherwise(0.0)
                             )
                             .withColumn('returning_visitor', 
                                f.when(f.col('session_index') > 1, 1.0).otherwise(0.0)
                             )
                             .select(
                                 'client_id',
                                 'session_id',
                                 f.array(
                                     f.col('session_duration'),
                                     f.col('unique_page_views'),
                                     f.col('transactions'),
                                     f.col('transaction_revenue'),
                                     f.col('unique_purchases'),
                                     f.col('days_since_last_session'),
                                     f.col('search_result_views'),
                                     f.col('search_uniques'),
                                     f.col('search_depth'),
                                     f.col('search_refinements'),
                                     f.col('with_site_search'),
                                     f.col('without_site_search'),
                                     f.col('new_visitor'),
                                     f.col('returning_visitor'),
                                 ).alias('sessions_features')
                             )
                             .withColumn('sessions_features', min_maxer_sessions('sessions_features'))
                             .withColumn('sessions_features', f.collect_list('sessions_features').over(sessions_window_by_user))
                             .groupBy('client_id')
                             .agg(f.last('sessions_features').alias('sessions_features')
                                  )
                             .repartition(32)
                             )

    # Get user arrays. Similar to sessions with data at user level.
    # user[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ga_epna_data_users = (users_df.
                          withColumn(
                              'is_mobile', f.when(
                                  f.col('device_category') == 'mobile', 1.0).otherwise(0.0)
                          ).
                          withColumn(
                              'is_tablet', f.when(
                                  f.col('device_category') == 'tablet', 1.0).otherwise(0.0)
                          ).
                          withColumn(
                              'is_desktop', f.when(
                                  f.col('device_category') == 'desktop', 1.0).otherwise(0.0)
                          ).
                          select(
                              'client_id',
                              f.array(
                                  f.col('session_count'),
                                  f.col('device_transactions_per_user'),
                                  f.col('device_revenue_per_transaction'),
                                  f.col('browser_transactions_per_user'),
                                  f.col('browser_revenue_per_transaction'),
                                  f.col('total_time_on_page'),
                                  f.col('avg_time_on_page'),
                                  f.col('is_desktop'),
                                  f.col('is_mobile'),
                                  f.col('is_tablet'),
                              ).alias('user_features')
                          )
                          .withColumn('user_features', min_maxer_users('user_features'))
                          .repartition(32)
                          )

    # Get the number of hits arrays
    # user[
    #     numHitsSession1,
    #     numHitsSession2,
    #     numHitsSession3
    # ]

    num_hits_window_by_user = Window.partitionBy(
        'client_id').orderBy('session_id')

    ga_epna_data_num_hits = (ga_epnah_features_filtered_df.
                             groupBy('session_id').
                             agg(
                                 f.first('client_id').alias('client_id'),
                                 f.count('date_hour_minute').alias(
                                     'hits_count')
                             ).
                             withColumn('sessions_hits_count', f.collect_list('hits_count').over(num_hits_window_by_user)).
                             groupBy('client_id').
                             agg(
                                 f.last('sessions_hits_count').alias(
                                     'sessions_hits_count')
                             ).repartition(32)
                             )

    save_data(ga_epna_data_hits, ga_epna_data_num_hits, ga_epna_data_sessions,
              ga_epna_data_users)


if __name__ == '__main__':
    main()

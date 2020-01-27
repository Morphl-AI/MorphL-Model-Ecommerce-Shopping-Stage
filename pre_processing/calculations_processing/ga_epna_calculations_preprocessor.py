from os import getenv
from pyspark.sql import functions as f, SparkSession, Window
from pyspark.sql.types import ArrayType, DoubleType, StructType, StructField, IntegerType, StringType


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


BROWSER_STATISTICS_CSV_FILE = '/opt/models/statistics/browser_statistics.csv'
MOBILE_BRAND_STATISTICS_CSV_FILE = '/opt/models/statistics/mobile_brand_statistics.csv'
CITY_STATISTICS_CSV_FILE = '/opt/models/statistics/city_statistics.csv'


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


# Add browser and device features from statistics csvs
def add_browser_device_features(users_df, spark_session):

    schema_browser = StructType([StructField('index', IntegerType(), True),
                                 StructField('browser', StringType(), True),
                                 StructField(
                                     'browser_transactions_per_user', DoubleType(), True),
                                 StructField('browser_revenue_per_transaction', DoubleType(), True)])

    schema_mobile_brand = StructType([StructField('index', IntegerType(), True),
                                      StructField(
                                          'mobile_device_branding', StringType(), True),
                                      StructField(
                                          'device_transactions_per_user', DoubleType(), True),
                                      StructField('device_revenue_per_transaction', DoubleType(), True)])

    browser_statistics_df = spark_session.read.csv(
        BROWSER_STATISTICS_CSV_FILE, header=True, schema=schema_browser).drop('index')
    mobile_brand_statistics_df = spark_session.read.csv(
        MOBILE_BRAND_STATISTICS_CSV_FILE, header=True, schema=schema_mobile_brand).drop('index')

    users_df = (
        users_df
        .join(
            mobile_brand_statistics_df,
            'mobile_device_branding',
            'left_outer'
        )
        .join(
            browser_statistics_df,
            'browser',
            'left_outer'
        )
        .fillna(
            0.0,
            [
                'browser_revenue_per_transaction',
                'browser_transactions_per_user',
                'device_revenue_per_transaction',
                'device_transactions_per_user',
            ]
        )
        .repartition(32)
    )

    return users_df


# Add city features from csv file
def add_city_features(users_df, spark_session):

    schema_city = StructType([StructField('index', IntegerType(), True),
                              StructField('city', StringType(), True),
                              StructField(
                                  'city_transactions_per_user', DoubleType(), True),
                              StructField('city_revenue_per_transaction', DoubleType(), True)])

    city_statistics_df = spark_session.read.csv(
        CITY_STATISTICS_CSV_FILE, header=True, schema=schema_city).drop('index')

    users_df = (
        users_df
        .join(
            city_statistics_df,
            'city',
            'left_outer'
        )
        .fillna(
            0.0,
            [
                'city_revenue_per_transaction',
                'city_transactions_per_user',
            ]
        )
        .repartition(32)
    )

    return users_df


# Calculate 'searches_per_session.
def calculate_search_features(user_features, session_features):
    search_features = (session_features
                       .drop_duplicates(subset=['client_id', 'session_id'])
                       .groupBy('client_id')
                       .agg(
                           f.count('session_id').alias('number_of_sessions'),
                           f.sum('results_pageviews').alias(
                               'number_of_searches')
                       )
                       )

    search_features = search_features.withColumn(
        'searches_per_session', search_features['number_of_searches']/search_features['number_of_sessions'])

    search_features = search_features.select(
        'client_id', 'searches_per_session')

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
    # ['time_on_page', 'product_detail_views']
    min = [0.0, 0.0]
    max = [1510.0, 2.0]

    for i in range(0, 2):
        hit_features[i] = clip((hit_features[i] - min[i]) / (max[i] - min[i]))

    return hit_features


# Normalizes session data.
def min_max_sessions(session_features):
    # [
    # 'days_since_last_session', 'results_pageviews', 'search_depth', 'search_refinements',
    # 'session_duration', 'site_search_status_visit_with_site_search', 'site_search_status_visit_without_site_search',
    # 'unique_searches', 'unique_pageviews'
    # ]
    min = [0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0]
    max = [130.0, 37.0, 135.0, 22.0, 11500.0, 1.0, 1.0, 20.0, 110.0]

    for i in range(0, 9):
        session_features[i] = clip(
            (session_features[i] - min[i]) / (max[i] - min[i]))

    return session_features


# Normalizes user data.
def min_max_users(users_features):
    # [
    # 'browser_revenue_per_transaction', browser_transactions_per_user', 'city_revenue_per_transaction', 'city_transactions_per_user',
    # 'device_category_desktop', 'device_category_mobile', 'device_category_tablet', 'device_revenue_per_transaction', 'device_transactions_per_user',
    # 'searches_per_session', 'total_number_of_sessions', 'total_products_viewed',
    # 'user_type_new_user', 'user_type_returning_user'
    # ]

    min = [1445.3, 0.038495857, 0.0, 0.0, 0.0, 0.0, 0.0,
           1000.0589, 0.027276857, 0.0, 1.0, 1.0, 0.0, 0.0]

    max = [3274.44, 0.073248476, 5707.6587, 0.100638375, 1.0, 1.0, 1.0,
           2391.0286, 0.06703993, 13.0, 6121.0, 1666.0, 1.0, 1.0]

    for i in range(0, 15):
        users_features[i] = clip(
            (users_features[i] - min[i]) / (max[i] - min[i]))

    return users_features


# Fills up session arrays with fake hits so we have a square array for the model.
def pad_with_zero(hits_features):
    max_hit_count = 0
    for session in hits_features:
        max_hit_count = max_hit_count if len(
            session) < max_hit_count else len(session)

    for session_count in range(len(hits_features)):
        hits_features[session_count] = hits_features[session_count] + \
            [[0.0] * 2] * \
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

    load_options = {
        'keyspace': MORPHL_CASSANDRA_KEYSPACE,
        'table': 'ga_epnap_features_raw',
        'spark.cassandra.input.fetch.size_in_rows': '150'}

    # Fetch dfs from hadoop
    ga_epnau_features_filtered_df = spark_session.read.parquet(
        HDFS_DIR_USER_FILTERED)

    ga_epnas_features_filtered_df = spark_session.read.parquet(
        HDFS_DIR_SESSION_FILTERED)

    ga_epnah_features_filtered_df = spark_session.read.parquet(
        HDFS_DIR_HIT_FILTERED)

    users_df = add_browser_device_features(
        ga_epnau_features_filtered_df, spark_session)
    users_df = add_city_features(
        users_df, spark_session
    )
    users_df = calculate_search_features(
        users_df, ga_epnas_features_filtered_df)

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
    #         hit[1.0, 2.0],
    #         hit[5.0, 6.0]
    #     ],
    #     session[
    #         hit[9.0, 10.0],
    #         hit[0.0, 0.0]
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
                                 'site_search_status_visit_with_site_search',
                                 f.when(f.col('search_used') == 'Visits With Site Search', 1.0).otherwise(
                                     0.0)
                             )
                             .withColumn(
                                 'site_search_status_visit_without_site_search',
                                 f.when(f.col('search_used') == 'Visits Without Site Search', 1.0).otherwise(
                                     0.0)
                             )
                             .select(
                                 'client_id',
                                 'session_id',
                                 f.array(
                                     f.col('days_since_last_session'),
                                     f.col('results_pageviews'),
                                     f.col('search_depth'),
                                     f.col('search_refinements'),
                                     f.col('session_duration'),
                                     f.col('site_search_status_visit_with_site_search'),
                                     f.col('site_search_status_visit_without_site_search'),
                                     f.col('total_unique_searches'),
                                     f.col('unique_pageviews'),
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
                              'device_category_mobile', f.when(
                                  f.col('device_category') == 'mobile', 1.0).otherwise(0.0)
                          ).
                          withColumn(
                              'device_category_tablet', f.when(
                                  f.col('device_category') == 'tablet', 1.0).otherwise(0.0)
                          ).
                          withColumn(
                              'device_category_desktop', f.when(
                                  f.col('device_category') == 'desktop', 1.0).otherwise(0.0)
                          ).
                          withColumn(
                              'user_type_new_user', f.when(
                                  f.col('first_collected_index') == 1.0, 1.0).otherwise(0.0)
                          ).
                          withColumn(
                              'user_type_returning_user', f.when(
                                  f.col('first_collected_index') > 1.0, 1.0).otherwise(0.0)
                          ).
                          select(
                              'client_id',
                              f.array(
                                  f.col('browser_revenue_per_transaction'),
                                  f.col('browser_transactions_per_user'),
                                  f.col('city_revenue_per_transaction'),
                                  f.col('city_transactions_per_user'),
                                  f.col('device_category_desktop'),
                                  f.col('device_category_mobile'),
                                  f.col('device_category_tablet'),
                                  f.col('device_revenue_per_transaction'),
                                  f.col('device_transactions_per_user'),
                                  f.col('searches_per_session'),
                                  f.col('total_number_of_sessions'),
                                  f.col('total_products_viewed'),
                                  f.col('user_type_new_user'),
                                  f.col('user_type_returning_user')
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
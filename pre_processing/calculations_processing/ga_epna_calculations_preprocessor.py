from os import getenv
from pyspark.sql import functions as f, SparkSession
from pyspark.sql.types import ArrayType, DoubleType, StringType


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

HDFS_DIR_DATA_HITS = f'hdfs://{MORPHL_SERVER_IP_ADDRESS}:{HDFS_PORT}/{PREDICTION_DAY_AS_STR}_{UNIQUE_HASH}_ga_data_hits'

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

# UDF to format shopping stages and cast them to string


def format_shopping_stages(stages):
    stages.sort()

    return '|'.join(stages)

# Replace all shopping stages that contain the TRANSACTION stage with just TRANSACTION


def aggregate_transactions(df):
    replace_stages_udf = f.udf(lambda stages: 'TRANSACTION' if stages.find(
        'TRANSACTION') != -1 else stages, 'string')

    return df.withColumn('shopping_stage', replace_stages_udf('shopping_stage'))

# Remove outlying shopping stages


def remove_outliers(df):

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

    stages_to_keep = ['ALL_VISITS', 'ALL_VISITS|PRODUCT_VIEW', 'ADD_TO_CART|ALL_VISITS|PRODUCT_VIEW',
                      'ADD_TO_CART|ALL_VISITS|CHECKOUT|PRODUCT_VIEW', 'ALL_VISITS|CHECKOUT|PRODUCT_VIEW', 'TRANSACTION']

    # Replace all outlying shopping stages with ALL_VISITS
    remove_outliers_udf = f.udf(
        lambda stages: stages if stages in stages_to_keep else 'ALL_VISITS', 'string')

    return df.withColumn('shopping_stage', remove_outliers_udf('shopping_stage'))

# Replace single PRODUCT_VIEW shopping stages with ALL_VISITS|PRODUCT_VIEW


def replace_single_product_views(df):

    replace_single_product_view_udf = f.udf(
        lambda stages: stages if stages != 'PRODUCT_VIEW' else 'ALL_VISITS|PRODUCT_VIEW')

    return df.withColumn('shopping_stage', replace_single_product_view_udf('shopping_stage'))

# Save the 5 dfs to Cassandra


def save_data(hits_data, hits_num_data, session_data, user_data, shopping_stages_data):

    save_options_ga_epna_data_hits = {
        'keyspace': MORPHL_CASSANDRA_KEYSPACE,
        'table': ('ga_epna_data_hits')
    }

    save_options_ga_epna_data_num_hits = {
        'keyspace': MORPHL_CASSANDRA_KEYSPACE,
        'table': ('ga_epna_data_num_hits')
    }

    save_options_ga_epna_data_sessions = {
        'keyspace': MORPHL_CASSANDRA_KEYSPACE,
        'table': ('ga_epna_data_sessions')
    }

    save_options_ga_epna_data_user = {
        'keyspace': MORPHL_CASSANDRA_KEYSPACE,
        'table': ('ga_epna_data_users')
    }

    save_options_ga_epna_data_shopping_stages = {
        'keyspace': MORPHL_CASSANDRA_KEYSPACE,
        'table': ('ga_epna_data_shopping_stages')
    }

    (hits_data
        .write
        .format('org.apache.spark.sql.cassandra')
        .mode('append')
        .options(**save_options_ga_epna_data_hits)
        .save())

    (hits_num_data
        .write
        .format('org.apache.spark.sql.cassandra')
        .mode('append')
        .options(**save_options_ga_epna_data_num_hits)
        .save())

    (session_data
        .write
        .format('org.apache.spark.sql.cassandra')
        .mode('append')
        .options(**save_options_ga_epna_data_sessions)
        .save())

    (user_data
        .write
        .format('org.apache.spark.sql.cassandra')
        .mode('append')
        .options(**save_options_ga_epna_data_user)
        .save())

    (shopping_stages_data
        .write
        .format('org.apache.spark.sql.cassandra')
        .mode('append')
        .options(**save_options_ga_epna_data_shopping_stages)
        .save())


# Udf function used to pad the session arrays with hits of zero value
def pad_with_zero(features, max_hit_count):

    for session_count in range(len(features)):
        features[session_count] = features[session_count] + \
            [[0.0, 0.0, 0.0, 0.0]] * max_hit_count

    return features


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

    shopping_stage_formatter = f.udf(format_shopping_stages, StringType())

    # Format shopping stages as strings
    hits_df = hits_df.withColumn(
        'shopping_stage', shopping_stage_formatter('shopping_stage'))

    # Aggregate all transactions to a single output
    hits_df = aggregate_transactions(hits_df)

    # Replace single product views with all visits
    hits_df = replace_single_product_views(
        hits_df)

    # Remove shopping stage outliers
    hits_df = remove_outliers(hits_df).repartition(32)

    # Get the maximum hit count for each session count
    hit_counts = (hits_df
                  .groupBy('session_id')
                  .agg(
                      f.first('client_id').alias('client_id'),
                      f.count('hit_id').alias('hit_count')
                  )
                  .groupBy('client_id')
                  .agg(
                      f.count('session_id').alias('session_count'),
                      f.max('hit_count').alias('hit_count')
                  )
                  .groupBy('session_count')
                  .agg(
                      f.max('hit_count').alias('max_hit_count')
                  ).repartition(32)
                  )

    # Get the session_counts for all client_ids
    session_counts = (ga_epnas_features_filtered_df
                      .groupBy('client_id')
                      .agg(
                          f.count('session_id').alias('session_count')
                      ))

    session_counts.cache()

    # Initialize udf
    zero_padder = f.udf(pad_with_zero, ArrayType(
        ArrayType(ArrayType(DoubleType()))))

    # Get the hit zero padded arrays
    # user[
    #     session[
    #         hit[1.0, 2.0, 3.0, 4.0],
    #         hit[5.0, 6.0, 7.0, 8.0]
    #     ],
    #     session[
    #         hit[9.0, 10.0, 11.0, 12.0],
    #         hit[0.0, 0.0, 0.0, 0.0]
    #     ]
    # ]
    ga_epna_data_hits = (hits_df
                         .select(
                             'client_id',
                             'session_id',
                             f.array(
                                 f.col('time_on_page'),
                                 f.col('product_list_clicks'),
                                 f.col('product_list_views'),
                                 f.col('product_detail_views')
                             ).alias('hits_features')
                         ).groupBy('session_id')
                         .agg(
                             f.first('client_id').alias('client_id'),
                             f.collect_list('hits_features').alias(
                                 'hits_features')
                         ).groupBy('client_id')
                         .agg(
                             f.collect_list(
                                 'hits_features').alias('features')
                         )
                         .join(session_counts, 'client_id', 'inner')
                         .join(hit_counts, 'session_count', 'inner')
                         .withColumn('features', zero_padder('features', 'max_hit_count'))
                         .drop('max_hit_count')
                         .repartition(32)
                         )

    # Get session arrays
    # user[
    #     session[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    #     session[13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
    # ]
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
                                     f.col('with_site_search'),
                                     f.col('without_site_search'),
                                     f.col('search_result_views'),
                                     f.col('search_uniques'),
                                     f.col('search_depth'),
                                     f.col('search_refinements')
                                 ).alias('session_features')
                             ).groupBy('client_id')
                             .agg(
                                 f.collect_list('session_features').alias(
                                     'features')
                             ).join(
                                 session_counts, 'client_id', 'inner'
                             )
                             .repartition(32)
                             )

    user_types = hits_df.groupBy('client_id').agg(
        f.last('user_type').alias('user_type'))

    # Get user arrays
    # user[1.0, 2.0, 3.0, 4.0, 5.0]
    ga_epna_data_users = (users_df.
                          join(user_types, 'client_id', 'inner').
                          withColumn(
                              'is_mobile', f.when(
                                  f.col('device_category') == 'desktop', 1.0).otherwise(0.0)
                          ).
                          withColumn(
                              'is_tablet', f.when(
                                  f.col('device_category') == 'tablet', 1.0).otherwise(0.0)
                          ).
                          withColumn(
                              'is_desktop', f.when(
                                  f.col('device_category') == 'mobile', 1.0).otherwise(0.0)
                          ).
                          withColumn(
                              'new_visitor', f.when(
                                  f.col('user_type') == 'New Visitor', 1.0).otherwise(0.0)
                          ).
                          withColumn(
                              'returning_visitor', f.when(
                                  f.col('user_type') == 'Returning Visitor', 1.0).otherwise(0.0)
                          ).
                          select(
                              'client_id',
                              f.array(
                                  f.col('is_mobile'),
                                  f.col('is_tablet'),
                                  f.col('is_desktop'),
                                  f.col('new_visitor'),
                                  f.col('returning_visitor'),
                                  f.col('device_revenue_per_transaction'),
                                  f.col('browser_revenue_per_transaction')
                              ).alias('features')
                          ).join(
                              session_counts, 'client_id', 'inner'
                          ).repartition(32)
                          )
    # Get the shopping stages arrays
    # user[
    #       session[stages],
    #       session[stages]
    # ]
    ga_epna_data_shopping_stages = (hits_df.
                                    groupBy('session_id').
                                    agg(
                                        f.first('client_id').alias(
                                            'client_id'),
                                        f.first('shopping_stage').alias(
                                            'shopping_stage')
                                    ).
                                    withColumn(
                                        'shopping_stage_1', f.when(
                                            f.col('shopping_stage') == 'ALL_VISITS', 1.0).otherwise(0.0)
                                    ).
                                    withColumn(
                                        'shopping_stage_2', f.when(
                                            f.col('shopping_stage') == 'ALL_VISITS|PRODUCT_VIEW', 1.0).otherwise(0.0)
                                    ).
                                    withColumn(
                                        'shopping_stage_3', f.when(
                                            f.col('shopping_stage') == 'ADD_TO_CART|ALL_VISITS|PRODUCT_VIEW', 1.0).otherwise(0.0)
                                    ).
                                    withColumn(
                                        'shopping_stage_4', f.when(
                                            f.col('shopping_stage') == 'ADD_TO_CART|ALL_VISITS|CHECKOUT|PRODUCT_VIEW', 1.0).otherwise(0.0)
                                    ).
                                    withColumn(
                                        'shopping_stage_5', f.when(
                                            f.col('shopping_stage') == 'ALL_VISITS|CHECKOUT|PRODUCT_VIEW', 1.0).otherwise(0.0)
                                    ).
                                    withColumn('shopping_stage_6', f.when(
                                        f.col('shopping_stage') == 'TRANSACTION', 1.0).otherwise(0.0)
                                    ).select(
                                        'client_id',
                                        'session_id',
                                        f.array(
                                            f.col('shopping_stage_1'),
                                            f.col('shopping_stage_2'),
                                            f.col('shopping_stage_3'),
                                            f.col('shopping_stage_4'),
                                            f.col('shopping_stage_5'),
                                            f.col('shopping_stage_6')
                                        ).alias('shopping_stage')
                                    ).
                                    groupBy('client_id').
                                    agg(f.collect_list('shopping_stage').alias('shopping_stages')).
                                    join(session_counts, 'client_id', 'inner').
                                    repartition(32)
                                    )

    # Get the number of hits arrays
    # user[
    #     numHitsSession1,
    #     numHitsSession2,
    #     numHitsSession3
    # ]
    ga_epna_data_num_hits = (hits_df.
                             groupBy('session_id').
                             agg(
                                 f.first('client_id').alias('client_id'),
                                 f.count('hit_id').alias('hits_count')
                             ).
                             groupBy('client_id').
                             agg(
                                 f.collect_list('hits_count').alias(
                                     'sessions_hits_count')
                             ).join(
                                 session_counts, 'client_id', 'inner'
                             ).repartition(32)
                             )

    save_data(ga_epna_data_hits, ga_epna_data_num_hits, ga_epna_data_sessions,
              ga_epna_data_users, ga_epna_data_shopping_stages)


if __name__ == '__main__':
    main()

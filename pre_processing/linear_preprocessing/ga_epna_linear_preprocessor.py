import datetime
from os import getenv
from pyspark.sql import functions as f, SparkSession


class LinearPreprocessor:
    def __init__(self):
        self.MASTER_URL = 'local[*]'
        self.APPLICATION_NAME = 'preprocessor'
        self.DAY_AS_STR = getenv('DAY_AS_STR')
        self.UNIQUE_HASH = getenv('UNIQUE_HASH')

        self.TRAINING_OR_PREDICTION = getenv('TRAINING_OR_PREDICTION')
        self.MODELS_DIR = getenv('MODELS_DIR')

        self.MORPHL_SERVER_IP_ADDRESS = getenv('MORPHL_SERVER_IP_ADDRESS')
        self.MORPHL_CASSANDRA_USERNAME = getenv('MORPHL_CASSANDRA_USERNAME')
        self.MORPHL_CASSANDRA_PASSWORD = getenv('MORPHL_CASSANDRA_PASSWORD')
        self.MORPHL_CASSANDRA_KEYSPACE = getenv('MORPHL_CASSANDRA_KEYSPACE')

        self.HDFS_PORT = 9000
        self.HDFS_DIR_TRAINING = f'hdfs://{self.MORPHL_SERVER_IP_ADDRESS}:{self.HDFS_PORT}/{self.DAY_AS_STR}_{self.UNIQUE_HASH}_ga_epna_preproc_training'
        self.HDFS_DIR_PREDICTION = f'hdfs://{self.MORPHL_SERVER_IP_ADDRESS}:{self.HDFS_PORT}/{self.DAY_AS_STR}_{self.UNIQUE_HASH}_ga_epna_preproc_prediction'

    def concatenate_hits(self, data, spark_session):

        data.cache()

        data.createOrReplaceTempView('data')

        queries = []

        for i in range(1, 4):
            queries.append([
                ", session_id as session_id_{0},".format(i),
                "LEAD(hit_id, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS hit_id_{0},".format(
                    i),
                "LEAD(transactions, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS transactions_{0},".format(
                    i),
                "LEAD(bounces, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS bounces_{0},".format(
                    i),
                "LEAD(date_hour_minute, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS date_hour_minute_{0},".format(
                    i),
                "LEAD(product_detail_views, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS product_detail_views_{0},".format(
                    i),
                "LEAD(product_list_clicks, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS product_list_clicks_{0},".format(
                    i),
                "LEAD(product_list_views, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS product_list_views_{0},".format(
                    i),
                "LEAD(shopping_stage, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS shopping_stage_{0},".format(
                    i),
                "LEAD(time_on_page, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS time_on_page_{0},".format(
                    i),
                "LEAD(user_type, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS user_type_{0},".format(
                    i),
                "LEAD(device_category, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS device_category_{0},".format(
                    i),
                "LEAD(days_since_last_session, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS days_since_last_session_{0},".format(
                    i),
                "LEAD(page_views, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS page_views_{0},".format(
                    i),
                "LEAD(search_depth, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS search_depth_{0},".format(
                    i),
                "LEAD(search_refinements, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS search_refinements_{0},".format(
                    i),
                "LEAD(search_result_views, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS search_result_views_{0},".format(
                    i),
                "LEAD(search_uniques, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS search_uniques_{0},".format(
                    i),
                "LEAD(search_used, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS search_used_{0},".format(
                    i),
                "LEAD(session_duration, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS session_duration_{0},".format(
                    i),
                "LEAD(transaction_revenue, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS transaction_revenue_{0},".format(
                    i),
                "LEAD(unique_page_views, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS unique_page_views_{0},".format(
                    i),
                "LEAD(unique_purchases, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS unique_purchases_{0},".format(
                    i),
                "LEAD(days_to_transaction, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS days_to_transaction_{0},".format(
                    i),
                "LEAD(sessions_to_transaction, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS sessions_to_transaction_{0},".format(
                    i),
                "LEAD(sessions, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS sessions_{0},".format(
                    i),
                "LEAD(revenue_per_user, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS revenue_per_user_{0},".format(
                    i),
                "LEAD(transactions_per_user, {0}) OVER (PARTITION BY session_id ORDER BY hit_id) AS transactions_per_user_{0}".format(
                    i)
            ])

        sql_parts = [
            "SELECT *",
            "FROM (SELECT",
            "client_id AS client_id_0,",
            "session_id AS session_id_0,",
            "hit_id AS hit_id_0,",
            "transactions as transactions_0,",
            "bounces as bounces_0,",
            "date_hour_minute as date_hour_minute_0,",
            "product_detail_views as product_detail_views_0,",
            "product_list_clicks as product_list_clicks_0,",
            "product_list_views as product_list_views_0,",
            "shopping_stage as shopping_stage_0,",
            "time_on_page as time_on_page_0,",
            "user_type as user_type_0,",
            "device_category as device_category_0,",
            "days_since_last_session as days_since_last_session_0,",
            "page_views as page_views_0,",
            "search_depth as search_depth_0,",
            "search_refinements as search_refinements_0,",
            "search_result_views as search_result_views_0,",
            "search_uniques as search_uniques_0,",
            "search_used as search_used_0,",
            "session_duration as session_duration_0,",
            "transaction_revenue as transaction_revenue_0,",
            "unique_page_views as unique_page_views_0,",
            "unique_purchases as unique_purchases_0,",
            "days_to_transaction as days_to_transaction_0,",
            "sessions_to_transaction as session_to_transactions_0,",
            "sessions as sessions_0,",
            "revenue_per_user as revenue_per_user_0,",
            "transactions_per_user as transactions_per_user_0",
            ' '.join(queries[0]),
            ' '.join(queries[1]),
            ' '.join(queries[2]),
            "FROM data",
            ") AS step_1",
            "WHERE hit_id_3 IS NOT NULL"
        ]

        sql = ' '.join(sql_parts)

        concat_df = spark_session.sql(sql)

        return concat_df

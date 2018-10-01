import datetime
from os import getenv
from pyspark.sql import functions as f, SparkSession


class BasicPreprocessor:

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

        self.init_keys()
        self.init_baselines()

    def init_keys(self):
        primary_key = {}

        primary_key['ga_epnau_df'] = ['client_id', 'day_of_data_capture']
        primary_key['ga_epnas_df'] = ['client_id',
                                      'day_of_data_capture', 'session_id']
        primary_key['ga_epnah_df'] = [
            'client_id', 'day_of_data_capture', 'session_id', 'hit_id']
        primary_key['ga_epnat_df'] = ['client_id',
                                      'day_of_data_capture', 'session_id', 'transaction_id']

        self.primary_key = primary_key

    def init_baselines(self):

        field_baselines = {}

        field_baselines['ga_epnau_df'] = [
            {'field_name': 'device_category',
             'original_name': 'ga:deviceCategory',
             'needs_conversion': False,
             },
            {'field_name': 'sessions',
             'original_name': 'ga:sessions',
             'needs_conversion': True,
             },
            {'field_name': 'bounces',
             'original_name': 'ga:bounces',
             'needs_conversion': True,
             },
            {'field_name': 'revenue_per_user',
             'original_name': 'ga:revenuePerUser',
             'needs_conversion': True,
             },
            {'field_name': 'transactions_per_user',
             'original_name': 'ga:transactionsPerUser',
             'needs_conversion': True,
             },
        ]

        field_baselines['ga_epnas_df'] = [
            {'field_name': 'session_duration',
             'original_name': 'ga:sessionDuration',
             'needs_conversion': True,
             },
            {'field_name': 'page_views',
             'original_name': 'ga:pageviews',
             'needs_conversion': True,
             },
            {'field_name': 'unique_page_views',
             'original_name': 'ga:uniquePageviews',
             'needs_conversion': True,
             },
            {'field_name': 'transactions',
             'original_name': 'ga:transactions',
             'needs_conversion': True,
             },
            {'field_name': 'transaction_revenue',
             'original_name': 'ga:transactionRevenue',
             'needs_conversion': True,
             },
            {'field_name': 'unique_purchases',
             'original_name': 'ga:uniquePurchases',
             'needs_conversion': True,
             },
            {'field_name': 'search_result_views',
             'original_name': 'ga:searchResultViews',
             'needs_conversion': True,
             },
            {'field_name': 'search_uniques',
             'original_name': 'ga:searchUniques',
             'needs_conversion': True,
             },
            {'field_name': 'search_depth',
             'original_name': 'ga:searchDepth',
             'needs_conversion': True,
             },
            {'field_name': 'search_refinements',
             'original_name': 'ga:searchRefinements',
             'needs_conversion': True,
             },
            {'field_name': 'search_used',
             'original_name': 'ga:searchUsed',
             'needs_conversion': False,
             },
            {'field_name': 'days_since_last_session',
             'original_name': 'ga:daysSinceLastSession',
             'needs_conversion': True,
             },
        ]

        field_baselines['ga_epnah_df'] = [
            {'field_name': 'time_on_page',
             'original_name': 'ga:timeOnPage',
             'needs_conversion': True,
             },
            {'field_name': 'product_list_clicks',
             'original_name': 'ga:productListClicks',
             'needs_conversion': True,
             },
            {'field_name': 'product_list_views',
             'original_name': 'ga:productListViews',
             'needs_conversion': True,
             },
            {'field_name': 'product_detail_views',
             'original_name': 'ga:productDetailViews',
             'needs_conversion': True,
             },
            {'field_name': 'user_type',
             'original_name': 'ga:userType',
             'needs_conversion': False,
             },
            {'field_name': 'shopping_stage',
             'original_name': 'ga:shoppingStage',
             'needs_conversion': False,
             },
            {'field_name': 'date_hour_minute',
             'original_name': 'ga:dateHourMinute',
             'needs_conversion': False,
             },
        ]

        field_baselines['ga_epnat_df'] = [
            {'field_name': 'days_to_transaction',
             'original_name': 'ga:daysToTransaction',
             'needs_conversion': True,
             },
            {'field_name': 'sessions_to_transaction',
             'original_name': 'ga:sessionsToTransaction',
             'needs_conversion': True,
             },
        ]

        self.field_baselines = field_baselines

    def fetch_from_cassandra(self, c_table_name, spark_session):
        load_options = {
            'keyspace': self.MORPHL_CASSANDRA_KEYSPACE,
            'table': c_table_name,
            'spark.cassandra.input.fetch.size_in_rows': '150'}

        df = (spark_session.read.format('org.apache.spark.sql.cassandra')
                                .options(**load_options)
                                .load())

        return df

    def get_json_schemas(self, df, spark_session):
        return {
            'json_meta_schema': spark_session.read.json(
                df.limit(10).rdd.map(lambda row: row.json_meta)).schema,
            'json_data_schema': spark_session.read.json(
                df.limit(10).rdd.map(lambda row: row.json_data)).schema}

    def zip_lists_full_args(self,
                            json_meta_dimensions,
                            json_meta_metrics,
                            json_data_dimensions,
                            json_data_metrics,
                            field_attributes,
                            schema_as_list):
        orig_meta_fields = json_meta_dimensions + json_meta_metrics
        orig_meta_fields_set = set(orig_meta_fields)
        for fname in schema_as_list:
            assert(field_attributes[fname]['original_name'] in orig_meta_fields_set), \
                'The field {} is not part of the input record'
        data_values = json_data_dimensions + json_data_metrics[0].values
        zip_list_as_dict = dict(zip(orig_meta_fields, data_values))
        values = [
            zip_list_as_dict[field_attributes[fname]['original_name']]
            for fname in schema_as_list]

        return values

    def process(self, df, primary_key, field_baselines):
        schema_as_list = [
            fb['field_name']
            for fb in field_baselines]

        field_attributes = dict([
            (fb['field_name'], fb)
            for fb in field_baselines])

        meta_fields = [
            'raw_{}'.format(
                fname) if field_attributes[fname]['needs_conversion'] else fname
            for fname in schema_as_list]

        schema_before_concat = [
            '{}: string'.format(mf) for mf in meta_fields]

        schema = ', '.join(schema_before_concat)

        def zip_lists(json_meta_dimensions,
                      json_meta_metrics,
                      json_data_dimensions,
                      json_data_metrics):
            return self.zip_lists_full_args(json_meta_dimensions,
                                            json_meta_metrics,
                                            json_data_dimensions,
                                            json_data_metrics,
                                            field_attributes,
                                            schema_as_list)

        zip_lists_udf = f.udf(zip_lists, schema)

        after_zip_lists_udf_df = (
            df.withColumn('all_values', zip_lists_udf('jmeta_dimensions',
                                                      'jmeta_metrics',
                                                      'jdata_dimensions',
                                                      'jdata_metrics')))

        interim_fields_to_select = primary_key + ['all_values.*']

        interim_df = after_zip_lists_udf_df.select(*interim_fields_to_select)

        to_float_udf = f.udf(lambda s: float(s), 'float')

        for fname in schema_as_list:
            if field_attributes[fname]['needs_conversion']:
                fname_raw = 'raw_{}'.format(fname)
                interim_df = interim_df.withColumn(
                    fname, to_float_udf(fname_raw))

        fields_to_select = primary_key + schema_as_list

        result_df = interim_df.select(*fields_to_select)

        return {'result_df': result_df,
                'schema_as_list': schema_as_list}

    def process_user_data(self, user_data, spark_session):

        user_data.cache()

        user_data.createOrReplaceTempView('user_data')

        groupe_by_client_id_sql = "SELECT client_id, device_category, SUM(sessions) OVER (PARTITION BY client_id) AS sessions, SUM(bounces) OVER (PARTITION BY client_id) AS bounces, SUM(revenue_per_user) OVER (PARTITION BY client_id) AS revenue_per_user, SUM(transactions_per_user) OVER (PARTITION BY client_id) AS transactions_per_user FROM user_data"

        grouped_by_client_id_df = spark_session.sql(
            groupe_by_client_id_sql)

        return grouped_by_client_id_df

    def process_sessions_and_transactions_data(self, sessions_data, transactions_data):
        sessions_data = sessions_data.drop('day_of_data_capture')

        transactions_data = transactions_data.drop(
            'day_of_data_capture', 'transaction_id')

        joined_data = sessions_data.join(
            transactions_data, on=['client_id', 'session_id'], how='outer')

        final_data = joined_data.filter(
            joined_data.session_duration > 0.0).na.fill(0)

        return final_data

    def process_hits_data(self, hits_data, spark_session):

        cart_and_transaction_df = hits_data[(
            hits_data['shopping_stage'] == 'CART_ABANDONMENT') | (hits_data['shopping_stage'] == 'TRANSACTION')].dropDuplicates()

        product_view_df = hits_data[hits_data['shopping_stage']
                                    == 'PRODUCT_VIEW'].dropDuplicates()

        cart_and_transaction_df.createOrReplaceTempView('cart_and_transaction')
        product_view_df.createOrReplaceTempView('product_view')

        clean_product_view_sql = 'SELECT * FROM product_view WHERE session_id NOT IN (SELECT session_id FROM cart_and_transaction)'

        cleaned_product_view = spark_session.sql(clean_product_view_sql)

        no_activity_df = hits_data[(
            hits_data['shopping_stage'] == 'NO_SHOPPING_ACTIVITY')].dropDuplicates()

        return cart_and_transaction_df.union(cleaned_product_view).union(no_activity_df)

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

    def get_parsed_jsons(self, json_schemas, dataframes):

        after_json_parsing_df = {}

        after_json_parsing_df['ga_epnau_df'] = (
            dataframes['ga_epnau_df']
            .withColumn('jmeta', f.from_json(
                f.col('json_meta'), json_schemas['ga_epnau_df']['json_meta_schema']))
            .withColumn('jdata', f.from_json(
                f.col('json_data'), json_schemas['ga_epnau_df']['json_data_schema']))
            .select(f.col('client_id'),
                    f.col('day_of_data_capture'),
                    f.col('jmeta.dimensions').alias('jmeta_dimensions'),
                    f.col('jmeta.metrics').alias('jmeta_metrics'),
                    f.col('jdata.dimensions').alias('jdata_dimensions'),
                    f.col('jdata.metrics').alias('jdata_metrics')))

        after_json_parsing_df['ga_epnas_df'] = (
            dataframes['ga_epnas_df']
            .withColumn('jmeta', f.from_json(
                f.col('json_meta'), json_schemas['ga_epnas_df']['json_meta_schema']))
            .withColumn('jdata', f.from_json(
                f.col('json_data'), json_schemas['ga_epnas_df']['json_data_schema']))
            .select(f.col('client_id'),
                    f.col('day_of_data_capture'),
                    f.col('session_id'),
                    f.col('jmeta.dimensions').alias('jmeta_dimensions'),
                    f.col('jmeta.metrics').alias('jmeta_metrics'),
                    f.col('jdata.dimensions').alias('jdata_dimensions'),
                    f.col('jdata.metrics').alias('jdata_metrics')))

        after_json_parsing_df['ga_epnah_df'] = (
            dataframes['ga_epnah_df']
            .withColumn('jmeta', f.from_json(
                f.col('json_meta'), json_schemas['ga_epnah_df']['json_meta_schema']))
            .withColumn('jdata', f.from_json(
                f.col('json_data'), json_schemas['ga_epnah_df']['json_data_schema']))
            .select(f.col('client_id'),
                    f.col('day_of_data_capture'),
                    f.col('session_id'),
                    f.col('hit_id'),
                    f.col('jmeta.dimensions').alias('jmeta_dimensions'),
                    f.col('jmeta.metrics').alias('jmeta_metrics'),
                    f.col('jdata.dimensions').alias('jdata_dimensions'),
                    f.col('jdata.metrics').alias('jdata_metrics')))

        after_json_parsing_df['ga_epnat_df'] = (
            dataframes['ga_epnat_df']
            .withColumn('jmeta', f.from_json(
                f.col('json_meta'), json_schemas['ga_epnat_df']['json_meta_schema']))
            .withColumn('jdata', f.from_json(
                f.col('json_data'), json_schemas['ga_epnat_df']['json_data_schema']))
            .select(f.col('client_id'),
                    f.col('day_of_data_capture'),
                    f.col('session_id'),
                    f.col('transaction_id'),
                    f.col('jmeta.dimensions').alias('jmeta_dimensions'),
                    f.col('jmeta.metrics').alias('jmeta_metrics'),
                    f.col('jdata.dimensions').alias('jdata_dimensions'),
                    f.col('jdata.metrics').alias('jdata_metrics')))

        return after_json_parsing_df

    def save_raw_data(self, user_data, session_data, hit_data, transaction_data):
        save_options_ga_epnau_features_raw = {
            'keyspace': self.MORPHL_CASSANDRA_KEYSPACE,
            'table': ('ga_epnau_features_raw')
        }
        save_options_ga_epnas_features_raw = {
            'keyspace': self.MORPHL_CASSANDRA_KEYSPACE,
            'table': ('ga_epnas_features_raw')
        }
        save_options_ga_epnah_features_raw = {
            'keyspace': self.MORPHL_CASSANDRA_KEYSPACE,
            'table': ('ga_epnah_features_raw')
        }
        save_options_ga_epnat_features_raw = {
            'keyspace': self.MORPHL_CASSANDRA_KEYSPACE,
            'table': ('ga_epnat_features_raw')
        }

        (user_data
            .write
            .format('org.apache.spark.sql.cassandra')
            .mode('append')
            .options(**save_options_ga_epnau_features_raw)
            .save())

        (session_data
            .write
            .format('org.apache.spark.sql.cassandra')
            .mode('append')
            .options(**save_options_ga_epnas_features_raw)
            .save())

        (hit_data
            .write
            .format('org.apache.spark.sql.cassandra')
            .mode('append')
            .options(**save_options_ga_epnah_features_raw)
            .save())

        (transaction_data
            .write
            .format('org.apache.spark.sql.cassandra')
            .mode('append')
            .options(**save_options_ga_epnat_features_raw)
            .save())

    def process_data(self, user_data, session_data, hit_data, transaction_data):
        spark_session = self.get_spark_session()

        processed_user_data = self.process_user_data(user_data, spark_session)
        processed_session_and_transaction_data = self.process_sessions_and_transactions_data(
            session_data, transaction_data)
        processed_hit_data = self.process_hits_data(hit_data, spark_session)

        user_and_sessions_data = processed_user_data.join(
            processed_session_and_transaction_data, on=['client_id'], how='inner').dropDuplicates()

        final_join_data = processed_hit_data.join(user_and_sessions_data, on=[
            'client_id', 'session_id'], how='inner').dropDuplicates()

        return final_join_data.na.fill(0)

    def main(self):

        spark_session = self.get_spark_session()

        # ga_config_df = (
        #     self.fetch_from_cassandra(
        #         'ga_epna_config_parameters', spark_session)
        #     .filter("morphl_component_name = 'ga_epna' AND parameter_name = 'days_worth_of_data_to_load'"))

        # days_worth_of_data_to_load = int(ga_config_df.first().parameter_value)

        start_date = ((
            datetime.datetime(year=2018, month=7, day=2) -
            datetime.timedelta(days=0))
            .strftime('%Y-%m-%d'))

        ga_epna_users_df = self.fetch_from_cassandra(
            'ga_epna_users', spark_session)

        ga_epna_sessions_df = self.fetch_from_cassandra(
            'ga_epna_sessions', spark_session)

        ga_epna_hits_df = self.fetch_from_cassandra(
            'ga_epna_hits', spark_session)

        ga_epna_transactions_df = self.fetch_from_cassandra(
            'ga_epna_transactions', spark_session)

        dataframes = {}

        dataframes['ga_epnau_df'] = (
            ga_epna_users_df
            .filter("day_of_data_capture >= '{}'".format(start_date)))

        dataframes['ga_epnas_df'] = (
            ga_epna_sessions_df
            .filter("day_of_data_capture >= '{}'".format(start_date)))

        dataframes['ga_epnah_df'] = (
            ga_epna_hits_df
            .filter("day_of_data_capture >= '{}'".format(start_date)))

        dataframes['ga_epnat_df'] = (
            ga_epna_transactions_df
            .filter("day_of_data_capture >= '{}'".format(start_date)))

        json_schemas = {}

        json_schemas['ga_epnau_df'] = self.get_json_schemas(
            dataframes['ga_epnau_df'], spark_session)
        json_schemas['ga_epnas_df'] = self.get_json_schemas(
            dataframes['ga_epnas_df'], spark_session)
        json_schemas['ga_epnah_df'] = self.get_json_schemas(
            dataframes['ga_epnah_df'], spark_session)
        json_schemas['ga_epnat_df'] = self.get_json_schemas(
            dataframes['ga_epnat_df'], spark_session)

        after_json_parsing_df = self.get_parsed_jsons(json_schemas, dataframes)

        processed_users_dict = self.process(after_json_parsing_df['ga_epnau_df'],
                                            self.primary_key['ga_epnau_df'],
                                            self.field_baselines['ga_epnau_df'])

        processed_sessions_dict = self.process(after_json_parsing_df['ga_epnas_df'],
                                               self.primary_key['ga_epnas_df'],
                                               self.field_baselines['ga_epnas_df'])

        processed_hits_dict = self.process(after_json_parsing_df['ga_epnah_df'],
                                           self.primary_key['ga_epnah_df'],
                                           self.field_baselines['ga_epnah_df'])

        processed_transactions_dict = self.process(after_json_parsing_df['ga_epnat_df'],
                                                   self.primary_key['ga_epnat_df'],
                                                   self.field_baselines['ga_epnat_df'])

        users_df = (
            processed_users_dict['result_df']
        )

        sessions_df = (
            processed_sessions_dict['result_df']
        )

        hits_df = (
            processed_hits_dict['result_df']
        )

        transactions_df = (
            processed_transactions_dict['result_df']
            .drop('transactions')
        )

        # self.save_raw_data(users_df, sessions_df, hits_df, transactions_df)
        a = self.process_data(users_df, sessions_df, hits_df, transactions_df)
        a.toPandas().to_csv('final-final.csv')


if __name__ == '__main__':
    preprocessor = BasicPreprocessor()
    preprocessor.main()

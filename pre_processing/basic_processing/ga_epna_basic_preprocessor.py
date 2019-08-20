import datetime
from os import getenv
from pyspark.sql import functions as f, SparkSession


class BasicPreprocessor:

    def __init__(self):

        self.MASTER_URL = 'local[*]'
        self.APPLICATION_NAME = 'preprocessor'

        self.MORPHL_SERVER_IP_ADDRESS = getenv('MORPHL_SERVER_IP_ADDRESS')
        self.MORPHL_CASSANDRA_USERNAME = getenv('MORPHL_CASSANDRA_USERNAME')
        self.MORPHL_CASSANDRA_PASSWORD = getenv('MORPHL_CASSANDRA_PASSWORD')
        self.MORPHL_CASSANDRA_KEYSPACE = getenv('MORPHL_CASSANDRA_KEYSPACE')

        self.init_keys()
        self.init_baselines()

    # Initialize Cassandra tables primary keys.
    def init_keys(self):

        primary_key = {}

        primary_key['ga_epnau_df'] = ['client_id', 'day_of_data_capture']
        primary_key['ga_epnas_df'] = ['client_id',
                                      'day_of_data_capture', 'session_id']
        primary_key['ga_epnah_df'] = [
            'client_id', 'day_of_data_capture', 'session_id', 'date_hour_minute']

        self.primary_key = primary_key

    # Get all the features that need to be parsed from the jsons.
    def init_baselines(self):

        field_baselines = {}

        # device_category: the type of device associated to the user: tablet, mobile or desktop;
        # browser: the type of browser used by a user;
        field_baselines['ga_epnau_df'] = [
            {'field_name': 'device_category',
             'original_name': 'ga:deviceCategory',
             'needs_conversion': False,
             },
            {'field_name': 'browser',
             'original_name': 'ga:browser',
             'needs_conversion': False,
             },
        ]

        # session_duration: the total duration of a session;
        # unique_page_views: the number of unique pageviews during a session;
        # transactions: the number of transactions completed in a session;
        # transaction_revenue: the amount of money spent on the transactions in a session;
        # search_result_views: the number of times a search result page was viewed in a session;
        # search_uniques: total number of unique keywords from internal searches during a session;
        # search_depth: total number of subsequent page views made after an internal search;
        # search_refinements: the number of times a transition occurs between internal keywords search within a session
        #                     ex: "shoes", "shoes", "pants", "pants" => 1 transition from shoes to pants
        # search_used: either Visits With Site Search or Visits Without Site Search;
        # days_since_last_session: the number of days elapsed since the last sessions.
        field_baselines['ga_epnas_df'] = [
            {'field_name': 'session_duration',
             'original_name': 'ga:sessionDuration',
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

        # time_on_page: the amount of time a user spent on the page;
        # product_detail_views: number of time the product details were viewed.
        field_baselines['ga_epnah_df'] = [
            {'field_name': 'time_on_page',
             'original_name': 'ga:timeOnPage',
             'needs_conversion': True,
             },
            {'field_name': 'product_detail_views',
             'original_name':'ga:productDetailViews',
             'needs_conversion': True,
            }
        ]


        self.field_baselines = field_baselines

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

    # Get the json schema of a df.
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

    # Get the parsed jsons as dfs.
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
                    f.col('date_hour_minute'),
                    f.col('jmeta.dimensions').alias('jmeta_dimensions'),
                    f.col('jmeta.metrics').alias('jmeta_metrics'),
                    f.col('jdata.dimensions').alias('jdata_dimensions'),
                    f.col('jdata.metrics').alias('jdata_metrics')))


        return after_json_parsing_df

    # Parse json data.
    def process_json_data(self, df, primary_key, field_baselines):
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

    # Save the raw dfs to Cassandra tables.
    def save_raw_data(self, user_data, session_data, hit_data):

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

    def main(self):

        spark_session = self.get_spark_session()

        # Get the number of days to process.
        ga_config_df = (
            self.fetch_from_cassandra(
                'ga_epna_config_parameters', spark_session)
            .filter("morphl_component_name = 'ga_epna' AND parameter_name = 'days_prediction_interval'"))

        days_prediction_interval = int(ga_config_df.first().parameter_value)

        start_date = ((datetime.datetime.now(
        ) - datetime.timedelta(days=days_prediction_interval + 1)).strftime('%Y-%m-%d'))

        # Fetch required tables from Cassandra.
        ga_epna_users_df = self.fetch_from_cassandra(
            'ga_epna_users', spark_session)

        ga_epna_sessions_df = self.fetch_from_cassandra(
            'ga_epna_sessions', spark_session)

        ga_epna_hits_df = self.fetch_from_cassandra(
            'ga_epna_hits', spark_session)

        dataframes = {}

        # Filter them by the date we need.
        dataframes['ga_epnau_df'] = (
            ga_epna_users_df
            .filter("day_of_data_capture >= '{}'".format(start_date)))

        dataframes['ga_epnas_df'] = (
            ga_epna_sessions_df
            .filter("day_of_data_capture >= '{}'".format(start_date)))

        dataframes['ga_epnah_df'] = (
            ga_epna_hits_df
            .filter("day_of_data_capture >= '{}'".format(start_date)))

        json_schemas = {}

        # Get each df's json schema.
        json_schemas['ga_epnau_df'] = self.get_json_schemas(
            dataframes['ga_epnau_df'], spark_session)
        json_schemas['ga_epnas_df'] = self.get_json_schemas(
            dataframes['ga_epnas_df'], spark_session)
        json_schemas['ga_epnah_df'] = self.get_json_schemas(
            dataframes['ga_epnah_df'], spark_session)

        after_json_parsing_df = self.get_parsed_jsons(json_schemas, dataframes)

        processed_users_dict = self.process_json_data(after_json_parsing_df['ga_epnau_df'],
                                                      self.primary_key['ga_epnau_df'],
                                                      self.field_baselines['ga_epnau_df'])

        processed_sessions_dict = self.process_json_data(after_json_parsing_df['ga_epnas_df'],
                                                         self.primary_key['ga_epnas_df'],
                                                         self.field_baselines['ga_epnas_df'])

        processed_hits_dict = self.process_json_data(after_json_parsing_df['ga_epnah_df'],
                                                     self.primary_key['ga_epnah_df'],
                                                     self.field_baselines['ga_epnah_df'])
                                    
                                    

        users_df = processed_users_dict['result_df']

        sessions_df = processed_sessions_dict['result_df']

        hits_df = processed_hits_dict['result_df']

        self.save_raw_data(users_df, sessions_df, hits_df)


if __name__ == '__main__':
    preprocessor = BasicPreprocessor()
    preprocessor.main()

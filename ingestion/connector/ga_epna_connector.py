"""Google Analytics Reporting API V4 Connector for the MorphL project"""

from time import sleep, strptime, mktime
from json import dumps
from os import getenv
from sys import exc_info

from apiclient.discovery import build
from google.oauth2 import service_account

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider


class CassandraPersistence:
    def __init__(self):
        self.DAY_OF_DATA_CAPTURE = getenv('DAY_OF_DATA_CAPTURE')
        self.MORPHL_SERVER_IP_ADDRESS = getenv('MORPHL_SERVER_IP_ADDRESS')
        self.MORPHL_CASSANDRA_USERNAME = getenv('MORPHL_CASSANDRA_USERNAME')
        self.MORPHL_CASSANDRA_PASSWORD = getenv('MORPHL_CASSANDRA_PASSWORD')
        self.MORPHL_CASSANDRA_KEYSPACE = getenv('MORPHL_CASSANDRA_KEYSPACE')
        self.CASS_REQ_TIMEOUT = 3600.0

        self.auth_provider = PlainTextAuthProvider(
            username=self.MORPHL_CASSANDRA_USERNAME, password=self.MORPHL_CASSANDRA_PASSWORD)
        self.cluster = Cluster(
            contact_points=[self.MORPHL_SERVER_IP_ADDRESS], auth_provider=self.auth_provider)
        self.session = self.cluster.connect(self.MORPHL_CASSANDRA_KEYSPACE)

        self.prepare_statements()

    def prepare_statements(self):
        """
            Prepare statements for database insert queries
        """
        self.prep_stmts = {}

        type_1_list = ['users']
        type_2_list = ['users_mobile_brand']
        type_3_list = ['sessions']
        type_4_list = ['sessions_shopping_stages']
        type_5_list = ['hits']
        type_6_list = ['product_info']
        # type_7_list = ['event_info']
        type_8_list = ['session_index']

        template_for_type_1 = 'INSERT INTO ga_epna_{} (client_id,day_of_data_capture,json_meta,json_data) VALUES (?,?,?,?)'
        template_for_type_2 = 'INSERT INTO ga_epna_{} (client_id,day_of_data_capture,mobile_device_branding) VALUES (?,?,?)'
        template_for_type_3 = 'INSERT INTO ga_epna_{} (client_id,day_of_data_capture,session_id,json_meta,json_data) VALUES (?,?,?,?,?)'
        template_for_type_4 = 'INSERT INTO ga_epna_{} (client_id,day_of_data_capture,session_id,shopping_stage) VALUES (?,?,?,?)'
        template_for_type_5 = 'INSERT INTO ga_epna_{} (client_id,day_of_data_capture,session_id,date_hour_minute,json_meta,json_data) VALUES (?,?,?,?,?,?)'
        template_for_type_6 = 'INSERT INTO ga_epna_{} (client_id,day_of_data_capture,session_id,product_name,date_hour_minute,json_meta,json_data) VALUES (?,?,?,?,?,?,?)'
        # template_for_type_7 = 'INSERT INTO ga_epna_{} (client_id,day_of_data_capture,session_id,event_action,event_category,date_hour_minute) VALUES (?,?,?,?,?,?)'
        template_for_type_8 = 'INSERT INTO ga_epna_{} (client_id,day_of_data_capture,session_id,session_index) VALUES (?,?,?,?)'

        for report_type in type_1_list:
            self.prep_stmts[report_type] = self.session.prepare(
                template_for_type_1.format(report_type))
        for report_type in type_2_list:
            self.prep_stmts[report_type] = self.session.prepare(
                template_for_type_2.format(report_type))
        for report_type in type_3_list:
            self.prep_stmts[report_type] = self.session.prepare(
                template_for_type_3.format(report_type))
        for report_type in type_4_list:
            self.prep_stmts[report_type] = self.session.prepare(
                template_for_type_4.format(report_type))
        for report_type in type_5_list:
            self.prep_stmts[report_type] = self.session.prepare(
                template_for_type_5.format(report_type))
        for report_type in type_6_list:
            self.prep_stmts[report_type] = self.session.prepare(
                template_for_type_6.format(report_type))
        # for report_type in type_7_list:
        #     self.prep_stmts[report_type] = self.session.prepare(
        #         template_for_type_7.format(report_type))
        for report_type in type_8_list:
            self.prep_stmts[report_type] = self.session.prepare(
                template_for_type_8.format(report_type))

        self.type_1_set = set(type_1_list)
        self.type_2_set = set(type_2_list)
        self.type_3_set = set(type_3_list)
        self.type_4_set = set(type_4_list)
        self.type_5_set = set(type_5_list)
        self.type_6_set = set(type_6_list)
        # self.type_7_set = set(type_7_list)
        self.type_8_set = set(type_8_list)

    def persist_dict_record(self, report_type, meta_dict, data_dict):
        day_of_data_capture_timestamp = str(
            mktime(strptime(self.DAY_OF_DATA_CAPTURE, '%Y-%m-%d'))).replace('.0', '')
        raw_cl_id = data_dict['dimensions'][0]
        client_id = raw_cl_id if raw_cl_id.startswith('GA') else 'UNKNOWN'
        json_meta = dumps(meta_dict)
        json_data = dumps(data_dict)

        # User related data
        if report_type in self.type_1_set:
            bind_list = [client_id, self.DAY_OF_DATA_CAPTURE,
                         json_meta, json_data]
            return {'cassandra_future': self.session.execute_async(self.prep_stmts[report_type],
                                                                   bind_list,
                                                                   timeout=self.CASS_REQ_TIMEOUT),
                    'client_id': client_id}

        # Mobile device related data for mobile users
        if report_type in self.type_2_set:
            mobile_device_branding = data_dict['dimensions'][1]
            bind_list = [client_id, self.DAY_OF_DATA_CAPTURE,
                         mobile_device_branding]

            return {'cassandra_future': self.session.execute_async(self.prep_stmts[report_type],
                                                                   bind_list,
                                                                   timeout=self.CASS_REQ_TIMEOUT),
                    'client_id': client_id}

        # Session related data
        if report_type in self.type_3_set:
            session_id = data_dict['dimensions'][1] + '.' + \
                str(client_id) + '.' + day_of_data_capture_timestamp
            bind_list = [client_id, self.DAY_OF_DATA_CAPTURE,
                         session_id, json_meta, json_data]
            return {'cassandra_future': self.session.execute_async(self.prep_stmts[report_type],
                                                                   bind_list,
                                                                   timeout=self.CASS_REQ_TIMEOUT),
                    'client_id': client_id,
                    'session_id': session_id}

        # Session shopping stage data
        if report_type in self.type_4_set:
            session_id = data_dict['dimensions'][1] + '.' + \
                str(client_id) + '.' + day_of_data_capture_timestamp
            shopping_stage = data_dict['dimensions'][2]
            bind_list = [client_id, self.DAY_OF_DATA_CAPTURE,
                         session_id, shopping_stage]

            return {'cassandra_future': self.session.execute_async(self.prep_stmts[report_type],
                                                                   bind_list,
                                                                   timeout=self.CASS_REQ_TIMEOUT),
                    'client_id': client_id,
                    'session_id': session_id,
                    'shopping_stage': shopping_stage}

        # Hit related data
        if report_type in self.type_5_set:
            session_id = data_dict['dimensions'][1] + '.' + \
                str(client_id) + '.' + day_of_data_capture_timestamp
            date_hour_minute = data_dict['dimensions'][2]
            bind_list = [client_id, self.DAY_OF_DATA_CAPTURE,
                         session_id, date_hour_minute, json_meta, json_data]

            return {'cassandra_future': self.session.execute_async(self.prep_stmts[report_type],
                                                                   bind_list,
                                                                   timeout=self.CASS_REQ_TIMEOUT),
                    'client_id': client_id,
                    'session_id': session_id,
                    'date_hour_minute': date_hour_minute}

        if report_type in self.type_6_set:
            session_id = data_dict['dimensions'][1] + '.' + \
                str(client_id) + '.' + day_of_data_capture_timestamp
            date_hour_minute = data_dict['dimensions'][2]
            product_name = data_dict['dimensions'][3]

            bind_list = [client_id, self.DAY_OF_DATA_CAPTURE,
                         session_id, product_name, date_hour_minute, json_meta, json_data]

            return {'cassandra_future': self.session.execute_async(self.prep_stmts[report_type],
                                                                   bind_list,
                                                                   timeout=self.CASS_REQ_TIMEOUT),
                    'client_id': client_id,
                    'session_id': session_id,
                    'product_name': product_name,
                    'date_hour_minute': date_hour_minute
                    }

        # if report_type in self.type_7_set:
        #     session_id = data_dict['dimensions'][1] + '.' + \
        #         str(client_id) + '.' + day_of_data_capture_timestamp
        #     date_hour_minute = data_dict['dimensions'][2]
        #     event_action = data_dict['dimensions'][3]
        #     event_category = data_dict['dimensions'][4]

        #     bind_list = [client_id, self.DAY_OF_DATA_CAPTURE,
        #                  session_id, event_action, event_category, date_hour_minute]

        #     return {'cassandra_future': self.session.execute_async(self.prep_stmts[report_type],
        #                                                            bind_list,
        #                                                            timeout=self.CASS_REQ_TIMEOUT),
        #             'client_id': client_id,
        #             'session_id': session_id,
        #             'event_action': event_action,
        #             'event_category': event_category,
        #             'date_hour_minute': date_hour_minute
        #             }

        if report_type in self.type_8_set:
            session_id = data_dict['dimensions'][1] + '.' + \
                str(client_id) + '.' + day_of_data_capture_timestamp

            session_index = int(data_dict['dimensions'][2])

            bind_list = [client_id, self.DAY_OF_DATA_CAPTURE,
                         session_id, session_index]

            return {'cassandra_future': self.session.execute_async(self.prep_stmts[report_type],
                                                                   bind_list,
                                                                   timeout=self.CASS_REQ_TIMEOUT),
                    'client_id': client_id,
                    'session_id': session_id,
                    }


class GoogleAnalytics:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
        self.KEY_FILE_LOCATION = getenv('GA_EPNA_KEY_FILE_LOCATION')
        self.VIEW_ID = getenv('GA_EPNA_VIEW_ID')
        self.API_PAGE_SIZE = 10000
        self.DAY_OF_DATA_CAPTURE = getenv('DAY_OF_DATA_CAPTURE')
        self.start_date = self.DAY_OF_DATA_CAPTURE
        self.end_date = self.DAY_OF_DATA_CAPTURE
        self.analytics = None
        self.store = CassandraPersistence()

    # Initializes an Analytics Reporting API V4 service object.
    def authenticate(self):
        credentials = service_account.Credentials \
            .from_service_account_file(self.KEY_FILE_LOCATION) \
            .with_scopes(self.SCOPES)
        # Build the service object.
        self.analytics = build('analyticsreporting',
                               'v4', credentials=credentials)

    # Transform list of dimensions names into objects with a 'name' property.
    def format_dimensions(self, dims):
        return [{'name': 'ga:' + dim} for dim in dims]

    # Transform list of metrics names into objects with an 'expression' property.
    def format_metrics(self, metrics):
        return [{'expression': 'ga:' + metric} for metric in metrics]

    # Make request to the GA reporting API and return paginated results.
    def run_report_and_store(self, report_type, dimensions, metrics, user_segment, dimensions_filters=None, metrics_filters=None):
        """Queries the Analytics Reporting API V4 and stores the results in a datastore.

        Args:
          analytics: An authorized Analytics Reporting API V4 service object
          report_type: The type of data being requested
          dimensions: A list with the GA dimensions
          metrics: A list with the metrics
          dimensions_filters: A list with the GA dimensions filters
          metrics_filters: A list with the GA metrics filters
        """
        query_params = {
            'viewId': self.VIEW_ID,
            'dateRanges': [{'startDate': self.start_date, 'endDate': self.end_date}],
            'dimensions': self.format_dimensions(dimensions),
            'metrics': self.format_metrics(metrics),
            'pageSize': self.API_PAGE_SIZE,
        }

        # Segment filter so we get users in batches based on the starting digit
        # of their client id, ex: GA1, GA2, GA3 etc.
        query_params['dimensionFilterClauses'] = [
            {
                "filters": [
                    {
                        "dimensionName": "ga:dimension8",
                        "operator": "BEGINS_WITH",
                        "expressions": [user_segment]
                    }
                ]

            }
        ]

        # Extra filters for dimensions and metrics based on need.
        if dimensions_filters is not None:
            query_params['dimensionFilterClauses'].append(dimensions_filters)

        if metrics_filters is not None:
            query_params['metricFilterClauses'] = metrics_filters

        complete_responses_list = []
        reports_object = self.analytics.reports()
        page_token = None
        while True:
            sleep(0.1)
            if page_token:
                query_params['pageToken'] = page_token
            data_chunk = reports_object.batchGet(
                body={'reportRequests': [query_params]}).execute()
            if 'rows' not in data_chunk['reports'][0]['data']:
                break
            data_rows = []
            meta_dict = {}
            try:
                data_rows = data_chunk['reports'][0]['data']['rows']
                meta = data_chunk['reports'][0]['columnHeader']
                d_names_list = meta['dimensions']
                m_names_list = [m_meta_dict['name']
                                for m_meta_dict in meta['metricHeader']['metricHeaderEntries']]
                meta_dict = {'dimensions': d_names_list,
                             'metrics': m_names_list}
            except Exception as ex:
                print('BEGIN EXCEPTION')
                print(report_type)
                print(exc_info()[0])
                print(str(ex))
                print(dumps(data_chunk['reports'][0]))
                print('END EXCEPTION')
            partial_rl = [self.store.persist_dict_record(
                report_type, meta_dict, data_dict) for data_dict in data_rows]
            complete_responses_list.extend(partial_rl)
            page_token = data_chunk['reports'][0].get('nextPageToken')
            if not page_token:
                break

        # Wait for acks from Cassandra
        [cr['cassandra_future'].result() for cr in complete_responses_list]

        return complete_responses_list

    # Get user level data
    def store_users(self, user_segment):
        dimensions = ['dimension8', 'deviceCategory', 'browser', 'city', 'country']
        metrics = ['revenuePerUser', 'transactionsPerUser', 'sessions']

        return self.run_report_and_store('users', dimensions, metrics, user_segment)

    # Get user device branding data separately from general user data because
    # the google reporting API would only return mobile users if requested together
    # with the rest of the data.
    def store_users_mobile_brand(self, user_segment):
        dimensions = ['dimension8', 'mobileDeviceBranding']
        metrics = ['sessions']

        return self.run_report_and_store('users_mobile_brand', dimensions, metrics, user_segment)

    # Get session level data
    def store_sessions(self, user_segment):
        dimensions = ['dimension8', 'dimension2', 'searchUsed', 'daysSinceLastSession']
        metrics = ['sessionDuration', 'uniquePageviews', 'transactions', 'transactionRevenue',
                   'uniquePurchases', 'searchResultViews', 'searchUniques', 'searchDepth', 'searchRefinements']

        return self.run_report_and_store('sessions', dimensions, metrics, user_segment)

    # Get sessions shopping stages separately from general session data because shopping stages show up
    # as one per row and would cause alot of data duplication for all the other columns.
    def store_sessions_shopping_stages(self, user_segment):
        dimensions = ['dimension8', 'dimension2', 'shoppingStage']
        metrics = ['pageviews']

        # Apply a filter when retrieving shopping stages so that we only get shopping stages relevant to
        # our model.
        dimensions_filters = {
            "filters": [
                {
                    "dimensionName": "ga:shoppingStage",
                    "operator": "IN_LIST",
                    "expressions": ["ALL_VISITS", "PRODUCT_VIEW", "ADD_TO_CART", "CHECKOUT", "TRANSACTION"]
                }
            ]
        }

        return self.run_report_and_store('sessions_shopping_stages', dimensions, metrics, user_segment, dimensions_filters)

    # Get hit level data
    def store_hits(self, user_segment):
        dimensions = [
            'dimension8', 'dimension2', 'dateHourMinute'
        ]

        # Pageviews is not used as a feature in the model since it is covered by
        # unique pageviews in the session's request. We add it here so that
        # hits that only have pageview events get retrieved aswell.
        metrics = ['timeOnPage',  'pageviews']

        return self.run_report_and_store('hits', dimensions, metrics, user_segment)

    def store_product_info(self, user_segment):
        dimensions = [
            'dimension8',
            'dimension2',
            'dateHourMinute',
            'productName'
        ]
        
        metrics = [
            'quantityAddedToCart', 
            'productAddsToCart',
            'productCheckouts',
            'itemQuantity',
            'itemRevenue',
            'productDetailViews',
            'cartToDetailRate'
        ]
        
        return self.run_report_and_store('product_info', dimensions, metrics, user_segment)

    # def store_event_info(self, user_segment):
    #     dimensions = [
    #         'dimension8',
    #         'dimension2',
    #         'dateHourMinute',
    #         'eventAction',
    #         'eventCategory'
    #     ]

        
    #     metrics = ['sessionDuration']
        
    #     return self.run_report_and_store('event_info', dimensions, metrics, user_segment)

    def store_session_index(self, user_segment):
        dimensions = ['dimension8', 'dimension2', 'sessionCount']
        
        metrics = ['hits']
        
        return self.run_report_and_store('session_index', dimensions, metrics, user_segment)

    def run(self):
        self.authenticate()

        for i in range(1, 10):
            user_segment = 'GA' + str(i)

            # Add sleep() calls in between requests so as
            # to not exceed the requests/time quota and get temporarily blocked.
            sleep(1)
            self.store_users(user_segment)
            sleep(1)
            self.store_users_mobile_brand(user_segment)
            sleep(1)
            self.store_sessions(user_segment)
            sleep(1)
            self.store_sessions_shopping_stages(user_segment)
            sleep(1)
            self.store_hits(user_segment)
            sleep(1)
            self.store_product_info(user_segment)
            # sleep(1)
            # self.store_event_info(user_segment)
            sleep(1)
            self.store_session_index(user_segment)


def main():
    google_analytics = GoogleAnalytics()
    google_analytics.run()


if __name__ == '__main__':
    main()

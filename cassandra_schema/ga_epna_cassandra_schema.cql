CREATE KEYSPACE IF NOT EXISTS morphl WITH REPLICATION = {'class': 'SimpleStrategy', 'replication_factor': 1};


CREATE TABLE morphl.ga_epna_users (
  client_id text,
  day_of_data_capture date,
  json_meta text,
  json_data text,
  PRIMARY KEY ((client_id), day_of_data_capture)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

CREATE TABLE morph.ga_epna_users_extra (
  client_id text,
  day_of_data_capture date,
  json_meta text,
  json_data text,
  PRIMARY KEY ((client_id), day_of_data_capture)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

CREATE TABLE morphl.ga_epna_sessions (
  client_id text,
  day_of_data_capture date,
  session_id text,
  json_meta text,
  json_data text,
  PRIMARY KEY ((client_id), day_of_data_capture, session_id)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

CREATE TABLE morphl.ga_epna_extra_sessions (
    client_id text,
    day_of_data_capture date,
    session_id text,
    shopping_stage text,
    json_data text,
    json_meta text,
    PRIMARY KEY (client_id, day_of_data_capture, session_id, shopping_stage)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC, session_id ASC, shopping_stage ASC)

CREATE TABLE morphl.ga_epna_hits (
  client_id text,
  day_of_data_capture date,
  session_id text,
  hit_id text,
  json_meta text,
  json_data text,
  PRIMARY KEY ((client_id), day_of_data_capture, session_id, hit_id)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);


CREATE TABLE morphl.ga_epna_transactions (
    client_id text,
    day_of_data_capture date,
    session_id text,
    transaction_id text,
    json_data text,
    json_meta text,
    PRIMARY KEY ((client_id), day_of_data_capture, session_id, transaction_id)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC)


CREATE TABLE morphl.ga_epnau_features_raw (
    client_id text,
    day_of_data_capture date,
    device_category text,
    sessions double, 
    bounces double, 
    revenue_per_user double,
    transactions_per_user double,
    PRIMARY KEY ((client_id), day_of_data_capture)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC)


CREATE TABLE morphl.ga_epnas_features_raw(
  client_id text,
  day_of_data_capture date,
  session_id text, 
  search_used text,
  days_since_last_session double,
  session_duration double, 
  page_views double,
  unique_page_views double,
  transactions double, 
  transaction_revenue double,
  unique_purchases double,
  search_result_views double,
  search_uniques double, 
  search_depth double,
  search_refinements double,
  PRIMARY KEY ((client_id), day_of_data_capture, session_id)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC)


CREATE TABLE morphl.ga_epnah_features_raw(
  client_id text,
  day_of_data_capture date,
  session_id text,
  hit_id text, 
  user_type text,
  shopping_stage text,
  date_hour_minute text,
  time_on_page double,
  product_list_clicks double,
  product_list_views double,
  product_detail_views double,
  PRIMARY KEY ((client_id), day_of_data_capture, session_id, hit_id)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC)


CREATE TABLE morphl.ga_epnat_features_raw(
  client_id text,
  day_of_data_capture date,
  session_id text,
  transaction_id text,
  days_to_transaction double, 
  sessions_to_transaction double,
  PRIMARY KEY ((client_id), day_of_data_capture, session_id, transaction_id)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC)


CREATE TABLE morphl.ga_epna_preprocessed_features_basic(
  client_id text, #ClientId
  session_id text, #SessionId
  hit_id text, #HitId
  day_of_data_capture date, #Day of Data Capture
  sessions double, #Sessions
  revenue_per_user double, #Revenue per User
  transactions_per_user double, #Transactions per User
  bounces double, #Bounces
  date_hour_minute text, #Date Hour Minute
  product_detail_views double, #Product Detail Views
  transactions double, #Transactions
  product_list_clicks double, #Product List Clicks
  product_list_views double, #Product List Views
  shopping_stage text, #Shopping Stage
  time_on_page double, #Time on Page
  user_type text, #User Type
  device_category text, #Device Category
  days_since_last_session double, #Days Since Last Session
  page_views double, #Pageviews
  search_depth double, #Search Depth
  search_refinements double, #Search Refinements
  search_result_views double, #Result Pageviews
  search_uniques double, #Total Unique Searches
  search_used text,  #Site Search Status
  session_duration double, #Session Duration
  transaction_revenue double, #Revenue
  unique_page_views double, #Unique Pageviews
  unique_purchases double, #Unique Purchases
  days_to_transaction double, #Days to Transaction
  sessions_to_transaction double, #Sessions to Transaction
  PRIMARY KEY ((client_id), day_of_data_capture, session_id, hit_id)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC)


CREATE TABLE morphl.ga_epna_config_parameters (
  morphl_component_name text,
  parameter_name text,
  parameter_value text,
  PRIMARY KEY ((morphl_component_name, parameter_name))
);

INSERT INTO morphl.ga_epna_config_parameters (morphl_component_name,parameter_name,parameter_value)
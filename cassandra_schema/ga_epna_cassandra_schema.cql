CREATE KEYSPACE IF NOT EXISTS morphl WITH REPLICATION = {'class': 'SimpleStrategy', 'replication_factor': 1};


CREATE TABLE morphl.ga_epna_users (
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
  client_id text,
  session_id text,
  hit_id text,
  day_of_data_capture date,
  sessions double,
  revenue_per_user double, 
  transactions_per_user double,
  bounces double, 
  date_hour_minute text, 
  product_detail_views double, 
  transactions double, 
  product_list_clicks double, 
  product_list_views double, 
  shopping_stage text,
  time_on_page double, 
  user_type text,
  device_category text, 
  days_since_last_session double, 
  page_views double, 
  search_depth double, 
  search_refinements double,
  search_result_views double, 
  search_uniques double, 
  search_used text,
  session_duration double,
  transaction_revenue double,
  unique_page_views double,
  unique_purchases double,
  days_to_transaction double, 
  sessions_to_transaction double,
  PRIMARY KEY ((client_id), day_of_data_capture, session_id, hit_id)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC)


CREATE TABLE morphl.ga_epna_config_parameters (
  morphl_component_name text,
  parameter_name text,
  parameter_value text,
  PRIMARY KEY ((morphl_component_name, parameter_name))
);

INSERT INTO morphl.ga_epna_config_parameters (morphl_component_name,parameter_name,parameter_value)
CREATE KEYSPACE IF NOT EXISTS morphl WITH REPLICATION = {'class': 'SimpleStrategy', 'replication_factor': 1};


CREATE TABLE morphl.ga_epna_users (
  client_id text,
  day_of_data_capture date,
  json_meta text,
  json_data text,
  PRIMARY KEY ((client_id), day_of_data_capture)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

CREATE TABLE morphl.ga_epna_users_mobile_brand (
  client_id text,
  day_of_data_capture date,
  mobile_device_branding text,
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

CREATE TABLE morphl.ga_epna_sessions_shopping_stages (
    client_id text,
    day_of_data_capture date,
    session_id text,
    shopping_stage text,
    PRIMARY KEY ((client_id), day_of_data_capture, session_id, shopping_stage)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC, session_id ASC);

CREATE TABLE morphl.ga_epna_hits (
  client_id text,
  day_of_data_capture date,
  session_id text,
  hit_id text,
  json_meta text,
  json_data text,
  PRIMARY KEY ((client_id), day_of_data_capture, session_id, hit_id)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

CREATE TABLE morphl.ga_epnau_features_raw (
    client_id text,
    day_of_data_capture date,
    device_category text,
    browser text,
    sessions double, 
    bounces double, 
    revenue_per_user double,
    transactions_per_user double,
    PRIMARY KEY ((client_id), day_of_data_capture)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

CREATE TABLE morphl.ga_epnas_features_raw(
  client_id text,
  day_of_data_capture date,
  session_id text, 
  user_type text,
  search_used text,
  days_since_last_session double,
  session_duration double, 
  unique_page_views double,
  transactions double, 
  transaction_revenue double,
  unique_purchases double,
  search_result_views double,
  search_uniques double, 
  search_depth double,
  search_refinements double,
  PRIMARY KEY ((client_id), day_of_data_capture, session_id)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);


CREATE TABLE morphl.ga_epnah_features_raw(
  client_id text,
  day_of_data_capture date,
  session_id text,
  hit_id text, 
  date_hour_minute text,
  time_on_page double,
  product_list_clicks double,
  product_list_views double,
  product_detail_views double,
  PRIMARY KEY ((client_id), day_of_data_capture, session_id, hit_id)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

CREATE TABLE morphl.ga_epnau_features_filtered (
    client_id text,
    device_category text,
    browser text,
    sessions double, 
    bounces double, 
    revenue_per_user double,
    transactions_per_user double,
    mobile_device_branding text,
    PRIMARY KEY ((client_id))
);

CREATE TABLE morphl.ga_epnas_features_filtered(
  client_id text,
  session_id text, 
  search_used text,
  user_type text,
  days_since_last_session double,
  session_duration double, 
  unique_page_views double,
  transactions double, 
  transaction_revenue double,
  unique_purchases double,
  search_result_views double,
  search_uniques double, 
  search_depth double,
  search_refinements double,
  PRIMARY KEY ((client_id), session_id)
);


CREATE TABLE morphl.ga_epnah_features_filtered(
  client_id text,
  session_id text,
  hit_id text, 
  date_hour_minute text,
  time_on_page double,
  product_list_clicks double,
  product_list_views double,
  product_detail_views double,
  shopping_stage set<text>,
  PRIMARY KEY ((client_id),session_id, hit_id)
);

CREATE TABLE morphl.ga_epna_calculated_features(
  client_id text,
  session_id text,
  hit_id text, 
  device_category text,
  browser text,
  sessions double, 
  bounces double,
  revenue_per_user double, 
  device_revenue_per_transaction double,
  browser_revenue_per_transaction double,
  transactions_per_user double,
  mobile_device_branding text,
  search_used text,
  days_since_last_session double,
  session_duration double, 
  unique_page_views double, 
  transactions double, 
  transaction_revenue double, 
  unique_purchases double,
  search_result_views double,
  search_uniques double,
  search_depth double, 
  search_refinements double, 
  user_type text,
  date_hour_minute text,
  time_on_page double,
  product_list_clicks double,
  product_list_views double,
  product_detail_views double,
  shopping_stage text,
  PRIMARY KEY ((client_id),session_id,hit_id)
)

CREATE TABLE morphl.ga_epna_config_parameters (
  morphl_component_name text,
  parameter_name text,
  parameter_value text,
  PRIMARY KEY ((morphl_component_name, parameter_name))
);

INSERT INTO morphl.ga_epna_config_parameters (morphl_component_name,parameter_name,parameter_value)
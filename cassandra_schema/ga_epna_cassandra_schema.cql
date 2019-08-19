CREATE KEYSPACE IF NOT EXISTS morphl WITH REPLICATION = {'class': 'SimpleStrategy', 'replication_factor': 1};

DROP TABLE IF EXISTS morphl.ga_epna_users;

CREATE TABLE morphl.ga_epna_users (
  client_id text,
  day_of_data_capture date,
  json_meta text,
  json_data text,
  PRIMARY KEY ((client_id), day_of_data_capture)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

DROP TABLE IF EXISTS morphl.ga_epna_users_mobile_brand;

CREATE TABLE morphl.ga_epna_users_mobile_brand (
  client_id text,
  day_of_data_capture date,
  mobile_device_branding text,
  PRIMARY KEY ((client_id), day_of_data_capture)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

DROP TABLE IF EXISTS morphl.ga_epna_sessions;

CREATE TABLE morphl.ga_epna_sessions (
  client_id text,
  day_of_data_capture date,
  session_id text,
  json_meta text,
  json_data text,
  PRIMARY KEY ((client_id), day_of_data_capture, session_id)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

DROP TABLE IF EXISTS morphl.ga_epna_sessions_shopping_stages;

CREATE TABLE morphl.ga_epna_sessions_shopping_stages (
    client_id text,
    day_of_data_capture date,
    session_id text,
    shopping_stage text,
    PRIMARY KEY ((client_id), day_of_data_capture, session_id, shopping_stage)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC, session_id ASC);

DROP TABLE IF EXISTS morphl.ga_epna_hits;

CREATE TABLE morphl.ga_epna_hits (
  client_id text,
  day_of_data_capture date,
  session_id text,
  date_hour_minute text,
  json_meta text,
  json_data text,
  PRIMARY KEY ((client_id), day_of_data_capture, session_id, date_hour_minute)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

DROP TABLE IF EXISTS morphl.ga_epna_product_info;

CREATE TABLE morphl.ga_epna_product_info (
  client_id text,
  day_of_data_capture date, 
  session_id text, 
  product_name text, 
  date_hour_minute text, 
  json_meta text, 
  json_data text,
  PRIMARY KEY ((client_id), day_of_data_capture, session_id, product_name)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

-- DROP TABLE IF EXISTS morphl.ga_epna_event_info;

-- CREATE TABLE morphl.ga_epna_event_info (
--   client_id text,
--   day_of_data_capture date,
--   session_id text,
--   event_action text,
--   event_category text,
--   date_hour_minute text,
--   PRIMARY KEY((client_id), day_of_data_capture, session_id, event_action, event_category, date_hour_minute)
-- ) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

DROP TABLE IF EXISTS morphl.ga_epna_session_index;

CREATE TABLE morphl.ga_epna_session_index (
  client_id text,
  day_of_data_capture date,
  session_id text,
  session_index int, 
  PRIMARY KEY((client_id), day_of_data_capture, session_id)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

DROP TABLE IF EXISTS morphl.ga_epnau_features_raw;

CREATE TABLE morphl.ga_epnau_features_raw (
    client_id text,
    day_of_data_capture date,
    device_category text,
    browser text,
    revenue_per_user double,
    transactions_per_user double,
    PRIMARY KEY ((client_id), day_of_data_capture)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

DROP TABLE IF EXISTS morphl.ga_epnas_features_raw;

CREATE TABLE morphl.ga_epnas_features_raw (
  client_id text,
  day_of_data_capture date,
  session_id text, 
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
  search_used text,
  PRIMARY KEY ((client_id), day_of_data_capture, session_id)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

DROP TABLE IF EXISTS morphl.ga_epnah_features_raw;

CREATE TABLE morphl.ga_epnah_features_raw (
  client_id text,
  day_of_data_capture date,
  session_id text,
  date_hour_minute text,
  time_on_page double,
  PRIMARY KEY ((client_id), day_of_data_capture, session_id, date_hour_minute)
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

DROP TABLE IF EXISTS morphl.ga_epnap_features_raw;

CREATE TABLE morphl.ga_epnap_features_raw (
  client_id text, 
  day_of_data_capture date,
  session_id text, 
  date_hour_minute text,
  product_name text, 
  quantity_added_to_cart double, 
  product_adds_to_cart double,
  product_checkouts double, 
  item_quantity double,
  item_revenue double, 
  product_detail_views double, 
  cart_to_detail_rate double,
  PRIMARY KEY ((client_id), day_of_data_capture, session_id, date_hour_minute, product_name )
) WITH CLUSTERING ORDER BY (day_of_data_capture DESC);

DROP TABLE IF EXISTS morphl.ga_epnau_features_filtered;

CREATE TABLE morphl.ga_epnau_features_filtered (
    client_id text,
    device_category text,
    browser text,
    session_count int,
    revenue_per_user double,
    transactions_per_user double,
    mobile_device_branding text,
    PRIMARY KEY ((client_id))
);

DROP TABLE IF EXISTS morphl.ga_epnas_features_filtered;

CREATE TABLE morphl.ga_epnas_features_filtered (
  client_id text,
  session_id text, 
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
  search_used text,
  session_index int, 
  PRIMARY KEY ((client_id), session_id)
);

DROP TABLE IF EXISTS morphl.ga_epnah_features_filtered;

CREATE TABLE morphl.ga_epnah_features_filtered (
  client_id text,
  session_id text,
  date_hour_minute text,
  time_on_page double,
  product_detail_views double,
  cart_to_detail_rate double,
  item_quantity double, 
  item_revenue double, 
  product_adds_to_cart double, 
  product_checkouts double, 
  quantity_added_to_cart double,
  PRIMARY KEY ((client_id),session_id, date_hour_minute)
);

DROP TABLE IF EXISTS morphl.ga_epna_shopping_stages_filtered;

CREATE TABLE morphl.ga_epna_shopping_stages_filtered (
  client_id text, 
  session_id text,
  shopping_stage text,
  PRIMARY KEY((client_id), session_id)
);

DROP TABLE IF EXISTS morphl.ga_epna_batch_inference_data;

CREATE TABLE morphl.ga_epna_batch_inference_data (
  client_id text, 
  user_features frozen<list<double>>,
  sessions_features  frozen<list<frozen<list<double>>>>,
  hits_features frozen<list<frozen<list<frozen<list<double>>>>>>,
  sessions_hits_count frozen<list<int>>,
  shopping_stages frozen<list<frozen<list<double>>>>,
  PRIMARY KEY(client_id)
);

DROP TABLE IF EXISTS morphl.ga_epna_predictions;

CREATE TABLE morphl.ga_epna_predictions(
  client_id text, 
  prediction_date date,
  all_visits double, 
  product_view double, 
  add_to_cart double, 
  checkout_with_add_to_cart double, 
  checkout_without_add_to_cart double,
  transaction double,
  PRIMARY KEY((client_id))
);

DROP TABLE IF EXISTS morphl.ga_epna_predictions_access_logs;

CREATE TABLE morphl.ga_epna_predictions_access_logs (
  client_id text,
  tstamp timestamp,
  all_visits double, 
  product_view double, 
  add_to_cart double, 
  checkout_with_add_to_cart double, 
  checkout_without_add_to_cart double,
  transaction double,
  PRIMARY KEY ((client_id), tstamp)
) WITH CLUSTERING ORDER BY (tstamp DESC);

DROP TABLE IF EXISTS morphl.ga_epna_predictions_statistics;

CREATE TABLE morphl.ga_epna_predictions_statistics(
  prediction_date date,
  total_predictions int,
  all_visits int, 
  product_view int, 
  add_to_cart int, 
  checkout_with_add_to_cart int, 
  checkout_without_add_to_cart int, 
  transaction int,
  PRIMARY KEY((prediction_date))
);

DROP TABLE IF EXISTS morphl.ga_epna_config_parameters;

CREATE TABLE morphl.ga_epna_config_parameters (
  morphl_component_name text,
  parameter_name text,
  parameter_value text,
  PRIMARY KEY ((morphl_component_name, parameter_name))
);

INSERT INTO morphl.ga_epna_config_parameters (morphl_component_name,parameter_name,parameter_value)
VALUES ('ga_epna','days_training_interval','60');

INSERT INTO morphl.ga_epna_config_parameters (morphl_component_name,parameter_name,parameter_value)
VALUES ('ga_epna','days_prediction_interval','60');
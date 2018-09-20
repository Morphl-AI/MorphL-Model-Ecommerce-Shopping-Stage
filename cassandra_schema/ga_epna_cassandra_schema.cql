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


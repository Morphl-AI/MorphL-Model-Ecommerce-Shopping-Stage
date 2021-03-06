HDFS_PORT=9000

HDFS_DIR_USER_FILTERED=hdfs://${MORPHL_SERVER_IP_ADDRESS}:${HDFS_PORT}/${PREDICTION_DAY_AS_STR}_${UNIQUE_HASH}_ga_epnau_filtered
HDFS_DIR_SESSION_FILTERED=hdfs://${MORPHL_SERVER_IP_ADDRESS}:${HDFS_PORT}/${PREDICTION_DAY_AS_STR}_${UNIQUE_HASH}_ga_epnas_filtered
HDFS_DIR_HIT_FILTERED=hdfs://${MORPHL_SERVER_IP_ADDRESS}:${HDFS_PORT}/${PREDICTION_DAY_AS_STR}_${UNIQUE_HASH}_ga_epnah_filtered
HDFS_DIR_STAGES_FILTERED=hdfs://${MORPHL_SERVER_IP_ADDRESS}:${HDFS_PORT}/${PREDICTION_DAY_AS_STR}_${UNIQUE_HASH}_ga_epna_shopping_stages_filtered

hdfs dfs -rm ${HDFS_DIR_USER_FILTERED}/*
hdfs dfs -rmdir ${HDFS_DIR_USER_FILTERED}

hdfs dfs -rm ${HDFS_DIR_SESSION_FILTERED}/*
hdfs dfs -rmdir ${HDFS_DIR_SESSION_FILTERED}

hdfs dfs -rm ${HDFS_DIR_HIT_FILTERED}/*
hdfs dfs -rmdir ${HDFS_DIR_HIT_FILTERED}

hdfs dfs -rm ${HDFS_DIR_STAGES_FILTERED}/*
hdfs dfs -rmdir ${HDFS_DIR_STAGES_FILTERED}

exit 0
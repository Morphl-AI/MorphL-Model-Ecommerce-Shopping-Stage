HDFS_PORT=9000

remove_hdfs() {
    hdfs dfs -test -e $1

    if [ $? -eq 0 ]; then
        hdfs dfs -rm $1
        hdfs dfs -rmdir $1
    fi
}

HDFS_PATHS=(hdfs://${MORPHL_SERVER_IP_ADDRESS}:${HDFS_PORT}/${PREVIOUS_PREDICTION_DAY_AS_STR}_${UNIQUE_HASH}_ga_epnau_filtered)
HDFS_PATHS+=(hdfs://${MORPHL_SERVER_IP_ADDRESS}:${HDFS_PORT}/${PREVIOUS_PREDICTION_DAY_AS_STR}_${UNIQUE_HASH}_ga_epnas_filtered)
HDFS_PATHS+=(hdfs://${MORPHL_SERVER_IP_ADDRESS}:${HDFS_PORT}/${PREVIOUS_PREDICTION_DAY_AS_STR}_${UNIQUE_HASH}_ga_epnah_filtered)
HDFS_PATHS+=(hdfs://${MORPHL_SERVER_IP_ADDRESS}:${HDFS_PORT}/${CURRENT_PREDICTION_DAY_AS_STR}_${UNIQUE_HASH}_ga_epnau_filtered)
HDFS_PATHS+=(hdfs://${MORPHL_SERVER_IP_ADDRESS}:${HDFS_PORT}/${CURRENT_PREDICTION_DAY_AS_STR}_${UNIQUE_HASH}_ga_epnas_filtered)
HDFS_PATHS+=(hdfs://${MORPHL_SERVER_IP_ADDRESS}:${HDFS_PORT}/${CURRENT_PREDICTION_DAY_AS_STR}_${UNIQUE_HASH}_ga_epnah_filtered)
    
for HDFS_PATH in ${HDFS_PATHS[@]}; do
    remove_hdfs $HDFS_PATH
done

exit 0
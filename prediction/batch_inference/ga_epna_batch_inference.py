from os import getenv
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import Window, functions as f

import numpy as np
import torch as tr
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict

MORPHL_SERVER_IP_ADDRESS = getenv('MORPHL_SERVER_IP_ADDRESS')
MORPHL_CASSANDRA_USERNAME = getenv('MORPHL_CASSANDRA_USERNAME')
MORPHL_CASSANDRA_PASSWORD = getenv('MORPHL_CASSANDRA_PASSWORD')
MORPHL_CASSANDRA_KEYSPACE = getenv('MORPHL_CASSANDRA_KEYSPACE')

PREDICTION_DAY_AS_STR = getenv('PREDICTION_DAY_AS_STR')

MASTER_URL = 'local[*]'
APPLICATION_NAME = 'batch-inference'


device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")


class ModelLSTM_V1(nn.Module):
    def __init__(self, inputShape, outputShape, hyperParameters={}, **kwargs):
        super().__init__(**kwargs)

        assert type(hyperParameters) == dict

        self.hyperParameters = hyperParameters

        self.appendPreviousOutput = hyperParameters["appendPreviousOutput"]
        self.outputType = hyperParameters["outputType"]
        baseNeurons = hyperParameters["baseNeurons"]
        self.hidenShape2 = baseNeurons + int(inputShape[1]) + outputShape \
            if self.appendPreviousOutput else baseNeurons + int(inputShape[1])

        self.lstm1 = nn.LSTM(input_size=int(
            inputShape[2]), hidden_size=baseNeurons, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=self.hidenShape2,
                             hidden_size=baseNeurons, num_layers=1)
        self.fc1 = nn.Linear(in_features=baseNeurons +
                             int(inputShape[0]), out_features=baseNeurons)
        self.fc2 = nn.Linear(in_features=baseNeurons, out_features=outputShape)

    def doLoadWeights(self, loadedState):
        if not "weights" in loadedState and "params" in loadedState:
            print(
                "Warning: Depcrecated model, using \"params\" key instead of \"weights\".")
            loadedState["weights"] = loadedState["params"]

        assert "weights" in loadedState
        params = loadedState["weights"]
        loadedParams, _ = self.getNumParams(params)
        trainableParams = self.getTrainableParameters()
        thisParams, _ = self.getNumParams(trainableParams)
        if loadedParams != thisParams:
            raise Exception("Inconsistent parameters: %d vs %d." %
                            (loadedParams, thisParams))

        for i, item in enumerate(trainableParams):
            with tr.no_grad():
                item[:] = params[i][:].to(device)
            item.requires_grad_(True)
        print("Succesfully loaded weights (%d parameters) " % (loadedParams))

    def loadModel(self, path, stateKeys):
        assert len(stateKeys) > 0
        try:
            loadedState = tr.load(path)
        except Exception:
            print("Exception raised while loading model with tr.load(). Forcing CPU load")
            loadedState = tr.load(
                path, map_location=lambda storage, loc: storage)

        print("Loading model from %s" % (path))
        if not "model_state" in loadedState:
            print(
                "Warning, no model state dictionary for this model (obsolete behaviour). Ignoring.")
            loadedState["model_state"] = None

        if not self.onModelLoad(loadedState["model_state"]):
            loaded = loadedState["model_state"]
            current = self.onModelSave()
            raise Exception(
                "Could not correclty load the model state loaded: %s vs. current: %s" % (loaded, current))

        self.doLoadWeights(loadedState)

        print("Finished loading model")

    def getNpData(self, results):
        npResults = None
        if results is None:
            return None

        if type(results) in (list, tuple):
            npResults = []
            for result in results:
                npResult = self.getNpData(result)
                npResults.append(npResult)
        elif type(results) in (dict, OrderedDict):
            npResults = {}
            for key in results:
                npResults[key] = self.getNpData(results[key])

        elif type(results) == tr.Tensor:
            npResults = results.detach().to('cpu').numpy()
        else:
            assert False, "Got type %s" % (type(results))
        return npResults

    def getTrData(self, data):
        trData = None
        if data is None:
            return None

        elif type(data) in (list, tuple):
            trData = []
            for item in data:
                trItem = self.getTrData(item)
                trData.append(trItem)
        elif type(data) in (dict, OrderedDict):
            trData = {}
            for key in data:
                trData[key] = self.getTrData(data[key])
        elif type(data) is np.ndarray:
            trData = tr.from_numpy(data).to(device)
        elif type(data) is tr.Tensor:
            trData = data.to(device)
        return trData

    def getNumParams(self, params):
        numParams, numTrainable = 0, 0
        for param in params:
            npParamCount = np.prod(param.data.shape)
            numParams += npParamCount
            if param.requires_grad:
                numTrainable += npParamCount
        return numParams, numTrainable

    def npForward(self, x):
        trInput = self.getTrData(x)
        trResult = self.forward(trInput)
        npResult = self.getNpData(trResult)
        return npResult

    def onModelSave(self):
        return self.hyperParameters

    def onModelLoad(self, state):
        if len(self.hyperParameters.keys()) != len(state.keys()):
            return False

        for key in state:
            if not key in self.hyperParameters:
                return False

            if not state[key] == self.hyperParameters[key]:
                return False

        return True

    def getTrainableParameters(self):
        return list(filter(lambda p: p.requires_grad, self.parameters()))

    def loadWeights(self, path):
        self.loadModel(path, stateKeys=["weights", "model_state"])

    def forward(self, trInputs):
        # print(["%s=>%s" % (x, trInputs[x].shape) for x in trInputs])
        hiddens = self.computeHiddens(trInputs)
        # print(hiddens.shape)

        # Append the features of previous session
        X = trInputs["dataSessions"].transpose(0, 1).float()
        X[1:] = X[0: -1]
        X[0] *= 0
        hiddens = tr.cat([X, hiddens], dim=-1)
        # print(hiddens.shape)

        # If using previous shopping stage as part of the hidden state, then we need to append it to the hidden state
        #  vector. However, we need to append it accordingly (i.e. the output of session 0 to the hidden state of 1,
        #  output of session 2 to the input state of 1 etc. The first session has a zeroed entry (no previous session).
        if self.appendPreviousOutput:
            X = trInputs["dataShoppingStage"].transpose(0, 1).float()
            X[1:] = X[0: -1]
            X[0] *= 0
            hiddens = tr.cat([X, hiddens], dim=-1)
        # print(hiddens.shape)

        # [0] is the hidden state.
        sess_hidden = self.lstm2(hiddens, None)[0]
        # print(sess_hidden.shape)

        # Append user features
        X = trInputs["dataUsers"].float()
        Y = tr.ones(sess_hidden.shape[0], *
                    X.shape).to(device).requires_grad_(False)
        Y = Y * X
        sess_hidden = tr.cat([Y, sess_hidden], dim=-1)
        # print(sess_hidden.shape)

        sess_hidden = sess_hidden.transpose(0, 1)
        y1 = F.relu(self.fc1(sess_hidden))
        y2 = self.fc2(y1)

        if self.outputType == "classification":
            y3 = F.softmax(y2, dim=-1)
        else:
            y3 = tr.sigmoid(y2)
        return y3

    def computeHiddens(self, trInputs):
        trData, trNums = trInputs["dataHits"], trInputs["dataNumItems"]
        hiddens = []
        numSessions = trData.shape[1]
        MB = trData.shape[0]

        # Session by session
        for t_sessions in range(numSessions):
            trSessData = trData[:, t_sessions]
            trSessNums = trNums[:, t_sessions]
            trSessData = tr.transpose(trSessData, 0, 1)
            _, prevHidden = self.lstm1(trSessData[0: 1], None)
            mask = tr.ones(prevHidden[0].shape).to(
                device).requires_grad_(False)
            N = tr.max(trSessNums)
            # Hit by hit
            for t_hits in range(1, N):
                trSessNums -= 1
                item = trSessData[t_hits: t_hits + 1]
                _, hidden = self.lstm1(item, prevHidden)
                hereMask = mask * \
                    (trSessNums > 0).unsqueeze(0).unsqueeze(-1).float()
                prevHidden = (prevHidden[0] * (1 - hereMask) + hidden[0] * hereMask,
                              prevHidden[1] * (1 - hereMask) + hidden[1] * hereMask)
            hiddens.append(prevHidden[0])
        hiddens = tr.cat(hiddens, dim=0)
        return hiddens


# Return a spark dataframe from a specified Cassandra table.
def fetch_from_cassandra(c_table_name, spark_session):

    load_options = {
        'keyspace': MORPHL_CASSANDRA_KEYSPACE,
        'table': c_table_name,
        'spark.cassandra.input.fetch.size_in_rows': '150'}

    df = (spark_session.read.format('org.apache.spark.sql.cassandra')
                            .options(**load_options)
                            .load())

    return df


def main():
    spark_session = (
        SparkSession.builder
        .appName(APPLICATION_NAME)
        .master(MASTER_URL)
        .config('spark.cassandra.connection.host', MORPHL_SERVER_IP_ADDRESS)
        .config('spark.cassandra.auth.username', MORPHL_CASSANDRA_USERNAME)
        .config('spark.cassandra.auth.password', MORPHL_CASSANDRA_PASSWORD)
        .config('spark.sql.shuffle.partitions', 16)
        .getOrCreate()
    )

    log4j = spark_session.sparkContext._jvm.org.apache.log4j
    log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)

    users_ingested = fetch_from_cassandra(
        'ga_epnau_features_raw', spark_session)

    current_day_ids = users_ingested.select('client_id').where(
        "day_of_data_capture = '{}'".format(PREDICTION_DAY_AS_STR))

    sessions = fetch_from_cassandra(
        'ga_epna_data_sessions', spark_session).join(current_day_ids, 'client_id', 'inner')
    hits = fetch_from_cassandra(
        'ga_epna_data_hits', spark_session).join(current_day_ids, 'client_id', 'inner')
    users = fetch_from_cassandra(
        'ga_epna_data_users', spark_session).join(current_day_ids, 'client_id', 'inner')
    num_items = fetch_from_cassandra(
        'ga_epna_data_num_hits', spark_session).join(current_day_ids, 'client_id', 'inner')
    shopping_stage = fetch_from_cassandra(
        'ga_epna_data_shopping_stages', spark_session).join(current_day_ids, 'client_id', 'inner')

    model = ModelLSTM_V1(inputShape=(9, 12, 2), outputShape=6, hyperParameters={"randomizeSessionSize": True,
                                                                                "appendPreviousOutput": True,
                                                                                "baseNeurons": 30,
                                                                                "outputType": "regression",
                                                                                'normalization': 'min_max',
                                                                                'inShape': (9, 12, 2),
                                                                                "attributionModeling": "linear"})
    model.loadWeights('/opt/models/ga_epna_model_weights.pkl')

    for session_count in range(1, 40):
        segment_range = range(10, 95, 5) if session_count < 5 else range(10, 100, 10)
        
        for segment_limit in segment_range:
            
            lower_limit = 'GA' + str(segment_limit)
            upper_limit = 'GA' + str(segment_limit + 5) if session_count < 5 else str(segment_limit + 10)

            condition_string = "session_count = {} and user_segment >= '{}' and user_segment < '{}'".format(
                session_count, lower_limit, upper_limit)

            if segment_limit == 90:

                condition_string = "session_count = {} and user_segment >= 'GA90'".format(
                    session_count)

            filtered_users_df = users.filter(condition_string)

            if len(filtered_users_df.head(1)) == 0:
                continue

            order_df = (filtered_users_df.
                        select('client_id').
                        withColumn(
                            'index',
                            f.row_number().over(Window.orderBy('client_id')) - 1
                        )
                        )

            users_array = np.array(filtered_users_df.orderBy('client_id').select(
                'features').rdd.flatMap(lambda x: x).collect()).astype(np.float32)

            sessions_array = np.array(sessions.filter(condition_string).orderBy('client_id').select(
                'features').rdd.flatMap(lambda x: x).collect()).astype(np.float32)

            hits_array = np.array(hits.filter(condition_string).orderBy('client_id').select(
                'features').rdd.flatMap(lambda x: x).collect()).astype(np.float32)

            num_items_array = np.array(num_items.filter(condition_string).orderBy(
                'client_id').select('sessions_hits_count').rdd.flatMap(lambda x: x).collect())

            shopping_stage_array = np.array(shopping_stage.filter(condition_string).orderBy(
                'client_id').select('shopping_stages').rdd.flatMap(lambda x: x).collect()).astype(np.float32)

            result = model.npForward({"dataSessions": sessions_array,
                                      "dataHits": hits_array,
                                      "dataUsers": users_array,
                                      "dataNumItems": num_items_array,
                                      "dataShoppingStage": shopping_stage_array})

            result = np.delete(
                result, np.s_[:session_count - 1], axis=1).reshape(result.shape[0], 6)

            to_int_udf = f.udf(lambda x: int(x), 'int')

            result_df = (
                spark_session.
                sparkContext.
                parallelize(result).
                zipWithIndex().
                map(lambda x: [float(item) for item in x[0]] + [float(x[1])]).
                toDF([
                    'all_visits',
                    'product_view',
                    'checkout_with_add_to_cart',
                    'transaction',
                    'add_to_cart',
                    'checkout_without_add_to_cart',
                    'index'
                ]).
                withColumn('index', to_int_udf('index')).
                join(order_df, 'index', 'inner').
                drop('index').
                repartition(32)
            )

            save_options_ga_epna_predictions = {
                'keyspace': MORPHL_CASSANDRA_KEYSPACE,
                'table': ('ga_epna_predictions')
            }

            (result_df.
             write.
             format('org.apache.spark.sql.cassandra').
             mode('append').
             options(**save_options_ga_epna_predictions).
             save()
             )


if __name__ == '__main__':
    main()

from os import getenv
import numpy as np

from pyspark.sql import SparkSession, Window, functions as f

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

import numpy as np
import torch as tr
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict

MORPHL_SERVER_IP_ADDRESS = getenv('MORPHL_SERVER_IP_ADDRESS')
MORPHL_CASSANDRA_USERNAME = getenv('MORPHL_CASSANDRA_USERNAME')
MORPHL_CASSANDRA_PASSWORD = getenv('MORPHL_CASSANDRA_PASSWORD')
MORPHL_CASSANDRA_KEYSPACE = getenv('MORPHL_CASSANDRA_KEYSPACE')

CASS_REQ_TIMEOUT = 3600.0

PREDICTION_DAY_AS_STR = getenv('PREDICTION_DAY_AS_STR')

MASTER_URL = 'local[*]'
APPLICATION_NAME = 'batch-inference'


device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")

# Model class used for predictions.


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


def insert_statistics(statistics):
    auth_provider = PlainTextAuthProvider(
        username=MORPHL_CASSANDRA_USERNAME,
        password=MORPHL_CASSANDRA_PASSWORD
    )

    cluster = Cluster(
        [MORPHL_SERVER_IP_ADDRESS], auth_provider=auth_provider)

    spark_session_cass = cluster.connect(MORPHL_CASSANDRA_KEYSPACE)

    insert_sql_parts = [
        'INSERT INTO ga_epna_predictions_statistics ',
        '(prediction_date, total_predictions, all_visits, product_view, checkout_with_add_to_cart,',
        'transaction, add_to_cart, checkout_without_add_to_cart)'
        'VALUES (?,?,?,?,?,?,?,?)',
    ]

    insert_sql = ' '.join(insert_sql_parts)

    prep_stmt_statistics = spark_session_cass.prepare(insert_sql)

    bind_list = [
        PREDICTION_DAY_AS_STR,
        statistics['total_predictions'],
        statistics['all_visits'],
        statistics['product_view'],
        statistics['checkout_with_add_to_cart'],
        statistics['transaction'],
        statistics['add_to_cart'],
        statistics['checkout_without_add_to_cart']
    ]

    spark_session_cass.execute(
        prep_stmt_statistics, bind_list, timeout=CASS_REQ_TIMEOUT)


def get_predictions(row):

    sessions_array = np.array([row.sessions_features]).astype(np.float32)
    hits_array = np.array([row.hits_features]).astype(np.float32)
    user_array = np.array([row.user_features]).astype(np.float32)
    sessions_hits_count_array = np.array([row.sessions_hits_count])
    shopping_stages = np.array([row.shopping_stages]).astype(np.float32)

    result = model.npForward({"dataSessions": sessions_array,
                              "dataHits": hits_array,
                              "dataUsers": user_array,
                              "dataNumItems": sessions_hits_count_array,
                              "dataShoppingStage": shopping_stages})

    result = np.delete(
        result, np.s_[:row.session_count - 1], axis=1).reshape(result.shape[0], 6)
    result = [float(i) for i in result.tolist()[0]]

    return (row.client_id, row.user_segment, row.session_count, result[0], result[1], result[2], result[3], result[4], result[5] ,PREDICTION_DAY_AS_STR)





model = ModelLSTM_V1(inputShape=(10, 12, 8), outputShape=6, hyperParameters={"randomizeSessionSize": True,
                                                                             "appendPreviousOutput": True,
                                                                             "baseNeurons": 30,
                                                                             "outputType": "regression",
                                                                             'normalization': 'min_max',
                                                                             'inShape': (10, 12, 8),
                                                                             "attributionModeling": "linear"})
# Load the model weights.
model.loadWeights('/opt/models/ga_epna_model_weights.pkl')

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

    # Get the ids of users from the current day of predictions.
    current_day_ids = users_ingested.select('client_id').where(
        "day_of_data_capture = '{}'".format(PREDICTION_DAY_AS_STR))

    batch_inference_data = fetch_from_cassandra('ga_epna_batch_inference_data', spark_session).join(
        current_day_ids, 'client_id', 'inner')

    ga_epna_predictions = (
        batch_inference_data.
        rdd.
        map(get_predictions).
        repartition(32)
        .toDF([
            'client_id',
            'user_segment',
            'session_count',
            'all_visits',
            'product_view',
            'add_to_cart',
            'checkout_with_add_to_cart',
            'checkout_without_add_to_cart',
            'transaction',
            'prediction_date',
        ])
    )

    ga_epna_predictions.cache()

    statistics = {
        'total_predictions': ga_epna_predictions.count(),
        'all_visits': ga_epna_predictions.where("all_visits > 0.5").count(),
        'product_view': ga_epna_predictions.where("product_view > 0.5").count(),
        'add_to_cart': ga_epna_predictions.where("add_to_cart > 0.5").count(),
        'checkout_with_add_to_cart': ga_epna_predictions.where("checkout_with_add_to_cart > 0.5").count(),
        'checkout_without_add_to_cart': ga_epna_predictions.where("checkout_without_add_to_cart > 0.5").count(),
        'transaction': ga_epna_predictions.where("transaction > 0.5").count(),
    }

    insert_statistics(statistics)

    save_options_ga_epna_predictions = {
        'keyspace': MORPHL_CASSANDRA_KEYSPACE,
        'table': ('ga_epna_predictions')
    }

    (ga_epna_predictions.
     drop('user_segment').
     write.
     forma('org.apache.spark.sql.cassandra').
     mode('append').
     options(**save_options_ga_epna_predictions).
     save()
     )


if __name__ == '__main__':
    main()

from os import getenv

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
class ModelTargetabilityRNN(nn.Module):
    def __init__(self, hyperParameters={}, **kwargs):
        super().__init__(**kwargs)

        assert type(hyperParameters) == dict

        self.hyperParameters = hyperParameters
        
        self.inShape = hyperParameters["inShape"]
        self.outShape = hyperParameters["outShape"]
        
        self.hiddenShape1 = hyperParameters["hiddenShape1"]
        self.inShape2 = self.hiddenShape1 + int(self.inShape[1])
        self.hiddenShape2 = hyperParameters["hiddenShape2"]

        self.lstm1 = nn.LSTM(input_size=int(self.inShape[2]), hidden_size=self.hiddenShape1, num_layers=1)
        self.lstm2 = nn.LSTM(input_size=self.inShape2,
                             hidden_size=self.hiddenShape2, num_layers=1)
        
        self.fc1 = nn.Linear(in_features=self.hiddenShape2 + int(self.inShape[0]), out_features=self.hiddenShape2)
        
        self.fc2 = nn.Linear(in_features=self.hiddenShape2, out_features=self.outShape)

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

        hiddens = tr.cat([X, hiddens], dim=-1)
        # print(hiddens.shape)

        # [0] is the hidden state.
        sess_hidden = self.lstm2(hiddens, None)[0][-1]
        # print(sess_hidden.shape)

        # Append user features
        X = trInputs["dataUsers"].float()
        user_sess_hidden = tr.cat([X, sess_hidden], dim=-1)

        y1 = F.relu(self.fc1(user_sess_hidden))
        y2 = self.fc2(y1)
        y3 = F.softmax(y2, dim=1)

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
        '(prediction_date, total_predictions, targetable, untargetable)'
        'VALUES (?,?,?,?)',
    ]

    insert_sql = ' '.join(insert_sql_parts)

    prep_stmt_statistics = spark_session_cass.prepare(insert_sql)

    bind_list = [
        PREDICTION_DAY_AS_STR,
        statistics['total_predictions'],
        statistics['targetable'],
        statistics['untargetable'],
    ]

    spark_session_cass.execute(
        prep_stmt_statistics, bind_list, timeout=CASS_REQ_TIMEOUT)

# Map function that makes a prediction for each user
def get_predictions(row):

    # Get all the relevant numpy arrays
    sessions_array = np.array([row.sessions_features]).astype(np.float32)
    hits_array = np.array([row.hits_features]).astype(np.float32)
    user_array = np.array([row.user_features]).astype(np.float32)
    sessions_hits_count_array = np.array([row.sessions_hits_count])

    # Input the numpy arrays into the modle
    result = model.npForward({"dataSessions": sessions_array,
                              "dataHits": hits_array,
                              "dataUsers": user_array,
                              "dataNumItems": sessions_hits_count_array,
                              })

    result = [float(i) for i in result[0].tolist()]

    # Return the new row to the dataframe
    return (row.client_id, result[0], result[1], PREDICTION_DAY_AS_STR)


# Load the model
model = ModelTargetabilityRNN(
    hyperParameters = {"randomizeSessionSize" : True, "hiddenShape1" : 20, "hiddenShape2" : 30, \
        "inShape" : (14, 9, 2), "outShape" : 2, "labelColumnName" : "Next Shopping Stage Binary Targetable"
    }
)
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

    # Load the batch inference data from Cassandra and filter it by the client ids from the prediction date
    batch_inference_data = fetch_from_cassandra('ga_epna_batch_inference_data', spark_session).join(
        current_day_ids, 'client_id', 'inner')

    # Convert the dataframe to an rdd so we can apply the mapping function to it
    ga_epna_predictions = (
        batch_inference_data.
        rdd.
        map(get_predictions).
        repartition(32)
        .toDF([
            'client_id',
            'targetable',
            'untargetable',
            'prediction_date',
        ])
    )

    ga_epna_predictions.cache()


    statistics = {
        'total_predictions': ga_epna_predictions.count(),
        'targetable': ga_epna_predictions.where("targetable > 0.5").count(),
        'untargetable': ga_epna_predictions.where("untargetable > 0.5").count(),
    }

    # Save the statistics to Cassandra
    insert_statistics(statistics)

    # Save the predictions to Cassandra
    save_options_ga_epna_predictions = {
        'keyspace': MORPHL_CASSANDRA_KEYSPACE,
        'table': ('ga_epna_predictions')
    }

    (ga_epna_predictions.
     write.
     format('org.apache.spark.sql.cassandra').
     mode('append').
     options(**save_options_ga_epna_predictions).
     save()
     )


if __name__ == '__main__':
    main()
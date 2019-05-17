import numpy as np
import torch as tr
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict


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
            if item.shape != params[i].shape:
                raise Exception("Inconsistent parameters: %d vs %d." %
                                (item.shape, params[i].shape))
            with tr.no_grad():
                item[:] = self.maybeCuda(params[i][:])
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
            npResults = self.maybeCpu(results.detach()).numpy()
        else:
            assert False, "Got type %s" % (type(results))
        return npResults

    def maybeCpu(self, x):
        return x.cpu() if tr.cuda.is_available() and hasattr(x, "cpu") else x

    def maybeCuda(self, x):
        return x.cuda() if tr.cuda.is_available() and hasattr(x, "cuda") else x

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
            trData = self.maybeCuda(tr.from_numpy(data))
        elif type(data) is tr.Tensor:
            trData = self.maybeCuda(data)
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

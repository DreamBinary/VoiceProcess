import os
import pickle
import numpy as np
import scipy.io.wavfile as wvf
from python_speech_features import mfcc
from hmmlearn.hmm import GMMHMM
import heapq
import logging

logging.getLogger("hmmlearn").setLevel("CRITICAL")

data_path = "../../data/data4"
model_path = "hmm_gmm_model.pkl"


def wav2mfcc(label, data_path):
    trng_data = {}
    mfccs = []
    rate, sig = wvf.read(data_path)
    mfcc_feat = mfcc(sig, rate)
    mfccs.append(mfcc_feat)
    trng_data[label] = mfccs
    return trng_data


def obtain_config(label):
    conf = {}
    conf[label] = {}
    conf[label]["n_components"] = 2
    conf[label]["n_mix"] = 2
    return conf


def get_hmm_gmm(label, trng_data=None, GMM_config=None, model_path="hmm_gmm_model.pkl", from_file=False):
    hmm_gmm = {}
    if not from_file:
        hmm_gmm[label] = GMMHMM(
            n_components=GMM_config[label]["n_components"],
            n_mix=GMM_config[label]["n_mix"])
        if trng_data[label]:
            hmm_gmm[label].fit(np.vstack(trng_data[label]))
        pickle.dump(hmm_gmm, open(model_path, "wb"))
    else:
        hmm_gmm = pickle.load(open(model_path, "rb"))
    return hmm_gmm


def train(data_path, model_path):
    with open(os.path.join(data_path, "label.txt")) as f:
        label = f.readline()
    data_path = os.path.join(data_path, "train.wav")
    train_data = wav2mfcc(label, data_path)
    GMM_config = obtain_config(label)
    hmm_gmm = get_hmm_gmm(label, train_data, GMM_config, model_path)
    return hmm_gmm


def test_file(test_file, hmm_gmm):
    rate, sig = wvf.read(test_file)
    mfcc_feat = mfcc(sig, rate)
    pred = {}
    for model in hmm_gmm:
        pred[model] = hmm_gmm[model].score(mfcc_feat)
    return get_nbest(pred, 2), pred


def get_nbest(d, n):
    return heapq.nlargest(n, d, key=lambda k: d[k])


def predict_label(file, hmm_gmm):
    predicted = test_file(file, hmm_gmm)
    return predicted


logging.getLogger("hmmlearn").setLevel("CRITICAL")
wave_path = os.path.join(data_path, "train.wav")
hmm_gmm = train(data_path, model_path)
predicted, probs = predict_label(wave_path, hmm_gmm)
print("PREDICTIED: %s" % predicted[0])

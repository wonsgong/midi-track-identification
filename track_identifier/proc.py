import os
import numpy as np
# from sklearn.externals import joblib 
import joblib
import pypianoroll

from track_identifier.utils import features
from track_identifier.utils.misc import traverse_dir

from miditoolkit.midi import parser

cur_path = os.path.dirname(os.path.realpath(__file__))
PATH_MODEL = os.path.join(cur_path, 'model/2022-6-10.pkl')

def testing(X):
    # load model
    loaded_model = joblib.load(PATH_MODEL)
    
    # prediction
    predict_loaded_y = loaded_model.predict(X)
    return list(predict_loaded_y)

def identify_single_track(pianoroll):
    X = features.extract_features(pianoroll)
    test_x = np.array([X])
    return testing(test_x)[0]

def identify_multiple_track(pianorolls):
    # extract features
    num = len(pianorolls)
    test_x = []
    for idx in range(num): 
        X = features.extract_features(pianorolls[idx])
        test_x.append(X)
    test_x = np.array(test_x)
    return testing(test_x)
   
def identify_song(input_obj):
    # loading
    if isinstance(input_obj, str):
        # midi_file = parser.MidiFile(input_obj)
        midi_file = pypianoroll.read(input_obj)
    else:
        midi_file = input_obj
    
    # processing
    # num_instr = len(midi_file.instruments)
    num_instr = len(midi_file.tracks)

    pianorolls = []
    for idx in range(num_instr):
        # pr = midi_file.get_instrument_pianoroll(idx, resample_resolution=24)
        pr = midi_file.tracks[idx].pianoroll
        pianorolls.append(pr)
    ys = identify_multiple_track(pianorolls)
    return ys

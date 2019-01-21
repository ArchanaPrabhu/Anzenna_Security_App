from scipy.io import loadmat
from keras.models import model_from_json

import cv2
import numpy.matlib as matlib
import numpy as np
from math import factorial
import matplotlib.pyplot as plt
import os
import requests
import http.client

weights_path = 'weights_L1L2.mat'
model_path = 'model.json'

seed = 7
np.random.seed(seed)

#  Adding twilio api 
def send_message():
    

    conn = http.client.HTTPConnection("api.msg91.com")
    payload = "{ \"sender\": \"SOCKET\", \"route\": \"4\", \"country\": \"91\", \"sms\": [ { \"message\": \"Alert: Anomaly detected in user deekal webcam. Please check portal.\", \"to\": [ \"9916983961\" ] } ] }"

    headers = {
        'authkey': "249847AcKPqVmd5c02068a",
        'content-type': "application/json"
        }

    conn.request("POST", "/api/v2/sendsms?campaign=&response=&afterminutes=&schtime=&unicode=&flash=&message=&encrypt=&authkey=&mobiles=&route=&sender=&country=91", payload, headers)

    res = conn.getresponse()
    data = res.read()

def load_model(json_path):
    model = model_from_json(open(json_path).read())
    return model


def load_weights(model, weight_path):
    """
    Functions loads pre-trained weights into current model layers
    """
    # weights are stored in a dictionary
    weights_of_model = loadmat(weight_path)
    weight_dict = mat_dict_to_python(weights_of_model)
    i = 0
    # Fill current model layers with pre-trained weights
    for layer in model.layers:
        current_layer_weights = weight_dict[str(i)]
        layer.set_weights(current_layer_weights)
        i += 1
    return model


def mat_dict_to_python(weights):
    """
    Helper function to convert the model weights from Matlab-format to python-readable
    :param weights: dictionary containing model weights
    """
    i = 0
    _dict = {}
    for i in range(len(weights)):
        # If integer 'i' is a key in the weights dictionary, store it as is in the new dictionary _dict
        #
        if str(i) in weights:
            if weights[str(i)].shape == (0, 0):
                _dict[str(i)] = weights[str(i)]
            else:
                # TODO: why is only the first row being saved?
                local_weights = weights[str(i)][0]
                temp_weights = []
                # TODO: This loop is probably just a hack to get around some bug in Matlab weights. Confirm?
                for weight in local_weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        temp_weights.append(weight[0])
                    else:
                        temp_weights.append(weight)
                _dict[str(i)] = temp_weights
    return _dict


def load_video_features(video_path):
    """
    Given a video_path, return c3d features of each of the 32 segments
    """
    f = open(video_path, "r")
    words = f.read().split()
    num_feat = len(words) / 4096  # 32

    count = -1
    # array of length num_feat containing segments (clips of features) stacked vertically
    video_features = []
    for feat in range(0, int(num_feat)):
        # 4096 feature elements for each clip
        feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
        count = count + 1
        if count == 0:
            video_features = feat_row1
        if count > 0:
            video_features = np.vstack((video_features, feat_row1))

    return video_features


def savitzky_golay_filter(y, window_size, order, deriv=0, rate=1):
    # try:
    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    # except ValueError, msg:
    #    raise ValueError("window_size and order have to be of type int")

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")

    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)

    half_window = (window_size - 1) // 2
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def anomaly_check(predictions, total_segments, video_path):
    output_video_name = 'outputs/anomaly_' + ((video_path.split('/')[-1]).split('_')[-1])
    frames__score = []
    count = -1
    for iv in range(0, 32):
        F_Score = matlib.repmat(predictions[iv], 1, (int(total_segments[iv + 1]) - int(total_segments[iv])))
        count = count + 1
        if count == 0:
            frames__score = F_Score
        if count > 0:
            frames__score = np.hstack((frames__score, F_Score))

    cap = cv2.VideoCapture(video_path)
    while not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
        cv2.waitKey(1000)
        print("Wait for the header")

    # Current frame position (pos_frame = 0)
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    Total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print("Anomaly Prediction")
    x = np.linspace(1, Total_frames, Total_frames)
    scores = frames__score
    scores1 = scores.reshape((scores.shape[1],))
    scores1 = savitzky_golay_filter(scores1, 101, 3)

    break_pt = min(scores1.shape[0], x.shape[0])

    i = 0
    anomaly = False
    j = 0
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                          (frame_width, frame_height))
    timestamp_of_anomaly = -1
    while True:
        flag, frame = cap.read()
        if flag:
            i = i + 1
            if i >= len(scores1):
                break
            if scores1[i] > 0.3:
                if not anomaly:
                    timestamp_of_anomaly = i
                    anomaly = True
            if anomaly and j < 100:
                j += 1

                overlay = cv2.imread("alert.png")
                overlay = cv2.resize(overlay, (0,0), fx=0.5, fy=0.5) 
                '''y_offset = x_offset = 50
                frame[y_offset:y_offset + s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img'''
                rows,cols,channels = overlay.shape
                overlay=cv2.addWeighted(frame[0:0+rows, 0:0+cols],0.5,overlay,0.5,0)
                frame[0:0+rows, 0:0+cols] = overlay
                out.write(frame)
            else:
                out.write(frame)

            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cap.get(cv2.CAP_PROP_POS_FRAMES) == break_pt:
            # cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break
    out.release()
    if anomaly:
        send_message()
        return [output_video_name, [], timestamp_of_anomaly]
    else:
        feature_file = 'features/' + ((video_path.split('/'))[-1])[:-4] + '.txt'
        print(feature_file)
        os.remove(feature_file)
        os.remove(output_video_name)
        os.remove(video_path)
        return "No Anomaly"


def init(video_path):
    """
    :param video_path: Stream video to be received here. Needs to be at least 32*16 frames where each segment is just
    one clip
    """
    
    model = load_model(model_path)
    load_weights(model, weights_path)
    cap = cv2.VideoCapture('videos/' + video_path)

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_segments = np.linspace(1, total_frames, num=33)
    total_segments = total_segments.round()

    # Assumes that features are already calculated
    feature_path = 'features/' + video_path[0:-4] + '.txt'
    # array of 32 segments
    inputs = load_video_features(feature_path)
    # predict for the 32 segments.
    # predict_on_batch works only if a single batch is passed.
    predictions = model.predict_on_batch(inputs)

    # Each frame of video is being given a score between 0 & 1
    return anomaly_check(predictions, total_segments, 'videos/' + video_path)
    
# init('output_1543577788.avi')

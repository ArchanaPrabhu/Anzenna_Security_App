from os import listdir, walk
from pathlib import Path
import numpy as np
import cv2

from keras.models import Model

from c3d.c3d import C3D
from c3d.sports1M_utils import preprocess_input
num_of_segs = 32

def extract_features(file='/home/deepti/Documents/AnomalyDetection/Rakathon/videos/output_1543502728.avi'):
    outDir = '/home/deepti/Documents/AnomalyDetection/Rakathon/features'
    outFile = (file.split('/'))[-1]
    base_model = C3D(weights='sports1M')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc6').output)

    if not Path(outDir + "/" + file + ".txt").exists():
        print("Processing file: " + file)
        print(outDir+"/"+outFile[:-4] + '.txt')
        video = cv2.VideoCapture(file)
        while not video.isOpened():
            video = cv2.VideoCapture(file)
            cv2.waitKey(1000)
        print([int(video.get(cv2.CAP_PROP_FRAME_COUNT))])
        frame_count = 0
        clip = []
        list_of_features = []
        pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        while True:
            flag, frame = video.read()
            if flag:
                # The frame is ready and already captured
                pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
                if pos_frame % 17 == 0:
                    # Pre-process video into clips of desired dimension
                    # print(len(clip))
                    x = preprocess_input(np.array(clip))
                    x = model.predict(x)
                    # Apply L2 normalization on the feature vector
                    x = np.divide(x, np.linalg.norm(x, ord=2))  # Shape of x = (1, 4096)
                    list_of_features.append(x)
                    clip = []
                else:
                    clip.append(frame)
            else:
                # The next frame is not ready, so we try to read it again
                video.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                cv2.waitKey(1000)
            if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break
        segment_wise_clip_division = np.round(np.linspace(0, len(list_of_features), num_of_segs + 1))
        seg_features = []
        for i in range(32):
            clips_for_current_segment = list_of_features[
                                        int(segment_wise_clip_division[i]): int(np.max(
                                            [segment_wise_clip_division[i] + 1,
                                             segment_wise_clip_division[i + 1]]))]
            if clips_for_current_segment is None:
                print("FOUND NONE")
                print([segment_wise_clip_division[i] + 1, segment_wise_clip_division[i + 1]])
                clips_for_current_segment = 0
            seg_features.append(np.mean(clips_for_current_segment, axis=0).flatten())
        # print(seg_features)
        np.savetxt(outDir+"/"+outFile[:-4] + '.txt', seg_features, fmt="%10.6f")
        print("Saved features for Video segs: ")
import os
import sys
import socket

## gain linux ip
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(('10.0.0.1',8080))
        ip= s.getsockname()[0]
    finally:
        s.close()
    return ip

############ For LINUX ##############
# path
DATA_DIR = {
	'CMUMOSI': '../dataset/CMUMOSI',   # for nlpr
	'CMUMOSEI': '../dataset/CMUMOSEI',# for nlpr
}
PATH_TO_RAW_AUDIO = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'subaudio'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'subaudio'),
}
PATH_TO_RAW_FACE = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'openface_face'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'transcription.csv'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'transcription.csv'),
}
PATH_TO_FEATURES = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'features'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'features'),
}
PATH_TO_LABEL = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'CMUMOSI_features_raw_2way.pkl'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'CMUMOSEI_features_raw_2way.pkl'),
}

# pre-trained models, including supervised and unsupervised
PATH_TO_PRETRAINED_MODELS = '../tools'

# dir
SAVED_ROOT = os.path.join('../saved')
DATA_DIR = os.path.join(SAVED_ROOT, 'data')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')



############ For Windows ##############
DATA_DIR_Win = {
	'CMUMOSI': 'E:\\Dataset\\CMU-MOSI\\Raw',
	'CMUMOSEI1': 'E:\\Dataset\\CMU-MOSEI', # extract openface in five subprocess
	'CMUMOSEI2': 'E:\\Dataset\\CMU-MOSEI', # extract openface in five subprocess
	'CMUMOSEI3': 'E:\\Dataset\\CMU-MOSEI', # extract openface in five subprocess
	'CMUMOSEI4': 'E:\\Dataset\\CMU-MOSEI', # extract openface in five subprocess
	'CMUMOSEI5': 'E:\\Dataset\\CMU-MOSEI', # extract openface in five subprocess
}

PATH_TO_RAW_FACE_Win = {
	'CMUMOSI': os.path.join(DATA_DIR_Win['CMUMOSI'], 'Video\\Segmented'),
	'CMUMOSEI1': os.path.join(DATA_DIR_Win['CMUMOSEI1'], 'subvideo1'),
	'CMUMOSEI2': os.path.join(DATA_DIR_Win['CMUMOSEI2'], 'subvideo2'),
	'CMUMOSEI3': os.path.join(DATA_DIR_Win['CMUMOSEI3'], 'subvideo3'),
	'CMUMOSEI4': os.path.join(DATA_DIR_Win['CMUMOSEI4'], 'subvideo4'),
	'CMUMOSEI5': os.path.join(DATA_DIR_Win['CMUMOSEI5'], 'subvideo5'),
}

PATH_TO_FEATURES_Win = {
	'CMUMOSI': os.path.join(DATA_DIR_Win['CMUMOSI'], 'features'),
	'CMUMOSEI1': os.path.join(DATA_DIR_Win['CMUMOSEI1'], 'features'),
	'CMUMOSEI2': os.path.join(DATA_DIR_Win['CMUMOSEI2'], 'features'),
	'CMUMOSEI3': os.path.join(DATA_DIR_Win['CMUMOSEI3'], 'features'),
	'CMUMOSEI4': os.path.join(DATA_DIR_Win['CMUMOSEI4'], 'features'),
	'CMUMOSEI5': os.path.join(DATA_DIR_Win['CMUMOSEI5'], 'features'),
}



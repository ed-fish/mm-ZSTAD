import numpy as np
import librosa
import torch
import laion_clap

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

class CLAPModel:
    def __init__(self, enable_fusion=False):
        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion)
        self.model.load_ckpt()

    def get_audio_embedding_from_filelist(self, audio_files, use_tensor=False):
        return self.model.get_audio_embedding_from_filelist(x=audio_files, use_tensor=use_tensor)

    def get_audio_embedding_from_data(self, audio_data, use_tensor=False):
        return self.model.get_audio_embedding_from_data(x=audio_data, use_tensor=use_tensor)

    def get_text_embedding(self, text_data, use_tensor=False):
        return self.model.get_text_embedding(text_data, use_tensor=use_tensor)

# Instantiate the CLAP model
# clap_model = CLAPModel()

#!python
# cython: language_level=3

import ffmpeg
import numpy as np
import requests
import os
from pathlib import Path

MODELS_DIR = str(Path('~/.ggml-models').expanduser())
print("Saving models to:", MODELS_DIR)


cimport numpy as cnp

cdef int SAMPLE_RATE = 16000
cdef char* TEST_FILE = 'test.wav'
cdef char* DEFAULT_MODEL = 'tiny'
cdef char* LANGUAGE = b'auto'
cdef int N_THREADS = os.cpu_count()

MODELS = {
    'ggml-tiny.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin',
    'ggml-base.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-base.bin',
    'ggml-small.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-small.bin',
    'ggml-medium.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin',
    'ggml-large.bin': 'https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-large.bin',
}

def model_exists(model_path) -> bool:
    return os.path.exists(model_path)

def download_model(model_folder_path: str, model: str) -> str:
    if model_folder_path is None:
        model_folder_path = MODELS_DIR
    model_path = str(Path(model_folder_path).joinpath(model))
    if not model_exists(model_path):
        print(f'Downloading {model}...')
        url = MODELS[model]
        r = requests.get(url, allow_redirects=True)
        os.makedirs(model_folder_path, exist_ok=True)
        with open(Path(model_folder_path).joinpath(model), 'wb') as f:
            f.write(r.content)
    return model_path


cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] load_audio(bytes file, int sr = SAMPLE_RATE):
    try:
        out = (
            ffmpeg.input(file, threads=0)
            .output(
                "-", format="s16le",
                acodec="pcm_s16le",
                ac=1, ar=sr
            )
            .run(
                cmd=["ffmpeg", "-nostdin"],
                capture_stdout=True,
                capture_stderr=True
            )
        )[0]
    except:
        raise RuntimeError(f"File '{file}' not found")

    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] frames = (
        np.frombuffer(out, np.int16)
        .flatten()
        .astype(np.float32)
    ) / pow(2, 15)

    return frames

cdef whisper_full_params default_params() nogil:
    cdef whisper_full_params params = whisper_full_default_params(
        whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
    )
    params.print_realtime = False
    params.print_progress = False
    params.translate = False
    params.language = <const char *> LANGUAGE
    params.token_timestamps = True
    params.print_timestamps = True
    n_threads = N_THREADS
    return params


cdef whisper_full_params custom_params(dict p) nogil:
    cdef whisper_full_params params = whisper_full_default_params(
        whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
    )
    with gil:
        params.print_realtime = <const bint> p["print_realtime"]
        params.print_progress = <const bint> p["print_progress"]
        params.translate = <const bint> p["translate"]
        params.language = <const char *> p["language"]
        params.token_timestamps = <const bint> p["token_timestamps"]
        params.print_timestamps = <const bint>  p["print_timestamps"]
    n_threads = N_THREADS
    return params

def prepare_parameters(parameters=None):
    if parameters:
        if "language" in parameters:
            language = parameters["language"]
            if type(language) == str:
                language = language.encode("utf-8")
            parameters["language"] = language
    return parameters


cdef class Whisper:
    cdef whisper_context * ctx
    cdef whisper_full_params params

    def __init__(self, model=DEFAULT_MODEL, parameters=None, model_folder_path=None):
        parameters = prepare_parameters(parameters=parameters)
        model_fullname = f'ggml-{model}.bin'
        model_path = download_model(model_folder_path=model_folder_path, model=model_fullname)
        if parameters is None:
            self.params = default_params()
        else:
            self.params = custom_params(p=parameters)
        cdef bytes model_b = str(model_path).encode('utf8')
        self.ctx = whisper_init_from_file(model_b)

    def __dealloc__(self):
        whisper_free(self.ctx)

    def free_memory(self):
        whisper_free(self.ctx)

    def full_transcribe(self, filename=TEST_FILE):
        cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] frames = load_audio(<bytes>filename)
        cdef int res = whisper_full(self.ctx, self.params, &frames[0], len(frames))
        return res

    def whisper_full_get_segment_text(self, int i_segment):
        return whisper_full_get_segment_text(self.ctx, i_segment).decode("utf-8")

    def whisper_full_get_segment_t0(self, int i_segment):
        return whisper_full_get_segment_t0(self.ctx, i_segment)

    def whisper_full_get_segment_t1(self, int i_segment):
        return whisper_full_get_segment_t1(self.ctx, i_segment)

    def whisper_token_to_str(self, int token):
        return whisper_token_to_str(self.ctx, token).decode("utf-8")

    def whisper_token_sot(self):
        yield whisper_token_sot(self.ctx)

    def whisper_token_eot(self):
        yield whisper_token_eot(self.ctx)

    def whisper_token_prev(self):
        yield whisper_token_prev(self.ctx)

    def whisper_token_solm(self):
        yield whisper_token_solm(self.ctx)

    def whisper_token_not(self):
        yield whisper_token_not(self.ctx)

    def whisper_token_beg(self):
        yield whisper_token_beg(self.ctx)

    def whisper_token_beg(self):
        yield whisper_token_beg(self.ctx)

    def whisper_full_n_tokens(self, int i_segment):
        return whisper_full_n_tokens(self.ctx, i_segment)

    def whisper_full_get_token_id(self, int i_segment, int i_token):
        return whisper_full_get_token_id(self.ctx, i_segment, i_token)

    def whisper_full_get_token_data(self, int i_segment, int i_token):
        return whisper_full_get_token_data(self.ctx, i_segment, i_token)

    def whisper_full_n_segments(self):
        return whisper_full_n_segments(self.ctx)

    def whisper_full_get_token_text(self, int i_segment, int i_token):
        return whisper_full_get_token_text(self.ctx, i_segment, i_token).decode("utf-8")

    def whisper_full_lang_id(self):
        return whisper_full_lang_id(self.ctx)

    def whisper_lang_str(self, int id):
        return whisper_lang_str(id).decode("utf-8")

    def get_language_string(self):
        l_id = whisper_full_lang_id(self.ctx)
        return whisper_lang_str(l_id).decode("utf-8")

    def get_segment_timestamp(self, int i_segment):
        t0 = int(whisper_full_get_segment_t0(self.ctx, i_segment))
        t1 = int(whisper_full_get_segment_t1(self.ctx, i_segment))
        return (t0, t1)

    def whisper_is_multilingual(self):
        return whisper_is_multilingual(self.ctx)




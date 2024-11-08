import speech_recognition as sr
import ollama
from gtts import gTTS
import gradio as gr
from io import BytesIO
import numpy as np
from dataclasses import dataclass, field
import time
import traceback
from pydub import AudioSegment
import librosa
from utils.vad import get_speech_timestamps, collect_chunks, VadOptions
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import os
import sys
import subprocess
from huggingface_hub import login
login(token=os.getenv("new_data_token"), write_permission=True)
r = sr.Recognizer()

access_token = "hf_..."
model = AutoModel.from_pretrained("meta-llama/Meta-Llama-3-8B", token=access_token, use_auth_token=True)
model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
text2text = pipeline("text-generation", model=model, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto", use_auth_token=True, tokenizer=tokenizer)
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
@dataclass
class AppState:
    stream: np.ndarray | None = None
    sampling_rate: int = 0
    pause_detected: bool = False
    started_talking: bool =  False
    stopped: bool = False
    conversation: list = field(default_factory=list)

def run_vad(ori_audio, sr):
    _st = time.time()
    try:
        audio = ori_audio
        audio = audio.astype(np.float32) / 32768.0
        sampling_rate = 16000
        if sr != sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
        vad_parameters = {}
        vad_parameters = VadOptions(**vad_parameters)
        speech_chunks = get_speech_timestamps(audio, vad_parameters)
        audio = collect_chunks(audio, speech_chunks)
        duration_after_vad = audio.shape[0] / sampling_rate
        if sr != sampling_rate:
            # resample to original sampling rate
            vad_audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=sr)
        else:
            vad_audio = audio
        vad_audio = np.round(vad_audio * 32768.0).astype(np.int16)
        vad_audio_bytes = vad_audio.tobytes()
        return duration_after_vad, vad_audio_bytes, round(time.time() - _st, 4)
    except Exception as e:
        msg = f"[asr vad error] audio_len: {len(ori_audio)/(sr*2):.3f} s, trace: {traceback.format_exc()}"
        print(msg)
        return -1, ori_audio, round(time.time() - _st, 4)

def determine_pause(audio: np.ndarray, sampling_rate: int, state: AppState) -> bool:
    """Take in the stream, determine if a pause happened"""
    temp_audio = audio
    dur_vad, _, time_vad = run_vad(temp_audio, sampling_rate)
    duration = len(audio) / sampling_rate
    if dur_vad > 0.5 and not state.started_talking:
        print("started talking")
        state.started_talking = True
        return False
    print(f"duration_after_vad: {dur_vad:.3f} s, time_vad: {time_vad:.3f} s")
    return (duration - dur_vad) > 1

def process_audio(audio:tuple, state:AppState):
    if state.stream is None:
        state.stream = audio[1]
        state.sampling_rate = audio[0]
    else:
        state.stream =  np.concatenate((state.stream, audio[1]))
    pause_detected = determine_pause(state.stream, state.sampling_rate, state)
    state.pause_detected = pause_detected
    if state.pause_detected and state.started_talking:
        return gr.Audio(recording=False), state
    return None, state

def response(state:AppState):
    if not state.pause_detected and not state.started_talking:
        return None, AppState()
    audio_buffer = BytesIO()
    segment = AudioSegment(
        state.stream.tobytes(),
        frame_rate=state.sampling_rate,
        sample_width=state.stream.dtype.itemsize,
        channels=(1 if len(state.stream.shape) == 1 else state.stream.shape[1]),
    )
    segment.export(audio_buffer, format="wav")
    textin = ""
    with sr.AudioFile(audio_buffer) as source:
        audio_data=r.record(source)
        try:
            textin=r.recognize_google(audio_data,language='vi')
        except:
            textin = ""
        state.conversation.append({"role": "user", "content": "Bạn: " + textin})   
    if textin != "":
        print("Đang nghĩ...")
        textout=str(text2text(textin))
        textout = textout.replace('*','')
        state.conversation.append({"role": "user", "content": "Trợ lý: " + textout})
    if textout != "":
        print("Đang đọc...")
        mp3 = gTTS(textout,tld='com.vn',lang='vi',slow=False)
        mp3_fp = BytesIO()
        mp3.write_to_fp(mp3_fp)
        srr=mp3_fp.getvalue()
        mp3_fp.close()
        #yield srr, state
    yield srr, AppState(conversation=state.conversation)

def start_recording_user(state: AppState):
    if not state.stopped:
        return gr.Audio(recording=True)

title = "vietnamese by tuphamkts"
description = "A vietnamese text-to-speech demo."

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(label="Nói cho tôi nghe nào", sources="microphone", type="numpy")
        with gr.Column():
            chatbot = gr.Chatbot(label="Nội dung trò chuyện", type="messages")
            output_audio = gr.Audio(label="Trợ lý", autoplay=True)
    state = gr.State(value=AppState())

    stream = input_audio.stream(
        process_audio,
        [input_audio, state],
        [input_audio, state],
        stream_every=0.50,
        time_limit=30,
    )
    respond = input_audio.stop_recording(
        response,
        [state],
        [output_audio, state],
    )
    respond.then(lambda s: s.conversation, [state], [chatbot])

    restart = output_audio.stop(
        start_recording_user,
        [state],
        [input_audio],
    )
    cancel = gr.Button("Stop Conversation", variant="stop")
    cancel.click(lambda: (AppState(stopped=True), gr.Audio(recording=False)), None,
                [state, input_audio], cancels=[respond, restart])
demo.launch()
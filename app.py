###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
# server.py
# from flask import Flask, render_template,send_from_directory,request, jsonify
# from flask_sockets import Sockets
# from flask_sslify import SSLify
# from flask_cors import CORS
# from OpenSSL import SSL
import ssl
import sys, os, select, socket
# import conf
# import base64
import json
import requests
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
# import re
# import numpy as np
from threading import Thread,Event
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpsender import RTCRtpSender

from asr_salute import CHUNK
from webrtc import HumanPlayer
from basereal import BaseReal
from llm import llm_response

import argparse
import random
# import shutil
import asyncio
# import torch
from typing import Dict
from logger import logger

import asr_salute as asr

# DEBUG = False
# DEBUG = True
##### ASR ##################################
# asr_url = 'http://localhost:8089/realtime_inference'  # Debug
coordinator_url = 'http://coordinator:8087/ask'  # Sber STAND
# coordinator_url = 'http://localhost:8087/ask'  # Debug
json_request_header = {'Content-Type': 'application/json'}
empty_bytes = b''
wait_question = True
audio_chunks = []
started = False
#
# app = Flask(__name__)
#sockets = Sockets(app)
nerfreals:Dict[int, BaseReal] = {} #sessionid:BaseReal
opt = None
model = None
avatar = None
inprogress = False


def format_prepped_request(prepped, encoding=None):
    # prepped has .method, .path_url, .headers and .body attribute to view the request
    encoding = encoding or requests.utils.get_encoding_from_headers(prepped.headers)
    body = prepped.body.decode(encoding) if encoding else '<binary data>'
    headers = '\n'.join(['{}: {}'.format(*hv) for hv in prepped.headers.items()])
    return f"""{prepped.method} {prepped.path_url} HTTP/1.1 {headers} {body}"""


async def record_audio_upload(request: requests):
    speach = empty_bytes
    while True:
        chunk = await request.content.read(CHUNK)
        if chunk == empty_bytes:
            break
        speach += chunk
    recognition = await speach_dialogue(speach)
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "status": recognition.status}
        ),
    )


async def ask_and_answer(request: requests):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    question  = params.get('question', '')
    return await question_and_answer(question, sessionid)


async def speach_dialogue(speach, sessionid: int = 0):
    question = asr.recognize_speach(speach)
    return await question_and_answer(question, sessionid)


async def question_and_answer(question:str = '', sessionid: int = 0):
    if question and question != '':
        question = question.strip()
        print("Вопрос: ", question)

        if question.capitalize() != 'STOP' and question.capitalize() != 'СТОП':
            try:
                answer_response = requests.post(coordinator_url, headers=json_request_header, json={"question": question}, verify=False)
                if answer_response.ok:
                    answer = answer_response.text
                else:
                    answer = 'Сервер GigaChat сообщает о технической ошибке'
            except:
                answer = 'Сервер Coordinator сообщает о технической ошибке'
            print("answer:", answer, " from ASR URL:", coordinator_url)
            inference = await realtime_inference(answer = answer, sessionid = sessionid)
            status = inference.status
        else:
            if sessionid in nerfreals:
                print("command: STOP AVATAR")
                nerfreals[sessionid].flush_talk()

    else:
        status = "ERROR: question is empty"

    print("Следующий вопрос")
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "status": status}
        ),
    )


async def record_audio_chunk(request: requests):
    global audio_chunks, started
    started = True
    while True:
        chunk = await request.content.read(CHUNK)
        if chunk == empty_bytes:
            break
        audio_chunks.append(chunk)

    print('audio has chunk received')
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 200, "status": "OK", "message": "audio has chunk received"}
        ),
    )


async def record_audio_end(request: requests):
    global audio_chunks, started
    started = False
    await speach_dialogue(b''.join(audio_chunks))
    audio_chunks = []
    print('audio has fully received')
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 200, "status": "OK", "message": "audio has fully received"}
        ),
    )


##### webrtc ###############################
pcs = set()

def randN(N)->int:
    ''' Сгенерировать случайное число длины N '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_nerfreal(sessionid:int)->BaseReal:
    global nerfreal
    opt.sessionid=sessionid
    if opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(opt,model,avatar)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt,model,avatar)
    elif opt.model == 'ernerf':
        from nerfreal import NeRFReal
        nerfreal = NeRFReal(opt,model,avatar)
    elif opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(opt,model,avatar)
    return nerfreal

#@app.route('/offer', methods=['POST'])
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    if len(nerfreals) >= opt.max_session:
        logger.info('reach max session')
        return -1
    sessionid = randN(6) #len(nerfreals)
    logger.info('sessionid=%d',sessionid)
    nerfreals[sessionid] = None
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal
    
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            del nerfreals[sessionid]
        if pc.connectionState == "closed":
            pcs.discard(pc)
            del nerfreals[sessionid]

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)
    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)

    await pc.setRemoteDescription(offer)
    # RTCSessionDescription(sdp='v=0\r\no=- 4207378497271027466 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\na=group:BUNDLE 0 1\r\na=extmap...PCMA/8000\r\na=rtpmap:13 CN/8000\r\na=rtpmap:110 telephone-event/48000\r\na=rtpmap:126 telephone-event/8000\r\n', type='offer')
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    # answer = RTCSessionDescription(sdp='v=0\r\no=- 3952236247 3952236247 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\na=group:BUNDLE 0 1\r\na=msid-sem...E:F0:19:18:B5:B0:6A:1C:CC:70:99:D2:02:61:CE:3E:1C:29:8A:0B:C1:9F:D9:1F:03:AF:25:10:F6:CF\r\na=setup:active\r\n', type='answer')
    #return jsonify({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid":sessionid}
        ),
    )


async def realtime_inference(sessionid: int = 0, answer: str = "Повторите вопрос, попробуйте его переформулировать"):
    global inprogress
    if inprogress:
        print("WARNING: skipping request")
        return "IN PROGRESS", 200
    inprogress = True
    message = '/realtime_inference - Request has gotten'
    print(message, "with answer:", answer,  ", sessionid:", sessionid)
    if nerfreals and nerfreals.keys() and sessionid == 0:
        sessionid = list(nerfreals.keys())[0]
    print("/realtime_inference - sessionid:", sessionid)
    if sessionid in nerfreals:
        try:
            nerfreals[sessionid].put_msg_txt(answer)
        except Exception as e:
            print(f'Error: {e}')
            # nerfreals[sessionid].flush_talk()
    else:
        message = "There are no any listen users"
        print(message)
    print('Processing request finished')
    inprogress = False

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "status":"ok", "sessionid": sessionid, "message": message}
        ),
    )


async def human(request):
    params = await request.json()
    logger.info(f"/human parameters: {params}")
    sessionid = params.get('sessionid', 0)
    logger.info(f"/human sessionid = {sessionid}")
    logger.info(f"/humah nerfreals.keys(): {nerfreals.keys()}")
    if nerfreals and nerfreals.keys() and sessionid == 0:
        sessionid = list(nerfreals.keys())[0]
    logger.info(f"/human sessionid = {sessionid}")

    if sessionid in nerfreals:
        if params.get('finish'):
            nerfreals[sessionid].flush_talk()
        else:
            if params.get('interrupt'):
                nerfreals[sessionid].break_talk()
            if params.get('text', None) and params.get('type', None):
                if params['type']=='echo':
                    nerfreals[sessionid].put_msg_txt(params['text'])
                elif params['type']=='chat':
                    res=await asyncio.get_event_loop().run_in_executor(None, llm_response, params['text'],nerfreals[sessionid])
                    #nerfreals[sessionid].put_msg_txt(res)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "data":"ok", "sessionid": sessionid}
            ),
        )

    else:
        message = "There are no any listen users"
        logger.info(message)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "data":"nok", "status":"nok", "sessionid": sessionid, "message": message}
            ),
    )

async def humanaudio(request):
    try:
        form= await request.post()
        sessionid = int(form.get('sessionid', 0))
        fileobj = form["file"]
        filename=fileobj.filename
        filebytes=fileobj.file.read()
        nerfreals[sessionid].put_audio_file(filebytes)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg":"err","data": ""+e.args[0]+""}
            ),
        )


async def set_audiotype(request):
    params = await request.json()

    sessionid = params.get('sessionid',0)    
    nerfreals[sessionid].set_curr_state(params['audiotype'],params['reinit'])

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data":"ok"}
        ),
    )


async def record(request):
    params = await request.json()
    sessionid = params.get('sessionid',0)
    if nerfreals and nerfreals.keys() and sessionid == 0:
        sessionid = list(nerfreals.keys())[0]
    if sessionid in nerfreals:

        if params['type']=='start_record':
            # nerfreals[sessionid].put_msg_txt(params['text'])
            nerfreals[sessionid].start_recording()
        elif params['type']=='end_record':
            nerfreals[sessionid].stop_recording()
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "data":"ok"}
            ),
        )

    else:
        message = "There are no any listen users"
        print(message)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "data": "nok", "status": "nok", "sessionid": sessionid, "message": message}
            ),
        )


async def is_speaking(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": nerfreals[sessionid].is_speaking()}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def post(url,data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url,data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        logger.info(f'Error: {e}')


async def run(push_url,sessionid):
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url,pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer,type='answer'))

##########################################

# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', type=str, default="data/data_kf.json", help="transforms.json, pose source")
    parser.add_argument('--au', type=str, default="data/au.csv", help="eye blink area")
    parser.add_argument('--torso_imgs', type=str, default="", help="torso images path")

    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --exp_eye")

    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="data range to use")
    parser.add_argument('--workspace', type=str, default='data/video')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--ckpt', type=str, default='data/pretrained/ngp_kf.pth')
   
    parser.add_argument('--num_rays', type=int, default=4096 * 16, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=16, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=16, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    ### loss set
    parser.add_argument('--warmup_step', type=int, default=10000, help="warm up steps")
    parser.add_argument('--amb_aud_loss', type=int, default=1, help="use ambient aud loss")
    parser.add_argument('--amb_eye_loss', type=int, default=1, help="use ambient eye loss")
    parser.add_argument('--unc_loss', type=int, default=1, help="use uncertainty loss")
    parser.add_argument('--lambda_amb', type=float, default=1e-4, help="lambda for ambient loss")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    
    parser.add_argument('--bg_img', type=str, default='white', help="background image")
    parser.add_argument('--fbg', action='store_true', help="frame-wise bg")
    parser.add_argument('--exp_eye', action='store_true', help="explicitly control the eyes")
    parser.add_argument('--fix_eye', type=float, default=-1, help="fixed eye area, negative to disable, set to 0-0.3 for a reasonable eye")
    parser.add_argument('--smooth_eye', action='store_true', help="smooth the eye area sequence")

    parser.add_argument('--torso_shrink', type=float, default=0.8, help="shrink bg coords to allow more flexibility in deform")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', type=int, default=0, help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means GPU.")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=4, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied (sigma)")
    parser.add_argument('--density_thresh_torso', type=float, default=0.01, help="threshold for density grid to be occupied (alpha)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    parser.add_argument('--init_lips', action='store_true', help="init lips region")
    parser.add_argument('--finetune_lips', action='store_true', help="use LPIPS and landmarks to fine tune lips region")
    parser.add_argument('--smooth_lips', action='store_true', help="smooth the enc_a in a exponential decay way...")

    parser.add_argument('--torso', action='store_true', help="fix head and train torso")
    parser.add_argument('--head_ckpt', type=str, default='', help="head model")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")
    parser.add_argument('--radius', type=float, default=3.35, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=21.24, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    ### else
    parser.add_argument('--att', type=int, default=2, help="audio attention mode (0 = turn off, 1 = left-direction, 2 = bi-direction)")
    parser.add_argument('--aud', type=str, default='', help="audio source (empty will load the default, else should be a path to a npy file)")
    parser.add_argument('--emb', action='store_true', help="use audio class + embedding instead of logits")

    parser.add_argument('--ind_dim', type=int, default=4, help="individual code dim, 0 to turn off")
    parser.add_argument('--ind_num', type=int, default=10000, help="number of individual codes, should be larger than training dataset size")

    parser.add_argument('--ind_dim_torso', type=int, default=8, help="individual code dim, 0 to turn off")

    parser.add_argument('--amb_dim', type=int, default=2, help="ambient dimension")
    parser.add_argument('--part', action='store_true', help="use partial training data (1/10)")
    parser.add_argument('--part2', action='store_true', help="use partial training data (first 15s)")

    parser.add_argument('--train_camera', action='store_true', help="optimize camera pose")
    parser.add_argument('--smooth_path', action='store_true', help="brute-force smooth camera pose trajectory with a window size")
    parser.add_argument('--smooth_path_window', type=int, default=7, help="smoothing window size")

    # asr
    parser.add_argument('--asr', action='store_true', help="load asr for real-time app")
    parser.add_argument('--asr_wav', type=str, default='', help="load the wav and use as input")
    parser.add_argument('--asr_play', action='store_true', help="play out the audio")

    #parser.add_argument('--asr_model', type=str, default='deepspeech')
    parser.add_argument('--asr_model', type=str, default='cpierse/wav2vec2-large-xlsr-53-esperanto') #
    # parser.add_argument('--asr_model', type=str, default='facebook/wav2vec2-large-960h-lv60-self')
    # parser.add_argument('--asr_model', type=str, default='facebook/hubert-large-ls960-ft')

    parser.add_argument('--asr_save_feats', action='store_true')
    # audio FPS
    parser.add_argument('--fps', type=int, default=50)
    # sliding window left-middle-right length (unit: 20ms)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    parser.add_argument('--fullbody', action='store_true', help="fullbody human")
    parser.add_argument('--fullbody_img', type=str, default='data/fullbody/img')
    parser.add_argument('--fullbody_width', type=int, default=580)
    parser.add_argument('--fullbody_height', type=int, default=1080)
    parser.add_argument('--fullbody_offset_x', type=int, default=0)
    parser.add_argument('--fullbody_offset_y', type=int, default=0)

    #musetalk opt
    parser.add_argument('--avatar_id', type=str, default='avatar_blonde')  # 'avator_1')
    parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)

    # parser.add_argument('--customvideo', action='store_true', help="custom video")
    # parser.add_argument('--customvideo_img', type=str, default='data/customvideo/img')
    # parser.add_argument('--customvideo_imgnum', type=int, default=1)

    parser.add_argument('--customvideo_config', type=str, default='')

    parser.add_argument('--tts', type=str, default='edgetts') #xtts gpt-sovits cosyvoice
    parser.add_argument('--REF_FILE', type=str, default=None)
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://localhost:9880') # http://localhost:9000
    # parser.add_argument('--CHARACTER', type=str, default='test')
    # parser.add_argument('--EMOTION', type=str, default='default')

    parser.add_argument('--model', type=str, default='wav2lip')  # 'ernerf') #musetalk wav2lip

    parser.add_argument('--transport', type=str, default='webrtc')  # 'rtcpush') #rtmp webrtc rtcpush
    parser.add_argument('--push_url', type=str, default='http://localhost:8089/offer')

    parser.add_argument('--max_session', type=int, default=10)  #multi session count
    parser.add_argument('--listenport', type=int, default=8089)

    opt = parser.parse_args()
    #app.config.from_object(opt)
    #print(app.config)
    opt.customopt = []
    if opt.customvideo_config!='':
        with open(opt.customvideo_config,'r') as file:
            opt.customopt = json.load(file)

    if opt.model == 'ernerf':       
        from nerfreal import NeRFReal,load_model,load_avatar
        model = load_model(opt)
        avatar = load_avatar(opt) 
        
        # we still need test_loader to provide audio features for testing.
        # for k in range(opt.max_session):
        #     opt.sessionid=k
        #     nerfreal = NeRFReal(opt, trainer, test_loader,audio_processor,audio_model)
        #     nerfreals.append(nerfreal)
    elif opt.model == 'musetalk':
        from musereal import MuseReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model()
        avatar = load_avatar(opt.avatar_id) 
        warm_up(opt.batch_size,model)      
        # for k in range(opt.max_session):
        #     opt.sessionid=k
        #     nerfreal = MuseReal(opt,audio_processor,vae, unet, pe,timesteps)
        #     nerfreals.append(nerfreal)
    elif opt.model == 'wav2lip':
        from lipreal import LipReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model("./models/wav2lip.pth")
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model,256)
        # for k in range(opt.max_session):
        #     opt.sessionid=k
        #     nerfreal = LipReal(opt,model)
        #     nerfreals.append(nerfreal)
    elif opt.model == 'ultralight':
        from lightreal import LightReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,avatar,160)

    if opt.transport=='rtmp':
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    appasync = web.Application()
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_post("/audio/upload", record_audio_upload)
    appasync.router.add_post("/audio/chunk", record_audio_chunk)
    appasync.router.add_post("/audio/end", record_audio_end)
    appasync.router.add_post("/ask_and_answer", ask_and_answer)


    appasync.router.add_static('/asr',path='web/asr')
    appasync.router.add_static('/avatar',path='web/avatar')
    appasync.router.add_static('/',path='web')

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='webrtcapi-asr.html'
    if opt.transport=='rtmp':
        pagename='echoapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    logger.info('start http server; http://localhost:'+str(opt.listenport)+'/'+pagename)
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(run(push_url,k))
        loop.run_forever()    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()

    dir = os.path.dirname(sys.argv[0])
    if dir == '':
        dir = os.curdir

    # from OpenSSL import SSL
    # context = SSL.Context(SSL.TLS_SERVER_METHOD)
    # context = SSL.Context(SSL.TLS_METHOD)
    # context.set_options(SSL.OP_NO_SSLv2)
    # context.use_privatekey_file('certificates/key.pem')
    # context.use_certificate_file('certificates/cert.pem')
    # ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    # ssl_context.load_cert_chain('certificates/ip.crt', 'certificates/ip.key')
    run_server(web.AppRunner(appasync, debug=True))  #, ssl_context=ssl_context))
    #app.on_shutdown.append(on_shutdown)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()
    
    

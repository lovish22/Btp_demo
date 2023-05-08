from django.shortcuts import render, get_object_or_404

from .models import audio

from . import algo

import numpy as np
from scipy.io import wavfile


def index(request):
    audios = audio.objects.all()
    context = {"audios": audios}
    return render(request, "abe/index.html", context)


def analysis(request):
    audio_name = request.POST["audio"]
    noise_type = request.POST["noise"]
    noise_level = request.POST["noise_level"]

    aud_add = audio_name + "_" + noise_type + "_" + noise_level + ".wav"
    aud = get_object_or_404(audio, pk=aud_add)
    np_arr, stoi_score = algo.bwe(aud.audio_file)

    normalized_arr = np_arr / np.max(np.abs(np_arr))
    scaled_arr = 0.9 * normalized_arr

    pred_path = "audio_output/pred_aud.wav"
    wavfile.write(pred_path, 16000, scaled_arr)

    # maxx = -1
    # minn = 1
    # for i in np_arr:
    #     maxx = max(maxx, i)
    #     minn = min(minn, i)
    # print(maxx, minn)

    return render(
        request,
        "abe/analysis.html",
        {"stoi_score": stoi_score, "aud": aud, "pred_path": pred_path},
    )

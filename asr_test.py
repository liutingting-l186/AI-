import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _build_asr_model(model: str, model_revision: str, vad_model: str, vad_model_revision: str, punc_model: str, punc_model_revision: str):
    from funasr import AutoModel

    model_path = str(model)
    is_local_model = Path(model_path).exists()

    kwargs = {"model": model_path, "trust_remote_code": True, "disable_update": True}
    if (not is_local_model) and model_revision:
        kwargs["model_revision"] = model_revision

    if vad_model:
        vad_path = str(vad_model)
        is_local_vad = Path(vad_path).exists()
        kwargs["vad_model"] = vad_path
        if (not is_local_vad) and vad_model_revision:
            kwargs["vad_model_revision"] = vad_model_revision

    if punc_model:
        punc_path = str(punc_model)
        is_local_punc = Path(punc_path).exists()
        kwargs["punc_model"] = punc_path
        if (not is_local_punc) and punc_model_revision:
            kwargs["punc_model_revision"] = punc_model_revision

    return AutoModel(**kwargs)


def _maybe_convert_to_16k_mono(audio_path: str, ffmpeg_path: str) -> str:
    ffmpeg_bin = (ffmpeg_path or "").strip() or shutil.which("ffmpeg") or ""
    if not ffmpeg_bin:
        return audio_path

    fd, tmp_wav = tempfile.mkstemp(prefix="asr_", suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                audio_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-vn",
                "-f",
                "wav",
                tmp_wav,
            ],
            check=True,
        )
        return tmp_wav
    except Exception:
        try:
            os.remove(tmp_wav)
        except Exception:
            pass
        return audio_path


def _extract_text(res):
    if isinstance(res, list) and res:
        first = res[0]
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, str):
                return text.strip()
        if isinstance(first, str):
            return first.strip()
    if isinstance(res, dict):
        text = res.get("text")
        if isinstance(text, str):
            return text.strip()
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="./test.wav")
    parser.add_argument("--hotword", default="")
    parser.add_argument("--model", default="./models/speech_paraformer")
    parser.add_argument("--model-revision", default="v2.0.4")
    parser.add_argument("--vad-model", default="")
    parser.add_argument("--vad-model-revision", default="v2.0.4")
    parser.add_argument("--punc-model", default="")
    parser.add_argument("--punc-model-revision", default="v2.0.4")
    parser.add_argument("--force-16k", action="store_true")
    parser.add_argument("--ffmpeg-path", default="")
    args = parser.parse_args()

    ffmpeg_path = (args.ffmpeg_path or "").strip()
    if ffmpeg_path:
        ffmpeg_dir = str(Path(ffmpeg_path).resolve().parent)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    audio_path = (args.audio or "").strip()
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    model = _build_asr_model(
        model=args.model,
        model_revision=args.model_revision,
        vad_model=args.vad_model,
        vad_model_revision=args.vad_model_revision,
        punc_model=args.punc_model,
        punc_model_revision=args.punc_model_revision,
    )

    tmp_wav = ""
    try:
        input_path = audio_path
        if args.force_16k:
            converted = _maybe_convert_to_16k_mono(audio_path, args.ffmpeg_path)
            if converted != audio_path:
                tmp_wav = converted
                input_path = converted

        res = model.generate(input=input_path, batch_size_s=300, hotword=args.hotword or "")
        print(json.dumps(res, ensure_ascii=False, indent=2))
        print("\n--- ASR Text ---")
        print(_extract_text(res))
    finally:
        if tmp_wav:
            try:
                os.remove(tmp_wav)
            except Exception:
                pass


if __name__ == "__main__":
    main()

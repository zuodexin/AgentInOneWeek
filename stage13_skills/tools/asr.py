"""ASR (Automatic Speech Recognition) tool interface.

Uses FunASR (Alibaba) as the default backend, with a pluggable interface
to swap in other backends (e.g. faster-whisper, whisper).
"""

import argparse
import json
from pathlib import Path
from typing import Protocol, Optional, List, Dict

# import ipdb  # Removed to avoid dependency issues


class ASRResult:
    """Structured ASR result with word-level timestamps."""

    def __init__(self, text: str, segments: List[Dict], language: str = "zh"):
        self.text = text
        self.segments = (
            segments  # [{text, start, end, words: [{word, start, end, confidence}]}]
        )
        self.language = language

    def to_dict(self) -> dict:
        return {"text": self.text, "segments": self.segments, "language": self.language}


class ASRBackend(Protocol):
    """Protocol for ASR backends."""

    def transcribe(self, audio_path: str, language: str = "zh") -> ASRResult: ...


class FunASRBackend:
    """FunASR-based ASR with word-level timestamps."""

    def __init__(self):
        self._pipeline = None

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return
        from funasr import AutoModel
        import os

        device = os.getenv("ECHO_DEVICE", "cuda")
        self._pipeline = AutoModel(
            model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
            device=device,
        )

    def transcribe(self, audio_path: str, language: str = "zh") -> ASRResult:
        self._ensure_loaded()
        result = self._pipeline.generate(
            input=audio_path,
            batch_size_s=300,
            hotword="",
        )
        if not result:
            return ASRResult("", [], language)

        segments = []
        for item in result:
            # FunASR with spk_model returns sentence_info with per-sentence speaker/timestamps
            sentence_info = item.get("sentence_info", [])
            if sentence_info:
                for sent in sentence_info:
                    text = sent.get("text", "")
                    timestamp = sent.get("timestamp", [])
                    words = []
                    chars = list(text.replace(" ", ""))
                    for i, ts in enumerate(timestamp):
                        if i < len(chars):
                            words.append(
                                {
                                    "word": chars[i],
                                    "start": ts[0] / 1000.0,
                                    "end": ts[1] / 1000.0,
                                    "confidence": 0.9,
                                }
                            )
                    seg = {
                        "text": text,
                        "start": timestamp[0][0] / 1000.0 if timestamp else 0,
                        "end": timestamp[-1][1] / 1000.0 if timestamp else 0,
                        "words": words,
                    }
                    if "spk" in sent:
                        seg["speaker"] = f"speaker_{sent['spk']}"
                    segments.append(seg)
            else:
                # Fallback: no sentence_info (no spk_model or old format)
                text = item.get("text", "")
                timestamp = item.get("timestamp", [])
                words = []
                chars = list(text.replace(" ", ""))
                for i, ts in enumerate(timestamp):
                    if i < len(chars):
                        words.append(
                            {
                                "word": chars[i],
                                "start": ts[0] / 1000.0,
                                "end": ts[1] / 1000.0,
                                "confidence": 0.9,
                            }
                        )
                seg = {
                    "text": text,
                    "start": timestamp[0][0] / 1000.0 if timestamp else 0,
                    "end": timestamp[-1][1] / 1000.0 if timestamp else 0,
                    "words": words,
                }
            # Speaker info if available
            if "spk" in item:
                seg["speaker"] = item["spk"]
            segments.append(seg)

        full_text = "".join(s["text"] for s in segments)
        return ASRResult(full_text, segments, language)


class WhisperBackend:
    """Faster-whisper based ASR backend (alternative)."""

    def __init__(self, model_size: str = "large-v3"):
        self._model = None
        self._model_size = model_size

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self._model_size, device="cuda", compute_type="float16"
        )

    def transcribe(self, audio_path: str, language: str = "zh") -> ASRResult:
        self._ensure_loaded()
        segments_iter, info = self._model.transcribe(
            audio_path, language=language, word_timestamps=True, beam_size=5
        )
        segments = []
        for seg in segments_iter:
            words = []
            for w in seg.words or []:
                words.append(
                    {
                        "word": w.word.strip(),
                        "start": w.start,
                        "end": w.end,
                        "confidence": w.probability,
                    }
                )
            segments.append(
                {
                    "text": seg.text.strip(),
                    "start": seg.start,
                    "end": seg.end,
                    "words": words,
                }
            )
        full_text = " ".join(s["text"] for s in segments)
        return ASRResult(full_text, segments, language)


_default_backend: Optional[ASRBackend] = None


def get_asr_backend(backend_name: str = "funasr") -> ASRBackend:
    global _default_backend
    if _default_backend is not None:
        return _default_backend
    if backend_name == "funasr":
        _default_backend = FunASRBackend()
    elif backend_name == "whisper":
        _default_backend = WhisperBackend()
    else:
        raise ValueError(f"Unknown ASR backend: {backend_name}")
    return _default_backend


def transcribe(
    audio_path: str,
    language: str = "zh",
    backend_name: str = "funasr",
    output_path: str = "asr_result.json",
) -> ASRResult:
    """
    Transcribe audio file with word-level timestamps. The result is saved as JSON.
    Args:
        audio_path: Path to the input audio file.
        language: Language of the audio (default: zh)
        backend_name: ASR backend to use (funasr or whisper, default: funasr)
        output_path: Path to save the ASR result as JSON (default: asr_result.json)
    Returns: None (result is saved to output_path)
    """
    backend = get_asr_backend(backend_name)
    result = backend.transcribe(audio_path, language)
    result_dict = result.to_dict()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)


# uv run -m skills.视频入库.scripts.asr  data/我记得.mp3 --language zh --backend funasr --output asr_result.json
if __name__ == "__main__":
    # Example usage
    # audio_file = "data/我记得.mp3"
    # result = transcribe(audio_file, language="zh", backend_name="funasr")
    # print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    # ipdb.set_trace()

    parser = argparse.ArgumentParser(description="ASR tool")
    parser.add_argument("audio_path", type=str, help="Path to the audio file")
    parser.add_argument(
        "--language", type=str, default="zh", help="Language of the audio (default: zh)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="funasr",
        help="ASR backend to use (funasr or whisper, default: funasr)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="asr_result.json",
        help="Path to save the ASR result as JSON",
    )
    args = parser.parse_args()
    transcribe(
        args.audio_path,
        language=args.language,
        backend_name=args.backend,
        output_path=args.output,
    )

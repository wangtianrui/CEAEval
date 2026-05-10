"""Data-side utilities for CEAeval inference.

This module groups together everything that the inference entry points
need in order to prepare a single sample for the Qwen2.5-Omni-Thinker
scorer:

* ``Qwen2_5OmniProcessor``  – multimodal processor (audio / text / chat
  template).  Copied verbatim from the training repo so that its
  ``apply_chat_template`` stays exactly in sync with training time.
* small audio / text helpers (``load_audio_ffmpeg``, ``fix_hyphen_spacing``,
  ``slugify_name``).
* ``build_scorer_messages`` – canonical ZH / EN chat construction that
  feeds the scorer; migrated from the now-removed ``dataset.py``.
* ``BiasConfig`` / ``make_bias_config`` – small wrapper over the
  attention-bias sentinel values used by the model's custom
  ``generate()``.

The module is intentionally free of any training / RL / preprocessing
code path.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import *  # noqa: F401,F403  (ProcessorMixin et al.)
from transformers.tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput


# ---------------------------------------------------------------------------
# 1) Low-level helpers
# ---------------------------------------------------------------------------
def fix_hyphen_spacing(text: str) -> str:
    """Collapse ``word - word`` into ``word-word``.

    The training data occasionally emits labels such as
    ``Relaxed - type`` that come back as ``Relaxed-type`` after this
    pass; applying it uniformly keeps the prompt deterministic.
    """
    if text is None:
        return ""
    return re.sub(r"(\w)\s*-\s*(\w)", r"\1-\2", text)


def slugify_name(name: str) -> str:
    """Turn a story/file name into a path-safe slug (training-compat)."""
    s = re.sub(r"[\\/:\*\?\"<>\|\s]+", "_", (name or "").strip())
    return s[:255]


def _is_m4a_path(path: str) -> bool:
    """Cheap suffix check — good enough as a fast-path skip."""
    return os.path.splitext(path)[1].lower() in (".m4a", ".mp4", ".aac")


def _transcode_to_m4a_bytes(path: str) -> bytes:
    """Re-encode any audio file into an in-memory M4A (AAC-in-MP4) blob.

    The training data was stored as ``.m4a``; routing every inference
    sample through the *same* AAC encode → decode chain guarantees the
    PCM that reaches the Whisper feature extractor matches the
    training-time distribution byte-for-byte (modulo the encoder being
    deterministic).  Nothing is written to disk.
    """
    cmd = [
        os.environ.get("FFMPEG_EXE", "ffmpeg"),
        "-nostdin", "-y",
        "-i", path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "aac",
        "-b:a", "64k",
        "-movflags", "frag_keyframe+empty_moov+faststart",
        "-f", "ipod",          # m4a container
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True)
    return proc.stdout


def _decode_pcm(input_arg: Union[str, bytes], sr: int, mono: bool
                ) -> Tuple[np.ndarray, int]:
    """Decode either a path or raw bytes into float32 PCM via ffmpeg."""
    cmd = [
        os.environ.get("FFMPEG_EXE", "ffmpeg"),
        "-nostdin",
    ]
    if isinstance(input_arg, (bytes, bytearray)):
        cmd += ["-i", "pipe:0"]
        stdin_data = bytes(input_arg)
    else:
        cmd += ["-i", input_arg]
        stdin_data = None
    cmd += [
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ar", str(sr),
        "-ac", "1" if mono else "2",
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        input=stdin_data,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        check=True,
    )
    audio = np.frombuffer(proc.stdout, np.float32)
    if mono:
        return audio, sr
    return audio.reshape(-1, 2).T, sr


# Whether to force every input audio through the training-time M4A/AAC
# codec before decoding to PCM.  Default ON to stay byte-compatible
# with the training distribution.  Override with ``CEAEVAL_FORCE_M4A=0``.
_FORCE_M4A_DEFAULT = os.environ.get("CEAEVAL_FORCE_M4A", "1").lower() not in (
    "0", "false", "no", ""
)


def load_audio_ffmpeg(
    path: str,
    sr: int = 16000,
    mono: bool = True,
    force_m4a: Optional[bool] = None,
) -> Tuple[np.ndarray, int]:
    """Decode any audio file into 16 kHz mono float32 PCM.

    By default (``force_m4a=True``) inputs that are not already
    ``.m4a/.mp4/.aac`` are first re-encoded to an in-memory M4A (AAC)
    blob and then decoded to PCM, so the codec round-trip matches the
    one applied to the training data.  Files that are already AAC pass
    through a single decode (a re-encode would only add more loss).

    Set ``force_m4a=False`` (or env ``CEAEVAL_FORCE_M4A=0``) to skip the
    round-trip and decode directly — useful when feeding lossless test
    material or when running outside the training distribution.
    """
    if force_m4a is None:
        force_m4a = _FORCE_M4A_DEFAULT

    if force_m4a and not _is_m4a_path(path):
        m4a_bytes = _transcode_to_m4a_bytes(path)
        return _decode_pcm(m4a_bytes, sr=sr, mono=mono)

    return _decode_pcm(path, sr=sr, mono=mono)


# Alias kept for backward compatibility with older scripts importing
# ``load_m4a_ffmpeg`` from this module.
load_m4a_ffmpeg = load_audio_ffmpeg


def normalise_novel_context(ctx: Union[str, Iterable[str], None]) -> str:
    """Accept either a string or a list of lines and return a single string."""
    if ctx is None:
        return ""
    if isinstance(ctx, str):
        return ctx.strip()
    # assume iterable of strings
    lines = [str(x).rstrip() for x in ctx if x is not None]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# 2) Prompt / message construction for the Qwen2.5-Omni-Thinker scorer
# ---------------------------------------------------------------------------
DEFAULT_SYS_PROMPT = (
    "你是一个用于评估语音表达质量的模型。"
    "|You are a model designed to evaluate the quality of speech expressiveness."
)

_PLACEHOLDER_ASSISTANT = "<think>placeholder</think><score>0.0</score>"


def build_scorer_messages(
    audio_path: str,
    ideal: dict,
    lang: str = "en",
    sys_prompt: str = DEFAULT_SYS_PROMPT,
) -> List[dict]:
    """Build the 3-turn chat consumed by ``Qwen2_5OmniProcessor.apply_chat_template``.

    The structure mirrors what the training / evaluation code used so
    that the tokenizer yields identical prompts at inference time.

    Parameters
    ----------
    audio_path
        Path to the target audio segment (wav / mp3 / m4a / ...).
    ideal
        Dict with keys ``emotion`` / ``rhythm`` / ``intonation`` /
        ``sound_effects``.  In the reference pipeline these come from a
        Qwen3-8B labeller; see ``qwen3_ideal_labeler.py``.
    lang
        ``"zh"`` or ``"en"`` — selects the prompt template.  ``sys_prompt``
        may contain ZH and EN halves separated by ``"|"`` (exact
        same convention as training time).
    sys_prompt
        The system prompt; the "|" delimiter is respected.
    """
    sys_zh, sys_en = (sys_prompt.split("|", 1) + [""])[:2]
    sys_base = sys_zh if lang == "zh" else (sys_en or sys_zh)

    emo = fix_hyphen_spacing(ideal.get("emotion", ""))
    rhy = fix_hyphen_spacing(ideal.get("rhythm", ""))
    ito = fix_hyphen_spacing(ideal.get("intonation", ""))
    eff = fix_hyphen_spacing(ideal.get("sound_effects")
                             or ("正常说话" if lang == "zh" else "Normal speaking"))

    if lang == "zh":
        user_prompt = (
            "请根据情感、节奏、语调和音效匹配度，对该语音进行 0.0–5.0 分评分。\n"
            "评分标准：\n"
            "0–1 分：完全不匹配\n"
            "1–2 分：较弱或平淡\n"
            "2–3 分：基本符合要求\n"
            "3–4 分：整体良好但有轻微不足\n"
            "4–5 分：表现优秀、生动贴合\n"
            f"理想表现：情感={emo}、节奏={rhy}、语调={ito}、音效={eff}"
        )
        sys_text = sys_base + "在给出分数前，请逐步思考并解释你的判断。"
    else:
        user_prompt = (
            "Rate this speech (0.0–5.0) by emotion, rhythm, intonation, and effect match.\n"
            "Score guide:\n"
            "0–1: mismatched\n"
            "1–2: weak or monotone\n"
            "2–3: roughly matches\n"
            "3–4: good with small flaws\n"
            "4–5: excellent and vivid\n"
            f"Ideal Performance: emotion={emo}, rhythm={rhy}, intonation={ito}, effect={eff}"
        )
        if not sys_base.endswith(" "):
            sys_base = sys_base + " "
        sys_text = sys_base + "Think step by step and explain before giving scores."

    return [
        {"role": "system",    "content": [{"type": "text", "text": sys_text}]},
        {"role": "user",      "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text",  "text": user_prompt},
        ]},
        {"role": "assistant", "content": [
            {"type": "text",  "text": _PLACEHOLDER_ASSISTANT},
        ]},
    ]


def build_scorer_inputs(processor, messages: List[dict]):
    """Tokenise + feature-extract the given chat for ``generate()``.

    Drops the trailing assistant placeholder and adds the generation
    prompt, matching the behaviour of the training-time dataset.
    """
    text_prompt = processor.apply_chat_template(
        messages[:-1], add_generation_prompt=True, tokenize=False
    )
    sr = processor.feature_extractor.sampling_rate
    audio_path = messages[1]["content"][0]["audio"]
    audio, _ = load_audio_ffmpeg(audio_path, sr=sr, mono=True)
    return processor(
        text=[text_prompt],
        audio=[audio],
        return_tensors="pt",
        sampling_rate=sr,
        padding=True,
    )


# ---------------------------------------------------------------------------
# 3) Attention-bias config (keeps the model's custom generate() happy)
# ---------------------------------------------------------------------------
# Sentinel values that tell the patched GatedAttenQwen2_5omnithinker /
# DynamicAttenQwen2_5omnithinker to use its *learnt* gate rather than the
# hand-set focus/suppress/instruct constants.  Must stay in sync with
# ``qwen2_5_omni/modeling_qwen2_5_omni.py``.
_DYNAMIC_FOCUS    = -99.0
_DYNAMIC_SUPPRESS = -100.0
_DYNAMIC_INSTRUCT = -101.0


@dataclass
class BiasConfig:
    audio_token_id: int
    score_start_id: int
    score_end_id: int
    focus_audio_start_id: int = -777
    focus_audio_end_id:   int = -777
    focus_strength:    float = -1.0
    suppress_strength: float = -2.0
    instruct_strength: float = -3.0

    def to_dict(self) -> dict:
        return dict(
            audio_token_id       = self.audio_token_id,
            score_start_id       = self.score_start_id,
            score_end_id         = self.score_end_id,
            focus_audio_start_id = self.focus_audio_start_id,
            focus_audio_end_id   = self.focus_audio_end_id,
            focus_strength       = self.focus_strength,
            suppress_strength    = self.suppress_strength,
            instruct_strength    = self.instruct_strength,
        )


def make_bias_config(
    processor,
    *,
    gated_fc: bool = True,
    dynamic_fac: bool = True,
    inner_strength: bool = True,
    focus_strength: float = -1.0,
    suppress_strength: float = -2.0,
    instruct_strength: float = -3.0,
) -> dict:
    """Assemble the per-step ``bias_config`` attached to the model before
    ``generate()``.  The defaults reflect the deployed training recipe
    (``--gated_fc --dynamic_fac --inner_strength``)."""
    tok = processor.tokenizer
    if inner_strength:
        focus_start = tok.convert_tokens_to_ids("<focus_audio>")
        focus_end   = tok.convert_tokens_to_ids("</focus_audio>")
    else:
        focus_start = focus_end = -777

    if dynamic_fac or gated_fc:
        fs, ss, is_ = _DYNAMIC_FOCUS, _DYNAMIC_SUPPRESS, _DYNAMIC_INSTRUCT
    else:
        fs, ss, is_ = focus_strength, suppress_strength, instruct_strength

    return BiasConfig(
        audio_token_id       = tok.convert_tokens_to_ids("<|AUDIO|>"),
        score_start_id       = tok.convert_tokens_to_ids("<score>"),
        score_end_id         = tok.convert_tokens_to_ids("</score>"),
        focus_audio_start_id = focus_start,
        focus_audio_end_id   = focus_end,
        focus_strength       = fs,
        suppress_strength    = ss,
        instruct_strength    = is_,
    ).to_dict()


# ---------------------------------------------------------------------------
# 4) Qwen2.5-Omni processor (unchanged from training repo)
# ---------------------------------------------------------------------------
class Qwen2_5_OmniVideosKwargs(VideosKwargs):
    fps: Optional[list[Union[int, float]]] = None
    use_audio_in_video: Optional[bool] = None
    seconds_per_chunk: Optional[float] = None
    position_id_per_seconds: Optional[int] = None
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


class Qwen2_5_OmniImagesKwargs(ImagesKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


class Qwen2_5OmniProcessorKwargs(ProcessingKwargs, total=False):
    videos_kwargs: Qwen2_5_OmniVideosKwargs
    images_kwargs: Qwen2_5_OmniImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "videos_kwargs": {
            "seconds_per_chunk": 2.0,
            "position_id_per_seconds": 25,
            "use_audio_in_video": False,
            "min_pixels": 128 * 28 * 28,
            "max_pixels": 768 * 28 * 28,
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": "max_length",
            "return_attention_mask": True,
        },
    }


Qwen2_5OmniProcessorKwargs.__annotations__["videos_kwargs"] = Qwen2_5_OmniVideosKwargs
Qwen2_5OmniProcessorKwargs.__annotations__["images_kwargs"] = Qwen2_5_OmniImagesKwargs


class Qwen2_5OmniProcessor(ProcessorMixin):
    """Qwen2.5-Omni multimodal processor.

    This class is identical to the one used during training so that the
    chat template, audio-token layout and special-token padding stay
    byte-for-byte compatible with the checkpoints.
    """

    attributes = ["image_processor", "video_processor", "feature_extractor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self, image_processor=None, video_processor=None, feature_extractor=None, tokenizer=None, chat_template=None
    ):
        super().__init__(image_processor, video_processor, feature_extractor, tokenizer, chat_template=chat_template)
        self.image_token = self.tokenizer.image_token
        self.audio_token = self.tokenizer.audio_token
        self.video_token = self.tokenizer.video_token
        self.vision_bos_token = self.tokenizer.vision_bos_token
        self.vision_eos_token = self.tokenizer.vision_eos_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token
        self.prompt_error_loged = False

    # ------------------------------------------------------------------ __call__
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        images: Optional[ImageInput] = None,
        videos: Optional = None,
        audio: Optional[AudioInput] = None,
        **kwargs: Unpack[Qwen2_5OmniProcessorKwargs],
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            Qwen2_5OmniProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        seconds_per_chunk = output_kwargs["videos_kwargs"].pop("seconds_per_chunk")
        position_id_per_seconds = output_kwargs["videos_kwargs"].pop("position_id_per_seconds")
        use_audio_in_video = output_kwargs["videos_kwargs"].pop("use_audio_in_video")

        if audio is not None:
            output_kwargs["audio_kwargs"]["padding"] = "max_length"
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")
            audio_inputs["input_features"] = audio_inputs.pop("input_features")
            input_lengths = (audio_inputs["feature_attention_mask"].sum(-1) - 1) // 2 + 1
            audio_lengths = iter((input_lengths - 2) // 2 + 1)
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        if images is not None:
            images_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = iter(images_inputs["image_grid_thw"])
        else:
            images_inputs = {}
            image_grid_thw = iter([])

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])

            fps = output_kwargs["videos_kwargs"].get("fps", 2.0)
            video_grid_thw = videos_inputs["video_grid_thw"]
            second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)
            videos_inputs["video_second_per_grid"] = second_per_grid_ts

            video_grid_thw = iter(video_grid_thw)
            video_second_per_grid = iter(second_per_grid_ts)
        else:
            videos_inputs = {}
            video_grid_thw = iter([])
            video_second_per_grid = iter([])

        if not isinstance(text, list):
            text = [text]
        if images is not None or videos is not None or audio is not None:
            text = self.replace_multimodal_special_tokens(
                text,
                audio_lengths,
                image_grid_thw,
                video_grid_thw,
                video_second_per_grid=video_second_per_grid,
                use_audio_in_video=use_audio_in_video,
                position_id_per_seconds=position_id_per_seconds,
                seconds_per_chunk=seconds_per_chunk,
            )
        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**texts_inputs, **images_inputs, **videos_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    # ------------------------------------------------------------------ helpers
    def replace_multimodal_special_tokens(
        self,
        text,
        audio_lengths,
        image_grid_thw,
        video_grid_thw,
        video_second_per_grid,
        use_audio_in_video,
        position_id_per_seconds,
        seconds_per_chunk,
    ):
        merge_length_image = self.image_processor.merge_size**2
        merge_length_video = self.video_processor.merge_size**2

        processed_text = []
        for sample in text:
            special_tokens = [re.escape(tok) for tok in
                              [self.audio_token, self.image_token, self.video_token]]
            pattern = "|".join(special_tokens)
            positions = sorted([(m.start(), m.group()) for m in re.finditer(pattern, sample)])
            positions.sort(key=lambda x: x[0])

            for _, special_token in positions:
                if special_token == self.audio_token:
                    sample = sample.replace(self.audio_token,
                                            "<|audio_placeholder|>" * next(audio_lengths), 1)
                elif special_token == self.image_token:
                    image_seq_length = next(image_grid_thw).prod() // merge_length_image
                    sample = sample.replace(self.image_token,
                                            "<|image_placeholder|>" * image_seq_length, 1)
                elif special_token == self.video_token:
                    if not use_audio_in_video:
                        video_seq_length = next(video_grid_thw).prod() // merge_length_video
                        sample = sample.replace(self.video_token,
                                                "<|video_placeholder|>" * video_seq_length, 1)
                    else:
                        audio_token_indices = np.arange(next(audio_lengths))
                        curr_video_grid_thw = next(video_grid_thw)
                        height = curr_video_grid_thw[1] // self.video_processor.merge_size
                        width = curr_video_grid_thw[2] // self.video_processor.merge_size
                        video_token_indices = np.arange(curr_video_grid_thw[0]).reshape(-1, 1, 1)
                        video_token_indices = np.broadcast_to(
                            video_token_indices, (video_token_indices.shape[0], height, width)
                        ).reshape(-1)
                        video_token_indices = (
                            video_token_indices * next(video_second_per_grid) * position_id_per_seconds
                        )

                        tokens_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        video_chunk_indexes = self.get_chunked_index(video_token_indices, tokens_per_chunk)
                        audio_chunk_indexes = self.get_chunked_index(audio_token_indices, tokens_per_chunk)

                        placeholder_string = self.vision_bos_token + self.audio_bos_token
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            if j < len(video_chunk_indexes):
                                video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                                placeholder_string += "<|video_placeholder|>" * video_seq_length
                            if j < len(audio_chunk_indexes):
                                audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                                placeholder_string += "<|audio_placeholder|>" * audio_seq_length
                        placeholder_string += self.audio_eos_token + self.vision_eos_token
                        sample = sample.replace(
                            self.vision_bos_token + self.video_token + self.vision_eos_token,
                            placeholder_string,
                            1,
                        )

            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            sample = sample.replace("<|image_placeholder|>", self.image_token)
            sample = sample.replace("<|video_placeholder|>", self.video_token)
            processed_text.append(sample)
        return processed_text

    def get_chunked_index(self, token_indices: np.ndarray, tokens_per_chunk: int) -> list[tuple[int, int]]:
        def _iter():
            i, start_idx = 0, 0
            current_chunk = 1
            while i < len(token_indices):
                if token_indices[i] >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))
        return list(_iter())

    # ------------------------------------------------------------------ chat template
    def apply_chat_template(self, conversations, chat_template=None, crop_infos=None, **kwargs):
        crop_infos = crop_infos or {}
        is_batched = False
        if isinstance(conversations[0], dict):
            conversations = [conversations]
            is_batched = True

        for conversation in conversations:
            if (
                conversation[0]["role"] != "system"
                or conversation[0]["content"][0]["text"]
                != "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                   "capable of perceiving auditory and visual inputs, as well as generating text and speech."
            ) and not self.prompt_error_loged:
                self.prompt_error_loged = True
        if is_batched:
            conversations = conversations[0]

        if chat_template is None:
            if isinstance(self.chat_template, dict) and "default" in self.chat_template:
                chat_template = self.chat_template["default"]
            elif isinstance(self.chat_template, dict):
                raise ValueError(
                    'The processor has multiple chat templates but none of them are named "default". '
                    "You need to specify which one to use by passing the `chat_template` argument."
                )
            elif self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "Cannot use apply_chat_template because this processor does not have a chat template."
                )
        else:
            if isinstance(self.chat_template, dict) and chat_template in self.chat_template:
                chat_template = self.chat_template[chat_template]

        is_tokenizers_fast = hasattr(self, "tokenizer") and self.tokenizer.__class__.__name__.endswith("Fast")

        if kwargs.get("continue_final_message", False):
            if kwargs.get("add_generation_prompt", False):
                raise ValueError(
                    "continue_final_message and add_generation_prompt are not compatible."
                )
            if kwargs.get("return_assistant_tokens_mask", False):
                raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

        if kwargs.get("return_assistant_tokens_mask", False):
            if not is_tokenizers_fast:
                raise ValueError("`return_assistant_tokens_mask` is not possible with slow tokenizers.")
            else:
                kwargs["return_offsets_mapping"] = True

        processed_kwargs = {"mm_load_kwargs": {}, "template_kwargs": {}}
        for kwarg_type in processed_kwargs:
            for key in AllKwargsForChatTemplate.__annotations__[kwarg_type].__annotations__:
                kwarg_type_defaults = AllKwargsForChatTemplate.__annotations__[kwarg_type]
                default_value = getattr(kwarg_type_defaults, key, None)
                value = kwargs.pop(key, default_value)
                if value is not None and not isinstance(value, dict):
                    processed_kwargs[kwarg_type][key] = value

        kwargs.pop("video_load_backend", None)
        processed_kwargs["template_kwargs"].update(kwargs)

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            is_batched = True
            conversations = conversation
        else:
            is_batched = False
            conversations = [conversation]

        tokenize = processed_kwargs["template_kwargs"].pop("tokenize", False)
        return_dict = processed_kwargs["template_kwargs"].pop("return_dict", False)
        mm_load_kwargs = processed_kwargs["mm_load_kwargs"]

        if tokenize:
            batch_images, batch_videos = [], []
            batch_audios = []
            for m_idx, conversation in enumerate(conversations):
                images, videos = [], []
                for e_idx, message in enumerate(conversation):
                    visuals = [content for content in message["content"]
                               if content["type"] in ["image", "video"]]
                    audio_fnames = [
                        content[key]
                        for content in message["content"]
                        for key in ["audio", "url", "path"]
                        if key in content and content["type"] == "audio"
                    ]
                    image_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["image", "url", "path", "base64"]
                        if key in vision_info and vision_info["type"] == "image"
                    ]
                    images.extend(image_fnames)
                    video_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["video", "url", "path"]
                        if key in vision_info and vision_info["type"] == "video"
                    ]
                    videos.extend(video_fnames)

                    if not mm_load_kwargs["load_audio_from_video"]:
                        for fname in audio_fnames:
                            temp = load_audio_ffmpeg(fname, sr=mm_load_kwargs["sampling_rate"])[0]
                            if fname in crop_infos:
                                temp = temp[crop_infos[fname]["start_sample"]:crop_infos[fname]["end_sample"]]
                            batch_audios.append(temp)
                    else:
                        for fname in video_fnames:
                            temp = load_audio_ffmpeg(fname, sr=mm_load_kwargs["sampling_rate"])[0]
                            if fname in crop_infos:
                                temp = temp[crop_infos[fname]["start_sample"]:crop_infos[fname]["end_sample"]]
                            batch_audios.append(temp)

                batch_images.append(images)
                batch_videos.append(videos)

        prompt, generation_indices = render_jinja_template(
            conversations=conversations,
            chat_template=chat_template,
            **processed_kwargs["template_kwargs"],
            **self.tokenizer.special_tokens_map,
        )

        if not is_batched:
            prompt = prompt[0]

        if tokenize:
            single_prompt = prompt[0] if is_batched else prompt
            if self.tokenizer.bos_token is not None and single_prompt.startswith(self.tokenizer.bos_token):
                kwargs["add_special_tokens"] = False

            if "do_sample_frames" not in kwargs and (
                kwargs.get("fps") is not None or kwargs.get("num_frames") is not None
            ):
                kwargs["do_sample_frames"] = True

            images_exist = any((im is not None) for im_list in batch_images for im in im_list)
            videos_exist = any((vid is not None) for vid_list in batch_videos for vid in vid_list)
            out = self(
                text=prompt,
                images=batch_images if images_exist else None,
                videos=batch_videos if videos_exist else None,
                audio=batch_audios if batch_audios else None,
                **kwargs,
            )
            if return_dict:
                if processed_kwargs["template_kwargs"].get("return_assistant_tokens_mask", False):
                    assistant_masks = []
                    offset_mapping = out.pop("offset_mapping")
                    input_ids = out["input_ids"]
                    for i in range(len(input_ids)):
                        current_mask = [0] * len(input_ids[i])
                        offsets = offset_mapping[i]
                        offset_starts = [start for start, end in offsets]
                        for assistant_start_char, assistant_end_char in generation_indices[i]:
                            start_pos = bisect.bisect_left(offset_starts, assistant_start_char)  # noqa: F821
                            end_pos = bisect.bisect_left(offset_starts, assistant_end_char)  # noqa: F821
                            if not (
                                start_pos >= 0
                                and offsets[start_pos][0] <= assistant_start_char < offsets[start_pos][1]
                            ):
                                continue
                            for token_id in range(start_pos, end_pos if end_pos else len(input_ids[i])):
                                current_mask[token_id] = 1
                        assistant_masks.append(current_mask)
                    out["assistant_masks"] = assistant_masks
                    out.convert_to_tensors(tensor_type=kwargs.get("return_tensors"))
                return out
            else:
                return out["input_ids"]
        return prompt

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + image_processor_input_names
                + ["feature_attention_mask"]
                + ["video_second_per_grid"]
            )
        )

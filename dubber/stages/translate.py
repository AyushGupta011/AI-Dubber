"""Stage 5: Neural Machine Translation with Contextual Batches & Memory Safety."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict
from dubber.config import DubberConfig
from dubber.models import Segment, TranslatedSegment
from dubber.errors import TranslationError, UnsupportedLanguageError
from dubber.utils.gpu import print_gpu_status, cleanup_gpu_memory, get_gpu_info
from dubber.utils.logger import logger


class BaseTranslator(ABC):
    """Abstract interface for machine translation backends."""

    @abstractmethod
    def translate_batch(self, texts: List[str], src_lang: str, dst_lang: str = "en") -> List[str]:
        """Translates a batch of source texts into target language."""
        pass

    @abstractmethod
    def condense_text(self, text: str, src_lang: str, dst_lang: str = "en") -> str:
        """Produces a shorter, concise translation for time-fitting constraints."""
        pass


class MarianTranslator(BaseTranslator):
    """Helsinki-NLP MarianMT translation backend (< 1.5 GB VRAM)."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._load_model()

    def _load_model(self):
        from transformers import MarianMTModel, MarianTokenizer
        import torch

        logger.info(f"Loading MarianMT model '{self.model_name}' on device '{self.device}'...")
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

    def translate_batch(self, texts: List[str], src_lang: str, dst_lang: str = "en") -> List[str]:
        import torch

        if not texts:
            return []
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=4,
                max_length=512,
                early_stopping=True,
            )

        decoded = [self.tokenizer.decode(t, skip_special_tokens=True).strip() for t in outputs]
        return decoded

    def condense_text(self, text: str, src_lang: str, dst_lang: str = "en") -> str:
        """Generates a concise translation using length penalty."""
        import torch

        inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=4,
                max_length=64,
                length_penalty=0.6,
                early_stopping=True,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def close(self):
        """Releases model and clears GPU memory."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        cleanup_gpu_memory()


class NLLBTranslator(BaseTranslator):
    """NLLB-200-1.3B backend for 200+ languages (Better translation quality)."""

    # Language code mappings to Flores-200
    LANG_MAP: Dict[str, str] = {
        "hi": "hin_Deva",
        "hin": "hin_Deva",
        "hindi": "hin_Deva",
        "fr": "fra_Latn",
        "fre": "fra_Latn",
        "fra": "fra_Latn",
        "french": "fra_Latn",
        "de": "deu_Latn",
        "ger": "deu_Latn",
        "deu": "deu_Latn",
        "german": "deu_Latn",
        "es": "spa_Latn",
        "spa": "spa_Latn",
        "spanish": "spa_Latn",
        "en": "eng_Latn",
        "eng": "eng_Latn",
        "english": "eng_Latn",
    }

    def __init__(self, model_name: str = "facebook/nllb-200-1.3B", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._load_model()

    def _load_model(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        logger.info(f"Loading NLLB model '{self.model_name}' on device '{self.device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

    def _map_lang(self, lang: str) -> str:
        return self.LANG_MAP.get(lang.lower().strip(), "hin_Deva")

    def translate_batch(self, texts: List[str], src_lang: str, dst_lang: str = "en") -> List[str]:
        import torch

        if not texts:
            return []
        src_code = self._map_lang(src_lang)
        tgt_code = self._map_lang(dst_lang)

        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=4,
                early_stopping=True,
            )

        decoded = [self.tokenizer.decode(t, skip_special_tokens=True).strip() for t in outputs]
        return decoded

    def condense_text(self, text: str, src_lang: str, dst_lang: str = "en") -> str:
        import torch

        src_code = self._map_lang(src_lang)
        tgt_code = self._map_lang(dst_lang)
        self.tokenizer.src_lang = src_code

        inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=64,
                length_penalty=0.6,
                num_beams=4,
                early_stopping=True,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def close(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        cleanup_gpu_memory()


def get_translator_for_language(src_lang: str, device: str = "cuda") -> BaseTranslator:
    """Selects the optimal, lightweight translation model based on source language."""
    norm_lang = src_lang.lower().strip()

    # German -> English: Helsinki-NLP/opus-mt-de-en
    if norm_lang in ["de", "ger", "deu", "german"]:
        return MarianTranslator("Helsinki-NLP/opus-mt-de-en", device=device)

    # French -> English: Helsinki-NLP/opus-mt-fr-en
    if norm_lang in ["fr", "fre", "fra", "french"]:
        return MarianTranslator("Helsinki-NLP/opus-mt-fr-en", device=device)

    # Hindi -> English: Try opus-mt-hi-en or NLLB
    if norm_lang in ["hi", "hin", "hindi"]:
        try:
            return MarianTranslator("Helsinki-NLP/opus-mt-hi-en", device=device)
        except Exception:
            return NLLBTranslator(device=device)

    # Fallback to NLLB-200 1.3B for any other language (better quality than 600M)
    return NLLBTranslator(model_name="facebook/nllb-200-1.3B", device=device)


def translate_segments(
    config: DubberConfig,
    segments: List[Segment],
    detected_lang: str,
) -> List[TranslatedSegment]:
    """
    Translates sentence segments with contextual batching and memory safety.
    Saves translations.json artifact.
    """
    translations_path = config.translations_path

    # Resumption check
    if config.resume and translations_path.exists():
        try:
            with open(translations_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            trans_segments = [TranslatedSegment.from_dict(t) for t in data]
            if trans_segments:
                logger.info(f"Resume: Loaded {len(trans_segments)} translations from {translations_path}")
                return trans_segments
        except Exception:
            logger.warning("Existing translations file invalid, re-translating...")

    src_lang = config.src_lang or detected_lang

    if getattr(config, "whisper_task", "transcribe") == "translate" or src_lang.lower().strip() in ["en", "eng", "english"]:
        logger.info("Whisper already translated text (or source is English). Bypassing NMT...")
        trans_segments = [
            TranslatedSegment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                src_text=seg.text,
                dst_text=seg.text,
                speaker_id=seg.speaker_id,
                estimated_gender=seg.estimated_gender,
            )
            for seg in segments
        ]
        with open(translations_path, "w", encoding="utf-8") as f:
            json.dump([t.to_dict() for t in trans_segments], f, indent=2, ensure_ascii=False)
        return trans_segments

    logger.info(f"Translating {len(segments)} segments from '{src_lang}' to '{config.target_lang}'...")

    # Hardware check
    print_gpu_status(stage_name="Translate")
    is_cuda_avail, _, _, _ = get_gpu_info()
    target_device = "cuda" if (config.device in ["auto", "cuda"] and is_cuda_avail) else "cpu"

    translator = get_translator_for_language(src_lang, device=target_device)
    translated_segments: List[TranslatedSegment] = []

    try:
        # Batch translation in chunks of 32
        batch_size = 32
        texts = [s.text for s in segments]
        dst_texts: List[str] = []

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            translated_chunk = translator.translate_batch(chunk, src_lang=src_lang, dst_lang=config.target_lang)
            dst_texts.extend(translated_chunk)

        for seg, dst_t in zip(segments, dst_texts):
            translated_segments.append(
                TranslatedSegment(
                    id=seg.id,
                    start=seg.start,
                    end=seg.end,
                    src_text=seg.text,
                    dst_text=dst_t,
                    speaker_id=seg.speaker_id,
                    estimated_gender=seg.estimated_gender,
                )
            )

        logger.info(f"Translation complete: {len(translated_segments)} segments translated.")

    finally:
        if hasattr(translator, "close"):
            translator.close()
        cleanup_gpu_memory()

    # Save artifact
    with open(translations_path, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in translated_segments], f, indent=2, ensure_ascii=False)

    return translated_segments

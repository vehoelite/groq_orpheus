"""
Setting up TTS entity.
"""
from __future__ import annotations
import logging
import os
import time
import asyncio
from asyncio import CancelledError

from homeassistant.components.tts import TextToSpeechEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from .const import (
    CONF_API_KEY,
    CONF_MODEL,
    CONF_VOICE,
    CONF_URL,
    DOMAIN,
    UNIQUE_ID,
    CONF_CHIME_ENABLE,
    CONF_CHIME_SOUND,
    CONF_NORMALIZE_AUDIO,
    CONF_CACHE_SIZE,
    DEFAULT_CACHE_SIZE,
)
from .groqtts_engine import GroqTTSEngine

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    api_key = config_entry.data.get(CONF_API_KEY)
    engine = GroqTTSEngine(
        api_key,
        config_entry.data[CONF_VOICE],
        config_entry.data[CONF_MODEL],
        config_entry.data[CONF_URL],
        cache_max=config_entry.options.get(CONF_CACHE_SIZE, DEFAULT_CACHE_SIZE),
    )
    async_add_entities([GroqTTSEntity(hass, config_entry, engine)])

class GroqTTSEntity(TextToSpeechEntity):
    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, hass: HomeAssistant, config: ConfigEntry, engine: GroqTTSEngine) -> None:
        self.hass = hass
        self._engine = engine
        self._config = config
        # Prefer the config entry unique_id; fall back to stored value for backward compatibility
        self._attr_unique_id = getattr(config, "unique_id", None) or config.data.get(UNIQUE_ID)
        if not self._attr_unique_id:
            self._attr_unique_id = f"{config.data.get(CONF_URL)}_{config.data.get(CONF_MODEL)}"
        # Let the registry generate the entity_id based on name/device info

    @property
    def default_language(self) -> str:
        return "en"

    @property
    def supported_options(self) -> list:
        # Must match option keys actually read from service/data
        return [CONF_CHIME_ENABLE, CONF_VOICE, CONF_NORMALIZE_AUDIO]

    @property
    def default_options(self) -> dict:
        """Advertise default options for the TTS service."""
        return {
            CONF_CHIME_ENABLE: False,
            CONF_NORMALIZE_AUDIO: False,
            CONF_VOICE: self._config.options.get(CONF_VOICE, self._config.data.get(CONF_VOICE)),
        }
        
    @property
    def supported_languages(self) -> list:
        return self._engine.get_supported_langs()

    @property
    def device_info(self) -> dict:
        return {
            "identifiers": {(DOMAIN, self._attr_unique_id)},
            "model": self._config.data.get(CONF_MODEL),
            "manufacturer": "Groq",
        }

    @property
    def name(self) -> str:
        return self._config.data.get(CONF_MODEL, "").upper()

    def _split_text(self, text: str, max_length: int = 200) -> list[str]:
        """Split text into chunks respecting the Orpheus API 200 character limit.
        
        Tries to split at sentence boundaries when possible, otherwise splits at word boundaries.
        """
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        remaining = text
        
        while len(remaining) > max_length:
            # Try to find a sentence boundary (., !, ?) within the last 50 chars of max_length
            search_start = max(0, max_length - 50)
            search_end = min(len(remaining), max_length)
            search_text = remaining[search_start:search_end]
            
            # Look for sentence endings
            sentence_end = -1
            for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                pos = search_text.rfind(punct)
                if pos != -1:
                    sentence_end = search_start + pos + len(punct.rstrip())
                    break
            
            if sentence_end > 0:
                # Split at sentence boundary
                chunk = remaining[:sentence_end].strip()
                remaining = remaining[sentence_end:].strip()
            else:
                # Try to split at word boundary
                word_boundary = remaining.rfind(' ', 0, max_length)
                if word_boundary > 0:
                    chunk = remaining[:word_boundary].strip()
                    remaining = remaining[word_boundary:].strip()
                else:
                    # Force split at max_length if no word boundary found
                    chunk = remaining[:max_length].strip()
                    remaining = remaining[max_length:].strip()
            
            if chunk:
                chunks.append(chunk)
        
        if remaining:
            chunks.append(remaining)
        
        return chunks

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict | None = None,
    ) -> tuple[str, bytes] | tuple[None, None]:
        """Generate TTS audio asynchronously and optionally merge chime or normalize."""
        overall_start = time.monotonic()

        options = options or {}

        try:
            # Orpheus API has a strict 200 character limit per request
            # We'll split longer messages and concatenate the audio
            effective_voice = options.get(
                CONF_VOICE,
                self._config.options.get(CONF_VOICE, self._config.data.get(CONF_VOICE)),
            )

            # Split message into chunks if needed
            text_chunks = self._split_text(message, max_length=200)
            _LOGGER.debug("Split message into %d chunks (total length: %d)", len(text_chunks), len(message))

            # Generate TTS for each chunk
            audio_chunks = []
            for i, chunk in enumerate(text_chunks):
                _LOGGER.debug("Creating TTS API request for chunk %d/%d (length: %d)", i + 1, len(text_chunks), len(chunk))
                api_start = time.monotonic()
                speech = await self._engine.async_get_tts(self.hass, chunk, voice=effective_voice)
                api_duration = (time.monotonic() - api_start) * 1000
                _LOGGER.debug("TTS API call for chunk %d completed in %.2f ms", i + 1, api_duration)
                audio_chunks.append(speech.content)
            
            # Concatenate all audio chunks if we have multiple
            if len(audio_chunks) > 1:
                _LOGGER.debug("Concatenating %d audio chunks", len(audio_chunks))
                # Use ffmpeg to concatenate WAV files using filter_complex
                async def concat_audio_chunks(chunks: list[bytes]) -> bytes:
                    import tempfile
                    temp_files = []
                    try:
                        # Write each chunk to a temp file
                        for i, chunk in enumerate(chunks):
                            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                            temp_file.write(chunk)
                            temp_file.close()
                            temp_files.append(temp_file.name)
                        
                        # Build ffmpeg command with filter_complex for concatenation
                        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
                        
                        # Add all input files
                        for temp_file in temp_files:
                            cmd.extend(["-i", temp_file])
                        
                        # Build concat filter - normalize all inputs to same format, then concatenate
                        # Process each input: resample to 44.1kHz and ensure stereo
                        filter_parts = []
                        for i in range(len(temp_files)):
                            # Resample to 44.1kHz, then ensure stereo (works for both mono and stereo inputs)
                            filter_parts.append(f"[{i}:a]aresample=44100,channels=2:channel_layout=stereo[ch{i}]")
                        # Concatenate all processed chunks
                        concat_inputs = "".join([f"[ch{i}]" for i in range(len(temp_files))])
                        filter_complex = ";".join(filter_parts) + f";{concat_inputs}concat=n={len(temp_files)}:v=0:a=1[out]"
                        
                        cmd.extend([
                            "-filter_complex", filter_complex,
                            "-map", "[out]",
                            "-ac", "2",  # Stereo for better Cast compatibility
                            "-ar", "44100",  # Standard sample rate for Cast
                            "-f", "wav",
                            "pipe:1",
                        ])
                        
                        process = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        stdout, stderr = await process.communicate()
                        if process.returncode != 0:
                            _LOGGER.error("ffmpeg concat error: %s", stderr.decode())
                            raise Exception("ffmpeg concat failed")
                        
                        return stdout
                    finally:
                        # Clean up temp files
                        for temp_file in temp_files:
                            try:
                                os.unlink(temp_file)
                            except Exception:
                                pass
                
                audio_content = await concat_audio_chunks(audio_chunks)
            else:
                audio_content = audio_chunks[0]

            chime_enabled = options.get(
                CONF_CHIME_ENABLE,
                self._config.options.get(CONF_CHIME_ENABLE, self._config.data.get(CONF_CHIME_ENABLE, False)),
            )
            normalize_audio = options.get(
                CONF_NORMALIZE_AUDIO,
                self._config.options.get(
                    CONF_NORMALIZE_AUDIO, self._config.data.get(CONF_NORMALIZE_AUDIO, False)
                ),
            )
            _LOGGER.debug("Chime enabled: %s", chime_enabled)
            _LOGGER.debug("Normalization option: %s", normalize_audio)

            async def run_ffmpeg(cmd, input_bytes):
                try:
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                except FileNotFoundError:
                    _LOGGER.error("ffmpeg executable not found. Please install ffmpeg or adjust PATH.")
                    raise Exception("ffmpeg not found")
                stdout, stderr = await process.communicate(input=input_bytes)
                if process.returncode != 0:
                    _LOGGER.error("ffmpeg error: %s", stderr.decode())
                    raise Exception("ffmpeg failed")
                return stdout

            # Orpheus API returns WAV format, but Home Assistant expects MP3
            # Always convert WAV to MP3, even when chime/normalization are disabled
            if chime_enabled or normalize_audio:
                if chime_enabled:
                    chime_file = self._config.options.get(
                        CONF_CHIME_SOUND, self._config.data.get(CONF_CHIME_SOUND, "threetone.mp3")
                    )
                    chime_path = os.path.join(os.path.dirname(__file__), "chime", chime_file)
                    if not os.path.exists(chime_path):
                        _LOGGER.error("Chime file not found: %s", chime_path)
                        return None, None

                    if normalize_audio:
                        cmd = [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-i",
                            chime_path,
                            "-i",
                            "pipe:0",
                            "-filter_complex",
                            "[1:a]loudnorm=I=-16:TP=-1:LRA=5[tts];[0:a][tts]concat=n=2:v=0:a=1[out]",
                            "-map",
                            "[out]",
                            "-ac",
                            "2",  # Stereo for better Cast compatibility
                            "-ar",
                            "44100",  # Standard sample rate for Cast
                            "-b:a",
                            "192k",  # Higher bitrate for better quality
                            "-f",
                            "mp3",
                            "-acodec",
                            "libmp3lame",  # Explicit codec for better compatibility
                            "pipe:1",
                        ]
                    else:
                        cmd = [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-i",
                            chime_path,
                            "-i",
                            "pipe:0",
                            "-filter_complex",
                            "[0:a][1:a]concat=n=2:v=0:a=1[out]",
                            "-map",
                            "[out]",
                            "-ac",
                            "2",  # Stereo for better Cast compatibility
                            "-ar",
                            "44100",  # Standard sample rate for Cast
                            "-b:a",
                            "192k",  # Higher bitrate for better quality
                            "-f",
                            "mp3",
                            "-acodec",
                            "libmp3lame",  # Explicit codec for better compatibility
                            "pipe:1",
                        ]
                else:
                    cmd = [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        "pipe:0",
                        "-ac",
                        "2",  # Stereo for better Cast compatibility
                        "-ar",
                        "44100",  # Standard sample rate for Cast
                        "-b:a",
                        "192k",  # Higher bitrate for better quality
                        "-af",
                        "loudnorm=I=-16:TP=-1:LRA=5",
                        "-f",
                        "mp3",
                        "-acodec",
                        "libmp3lame",  # Explicit codec for better compatibility
                        "pipe:1",
                    ]

                audio_content = await run_ffmpeg(cmd, audio_content)
            else:
                # Convert WAV to MP3 when no chime or normalization is needed
                # Use Cast-compatible settings: stereo, 44.1kHz, higher bitrate
                cmd = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    "pipe:0",
                    "-ac",
                    "2",  # Stereo for better Cast compatibility
                    "-ar",
                    "44100",  # Standard sample rate for Cast
                    "-b:a",
                    "192k",  # Higher bitrate for better quality
                    "-f",
                    "mp3",
                    "-acodec",
                    "libmp3lame",  # Explicit codec for better compatibility
                    "pipe:1",
                ]
                audio_content = await run_ffmpeg(cmd, audio_content)

            overall_duration = (time.monotonic() - overall_start) * 1000
            _LOGGER.debug("Overall TTS processing time: %.2f ms", overall_duration)
            return "mp3", audio_content

        except CancelledError:
            _LOGGER.debug("TTS task cancelled")
            return None, None
        except Exception:
            _LOGGER.exception("Unknown error in async_get_tts_audio")
        return None, None

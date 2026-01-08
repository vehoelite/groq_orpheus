"""
Constants for Groq TTS custom component
"""

DOMAIN = "groq_tts"
VERSION = "0.1"
CONF_API_KEY = "api_key"
CONF_MODEL = "model"
CONF_VOICE = "voice"
CONF_URL = "url"
UNIQUE_ID = "unique_id"

MODELS = ["canopylabs/orpheus-v1-english", "canopylabs/orpheus-arabic-saudi"]
# English voices for canopylabs/orpheus-v1-english
VOICES_ENGLISH = [
    "autumn",
    "diana",
    "hannah",
    "austin",
    "Daniel",
    "troy",
]
# Arabic voices for canopylabs/orpheus-arabic-saudi
VOICES_ARABIC = [
    "fahad",
    "sultan",
    "lulwa",
    "noura",
]
# Combined list of all available voices
VOICES = VOICES_ENGLISH + VOICES_ARABIC

CONF_CHIME_ENABLE = "chime"
CONF_CHIME_SOUND = "chime_sound"
CONF_NORMALIZE_AUDIO = "normalize_audio"
CONF_CACHE_SIZE = "cache_size"
DEFAULT_CACHE_SIZE = 256

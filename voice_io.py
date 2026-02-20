"""
Voice I/O abstraction layer for the Diet Assistant.

Provides InputProvider interface so the conversation engine can accept
input from keyboard, microphone, or (future) Alexa without changes.
"""

import re
import time
from abc import ABC, abstractmethod

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

# ── Shared constants ──────────────────────────────────────────────────────

FOOD_CONTEXT_PROMPT = (
    "Diet food logging conversation. Common foods and brands: "
    "Drizzlicious, Banza, protein bar, protein shake, "
    "cafe latte, chai, Greek yogurt, chicken breast, "
    "asparagus, quinoa, Ragu, MyFitnessPal."
)


# ── Shared VAD recording helper ──────────────────────────────────────────

def record_until_silence(
    stream, vad, sample_rate, frame_size,
    voice_start_threshold=3, silence_threshold=30, max_duration=30.0,
):
    """Record audio from an open stream until silence is detected.

    Returns raw PCM bytes, or b"" if nothing was recorded.
    """
    audio_buffer = []
    voiced_count = 0
    silent_count = 0
    recording = False
    start_time = None

    while True:
        frame_data, _ = stream.read(frame_size)
        frame_bytes = frame_data.tobytes()
        is_speech = vad.is_speech(frame_bytes, sample_rate)

        if not recording:
            if is_speech:
                voiced_count += 1
                if voiced_count >= voice_start_threshold:
                    recording = True
                    start_time = time.time()
                    audio_buffer.append(frame_bytes)
            else:
                voiced_count = 0
        else:
            audio_buffer.append(frame_bytes)
            if is_speech:
                silent_count = 0
            else:
                silent_count += 1

            if silent_count >= silence_threshold:
                break
            if time.time() - start_time >= max_duration:
                break

    return b"".join(audio_buffer), recording


def transcribe_audio(model, raw_audio, sample_rate, initial_prompt=None, beam_size=1):
    """Convert raw PCM bytes to text via faster-whisper."""
    duration = len(raw_audio) / (sample_rate * 2)
    if duration < 0.3:
        return ""

    audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = model.transcribe(
        audio_np,
        language="en",
        initial_prompt=initial_prompt,
        beam_size=beam_size,
        vad_filter=True,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


# ── Input providers ──────────────────────────────────────────────────────

class InputProvider(ABC):
    """Abstract input source. Subclassed for text, voice, Alexa, etc."""

    @abstractmethod
    def get_input(self, prompt: str = "") -> str:
        """Block until user provides input. Returns transcribed/typed text."""
        ...

    def cleanup(self):
        """Release resources (mic streams, etc.)."""
        pass


class TextInputProvider(InputProvider):
    """Standard keyboard input — wraps built-in input()."""

    def get_input(self, prompt: str = "") -> str:
        return input(prompt).strip()


class VoiceInputProvider(InputProvider):
    """Microphone input with faster-whisper STT.

    Flow: listen → detect speech via VAD → record → transcribe locally.
    """

    def __init__(
        self,
        whisper_model: str = "base",
        sample_rate: int = 16000,
        vad_aggressiveness: int = 2,
        silence_threshold_frames: int = 30,
        voice_start_frames: int = 3,
        max_duration_sec: float = 30.0,
        io_lock=None,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = 30
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.silence_threshold = silence_threshold_frames
        self.voice_start_threshold = voice_start_frames
        self.max_duration = max_duration_sec
        self.io_lock = io_lock

        print(f"  [voice] Loading faster-whisper '{whisper_model}' model...")
        self.model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
        print(f"  [voice] Model loaded. Ready for voice input.")

    def get_input(self, prompt: str = "") -> str:
        if self.io_lock:
            self.io_lock.acquire()
        try:
            return self._listen_and_transcribe(prompt)
        finally:
            if self.io_lock:
                self.io_lock.release()

    def _listen_and_transcribe(self, prompt: str) -> str:
        print(f"{prompt}[listening...]", end="", flush=True)

        with sd.InputStream(
            samplerate=self.sample_rate, channels=1,
            dtype="int16", blocksize=self.frame_size,
        ) as stream:
            raw_audio, did_record = record_until_silence(
                stream, self.vad, self.sample_rate, self.frame_size,
                self.voice_start_threshold, self.silence_threshold, self.max_duration,
            )

        if not did_record:
            return ""

        print(f"\r{prompt}[transcribing...]   ", end="", flush=True)

        text = transcribe_audio(
            self.model, raw_audio, self.sample_rate,
            initial_prompt=FOOD_CONTEXT_PROMPT,
        )

        if not text:
            print(f"\r{prompt}(too short, try again)   ")
            return ""

        print(f"\r{prompt}{text}                    ")
        return text

    def cleanup(self):
        pass


class AlwaysOnInputProvider(InputProvider):
    """Always-on wake word provider.

    Mic stays open. VAD detects speech, quick Whisper transcription checks
    for wake phrase ("hey claude"). Only returns text when wake word is found.

    Uses tiny model for fast wake detection, base model for full commands.
    """

    # Fuzzy patterns to match the wake phrase
    WAKE_PATTERNS = [
        r"\bhey\s*,?\s*claude?\b",
        r"\bhey\s*,?\s*claud\b",
        r"\bhey\s*,?\s*cloud\b",    # common misheard
        r"\bay\s*,?\s*claude?\b",    # accent variation
    ]

    def __init__(
        self,
        whisper_model: str = "base",
        wake_phrase: str = "hey claude",
        sample_rate: int = 16000,
        vad_aggressiveness: int = 2,
        silence_threshold_frames: int = 30,
        voice_start_frames: int = 3,
        max_duration_sec: float = 30.0,
        io_lock=None,
        speaker=None,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = 30
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.silence_threshold = silence_threshold_frames
        self.voice_start_threshold = voice_start_frames
        self.max_duration = max_duration_sec
        self.io_lock = io_lock
        self.speaker = speaker
        self.wake_phrase = wake_phrase.lower().strip()

        # Compile wake patterns
        self._wake_re = [re.compile(p, re.IGNORECASE) for p in self.WAKE_PATTERNS]

        # Tiny model for fast wake-word screening
        print(f"  [always-on] Loading 'tiny' model for wake detection...")
        self.wake_model = WhisperModel("tiny", device="cpu", compute_type="int8")

        # Full model for accurate command transcription
        print(f"  [always-on] Loading '{whisper_model}' model for commands...")
        self.cmd_model = WhisperModel(whisper_model, device="cpu", compute_type="int8")

        # Conversation state — when True, listen without requiring wake word
        self._conversation_active = False

        # Reminder response window — set by reminder thread to temporarily
        # accept any speech (not just wake word) for a short period
        self._reminder_response_until = 0.0  # timestamp; 0 = no window

        # Goodbye phrases that end the conversation and return to passive listening
        self._goodbye_phrases = [
            "good day", "good night", "see you", "see ya", "talk later",
            "talk to you later", "gotta go", "i'm done", "that's all",
            "that's it", "peace", "later", "cya", "goodbye", "good bye",
            "have a good", "take care", "catch you later", "im out",
            "nope that's it", "nope thats it", "that should be it",
            "bye", "quit", "exit", "sounds good", "alright thanks",
            "all right thanks", "all right sounds good", "thanks that's it",
            "thank you", "thanks bye", "ok thanks", "ok bye", "okay bye",
            "perfect thanks", "cool thanks", "great thanks", "appreciate it",
        ]

        print(f"  [always-on] Ready. Say '{wake_phrase}' to activate.")

    def open_reminder_response_window(self, seconds=15):
        """Called by reminder thread — temporarily accept any speech for N seconds."""
        self._reminder_response_until = time.time() + seconds

    def _is_goodbye(self, text: str) -> bool:
        """Check if the text is a goodbye/end-conversation phrase."""
        import string
        # Strip punctuation so "Alright, thanks." matches "alright thanks"
        lower = text.lower().strip()
        clean = lower.translate(str.maketrans("", "", string.punctuation))
        return any(phrase in clean for phrase in self._goodbye_phrases)

    def _check_wake_word(self, text: str):
        """Check if text starts with the wake phrase.

        Returns the command portion after the wake phrase, or None if no match.
        """
        lower = text.lower().strip()
        for pattern in self._wake_re:
            match = pattern.search(lower)
            if match:
                # Everything after the wake phrase is the command
                command = text[match.end():].strip().lstrip(",").strip()
                return command
        return None

    def get_input(self, prompt: str = "") -> str:
        """Block until input is ready.

        If conversation is active: listen like normal voice input (no wake word).
        If passive: wait for wake word, then activate conversation.
        """
        while True:
            if self.io_lock:
                self.io_lock.acquire()
            try:
                if self._conversation_active:
                    result = self._listen_conversation(prompt)
                else:
                    result = self._listen_for_wake(prompt)
                if result is not None:
                    return result
            finally:
                if self.io_lock:
                    self.io_lock.release()

    def _listen_conversation(self, prompt: str):
        """Active conversation mode — listen and transcribe without wake word."""
        print(f"{prompt}[listening...]", end="", flush=True)

        with sd.InputStream(
            samplerate=self.sample_rate, channels=1,
            dtype="int16", blocksize=self.frame_size,
        ) as stream:
            raw_audio, did_record = record_until_silence(
                stream, self.vad, self.sample_rate, self.frame_size,
                self.voice_start_threshold, self.silence_threshold, self.max_duration,
            )

        if not did_record:
            return None

        print(f"\r{prompt}[transcribing...]   ", end="", flush=True)
        text = transcribe_audio(
            self.cmd_model, raw_audio, self.sample_rate,
            initial_prompt=FOOD_CONTEXT_PROMPT,
        )

        if not text:
            print(f"\r{prompt}(too short, try again)   ")
            return None

        print(f"\r{prompt}{text}                    ")

        # Check if user is saying goodbye — end conversation, go back to passive
        if self._is_goodbye(text):
            self._conversation_active = False
            return text  # Let the main loop handle the goodbye message + break

        return text

    def _listen_for_wake(self, prompt: str):
        """Listen for one utterance and check for wake word.

        Returns command string if wake word found, None otherwise.
        """
        # Silently listen (no [listening...] prompt in always-on mode)
        with sd.InputStream(
            samplerate=self.sample_rate, channels=1,
            dtype="int16", blocksize=self.frame_size,
        ) as stream:
            raw_audio, did_record = record_until_silence(
                stream, self.vad, self.sample_rate, self.frame_size,
                self.voice_start_threshold, self.silence_threshold, self.max_duration,
            )

        if not did_record:
            return None

        # Quick transcription with tiny model
        text = transcribe_audio(
            self.wake_model, raw_audio, self.sample_rate,
            initial_prompt="Hey Claude. " + FOOD_CONTEXT_PROMPT,
        )

        if not text:
            return None

        # Check for wake word
        command = self._check_wake_word(text)

        # If in reminder response window, accept any speech (no wake word needed)
        if command is None and time.time() < self._reminder_response_until:
            self._reminder_response_until = 0.0  # Clear the window
            self._conversation_active = True
            # Re-transcribe with full model for accuracy
            full_text = transcribe_audio(
                self.cmd_model, raw_audio, self.sample_rate,
                initial_prompt=FOOD_CONTEXT_PROMPT,
            )
            result = full_text if full_text else text
            print(f"  {result}                    ")
            return result

        if command is None:
            # Not for us — silently ignore
            return None

        # Wake word detected — activate conversation mode
        self._conversation_active = True

        if command:
            # "Hey Claude, log my snack" → re-transcribe with full model for accuracy
            print(f"{prompt}[heard: {text}]")
            full_text = transcribe_audio(
                self.cmd_model, raw_audio, self.sample_rate,
                initial_prompt="Hey Claude. " + FOOD_CONTEXT_PROMPT,
            )
            # Extract command from full transcription
            full_command = self._check_wake_word(full_text)
            if full_command:
                print(f"\r{prompt}{full_command}                    ")
                return full_command
            # Fallback to tiny model's command
            print(f"\r{prompt}{command}                    ")
            return command
        else:
            # Just "Hey Claude" with no command — ask what they want
            print(f"\n  Assistant: Yeah?\n")
            if self.speaker:
                self.speaker.speak_and_wait("Yeah?")

            # Now listen for the actual command with full model
            print(f"{prompt}[listening...]", end="", flush=True)
            with sd.InputStream(
                samplerate=self.sample_rate, channels=1,
                dtype="int16", blocksize=self.frame_size,
            ) as stream:
                raw_audio, did_record = record_until_silence(
                    stream, self.vad, self.sample_rate, self.frame_size,
                    self.voice_start_threshold, self.silence_threshold, self.max_duration,
                )

            if not did_record:
                print(f"\r{prompt}(didn't catch that)   ")
                return None

            print(f"\r{prompt}[transcribing...]   ", end="", flush=True)
            text = transcribe_audio(
                self.cmd_model, raw_audio, self.sample_rate,
                initial_prompt=FOOD_CONTEXT_PROMPT,
            )
            if not text:
                print(f"\r{prompt}(too short)   ")
                return None

            print(f"\r{prompt}{text}                    ")
            return text

    def cleanup(self):
        pass

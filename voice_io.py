"""
Voice I/O abstraction layer for the Diet Assistant.

Provides InputProvider interface so the conversation engine can accept
input from keyboard, microphone, or (future) Alexa without changes.
"""

import re
import select
import sys
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
    max_idle_sec=None,
):
    """Record audio from an open stream until silence is detected.

    Returns raw PCM bytes, or b"" if nothing was recorded.
    """
    audio_buffer = []
    voiced_count = 0
    silent_count = 0
    recording = False
    start_time = None
    idle_start = time.time()

    while True:
        frame_data, _ = stream.read(frame_size)
        frame_bytes = frame_data.tobytes()
        is_speech = vad.is_speech(frame_bytes, sample_rate)

        if not recording:
            if max_idle_sec is not None and (time.time() - idle_start) >= max_idle_sec:
                break
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


class HybridInputProvider(InputProvider):
    """Type OR hold Option key to speak — whichever comes first.

    Runs a pynput listener in the background. While waiting for typed input,
    if the user holds Option, it records audio, transcribes, and returns text.
    """

    def __init__(self, whisper_model: str = "base"):
        import threading
        from pynput import keyboard as kb

        self._result_ready = threading.Event()
        self._voice_text = None
        self._recording = False
        self._audio_buffer = []
        self._stream = None
        self._lock = threading.Lock()
        self.last_was_voice = False  # True when last input came from Option key

        print(f"  [hybrid] Loading Whisper '{whisper_model}' model...")
        self.model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
        print(f"  [hybrid] Ready. Type or hold Option key to speak.")

        self.sample_rate = 16000
        self.blocksize = 1024

        # Start pynput listener in background
        self._listener = kb.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        from pynput import keyboard as kb
        if key in (kb.Key.alt, kb.Key.alt_l, kb.Key.alt_r):
            self._start_recording()

    def _on_release(self, key):
        from pynput import keyboard as kb
        if key in (kb.Key.alt, kb.Key.alt_l, kb.Key.alt_r):
            self._stop_recording()

    def _audio_callback(self, indata, frames, time_info, status):
        if self._recording:
            self._audio_buffer.append(indata.copy())

    def _start_recording(self):
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._audio_buffer = []

        print("\n  [recording...] ", end="", flush=True)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.blocksize,
            callback=self._audio_callback,
        )
        self._stream.start()

    def _stop_recording(self):
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            buffer = list(self._audio_buffer)

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not buffer:
            print("\r                          ", end="\r", flush=True)
            return

        audio_np = np.concatenate(buffer, axis=0)
        raw_audio = audio_np.tobytes()
        duration = len(raw_audio) / (self.sample_rate * 2)

        if duration < 0.3:
            print("\r  (too short)             ", end="\r", flush=True)
            return

        print(f"\r  [transcribing...]       ", end="", flush=True)

        text = transcribe_audio(
            self.model, raw_audio, self.sample_rate,
            initial_prompt=FOOD_CONTEXT_PROMPT,
        )

        if text and len(text.strip()) >= 2:
            print(f"\r                          ", end="\r", flush=True)
            self._voice_text = text.strip()
            self._result_ready.set()
        else:
            print(f"\r  (couldn't transcribe)   ", end="\r", flush=True)

    def get_input(self, prompt: str = "") -> str:
        """Block until user types or speaks via Option key."""
        import threading

        self._result_ready.clear()
        self._voice_text = None

        # Read keyboard in a background thread so we can also wait for voice
        typed_text = [None]

        def read_stdin():
            try:
                text = input(prompt).strip()
                typed_text[0] = text
                self._result_ready.set()
            except (EOFError, KeyboardInterrupt):
                typed_text[0] = ""
                self._result_ready.set()

        t = threading.Thread(target=read_stdin, daemon=True)
        t.start()

        # Wait for either voice or keyboard
        self._result_ready.wait()

        if self._voice_text:
            result = self._voice_text
            self._voice_text = None
            self.last_was_voice = True
            # Print what was said so it appears in the conversation
            print(f"{prompt}{result}")
            return result

        self.last_was_voice = False
        return typed_text[0] or ""

    def cleanup(self):
        if hasattr(self, '_listener'):
            self._listener.stop()
        # Suppress the stdin lock error on exit
        import os
        os._exit(0)


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
                max_idle_sec=0.8,
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
            "thats it", "thats all", "im done",
        ]

        print(f"  [always-on] Ready. Say '{wake_phrase}' to activate.")
        print("  [always-on] You can also type and press Enter at any time.")

    def open_reminder_response_window(self, seconds=15):
        """Called by reminder thread — temporarily accept any speech for N seconds."""
        self._reminder_response_until = time.time() + seconds

    def _is_goodbye(self, text: str) -> bool:
        """Check if the text is a goodbye/end-conversation phrase."""
        import string
        # Strip punctuation from both input and phrases so "that's it" matches "thats it"
        trans = str.maketrans("", "", string.punctuation)
        clean = text.lower().strip().translate(trans)
        return any(phrase.translate(trans) in clean for phrase in self._goodbye_phrases)

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

    def _poll_typed_input(self):
        """Return a typed line if available, otherwise None (non-blocking)."""
        try:
            if not sys.stdin or not sys.stdin.isatty():
                return None
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if not ready:
                return None
            line = sys.stdin.readline()
            if not line:
                return None
            text = line.strip()
            return text or None
        except Exception:
            return None

    def get_input(self, prompt: str = "") -> str:
        """Block until input is ready.

        If conversation is active: listen like normal voice input (no wake word).
        If passive: wait for wake word, then activate conversation.
        """
        while True:
            typed = self._poll_typed_input()
            if typed:
                print(f"{prompt}{typed}")
                return typed

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
                max_idle_sec=0.8,
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

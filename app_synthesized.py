import os
import numpy as np
import io # Import io for BytesIO
import base64 # Import base64 for encoding
from scipy.signal import chirp, sawtooth, square
from scipy.io.wavfile import write as write_wav # Import write_wav to create WAV data
from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Initialize Flask app and SocketIO
app = Flask(__name__)

# --- IMPORTANT: Configure Secret Key ---
# In a real production environment, use a strong, randomly generated key from environment variables.
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_secret_key_for_drum_machine')

# --- Configure SocketIO for CORS ---
# For development/testing, '*' allows connections from any origin.
# For production, replace '*' with your specific frontend domain(s) for security.
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Configure Flask-CORS for regular HTTP routes (if you add them later) ---
CORS(app)

# --- Audio Constants ---
SAMPLE_RATE = 44100  # samples per second (Hz)
BIT_DEPTH = 16       # 16-bit audio
MAX_AMP = 32767      # Max amplitude for 16-bit signed audio (short int)

# --- Sound Synthesis Functions (808-like) ---

def generate_kick(duration=0.5, attack=0.005, decay=0.3, start_freq=60, end_freq=30):
    """
    Generates an 808-style kick drum sound.
    Uses a frequency sweep (chirp) and an amplitude envelope.
    """
    num_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Frequency envelope (pitch bend down)
    freq_env = np.interp(t, [0, decay], [start_freq, end_freq])
    
    # Generate sine wave with dynamic frequency
    # Integrated phase to handle frequency changes correctly
    phase = np.cumsum(2 * np.pi * freq_env / SAMPLE_RATE)
    kick_wave = np.sin(phase)

    # Amplitude envelope
    amplitude_env = np.ones(num_samples)
    
    # Attack phase
    attack_samples = int(attack * SAMPLE_RATE)
    amplitude_env[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Decay phase
    decay_samples = int(decay * SAMPLE_RATE)
    if decay_samples > attack_samples:
        amplitude_env[attack_samples:decay_samples] *= np.linspace(1, 0, decay_samples - attack_samples)
    
    # Apply amplitude envelope
    kick_sound = kick_wave * amplitude_env
    
    # Ensure sound is within -1 to 1 range and convert to float64 for WAV
    return np.array(kick_sound, dtype=np.float64)

def generate_snare(duration=0.3, noise_mix=0.6, tone_freq=200, decay=0.2):
    """
    Generates an 808-style snare drum sound.
    Mixes white noise with a tone, both with decay.
    """
    num_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # White noise component
    noise = np.random.uniform(-1, 1, num_samples)

    # Tone component (slightly decaying)
    tone_decay_env = np.exp(-t / (decay / 4)) # Faster decay for tone
    tone = np.sin(2 * np.pi * tone_freq * t) * tone_decay_env

    # Overall amplitude envelope for the snare
    amplitude_env = np.exp(-t / decay)

    # Mix noise and tone
    snare_sound = (noise * noise_mix + tone * (1 - noise_mix)) * amplitude_env
    
    # Ensure sound is within -1 to 1 range and convert to float64 for WAV
    return np.array(snare_sound, dtype=np.float64)

def generate_hihat(duration=0.1, decay=0.05):
    """
    Generates an 808-style hi-hat sound.
    Uses filtered white noise with a sharp decay.
    """
    num_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # High-passed noise (simplified simulation by mixing high-frequency sines)
    # This is a common trick for a quick noise-like high-hat without actual filtering
    hihat_wave = np.zeros(num_samples)
    frequencies = [4000, 5000, 6000, 7000, 8000, 9000, 10000] # High frequencies
    for freq in frequencies:
        hihat_wave += np.sin(2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)
    hihat_wave /= len(frequencies) # Normalize sum

    # Amplitude envelope (very sharp decay)
    amplitude_env = np.exp(-t / decay)
    
    hihat_sound = hihat_wave * amplitude_env

    # Ensure sound is within -1 to 1 range and convert to float64 for WAV
    return np.array(hihat_sound, dtype=np.float64)


# --- Socket.IO Events ---

@socketio.on('connect')
def test_connect():
    """Handle new client connections."""
    print('Client connected:', request.sid)
    # Optionally send a 'ready' message or initial state
    emit('status', {'message': 'Connected to drum machine backend!'})

@socketio.on('disconnect')
def test_disconnect():
    """Handle client disconnections."""
    print('Client disconnected:', request.sid)

@socketio.on('trigger_drum')
def handle_trigger_drum(data):
    """
    Receives a drum type from the client, generates the corresponding sound,
    and sends the raw audio data as Base64 encoded WAV back to the client.
    """
    drum_type = data.get('type')
    
    sound_samples = np.array([]) # Initialize as empty numpy array
    if drum_type == 'kick':
        sound_samples = generate_kick()
    elif drum_type == 'snare':
        sound_samples = generate_snare()
    elif drum_type == 'hihat':
        sound_samples = generate_hihat()
    else:
        print(f"Unknown drum type received: {drum_type}")
        return # Do not emit if drum type is unknown

    # Convert normalized float samples (-1.0 to 1.0) to 16-bit PCM
    # Ensure they are within the int16 range
    audio_int16 = (sound_samples * MAX_AMP).astype(np.int16)

    # Create an in-memory binary stream
    wav_file_buffer = io.BytesIO()
    write_wav(wav_file_buffer, SAMPLE_RATE, audio_int16)
    wav_file_buffer.seek(0) # Rewind to the beginning of the buffer

    # Read the WAV data and encode it as Base64
    wav_base64 = base64.b64encode(wav_file_buffer.read()).decode('utf-8')

    # Emit the Base64 encoded WAV data
    emit('audio_data', {'type': drum_type, 'wav_base64': wav_base64, 'sample_rate': SAMPLE_RATE})
    print(f"Generated and sent {drum_type} sound as Base64 WAV to {request.sid}")


# --- Main Execution Block (for local development) ---
if __name__ == '__main__':
    # Ensure a 'templates' directory exists if you plan to serve HTML files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(script_dir, 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
        print(f"Created templates directory: {template_dir}")

    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    print(f"Running 808 Drum Machine Flask-SocketIO server locally on {host}:{port}...")
    print("For production deployment, use Gunicorn with the specified start command.")
    
    # For local testing, you can use simpleaudio to play sounds directly in Python
    # This requires: pip install simpleaudio
    try:
        import simpleaudio as sa
        print("simpleaudio detected. You can test sound generation locally by adding playback code.")
    except ImportError:
        print("simpleaudio not found. Install 'pip install simpleaudio' to test local playback.")

    socketio.run(app, debug=True, host=host, port=port, allow_unsafe_werkzeug=True)


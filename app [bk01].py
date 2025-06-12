import os
import numpy as np
import io # Import io for BytesIO
import base64 # Import base64 for encoding
# Removed scipy.signal imports as we're loading WAVs now
# from scipy.signal import chirp, sawtooth, square
from scipy.io.wavfile import write as write_wav # Still used for encoding
import soundfile as sf # New: For loading WAV files

from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Initialize Flask app and SocketIO
app = Flask(__name__)

# --- IMPORTANT: Configure Secret Key ---
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_secret_key_for_drum_machine')

# --- Configure SocketIO for CORS ---
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# --- Audio Constants ---
SAMPLE_RATE = 44100  # samples per second (Hz) - Used for loading/resampling if necessary
BIT_DEPTH = 16       # 16-bit audio
MAX_AMP = 32767      # Max amplitude for 16-bit signed audio (short int)

# --- Global dictionary to store pre-loaded and pre-encoded drum sounds ---
loaded_drum_samples_base64 = {}

def load_and_encode_wav(filepath, target_sample_rate=SAMPLE_RATE):
    """
    Loads a WAV file, converts it to target_sample_rate and float64,
    then encodes it to Base64 WAV data.
    """
    if not os.path.exists(filepath):
        print(f"WARNING: WAV file not found at {filepath}")
        return None

    try:
        data, current_samplerate = sf.read(filepath, dtype='float64')

        # If stereo, convert to mono by taking the average of channels
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample if current sample rate is different from target sample rate
        if current_samplerate != target_sample_rate:
            from scipy.signal import resample
            num_samples = int(len(data) * target_sample_rate / current_samplerate)
            data = resample(data, num_samples)
            print(f"Resampled {filepath} from {current_samplerate}Hz to {target_sample_rate}Hz")

        # Convert normalized float samples (-1.0 to 1.0) to 16-bit PCM
        audio_int16 = (data * MAX_AMP).astype(np.int16)

        # Create an in-memory binary stream
        wav_file_buffer = io.BytesIO()
        write_wav(wav_file_buffer, target_sample_rate, audio_int16)
        wav_file_buffer.seek(0) # Rewind to the beginning of the buffer

        # Read the WAV data and encode it as Base64
        wav_base64 = base64.b64encode(wav_file_buffer.read()).decode('utf-8')
        print(f"Successfully loaded and encoded {filepath}")
        return wav_base64

    except Exception as e:
        print(f"ERROR: Could not load or encode {filepath}: {e}")
        return None

# --- Pre-load drum sounds on server startup ---
def preload_drum_sounds():
    """
    Loads drum samples from WAV files into memory.
    Ensure 'sounds' directory exists in the same location as app.py
    and contains 'kick.wav', 'snare.wav', 'hihat.wav'.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sounds_dir = os.path.join(script_dir, 'sounds')

    if not os.path.exists(sounds_dir):
        os.makedirs(sounds_dir)
        print(f"Created sounds directory: {sounds_dir}")
        print("Please place 'kick.wav', 'snare.wav', and 'hihat.wav' inside this directory.")
        return

    drum_files = {
        'kick': os.path.join(sounds_dir, 'kick.wav'),
        'snare': os.path.join(sounds_dir, 'snare.wav'),
        'hihat': os.path.join(sounds_dir, 'hihat.wav'),
    }

    for drum_type, filepath in drum_files.items():
        encoded_data = load_and_encode_wav(filepath)
        if encoded_data:
            loaded_drum_samples_base64[drum_type] = encoded_data
        else:
            print(f"WARNING: Could not load {drum_type}. It might not be available.")
    
    if not loaded_drum_samples_base64:
        print("CRITICAL WARNING: No drum sounds were loaded. Ensure WAV files are present and valid.")


# --- Socket.IO Events ---

@socketio.on('connect')
def test_connect():
    """Handle new client connections."""
    print('Client connected:', request.sid)
    emit('status', {'message': 'Connected to drum machine backend!'})
    # On connect, pre-load sounds if not already loaded (useful for dev reload)
    if not loaded_drum_samples_base64:
        preload_drum_sounds()


@socketio.on('disconnect')
def test_disconnect():
    """Handle client disconnections."""
    print('Client disconnected:', request.sid)

@socketio.on('trigger_drum')
def handle_trigger_drum(data):
    """
    Receives a drum type from the client, sends the pre-encoded WAV data back.
    """
    drum_type = data.get('type')
    
    wav_base64_data = loaded_drum_samples_base64.get(drum_type)

    if wav_base64_data:
        emit('audio_data', {'type': drum_type, 'wav_base64': wav_base64_data, 'sample_rate': SAMPLE_RATE})
        print(f"Sent pre-loaded {drum_type} sound (Base64 WAV) to {request.sid}")
    else:
        print(f"ERROR: Drum sound '{drum_type}' not pre-loaded or found.")
        emit('error', {'message': f"Drum sound '{drum_type}' not available on backend."})


# --- Main Execution Block (for local development) ---
if __name__ == '__main__':
    # Ensure a 'templates' directory exists if you plan to serve HTML files (for Flask default route)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(script_dir, 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
        print(f"Created templates directory: {template_dir}")

    # Pre-load sounds immediately when the server starts
    preload_drum_sounds()

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


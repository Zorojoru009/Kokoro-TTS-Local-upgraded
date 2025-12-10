"""
Kokoro-TTS Local Generator
-------------------------
A Gradio interface for the Kokoro-TTS-Local text-to-speech system.
Supports multiple voices and audio formats, with cross-platform compatibility.

Key Features:
- Multiple voice models support (54 voices across 8 languages)
- Real-time generation with progress logging
- WAV, MP3, and AAC output formats
- Network sharing capabilities
- Cross-platform compatibility (Windows, macOS, Linux)

Dependencies:
- kokoro: Official Kokoro TTS library
- gradio: Web interface framework
- soundfile: Audio file handling
- pydub: Audio format conversion
"""

import gradio as gr
import os
import sys
import platform
from datetime import datetime
import shutil
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
import torch
import numpy as np
import argparse
import json
from typing import Union, List, Optional, Tuple, Dict, Any
from models import (
    list_available_voices, build_model,
    generate_speech, download_voice_files, EnhancedKPipeline
)
import speed_dial

# Constants
MAX_TEXT_LENGTH = 50000  # Increased from 5000 to support long texts with chunking
CHUNK_SIZE = 2000  # Size of each text chunk for processing
DEFAULT_SAMPLE_RATE = 24000
MIN_SPEED = 0.1
MAX_SPEED = 3.0
DEFAULT_SPEED = 1.0
VOICE_PRESETS_FILE = Path("voice_presets.json")

# Voice blending presets for authoritative content
DEFAULT_VOICE_PRESETS = {
    "presets": {
        "philosopher": {
            "name": "Philosopher's Voice",
            "description": "Deep, contemplative voice perfect for philosophical discourse",
            "voices": ["am_adam", "am_michael"],
            "weights": [0.7, 0.3],
            "mode": "blend"
        },
        "educator": {
            "name": "Educator's Voice",
            "description": "Clear, powerful voice for educational and scientific content",
            "voices": ["am_adam", "am_echo"],
            "weights": [0.6, 0.4],
            "mode": "blend"
        },
        "authority": {
            "name": "Transatlantic Authority",
            "description": "Blend of American confidence and British reliability",
            "voices": ["am_adam", "bm_george"],
            "weights": [0.5, 0.5],
            "mode": "blend"
        },
        "pure_deep": {
            "name": "Pure Deep Voice",
            "description": "Single voice maximum gravitas (100% Adam)",
            "voices": ["am_adam"],
            "weights": [1.0],
            "mode": "single"
        },
        "warm_santa": {
            "name": "Warm Santa Voice",
            "description": "Deep warm voice with subtle jolly undertones (80% Michael + 20% Santa)",
            "voices": ["am_michael", "am_santa"],
            "weights": [0.8, 0.2],
            "mode": "blend"
        }
    }
}

# Define path type for consistent handling
PathLike = Union[str, Path]

# Configuration validation
def validate_sample_rate(rate: int) -> int:
    """Validate sample rate is within acceptable range"""
    valid_rates = [16000, 22050, 24000, 44100, 48000]
    if rate not in valid_rates:
        print(f"Warning: Unusual sample rate {rate}. Valid rates are {valid_rates}")
        return 24000  # Default to safe value
    return rate

# Global configuration
CONFIG_FILE = Path("tts_config.json")  # Stores user preferences and paths
DEFAULT_OUTPUT_DIR = Path("outputs")    # Directory for generated audio files
SAMPLE_RATE = validate_sample_rate(24000)  # Validated sample rate

# Initialize model globally
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = None

LANG_MAP = {
    "af_": "a", "am_": "a",
    "bf_": "b", "bm_": "b",
    "jf_": "j", "jm_": "j",
    "zf_": "z", "zm_": "z",
    "ef_": "e", "em_": "e",
    "ff_": "f",
    "hf_": "h", "hm_": "h",
    "if_": "i", "im_": "i",
    "pf_": "p", "pm_": "p",
}
pipelines = {}

def get_available_voices():
    """Get list of available voice models."""
    try:
        # Initialize model to trigger voice downloads
        global model
        if model is None:
            print("Initializing model and downloading voices...")
            model = build_model(None, device)

        voices = list_available_voices()
        if not voices:
            print("No voices found after initialization. Attempting to download...")
            download_voice_files()  # Try downloading again
            voices = list_available_voices()

        print("Available voices:", voices)
        return voices
    except Exception as e:
        print(f"Error getting voices: {e}")
        return []

def get_pipeline_for_voice(voice_name: str) -> EnhancedKPipeline:
    """
    Determine the language code from the voice prefix and return the associated pipeline.
    """
    prefix = voice_name[:3].lower()
    lang_code = LANG_MAP.get(prefix, "a")
    if lang_code not in pipelines:
        print(f"[INFO] Creating pipeline for lang_code='{lang_code}'")
        pipelines[lang_code] = EnhancedKPipeline(lang_code=lang_code, model=True)
    return pipelines[lang_code]

def convert_audio(input_path: PathLike, output_path: PathLike, format: str) -> Optional[PathLike]:
    """Convert audio to specified format.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        format: Output format ('wav', 'mp3', or 'aac')

    Returns:
        Path to output file or None on error
    """
    try:
        # Normalize paths
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        # Validate input file
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # For WAV format, just return the input path
        if format.lower() == "wav":
            return input_path

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert format
        audio = AudioSegment.from_wav(str(input_path))

        # Select proper format and options
        if format.lower() == "mp3":
            audio.export(str(output_path), format="mp3", bitrate="192k")
        elif format.lower() == "aac":
            audio.export(str(output_path), format="aac", bitrate="192k")
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Verify file was created
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise IOError(f"Failed to create {format} file")

        return output_path

    except (IOError, FileNotFoundError, ValueError) as e:
        print(f"Error converting audio: {type(e).__name__}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error converting audio: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_voice_presets() -> Dict[str, Any]:
    """Load voice presets from file or use defaults"""
    if VOICE_PRESETS_FILE.exists():
        try:
            with open(VOICE_PRESETS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading voice presets: {e}. Using defaults.")
            return DEFAULT_VOICE_PRESETS
    return DEFAULT_VOICE_PRESETS

def save_voice_presets(presets: Dict[str, Any]):
    """Save voice presets to file"""
    try:
        with open(VOICE_PRESETS_FILE, 'w', encoding='utf-8') as f:
            json.dump(presets, f, indent=2, ensure_ascii=False)
        print(f"Voice presets saved to {VOICE_PRESETS_FILE}")
    except IOError as e:
        print(f"Error saving voice presets: {e}")

def create_voice_blend_preset(preset_name: str, preset_description: str, voices: List[str],
                              weights: List[float], blend_method: str = "linear") -> Tuple[bool, str]:
    """
    Create and save a custom voice blend preset.

    Args:
        preset_name: Name for the preset (e.g., "my_deep_voice")
        preset_description: Description of the blend
        voices: List of voice names (e.g., ["am_adam", "am_michael"])
        weights: List of weights for each voice (should sum to ~100)
        blend_method: "linear" or "slerp"

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Validate inputs
        if not preset_name or not preset_name.strip():
            return False, "Preset name cannot be empty"

        if not voices or len(voices) == 0:
            return False, "At least one voice must be selected"

        if len(voices) != len(weights):
            return False, "Number of voices must match number of weights"

        if not all(w > 0 for w in weights):
            return False, "All weights must be greater than 0"

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Load existing presets
        presets = load_voice_presets()

        # Check if preset already exists
        if preset_name in presets.get("presets", {}):
            return False, f"Preset '{preset_name}' already exists. Use a different name."

        # Create preset entry
        new_preset = {
            "name": preset_name.replace("_", " ").title(),
            "description": preset_description,
            "voices": voices,
            "weights": normalized_weights,
            "mode": "blend",
            "blend_method": blend_method,
            "created": datetime.now().isoformat()
        }

        # Add to presets
        if "presets" not in presets:
            presets["presets"] = {}

        presets["presets"][preset_name] = new_preset

        # Save presets
        save_voice_presets(presets)

        return True, f"Preset '{new_preset['name']}' created successfully!"

    except Exception as e:
        return False, f"Error creating preset: {str(e)}"

def delete_voice_blend_preset(preset_name: str) -> Tuple[bool, str]:
    """
    Delete a voice blend preset.

    Args:
        preset_name: Name of the preset to delete

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        presets = load_voice_presets()

        if preset_name not in presets.get("presets", {}):
            return False, f"Preset '{preset_name}' not found"

        # Check if it's a built-in preset
        built_in = ["philosopher", "educator", "authority", "pure_deep"]
        if preset_name in built_in:
            return False, f"Cannot delete built-in preset '{preset_name}'"

        del presets["presets"][preset_name]
        save_voice_presets(presets)

        return True, f"Preset '{preset_name}' deleted successfully!"

    except Exception as e:
        return False, f"Error deleting preset: {str(e)}"

def blend_voices(voice_names: List[str], weights: List[float]) -> torch.Tensor:
    """
    Blend multiple voices using weighted averaging.

    Args:
        voice_names: List of voice file names (without .pt extension)
        weights: List of weights for each voice (should sum to 1.0)

    Returns:
        Blended voice tensor
    """
    if not voice_names or not weights:
        raise ValueError("Voice names and weights cannot be empty")

    if len(voice_names) != len(weights):
        raise ValueError("Number of voices must match number of weights")

    # Normalize weights to sum to 1.0
    weights = [w / sum(weights) for w in weights]

    voices_dir = Path("voices").resolve()
    blended_voice = None

    for voice_name, weight in zip(voice_names, weights):
        voice_path = voices_dir / f"{voice_name}.pt"

        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")

        try:
            voice_data = torch.load(voice_path, weights_only=True)

            if blended_voice is None:
                blended_voice = weight * voice_data
            else:
                blended_voice = blended_voice + (weight * voice_data)

            print(f"Loaded {voice_name}: weight={weight:.2f}")
        except Exception as e:
            raise Exception(f"Error loading voice {voice_name}: {e}")

    print(f"Successfully blended {len(voice_names)} voices")
    return blended_voice

def blend_voices_slerp(voice_names: List[str], weights: List[float]) -> torch.Tensor:
    """
    Blend voices using Spherical Linear Interpolation (SLERP) for smoother transitions.
    Only works with 2 voices for interpolation. For more, uses weighted average.

    Args:
        voice_names: List of voice file names
        weights: List of blend weights

    Returns:
        Blended voice tensor
    """
    if len(voice_names) != 2 or len(weights) != 2:
        print("SLERP requires exactly 2 voices. Falling back to linear blend.")
        return blend_voices(voice_names, weights)

    voices_dir = Path("voices").resolve()

    # Load voices
    v1_path = voices_dir / f"{voice_names[0]}.pt"
    v2_path = voices_dir / f"{voice_names[1]}.pt"

    if not v1_path.exists() or not v2_path.exists():
        raise FileNotFoundError("One or both voice files not found")

    v1 = torch.load(v1_path, weights_only=True)
    v2 = torch.load(v2_path, weights_only=True)

    # Normalize weights
    t = weights[1] / sum(weights)

    # Normalize vectors
    v1_norm = torch.nn.functional.normalize(v1.unsqueeze(0), dim=1).squeeze(0)
    v2_norm = torch.nn.functional.normalize(v2.unsqueeze(0), dim=1).squeeze(0)

    # Compute angle between vectors
    dot_product = torch.clamp((v1_norm * v2_norm).sum(), -1.0, 1.0)
    theta = torch.acos(dot_product)

    # Spherical linear interpolation
    if torch.sin(theta) > 1e-6:
        w1 = torch.sin((1 - t) * theta) / torch.sin(theta)
        w2 = torch.sin(t * theta) / torch.sin(theta)
        result = w1 * v1_norm + w2 * v2_norm
    else:
        # Fallback to linear blend if vectors are nearly parallel
        result = (1 - t) * v1_norm + t * v2_norm

    print(f"SLERP blended {voice_names[0]} (t={1-t:.2f}) and {voice_names[1]} (t={t:.2f})")
    return result

def split_long_text(text: str, max_chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Split long text into chunks at natural boundaries.

    Attempts to split at sentence boundaries to maintain coherence.
    Falls back to word boundaries if a sentence is too long.

    Args:
        text: Text to split
        max_chunk_size: Maximum size of each chunk in characters

    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    current_chunk = ""

    # Split by paragraphs first
    paragraphs = text.split('\n\n')

    for paragraph in paragraphs:
        # Split paragraph by sentences (. ! ?)
        sentences = []
        current_sentence = ""

        for char in paragraph:
            current_sentence += char
            if char in '.!?':
                sentences.append(current_sentence.strip())
                current_sentence = ""

        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        # Add sentences to current chunk
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        # Add paragraph break
        if current_chunk:
            current_chunk += "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]

def generate_tts_with_logs(voice_selection: str, text: str, format: str, speed: float = 1.0,
                          use_blend: bool = False, custom_voices: Optional[List[str]] = None,
                          custom_weights: Optional[List[float]] = None) -> Optional[PathLike]:
    """Generate TTS audio with progress logging, memory management, and voice blending.

    Handles long texts (up to 50,000 characters) by splitting into chunks and processing sequentially.

    Args:
        voice_selection: Name of the voice preset or single voice to use
        text: Text to convert to speech (up to 50,000 characters)
        format: Output format ('wav', 'mp3', 'aac')
        speed: Speech speed multiplier
        use_blend: Whether to use voice blending
        custom_voices: List of voices for custom blending
        custom_weights: Weights for custom voice blending

    Returns:
        Path to generated audio file or None on error
    """
    global model
    import psutil
    import gc

    try:
        # Check available memory before processing
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)

        if available_gb < 1.0:  # Less than 1GB available
            print(f"Warning: Low memory available ({available_gb:.1f}GB). Consider closing other applications.")
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Initialize model if needed
        if model is None:
            print("Initializing model...")
            model = build_model(None, device)

        # Create output directory
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Validate input text
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        # Dynamic text length limit based on available memory
        MAX_CHARS = MAX_TEXT_LENGTH
        if available_gb < 2.0:  # Less than 2GB available
            MAX_CHARS = min(MAX_CHARS, 2000)  # Reduce limit for low memory
            print(f"Reduced text limit to {MAX_CHARS} characters due to low memory")

        if len(text) > MAX_CHARS:
            print(f"Warning: Text exceeds {MAX_CHARS} characters. Truncating to prevent memory issues.")
            text = text[:MAX_CHARS] + "..."

        # Generate base filename from text
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"tts_{timestamp}"
        wav_path = DEFAULT_OUTPUT_DIR / f"{base_name}.wav"

        # Generate speech
        print(f"\nGenerating speech for: '{text}'")
        print(f"Speed: {speed}x")

        # Handle voice blending
        voice_path = None
        blend_description = ""

        if use_blend and custom_voices and custom_weights:
            # Custom voice blending
            print(f"Using custom voice blend: {custom_voices} with weights {custom_weights}")
            blended_voice = blend_voices(custom_voices, custom_weights)
            # Save blended voice temporarily
            blend_voice_path = Path("voices").resolve() / "__blend_temp.pt"
            torch.save(blended_voice, blend_voice_path)
            voice_path = blend_voice_path
            blend_description = f"Custom blend ({', '.join(custom_voices)})"
        else:
            # Check if voice_selection is a preset
            presets = load_voice_presets()
            if voice_selection in presets.get("presets", {}):
                preset = presets["presets"][voice_selection]
                print(f"Using voice preset: {preset['name']}")
                print(f"Description: {preset['description']}")

                if preset.get("mode") == "blend":
                    blended_voice = blend_voices(preset["voices"], preset["weights"])
                    # Save blended voice temporarily
                    blend_voice_path = Path("voices").resolve() / "__blend_temp.pt"
                    torch.save(blended_voice, blend_voice_path)
                    voice_path = blend_voice_path
                    blend_description = preset["name"]
                else:
                    # Single voice from preset
                    voice_path = Path("voices").resolve() / f"{preset['voices'][0]}.pt"
                    blend_description = preset["name"]
            else:
                # Regular voice selection
                voice_path = Path("voices").resolve() / f"{voice_selection}.pt"
                blend_description = voice_selection

        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")

        print(f"Voice: {blend_description}")

        try:
            # Determine language from voice path or use voice_selection for presets
            voice_prefix = voice_selection[:3].lower() if voice_selection else "am_"

            if voice_prefix.startswith(tuple(LANG_MAP.keys())):
                pipeline = get_pipeline_for_voice(voice_prefix)
                generator = pipeline(text, voice=voice_path, speed=speed, split_pattern=r'\n+')
            else:
                generator = model(text, voice=voice_path, speed=speed, split_pattern=r'\n+')

            all_audio = []
            max_segments = 100  # Safety limit for very long texts
            segment_count = 0

            for gs, ps, audio in generator:
                segment_count += 1
                if segment_count > max_segments:
                    print(f"Warning: Reached maximum segment limit ({max_segments})")
                    break

                if audio is not None:
                    if isinstance(audio, np.ndarray):
                        audio = torch.from_numpy(audio).float()
                    all_audio.append(audio)
                    print(f"Generated segment: {gs}")
                    if ps:  # Only print phonemes if available
                        print(f"Phonemes: {ps}")

            if not all_audio:
                raise Exception("No audio generated")
        except Exception as e:
            raise Exception(f"Error in speech generation: {e}")

        # Combine audio segments and save
        if not all_audio:
            raise Exception("No audio segments were generated")

        # Handle single segment case without concatenation
        if len(all_audio) == 1:
            final_audio = all_audio[0]
        else:
            try:
                final_audio = torch.cat(all_audio, dim=0)
            except RuntimeError as e:
                raise Exception(f"Failed to concatenate audio segments: {e}")

        # Save audio file
        try:
            sf.write(wav_path, final_audio.numpy(), SAMPLE_RATE)
        except Exception as e:
            raise Exception(f"Failed to save audio file: {e}")

        # Convert to requested format if needed
        if format.lower() != "wav":
            output_path = DEFAULT_OUTPUT_DIR / f"{base_name}.{format.lower()}"
            return convert_audio(wav_path, output_path, format.lower())

        return wav_path

    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_interface(server_name="127.0.0.1", server_port=7860):
    """Create and launch the Gradio interface."""

    # Get available voices
    voices = get_available_voices()
    if not voices:
        print("No voices found! Please check the voices directory.")
        return

    # Get speed dial presets
    preset_names = speed_dial.get_preset_names()

    # Load voice blend presets
    voice_presets = load_voice_presets()
    preset_keys = list(voice_presets.get("presets", {}).keys())
    preset_descriptions = {k: v.get("name", k) for k, v in voice_presets.get("presets", {}).items()}

    # Create interface
    with gr.Blocks(title="Kokoro TTS Generator - With Voice Blending", fill_height=True) as interface:
        gr.Markdown("# 🎙️ Kokoro TTS Generator with Voice Blending")
        gr.Markdown("Generate high-quality speech with customizable voice blending for authoritative, educational, and philosophical content.")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## TTS Controls")

            with gr.Column(scale=1):
                gr.Markdown("## Voice Blend Presets")

            with gr.Column(scale=1):
                gr.Markdown("## Speed Dial")

        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                # Main TTS controls
                text = gr.Textbox(
                    lines=4,
                    placeholder="Enter text to convert to speech...",
                    label="Text"
                )

            with gr.Column(scale=1):
                # Voice blend preset section
                blend_preset = gr.Dropdown(
                    choices=preset_keys,
                    value=preset_keys[0] if preset_keys else None,
                    label="Voice Blend Preset",
                    interactive=True
                )
                blend_info = gr.Markdown("### Philosopher's Voice\nDeep, contemplative voice perfect for philosophical discourse")

                # Quick blend adjusters - Voice 1
                with gr.Row(scale=1):
                    quick_voice1 = gr.Dropdown(
                        choices=voices,
                        value="am_adam",
                        label="Voice 1",
                        scale=2
                    )
                    quick_weight1 = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=70,
                        step=1,
                        label="Weight",
                        scale=1
                    )

                # Quick blend adjusters - Voice 2
                with gr.Row(scale=1):
                    quick_voice2 = gr.Dropdown(
                        choices=voices,
                        value="am_michael",
                        label="Voice 2",
                        scale=2
                    )
                    quick_weight2 = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=30,
                        step=1,
                        label="Weight",
                        scale=1
                    )

                # Quick preview
                quick_blend_preview = gr.Markdown("**70% Voice 1 + 30% Voice 2**")

                # Apply blend button
                apply_blend_btn = gr.Button("✓ Apply Blend", scale=1, variant="primary")
                blend_status = gr.Textbox(
                    value="Ready",
                    interactive=False,
                    label="Blend Status",
                    scale=1
                )

            with gr.Column(scale=1):
                # Speed dial section
                preset_dropdown = gr.Dropdown(
                    choices=preset_names,
                    value=preset_names[0] if preset_names else None,
                    label="Saved Presets",
                    interactive=True
                )
                preset_name = gr.Textbox(
                    placeholder="Enter preset name...",
                    label="New Preset Name"
                )

        # Voice Blending Options (Advanced)
        with gr.Accordion("Advanced - Custom Voice Blending & Presets", open=False):
            gr.Markdown("### Create & Manage Custom Voice Blends")

            # Blending Parameters Section
            gr.Markdown("#### 🎛️ Blend Parameters")

            with gr.Row():
                voice1 = gr.Dropdown(
                    choices=voices,
                    value="am_adam",
                    label="Voice 1"
                )
                weight1 = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=70,
                    step=1,
                    label="Voice 1 Weight (%)"
                )

            with gr.Row():
                voice2 = gr.Dropdown(
                    choices=voices,
                    value="am_michael",
                    label="Voice 2"
                )
                weight2 = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=30,
                    step=1,
                    label="Voice 2 Weight (%)"
                )

            with gr.Row():
                blend_method = gr.Radio(
                    choices=["Linear Blend", "Spherical Interpolation (SLERP)"],
                    value="Linear Blend",
                    label="Blending Method"
                )
                use_custom_blend = gr.Checkbox(
                    value=False,
                    label="Use This Custom Blend"
                )

            # Blend Preview
            blend_preview = gr.Markdown("**Blend Composition:** 70% Voice 1 + 30% Voice 2")

            gr.Markdown("---")

            # Save Custom Preset Section
            gr.Markdown("#### 💾 Save As Preset")

            with gr.Row():
                custom_preset_name = gr.Textbox(
                    placeholder="e.g., deep_narrator, warm_educator",
                    label="Preset Name",
                    info="Unique identifier (lowercase, no spaces)"
                )
                custom_preset_desc = gr.Textbox(
                    placeholder="e.g., Deep and warm voice for documentaries",
                    label="Description",
                    info="Brief description of this blend"
                )

            with gr.Row():
                save_blend_btn = gr.Button("💾 Save Blend As Preset", variant="primary")
                blend_save_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Ready to save"
                )

            gr.Markdown("---")

            # Manage Presets Section
            gr.Markdown("#### 🗑️ Manage Saved Presets")

            with gr.Row():
                saved_blend_presets = gr.Dropdown(
                    choices=preset_keys,
                    label="Select Preset to Delete",
                    info="(Built-in presets cannot be deleted)"
                )
                delete_blend_btn = gr.Button("🗑️ Delete Preset", variant="stop")
                delete_status = gr.Textbox(
                    label="Delete Status",
                    interactive=False,
                    value="Select a preset to delete"
                )

        # Additional options
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                with gr.Row():
                    format = gr.Radio(
                        choices=["wav", "mp3", "aac"],
                        value="wav",
                        label="Output Format"
                    )
                    speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speed"
                    )

            with gr.Column(scale=1):
                load_preset = gr.Button("Load")
                save_preset = gr.Button("Save Current")

        with gr.Row():
            with gr.Column(scale=2):
                generate = gr.Button("Generate Speech", size="lg", variant="primary")

            with gr.Column(scale=1):
                delete_preset = gr.Button("Delete")

        with gr.Row():
            # Output section
            output = gr.Audio(label="Generated Audio")

        # Blend preset info updater
        def update_blend_info(preset_key):
            if preset_key in voice_presets.get("presets", {}):
                preset_data = voice_presets["presets"][preset_key]
                name = preset_data.get("name", preset_key)
                description = preset_data.get("description", "")
                voices_list = ", ".join(preset_data.get("voices", []))
                weights_list = preset_data.get("weights", [])
                weights_str = ", ".join([f"{w*100:.0f}%" for w in weights_list])
                return f"### {name}\n{description}\n\n**Composition:** {voices_list}\n**Weights:** {weights_str}"
            return "Select a preset"

        blend_preset.change(
            fn=update_blend_info,
            inputs=blend_preset,
            outputs=blend_info
        )

        # Function to load a speed dial preset
        def load_preset_fn(preset_name):
            if not preset_name:
                return None, None, None, None

            preset = speed_dial.get_preset(preset_name)
            if not preset:
                return None, None, None, None

            # Return: blend_preset, text, format, speed
            # Use voice blend preset if it matches, otherwise use default
            blend_key = None
            for key, preset_data in voice_presets.get("presets", {}).items():
                if preset_data.get("name") == preset.get("voice"):
                    blend_key = key
                    break

            return blend_key or preset_keys[0], preset["text"], preset["format"], preset["speed"]

        # Function to save a speed dial preset
        def save_preset_fn(name, blend_preset_name, text, format, speed):
            if not name or not blend_preset_name or not text:
                return gr.update(value="Please provide a name, voice preset, and text")

            # Get the preset name/description for saving
            if blend_preset_name in voice_presets.get("presets", {}):
                voice_name = voice_presets["presets"][blend_preset_name].get("name", blend_preset_name)
            else:
                voice_name = blend_preset_name

            success = speed_dial.save_preset(name, voice_name, text, format, speed)

            # Update the dropdown with the new preset list
            preset_names = speed_dial.get_preset_names()

            if success:
                return gr.update(choices=preset_names, value=name)
            else:
                return gr.update(choices=preset_names)

        # Function to delete a speed dial preset
        def delete_preset_fn(name):
            if not name:
                return gr.update(value="Please select a preset to delete")

            success = speed_dial.delete_preset(name)

            # Update the dropdown with the new preset list
            preset_names = speed_dial.get_preset_names()

            if success:
                return gr.update(choices=preset_names, value=None)
            else:
                return gr.update(choices=preset_names)

        # Function to update blend preview
        def update_blend_preview(v1_choice, w1_val, v2_choice, w2_val):
            """Update the blend preview text"""
            total = w1_val + w2_val
            if total == 0:
                total = 100
            w1_pct = (w1_val / total) * 100
            w2_pct = (w2_val / total) * 100
            return f"**Blend Composition:** {w1_pct:.0f}% {v1_choice} + {w2_pct:.0f}% {v2_choice}"

        # Function to apply blend (sets which blend will be used)
        def apply_blend_fn(v1, w1, v2, w2):
            """Apply the selected blend - ready for generation"""
            total = w1 + w2
            if total == 0:
                total = 100
            w1_pct = (w1 / total) * 100
            w2_pct = (w2 / total) * 100
            message = f"✓ Blend Applied: {w1_pct:.0f}% {v1} + {w2_pct:.0f}% {v2} - Ready to Generate!"
            print(message)
            return message

        # Function to save a custom voice blend preset
        def save_blend_preset_fn(preset_name, preset_desc, v1, w1, v2, w2):
            """Save a custom voice blend as a preset"""
            if not preset_name:
                return "❌ Error: Preset name is required"

            # Normalize name (lowercase, replace spaces with underscore)
            clean_name = preset_name.lower().replace(" ", "_")

            success, message = create_voice_blend_preset(
                clean_name,
                preset_desc or f"Custom blend of {v1} and {v2}",
                [v1, v2],
                [w1, w2],
                blend_method="linear"
            )

            if success:
                # Reload preset dropdown
                updated_presets = load_voice_presets()
                updated_keys = list(updated_presets.get("presets", {}).keys())
                return f"✅ {message}"
            else:
                return f"❌ {message}"

        # Function to delete a custom voice blend preset
        def delete_blend_preset_fn(preset_to_delete):
            """Delete a custom voice blend preset"""
            if not preset_to_delete:
                return "⚠️ Please select a preset to delete"

            success, message = delete_voice_blend_preset(preset_to_delete)

            if success:
                return f"✅ {message}"
            else:
                return f"❌ {message}"

        # Function to generate speech with voice blending
        def generate_with_blending(blend_preset_name, text, format, speed,
                                  v1, w1, v2, w2, blend_method):
            """Generate speech using voice blend from quick controls"""
            # ALWAYS use the quick blend sliders (v1, w1, v2, w2) from main interface
            # These are the primary controls for blending
            custom_voices = [v1, v2]
            custom_weights = [w1, w2]

            print(f"✓ Using quick blend sliders: {w1}% {v1} + {w2}% {v2}")

            return generate_tts_with_logs(
                blend_preset_name,
                text,
                format,
                speed,
                use_blend=True,
                custom_voices=custom_voices,
                custom_weights=custom_weights
            )

        # Connect the buttons to their functions
        load_preset.click(
            fn=load_preset_fn,
            inputs=preset_dropdown,
            outputs=[blend_preset, text, format, speed]
        )

        save_preset.click(
            fn=save_preset_fn,
            inputs=[preset_name, blend_preset, text, format, speed],
            outputs=preset_dropdown
        )

        delete_preset.click(
            fn=delete_preset_fn,
            inputs=preset_dropdown,
            outputs=preset_dropdown
        )

        # Voice blend preset event handlers
        # Update blend preview when weights/voices change
        weight1.change(
            fn=update_blend_preview,
            inputs=[voice1, weight1, voice2, weight2],
            outputs=blend_preview
        )
        weight2.change(
            fn=update_blend_preview,
            inputs=[voice1, weight1, voice2, weight2],
            outputs=blend_preview
        )
        voice1.change(
            fn=update_blend_preview,
            inputs=[voice1, weight1, voice2, weight2],
            outputs=blend_preview
        )
        voice2.change(
            fn=update_blend_preview,
            inputs=[voice1, weight1, voice2, weight2],
            outputs=blend_preview
        )

        # Quick blend controls event handlers (main interface)
        quick_weight1.change(
            fn=update_blend_preview,
            inputs=[quick_voice1, quick_weight1, quick_voice2, quick_weight2],
            outputs=quick_blend_preview
        )
        quick_weight2.change(
            fn=update_blend_preview,
            inputs=[quick_voice1, quick_weight1, quick_voice2, quick_weight2],
            outputs=quick_blend_preview
        )
        quick_voice1.change(
            fn=update_blend_preview,
            inputs=[quick_voice1, quick_weight1, quick_voice2, quick_weight2],
            outputs=quick_blend_preview
        )
        quick_voice2.change(
            fn=update_blend_preview,
            inputs=[quick_voice1, quick_weight1, quick_voice2, quick_weight2],
            outputs=quick_blend_preview
        )

        # Apply blend button
        apply_blend_btn.click(
            fn=apply_blend_fn,
            inputs=[quick_voice1, quick_weight1, quick_voice2, quick_weight2],
            outputs=blend_status
        )

        # Save custom voice blend preset
        save_blend_btn.click(
            fn=save_blend_preset_fn,
            inputs=[custom_preset_name, custom_preset_desc, voice1, weight1, voice2, weight2],
            outputs=blend_save_status
        )

        # Delete voice blend preset
        delete_blend_btn.click(
            fn=delete_blend_preset_fn,
            inputs=saved_blend_presets,
            outputs=delete_status
        )

        # Connect the generate button with voice blending support
        generate.click(
            fn=generate_with_blending,
            inputs=[blend_preset, text, format, speed,
                   quick_voice1, quick_weight1, quick_voice2, quick_weight2, blend_method],
            outputs=output
        )

    # Launch interface
    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=True
    )

def cleanup_resources():
    """Properly clean up resources when the application exits"""
    global model

    try:
        print("Cleaning up resources...")

        # Clean up model resources
        if model is not None:
            print("Releasing model resources...")

            # Clear voice dictionary to release memory
            if hasattr(model, 'voices') and model.voices is not None:
                try:
                    voice_count = len(model.voices)
                    for voice_name in list(model.voices.keys()):
                        try:
                            # Release each voice explicitly
                            model.voices[voice_name] = None
                        except:
                            pass
                    model.voices.clear()
                    print(f"Cleared {voice_count} voice references")
                except Exception as ve:
                    print(f"Error clearing voices: {type(ve).__name__}: {ve}")

            # Clear model attributes that might hold tensors
            for attr_name in dir(model):
                if not attr_name.startswith('__') and hasattr(model, attr_name):
                    try:
                        attr = getattr(model, attr_name)
                        # Handle specific tensor attributes
                        if isinstance(attr, torch.Tensor):
                            if attr.is_cuda:
                                print(f"Releasing CUDA tensor: {attr_name}")
                                setattr(model, attr_name, None)
                        elif hasattr(attr, 'to'):  # Module or Tensor-like object
                            setattr(model, attr_name, None)
                    except:
                        pass

            # Delete model reference
            try:
                del model
                model = None
                print("Model reference deleted")
            except Exception as me:
                print(f"Error deleting model: {type(me).__name__}: {me}")

        # Clear CUDA memory explicitly
        if torch.cuda.is_available():
            try:
                # Get initial memory usage
                try:
                    initial = torch.cuda.memory_allocated()
                    initial_mb = initial / (1024 * 1024)
                    print(f"CUDA memory before cleanup: {initial_mb:.2f} MB")
                except:
                    pass

                # Free memory
                print("Clearing CUDA cache...")
                torch.cuda.empty_cache()

                # Force synchronization
                try:
                    torch.cuda.synchronize()
                except:
                    pass

                # Get final memory usage
                try:
                    final = torch.cuda.memory_allocated()
                    final_mb = final / (1024 * 1024)
                    freed_mb = (initial - final) / (1024 * 1024)
                    print(f"CUDA memory after cleanup: {final_mb:.2f} MB (freed {freed_mb:.2f} MB)")
                except:
                    pass
            except Exception as ce:
                print(f"Error clearing CUDA memory: {type(ce).__name__}: {ce}")

        # Final garbage collection
        try:
            import gc
            collected = gc.collect()
            print(f"Garbage collection completed: {collected} objects collected")
        except Exception as gce:
            print(f"Error during garbage collection: {type(gce).__name__}: {gce}")

        print("Cleanup completed")

    except Exception as e:
        print(f"Error during cleanup: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# Register cleanup for normal exit
import atexit
atexit.register(cleanup_resources)

# Register cleanup for signals
import signal
import sys

def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}, shutting down...")
    cleanup_resources()
    sys.exit(0)

# Register for common signals
for sig in [signal.SIGINT, signal.SIGTERM]:
    try:
        signal.signal(sig, signal_handler)
    except (ValueError, AttributeError):
        # Some signals might not be available on all platforms
        pass

def parse_arguments():
    """Parse command line arguments for host and port configuration."""
    parser = argparse.ArgumentParser(
        description="Kokoro TTS Local Generator - Gradio Web Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number to run the server on"
    )
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        create_interface(server_name=args.host, server_port=args.port)
    finally:
        # Ensure cleanup even if Gradio encounters an error
        cleanup_resources()

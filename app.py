"""Streamlit UI for Urdu TTS Pipeline."""
import streamlit as st
from pathlib import Path
import tempfile
import os

# Custom CSS for professional look
st.markdown("""
<style>
.main-header { 
    font-size: 3rem; 
    color: #1f77b4; 
    text-align: center; 
    margin-bottom: 2rem;
    font-weight: 300;
}
.section-header {
    font-size: 1.8rem;
    color: #333;
    margin-top: 2rem;
    border-bottom: 2px solid #eee;
    padding-bottom: 0.5rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.stButton > button {
    background: linear-gradient(45deg, #1f77b4, #4CAF50);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.8rem 2rem;
    font-weight: bold;
    font-size: 1.1rem;
    transition: transform 0.2s;
}
.stButton > button:hover {
    transform: scale(1.05);
}
.audio-player {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Core imports
from src.pipeline import UrduSpeechPipeline
from src.utils import ensure_dir, validate_urdu_text
from src.config import TTSConfig, TranscriberConfig

st.markdown('<h1 class="main-header">Urdu Speech AI Assistant</h1>', unsafe_allow_html=True)

# Sidebar config
st.sidebar.header("⚙️ Configuration")
tts_steps = st.sidebar.slider("TTS Steps", 10, 100, 40)
tts_cfg = TTSConfig(num_steps=tts_steps)

model_size = st.sidebar.selectbox("ASR Model", ["tiny", "base", "small", "medium"])
trans_cfg = TranscriberConfig(model_size=model_size)

# Initialize pipeline
@st.cache_resource
def load_pipeline():
    return UrduSpeechPipeline(tts_config=tts_cfg, transcriber_config=trans_cfg)

pipeline = load_pipeline()

tab1, tab2 = st.tabs(["🎤 Speech to Speech", "✍️ Text to Speech"])

with tab1:
    st.markdown('<h2 class="section-header">Speech Processing Pipeline</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Urdu audio", type=['wav', 'mp3', 'flac', 'ogg', 'm4a'])
    
    if uploaded_file:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.read())
            input_path = tmp.name
        
        col1, col2 = st.columns(2)
        with col1:
            st.audio(input_path, format='audio/wav')
            st.info(f"📁 Input: {Path(uploaded_file.name).name}")
        
        if st.button("🚀 Process Audio", type="primary"):
            with st.spinner("Processing..."):
                try:
                    # Run pipeline
                    output_path = ensure_dir("temp_output.wav")
                    text = pipeline.process_audio(input_path, str(output_path))
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Transcribed Text Length", len(text))
                    with col2:
                        st.metric("Status", "✅ Success")
                    with col3:
                        st.metric("Output File", "temp_output.wav")
                    
                    st.markdown('<div class="audio-player">', unsafe_allow_html=True)
                    st.audio(output_path)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.success(f"Generated speech: **{text[:100]}...**")
                    os.unlink(input_path)  # Cleanup
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab2:
    st.markdown('<h2 class="section-header">Text to Speech</h2>', unsafe_allow_html=True)
    
    text_input = st.text_area("Urdu text", height=150, 
                             placeholder="Enter Urdu text to synthesize...")
    
    ref_audio = st.file_uploader("Reference speaker audio (3-10s)", 
                                type=['wav', 'mp3'])
    
    if text_input and ref_audio:
        if st.button("🎯 Generate Speech", type="secondary"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as ref_tmp:
                    ref_tmp.write(ref_audio.read())
                    ref_path = ref_tmp.name
                
                output_path = ensure_dir("tts_output.wav")
                validate_urdu_text(text_input)
                
                sr = pipeline.synthesize_only(text_input, ref_path, str(output_path))
                
                st.audio(output_path)
                st.balloons()
                os.unlink(ref_path)
                
            except Exception as e:
                st.error(f"TTS Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Powered by Whisper + DeepSpeak | Professional Urdu AI Voice Pipeline*")

if __name__ == "__main__":
    pass


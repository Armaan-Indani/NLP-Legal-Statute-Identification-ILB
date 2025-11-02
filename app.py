import streamlit as st
import json
# from transformers import pipeline
from faster_whisper import WhisperModel
import pandas as pd
import io
import tempfile
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer, AutoModel

from googletrans import Translator
translator = Translator()

# Language Mappings
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Tamil": "ta",
    "Kannada": "kn"
}
TARGET_LANG_CODE = "en"

# -----------------
# Custom Model Classes
# -----------------
class LstmAttn(nn.Module):
    def __init__(self, hidden_size, drop=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Parameter(torch.rand(hidden_size))
        self.dropout = nn.Dropout(drop)
        
    def forward(self, inputs=None, attention_mask=None, dynamic_context=None):
        if attention_mask is None:
            attention_mask = torch.ones(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
        lengths = attention_mask.float().sum(dim=1)
        inputs_packed = pack_padded_sequence(inputs, torch.clamp(lengths, min=1).cpu(), enforce_sorted=False, batch_first=True)  
        outputs_packed = self.lstm(inputs_packed)[0]
        outputs = pad_packed_sequence(outputs_packed, batch_first=True)[0]
        
        activated_outputs = torch.tanh(self.dropout(self.attn_fc(outputs)))
        context = dynamic_context if dynamic_context is not None else self.context.expand(inputs.size(0), self.hidden_size)
        scores = torch.bmm(activated_outputs, context.unsqueeze(2)).squeeze(2)
        masked_scores = scores.masked_fill(~attention_mask, -1e-32)
        masked_scores = F.softmax(masked_scores, dim=1)
        hidden = torch.sum(outputs * masked_scores.unsqueeze(2), dim=1)
        return outputs, hidden

class HierBert(nn.Module):
    def __init__(self, encoder, drop=0.5):
        super().__init__()
        self.bert_encoder = encoder
        self.hidden_size = encoder.config.hidden_size
        self.segment_encoder = LstmAttn(self.hidden_size, drop=drop)
        self.dropout = nn.Dropout(drop)
    
    def forward(self, input_ids=None, attention_mask=None):
        batch_size, max_num_segments, max_segment_size = input_ids.shape
        input_ids_flat = input_ids.view(-1, max_segment_size)
        attention_mask_flat = attention_mask.view(-1, max_segment_size)
        encoder_outputs = self.bert_encoder(input_ids=input_ids_flat, attention_mask=attention_mask_flat).last_hidden_state[:, 0, :]
        encoder_outputs = encoder_outputs.view(batch_size, max_num_segments, self.hidden_size)
        attention_mask = attention_mask.any(dim=2)
        _, hidden = self.segment_encoder(inputs=encoder_outputs, attention_mask=attention_mask)
        return hidden

class HierBertForTextClassification(nn.Module):
    def __init__(self, hier_encoder, num_labels):
        super().__init__()
        self.hier_encoder = hier_encoder
        self.hidden_size = hier_encoder.hidden_size
        self.classifier_fc = nn.Linear(self.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask):
        hidden = self.dropout(self.hier_encoder(input_ids=input_ids, attention_mask=attention_mask))
        logits = torch.sigmoid(self.classifier_fc(hidden))
        return logits

# -----------------
# 1. Model and Data Loading Functions
# -----------------

@st.cache_resource
def load_lsi_model_and_tokenizer():
   
    model_ckpt = "./output/pytorch_model.bin"
    model_src = "law-ai/InLegalBERT"
    root = "./data/"

    try:
        # Load label vocab for id-to-label mapping
        with open(os.path.join(root, "label_vocab.json")) as fr:
            label_vocab = json.load(fr)
        id2label = {v: k for k, v in label_vocab.items()}
        num_labels = len(label_vocab)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_src)
        special_tokens = {'additional_special_tokens': ['<ENTITY>', '<ACT>', '<SECTION>']}
        tokenizer.add_special_tokens(special_tokens)

        # Load BERT and custom hierarchical model
        bert = AutoModel.from_pretrained(model_src)
        bert.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        hier_bert = HierBert(bert)
        model = HierBertForTextClassification(hier_bert, num_labels)
        
        # Load weights
        model.load_state_dict(torch.load(model_ckpt, map_location="cpu"), strict=False)
        model.eval()

        return model, tokenizer, id2label

    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Ensure all model files are present.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading custom LSI model: {e}")
        return None, None, None

@st.cache_resource
def load_label_descriptions():
    try:
        with open('label_descriptions.json', 'r', encoding="utf-8") as f:
            label_dict = json.load(f)
        return label_dict
        
    except FileNotFoundError:
        st.error("label_descriptions.json not found. Please ensure it's in the root directory.")
        return {}
    except json.JSONDecodeError:
        st.error("Error decoding JSON from label_descriptions.json. Check file syntax.")
        return {}
    except Exception as e:
        st.error(f"Error loading label descriptions: {e}")
        return {}

@st.cache_resource
def load_asr_pipeline():
    model_size = "tiny"
    try:
        asr_model = WhisperModel(
            model_size, 
            device="cpu", 
            compute_type="int8"
        )
        return asr_model
    except Exception as e:
        st.error(f"Error loading ASR model: {e}")
        return None


# -----------------
# 2. Pipeline Processing Functions
# -----------------
# Add language_code as an argument
def transcribe_audio(audio_file, asr_model, language_code):
    if not asr_model:
        st.error("ASR model not loaded.")
        return None
    
    # 1. Save the uploaded file to a temporary path on the disk
    try:
        audio_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_file.name) as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
    except Exception as e:
        st.error(f"Failed to save temporary audio file: {e}")
        return None

    # 2. Perform transcription using faster-whisper's transcribe method
    try:
        segments, info = asr_model.transcribe(
            tmp_path,
            language=language_code,
            beam_size=5,
            vad_filter=True
        )
        full_transcript = " ".join(segment.text for segment in segments)
        return full_transcript
        
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None

def translate_to_english_googletrans(text, language_code):
    try:
        result = translator.translate(text, dest='en', src=language_code)
        result.text = result.text.replace(".", ". ")
        return result.text
    except Exception as e:
        st.error(f"Translation failed via googletrans: {e}")
        return None

def get_predictions(text, lsi_model, lsi_tokenizer, id2label, label_dict, score_threshold):
    if not text or not lsi_model or not lsi_tokenizer:
        return []

    # 1. Segment the text (based on predictor_final.py logic)
    # Using "." as a simple delimiter for segmentation
    sentences = text.split(".") 
    
    if not sentences or (len(sentences) == 1 and not sentences[0].strip()):
        st.warning("Input text is empty or could not be segmented.")
        return []
    
    # 2. Tokenize and prepare input
    encoded = lsi_tokenizer(
        sentences, 
        padding=True, 
        truncation=True, 
        return_tensors="pt", 
        return_token_type_ids=False
    )
    
    # Reshape for the Hierarchical model: [1, num_segments, seq_len]
    input_ids = encoded["input_ids"].unsqueeze(0)  
    attention_mask = encoded["attention_mask"].unsqueeze(0)

    # 3. Get predictions (logits are sigmoid outputs [0-1])
    try:
        with torch.no_grad():
            # Model returns logits [1, num_labels], take the first element [num_labels]
            logits = lsi_model(input_ids, attention_mask)[0]
    except Exception as e:
        st.error(f"Prediction failed with custom model: {e}")
        return []
    
    # 4. Get top-k results (set k=50 to capture all likely labels for filtering)
    k_max = min(100, logits.size(0))
    topk = torch.topk(logits, k_max)
    
    # 5. Process and filter the results
    filtered_results = []
    for score_tensor, label_id_tensor in zip(topk.values, topk.indices):
        score = score_tensor.item()
        label_id = int(label_id_tensor.item())
        label = id2label.get(label_id, "UNKNOWN_LABEL")

        if score >= score_threshold:
            description = label_dict.get(label, "Description not found.")
            filtered_results.append({
                'Statute Section': label,
                'Prediction Score': f"{score:.4f}",
                'Description': description
            })
            
   
    filtered_results.sort(key=lambda x: x['Prediction Score'], reverse=True)
    return filtered_results[:10]

# -----------------
# 3. Streamlit UI (Main Application Logic)
# -----------------

def main():
    st.set_page_config(page_title="Legal Statute Identification (LSI)", layout="wide")
    st.title("Legal Statute Identification (LSI)")
    st.markdown("Predict relevant sections of **The Indian Penal Code** from text or audio input.")

    # Load resources
    lsi_model, lsi_tokenizer, id2label = load_lsi_model_and_tokenizer()
    label_dict = load_label_descriptions()
    asr_model = load_asr_pipeline()

    if lsi_model is None or lsi_tokenizer is None or not label_dict:
        st.error("Essential resources (LSI model, tokenizer, or labels) failed to load. Please check file paths.")
        st.stop()

    # --- Sidebar for parameters ---
    with st.sidebar:
        st.header("Model Parameters")
        score_threshold = st.slider(
            "Prediction Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Only display statutes with a prediction score above this value."
        )
        
    # --- Input Configuration (Step 1: Type and Language Selection) ---
    st.header("1. Choose Input Type and Language")
    col1, col2 = st.columns(2)

    input_type = col1.radio(
        "Select Input Type:", 
        ("Text Input", "Audio Input (File Upload)"),
        horizontal=True
    )

    selected_language_name = col2.selectbox(
        "Select Input Language:",
        list(LANGUAGES.keys())
    )
    selected_lang_code = LANGUAGES[selected_language_name]
    is_english = (selected_lang_code == 'en')

    # --- Data Input (Step 2: Data Entry) ---
    st.header("2. Provide Input Data")
    
    input_text = None
    original_input = None
    audio_file = None

    if input_type == "Text Input":
        raw_input = st.text_area(
            f"Paste your {selected_language_name} legal text/case summary here:",
            height=200,
            placeholder="E.g., Accused poured kerosene and set her on fire."
        )
        if raw_input:
            original_input = raw_input
            if is_english:
                input_text = raw_input
                st.caption("Flow: **English Text** $\\to$ **Statute Prediction**")
            else:
                st.caption(f"Flow: **Non-English Text** ({selected_language_name}) $\\to$ **Translate** $\\to$ **Statute Prediction**")
    
    elif input_type == "Audio Input (File Upload)":
        if asr_model is None:
            st.warning("ASR model failed to load. Audio input is disabled.")
            st.stop()
            
        audio_file = st.file_uploader(
            f"Upload an audio file in {selected_language_name} (.wav, .mp3, etc.):",
            type=['wav', 'mp3', 'flac']
        )
        
        if audio_file:
            st.audio(audio_file)
            if is_english:
                st.caption("Flow: **English Audio** $\\to$ **Transcript** $\\to$ **Statute Prediction**")
            else:
                st.caption(f"Flow: **Non-English Audio** ({selected_language_name}) $\\to$ **Transcript** $\\to$ **Translate** $\\to$ **Statute Prediction**")

    st.markdown("---")
    
    # --- Prediction Execution (Step 3: Processing) ---
    if st.button("Identify Statutes", use_container_width=True, type="primary"):
        
        # --- Validation ---
        if input_type == "Text Input" and not original_input:
            st.warning("Please enter some text to identify statutes.")
            return
        if input_type == "Audio Input (File Upload)" and not audio_file:
            st.warning("Please upload an audio file to identify statutes.")
            return

        with st.spinner("Processing input and identifying statutes..."):
            
            # 1. Transcription (If Audio)
            if input_type == "Audio Input (File Upload)":
                st.markdown("##### Step 1: Transcribing Audio...")
                transcribed_text = transcribe_audio(audio_file, asr_model, selected_lang_code)
                
                if transcribed_text:
                    original_input = transcribed_text
                else:
                    st.error("Failed to transcribe audio or no speech was detected. Please check the audio quality or try a shorter file.")
                    return
                   
            
            # 2. Translation (If Non-English)
           
            if not is_english and original_input:
                st.markdown("##### Step 2: Translating to English...")
                translated_text = translate_to_english_googletrans(original_input, selected_lang_code)
                
                if translated_text:
                    input_text = translated_text
                else:
                    return

            if is_english and input_type == "Text Input":
                 input_text = original_input
            elif is_english and input_type == "Audio Input (File Upload)":
                 input_text = original_input

            if not input_text:
                st.error("Processing failed: Could not obtain text for statute prediction.")
                return

            # 3. LSI Prediction
            st.markdown("##### Step 3: Identifying Statutes...")
            predictions = get_predictions(
                input_text, 
                lsi_model, 
                lsi_tokenizer, 
                id2label, 
                label_dict, 
                score_threshold
            )

            # --- Results Display ---
            st.header("Results")
            st.markdown(f"**Input Language:** `{selected_language_name}` | **Input Type:** `{input_type}`")
            
            if not is_english or input_type == "Audio Input (File Upload)":
                st.subheader("Intermediate Output")
                if input_type == "Audio Input (File Upload)":
                    st.info(f"**Transcription:** {original_input}")
                if not is_english:
                    st.info(f"**English Translation (for LSI):** {input_text}")
            
            st.subheader("Statute Prediction Output")
            if predictions:
                st.success(f"Found **{len(predictions)}** relevant statute sections (Score > {score_threshold:.2f}):")
                
                df_results = pd.DataFrame(predictions)
                st.dataframe(df_results, use_container_width=True)
                
            else:
                st.warning("No statutes found above the specified score threshold. Try lowering the threshold in the sidebar.")

if __name__ == "__main__":
    main()
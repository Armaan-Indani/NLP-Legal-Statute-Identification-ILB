# Legal Statute Identification (LSI)

A **Streamlit**-based application that identifies relevant sections of **The Indian Penal Code (IPC)** from **text or audio input** in multiple Indian languages.  
It integrates **speech recognition**, **machine translation**, and a **hierarchical BERT-based legal classifier** to perform end-to-end statute prediction.

---

## 🧠 Core Features

- **Multimodal Input**: Accepts both text and audio.
- **Multilingual Support**: Handles English, Hindi, Marathi, Tamil, and Kannada.
- **Automatic Speech Recognition (ASR)**: Uses `faster-whisper` for transcribing audio.
- **Translation**: Uses `googletrans` to convert text/audio to English before prediction.
- **Hierarchical BERT Model**: A custom `HierBert` model trained for statute classification.
- **Interactive UI**: Streamlit interface for easy experimentation and visualization.

---

## 📁 Project Structure

```
.
├── app.py                      # Main Streamlit application
├── data/
│   └── label_vocab.json        # Label-to-ID mapping for statutes
├── output/
│   └── pytorch_model.bin       # Trained HierBert model weights
├── label_descriptions.json     # Descriptions for each IPC section
├── requirements.txt            # Dependencies list
└── README.md                   # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```
   git clone https://github.com/Armaan-Indani/NLP-Legal-Statute-Identification-ILB
   cd legal-statute-identification
   ```

2. **Create a virtual environment**
   ```
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Download required model files**
   * Download `pytorch_model.bin` from https://drive.google.com/file/d/1Aj-L1qz66BgMyYuz0-vaatN4OmDXMiPI/view?usp=sharing
   * Place `pytorch_model.bin` in `./output/`

---

## 🧩 Dependencies

* `streamlit`
* `transformers`
* `torch`
* `faster-whisper`
* `pandas`
* `googletrans==4.0.0-rc1`

Install them manually if needed:
```
pip install streamlit transformers torch faster-whisper pandas googletrans==4.0.0-rc1
```

---

## 🚀 Usage

### Run the Application
```
streamlit run app.py
```

### Interface Overview

1. **Input Type**
   * Text or Audio file upload.

2. **Language Selection**
   * Choose input language from the supported set.

3. **Prediction Score Threshold**
   * Adjust sensitivity for predicted IPC sections.

4. **Output**
   * Transcription (for audio).
   * Translation (for non-English inputs).
   * Top predicted IPC sections with confidence scores and descriptions.

---

## 🧱 Model Architecture

The app uses a **Hierarchical BERT** with an **LSTM-Attention** layer for segment-level aggregation.

**Key components:**

* `HierBert`: Hierarchical encoder using `InLegalBERT` as a base.
* `LstmAttn`: BiLSTM + attention pooling over BERT segment embeddings.
* `HierBertForTextClassification`: Adds a sigmoid classifier for multilabel prediction.

---

## 🌐 Supported Languages

| Language | Code |
| -------- | ---- |
| English  | en   |
| Hindi    | hi   |
| Marathi  | mr   |
| Tamil    | ta   |
| Kannada  | kn   |

---

## 🧮 Example Workflow

**Audio Input (Hindi)**
→ Transcription via Whisper
→ Translation to English via Google Translate
→ HierBert prediction of relevant IPC sections
→ Display top matches with scores and legal descriptions

---

## ⚠️ Notes

* Ensure all required files are present before running (`pytorch_model.bin`, vocab, and description files).
* Audio transcription depends on Whisper model size and quality.
* Translation accuracy may vary with Google Translate.


# Yu-Gi-Oh Chatbot

An intelligent, multimodal assistant built with **LangGraph**, **LLM function-calling**, and **vector-based retrieval**, designed to help users interact with the Yu-Gi-Oh TCG.  
The bot can answer ruling questions, build decks, identify cards from images, retrieve accurate card stats/effects, and more.

---

## ✨ Features

### 🔍 Card Recognition
- Image-based card identification using a custom dataset  
- Supports different rarities, holo effects, and long card names  
- More robust than OCR due to direct image vector embedding  

### 📚 Card Effects + Stats Retrieval
- Retrieves card data from a curated card database  
- Pinecone (or other vector DBs) used for similarity search  
- Custom ranking reduces hallucinations and ensures accurate effects

### 🤖 LLM-Powered Reasoning
- GPT as the primary fallback model  
- Hugging Face models (Mistral, LLaMA, etc.) supported when available  
- Graceful fallback logic ensures consistent responses even if a backend model fails

### 🧠 LangGraph Orchestration
- Coordinates:
  - Model selection  
  - Retrieval  
  - Image understanding  
  - Tool calls  
  - Error handling  
- Ensures deterministic, extensible chatbot behavior

### 🖼 Image Dataset Integration
- Uses the **FabioArdi/yugioh_images** dataset  
- Loaded via Parquet for efficiency  
- Optional sub-sampling for development

---

## 📁 Project Structure

```
/
├── src/
│   ├── graph/             # LangGraph workflow definition
│   ├── models/            # LLM backends and fallback logic
│   ├── retrieval/         # Pinecone or other vector DB integration
│   ├── tools/             # Structured function-calling tools
│   ├── images/            # Local card image cache (optional)
│   └── utils/
│
├── data/
│   ├── yugioh.parquet     # Image embeddings dataset
│   ├── cards.json         # Structured card DB
│
├── .env                   # API keys
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Configure environment variables
Create a `.env` file:

```
OPENAI_API_KEY=your_key
HUGGINGFACEHUB_API_TOKEN=your_key
PINECONE_API_KEY=your_key
```

### 3️⃣ Run the chatbot
```bash
python main.py
```

---

## 🧩 Model Backends

### Primary
- **GPT-4.x / GPT-5.x** via OpenAI

### Secondary (optional)
- **Mistral Instruct**  
- **LLaMA Instruct**  
- Other HuggingFace text-generation endpoints that support the required task  

If a model errors or doesn’t support the expected task, the system automatically **falls back to GPT**.

---

## 📦 Card Image Processing

OCR is unreliable for Yu-Gi-Oh cards due to:
- holo foil glare  
- tiny fonts  
- rarity variations  
- long names  

Instead, card identification uses **image embeddings**:

1. User uploads an image  
2. CLIP/SigLIP generates an embedding  
3. Embedding is matched in Pinecone  
4. Best-matching card(s) returned  
5. Effects fetched from the card database  

---

## 🔧 Retrieval + RAG

- Pinecone index stores embeddings for images + effect text  
- Hybrid semantic search ensures accurate matches  
- LangGraph guards against hallucinations and enforces correct tool usage  

---

## 🐞 Common Issues

### ❗ HuggingFace models failing with:
`Model X is not supported for task text-generation`
- Some providers (Together, Featherless, etc.) only allow **conversational** tasks  
- Use the exact model ID from HuggingFace  
- Or rely on GPT fallback

### ❗ Image recognition not matching cards
Check:
- parquet files fully downloaded  
- subset indexing correct  
- embeddings stored in proper namespace  

---

## 🛠 Development Tips

### Update card dataset
```bash
python scripts/update_cards.py
```

### Regenerate embeddings
```bash
python scripts/generate_embeddings.py
```

### Visualize LangGraph
```bash
python scripts/visualize_graph.py
```

---

## 📌 Roadmap

- [ ] Hybrid (text + image) retrieval  
- [ ] Ruling citations from Konami sources  
- [ ] Archetype-aware deck-building engine  
- [ ] Video frame card detection  
- [ ] Local inference mode  
- [ ] Graph database for card relationships  

---

## 🤝 Contributing

Pull requests and suggestions are welcome.  
For major changes, please open an issue first to discuss the proposal.

---

## 📝 License

MIT License — free for personal and commercial use.
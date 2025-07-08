# 🔍 Qubic Smart Contract Auditor

A smart contract audit tool for the **Qubic blockchain**. This tool automatically analyzes Qubic contracts using a powerful RAG (Retrieval-Augmented Generation) pipeline to detect syntax issues or deviations from Qubic’s smart contract standards.
> 🔗 Built for the [Qubic BLockchain](https://qubic.org/) — the world’s fastest blockchain (15.5M TPS, verified by CertiK)
---

## 🚀 Features

- ✅ **Static Analysis + AI**: Detect potential flaws in your Qubic smart contracts.
- 📂 **Import Contracts**: Audit contracts directly from GitHub URLs or upload from your local system.
- 📜 **Qubic-Aware Validation**: Ensures syntax and structure compliance with Qubic's contract standards.
- 🤖 **LLM-powered**: Uses Google Gemini model via LangChain and RAG.
- 🌐 **Modern Stack**:
  - Frontend: React
  - Backend: FastAPI
  - LLM: Google Gemini (via LangChain)
  - RAG-based smart context-aware evaluation

---

## ⚙️ Tech Stack

- **Frontend**: React
- **Backend**: FastAPI
- **LLM**: Google Gemini via LangChain
- **Architecture**: Retrieval-Augmented Generation (RAG)

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/anubhavsultania/Qubic.git
cd Qubic
```
### 2. Install Python Requirements
```bash
pip install -r requirements.txt
```
📦 Make sure you have Python 3.8+ and pip installed.

### 3. Start FastAPI Backend
```bash
uvicorn main:app --reload
```
### 4. Start React Frontend
```bash
cd frontend
npm install
npm start
```
## 🧪 Usage
* Upload a .qubic smart contract file locally OR

* Paste a GitHub link to fetch the contract automatically.

* Click "Analyze".

* The tool will show:

  * Detected syntax issues

  * Violations of Qubic contract standards

  * Any other potential flaws

## 📷 Screenshots
![image](https://github.com/user-attachments/assets/b9a9e18e-878f-4ebb-b237-5739b50f26e0)


## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📫 Contact
For questions or support:

* GitHub Issues: [Submit an Issue](https://github.com/anubhavsultania/Qubic/issues/new)

## 📝 License
This project is licensed under the MIT License.

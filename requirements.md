```txt
# SBSE + LLM Lab - Requirements for Local and Colab Execution

## Local Requirements

### System (tested on Ubuntu with 24GB RAM, 8-core Intel i7-1165G7)
- Python 3.10 or 3.11 (recommended)
- RAM: 24GB (to run mistral/llama 8B locally)
- CPU: 8 cores (tested with 11th Gen Intel® Core™ i7-1165G7 @ 2.80GHz × 8)

### Base Python Packages
ollama>=0.1.9
deap>=1.3.3
numpy>=1.24
matplotlib>=3.7

### Ollama Models (install via terminal)
# For open-source LLM execution (about ~5 minutes per run with this spec):
# - Mistral (7B): ~7.1GB RAM
# - Llama3 (8B): ~7GB RAM
# Example install:
ollama pull mistral
ollama pull llama3

## Colab Requirements

### Python Packages
google-generativeai>=0.3.1
deap>=1.3.3
numpy>=1.24
matplotlib>=3.7

## Installation (local)
pip install ollama deap numpy matplotlib

## Installation (Colab)
!pip install -q google-generativeai deap numpy matplotlib

## Notes
- Ollama must be installed and the server running: https://github.com/ollama/ollama
- For Gemini API, you need a valid API key and access via Google Colab.
- Test configuration: Each open source model (Mistral, Llama3) ran in approx. 5 min with 24GB RAM.
- Gemini Pro (API via Google Colab): execution under 1 minute.

# Example: Launch Ollama server locally (Ubuntu)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve  # in one terminal
# then execute Python scripts as provided

```

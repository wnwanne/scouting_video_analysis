# Film Analysis & Scouting Reports with Azure OpenAI and AI Search

This app lets you upload a sports game or practice film, then automatically analyzes and summarizes player performance. Results are indexed in Azure AI Search, so you can instantly search or chat over the generated scouting reports with Retrieval-Augmented Generation (RAG).

---

## Features

- **Video Upload & Segmentation**  
  Upload a football video file, which is automatically split into 30-second (or configurable) segments for analysis.

- **Automated Scouting Reports**  
  For each video segment, GPT-4o provides an NFL-style scouting report on player physical traits, technique, decision-making, and more.

- **Intelligent Summary**  
  All segment analyses are combined and summarized into a single, concise scouting report.

- **Azure AI Search Integration**  
  Instantly index all scouting analyses and summaries into Azure AI Search with one click.

- **RAG-Style Q&A**  
  Ask questions or run searches over the indexed analyses—powered by OpenAI’s GPT-4o using the most relevant scouting report content as context.

- **Modern Streamlit UI**  
  Seamless tabbed interface for analysis, indexing, and search. Live progress feedback as the video processes.

---

## Setup

### Prerequisites

- Python 3.8+
- Azure OpenAI resource (with GPT-4.1 and Whisper deployed)
- Azure AI Search resource

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/scouting_video_analysis.git
cd scouting_video_analysis

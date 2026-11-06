---
title: distillo
emoji: 🎓
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: From a recorded lecture to the key points worth learning
---

# distillo: web demo

Upload a lecture (audio, video or subtitles) or paste a link, choose how many key points you want, and get a revision
sheet where every point links to the minute it was said and quotes the exact sentence it comes from.

This is the web demo of [distillo](https://github.com/AndreaAcanfora/distillo), an open-source command-line tool.

## Running it

```bash
pip install -r requirements.txt
GEMINI_API_KEY=... GROQ_API_KEY=... python app.py   # http://localhost:7860
```

On Hugging Face Spaces, set `GEMINI_API_KEY` and `GROQ_API_KEY` as **secrets**. Visitors using the shared keys get
`DISTILLO_DEMO_RUNS_PER_HOUR` runs per hour (default 4) and recordings up to `DISTILLO_DEMO_MAX_MINUTES` (default
90); visitors who paste their own free keys have no limits. Keys pasted by visitors are used for that request only.

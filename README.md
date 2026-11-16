# scary-videos

Daily Shorts for YouTube & TikTok – hosting for automated uploads.

- `engine/make_survival.py` – the animation engine (two styles: `survival` and `whatif`), story format in its docstring
- `engine/example_survival.json`, `engine/example_whatif.json` – complete examples
- `engine/make_video.py` – narration (Kokoro), sound effects and ambience helpers
- `history/` – one file per video: format, title, premise, recipe, stats
- `videos/` – the latest video (older ones are removed after they are published)

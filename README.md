# AI Exam Analyzer

Dieses Projekt prüft Fragen-/Antwort-Daten über die OpenAI API inhaltlich und ergänzt die Ergebnisse in einer neuen JSON-Datei.

## Struktur

- `classify_topics_merged_config_fixed.py`: Kompatibler Startpunkt (wie bisher).
- `ai_exam_analyzer/cli.py`: CLI-Argumente und Pipeline-Start.
- `ai_exam_analyzer/processor.py`: Hauptverarbeitung je Frage.
- `ai_exam_analyzer/passes.py`: Pass A / Pass B Prompt- und Lauf-Logik.
- `ai_exam_analyzer/openai_client.py`: OpenAI Responses API Wrapper mit JSON-Schema.
- `ai_exam_analyzer/schemas.py`: Structured-Output-Schemata.
- `ai_exam_analyzer/topic_catalog.py`: Topic-Katalog und Prompt-Formatierung.
- `ai_exam_analyzer/payload.py`: Nutzdatenaufbereitung pro Frage.
- `ai_exam_analyzer/io_utils.py`: JSON laden/speichern.
- `ai_exam_analyzer/config.py`: Default-Konfiguration.

## Ausführen

```bash
export OPENAI_API_KEY="..."
python classify_topics_merged_config_fixed.py \
  --input export.json \
  --topics topic-tree.json \
  --output export.AIannotated.json
```

## Hinweise zu potenziellen Problemen (ohne Funktionsänderung)

Siehe `KNOWN_ISSUES.md`.

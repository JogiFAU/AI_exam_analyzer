# Auto-Tuning Kontext: Workflow, Parameter und Qualitätslogik

Diese Dokumentation beschreibt, wie der Analyzer arbeitet und wie Parameter die Ergebnisqualität beeinflussen.

## 1) Pipeline-Überblick
1. **Preprocessing/Gates**: Ermittelt Datenqualitätsprobleme (z. B. fehlende korrekte Antwort, fehlendes Bild bei Bildhinweis). Kann LLM-Lauf einschränken oder Auto-Change sperren.
2. **Pass A**: Initiale fachliche Bewertung + Topic-Zuordnung + Änderungs-Empfehlung.
3. **Pass B (Verifier)**: Läuft bei Unsicherheit/Triggern und validiert Pass-A-Änderungen unabhängig.
4. **Optional Pass C Review**: Für wartungsintensive Fälle, konservative Korrekturstufe.
5. **Optional Reconstruction**: Verbesserung/Rekonstruktion v. a. bei Legacy/Low-Quality-Fragen.
6. **Optional Repeat-Reconstruction**: Nutzt ähnliche Fragen über Jahrgänge als Anker.

## 2) Bedeutung der Auto-Tuning-Parameter
- `trigger_answer_conf` / `trigger_topic_conf`:
  Niedrige Werte => seltener Pass B (höhere Geschwindigkeit, mehr Risiko).
  Höhere Werte => häufiger Pass B (robuster, aber teurer/langsamer).
- `apply_change_min_conf_b`:
  Mindestvertrauen, um Pass-B-Änderungen automatisch zu übernehmen.
  Höher = konservativer.
- `low_conf_maintenance_threshold`:
  Unterhalb dessen wird Wartungsbedarf markiert.
  Höher = mehr konservative Markierungen.
- `knowledge_top_k`, `knowledge_max_chars`, `knowledge_min_score`:
  Steuern Evidenzmenge/-qualität aus Knowledge Base.
- `enable_review_pass`, `enable_repeat_reconstruction`, `enable_explainer_pass`:
  Zusätzliche Qualitätsnetze für schwierige/legacy-lastige Datensätze und didaktische Nachvollziehbarkeit.
- `enable_reconstruction_pass`:
  Im Standard-Workflow deaktiviert lassen; der Pass ist teuer und soll vom Auto-Tuning nicht empfohlen werden.

## 3) Qualitätsabhängige Heuristik
- Bei **hoher Datenqualität** (klare Frage, valide Antworten, vorhandene korrekte Indizes, wenig Uneindeutigkeit):
  - Trigger etwas niedriger, Review selektiver, Reconstruction deaktiviert lassen, Explainer aktiv lassen.
- Bei **heterogenen oder schwachen Datensätzen** (kurze Texte, fehlende Korrektheiten, unklare Formulierungen, Bildlücken):
  - Trigger höher (mehr Pass B), konservativere Auto-Apply-Schwelle,
  - mehr Maintenance-Markierungen,
  - Review/Reconstruction eher aktivieren.
- Bei **niedriger Retrieval-Qualität**:
  - eher mehr Kontext (Top-K / Max-Chars erhöhen), Min-Score leicht senken,
  - Entscheidungen konservativ halten.

## 4) Ziel des Auto-Tunings
Nicht maximale Änderungsrate, sondern **hohe Verlässlichkeit**:
- Unsicherheit sichtbar machen,
- spekulative Korrekturen vermeiden,
- robuste Parameter für den konkreten Datensatz wählen.

## 5) Erwartete Ausgabe des Tuners
- Parameterempfehlung im erlaubten Wertebereich.
- Kurzer Bericht mit Begründung, der Dataset-Qualität + Workflowlogik miteinander verknüpft.

## 6) Knowledge-Base-Analyse für Auto-Tuning
Die Parameterauswahl MUSS die Qualität der Knowledge Base berücksichtigen:
- Umfang/Struktur: Anzahl Chunks, Quellenvielfalt, durchschnittliche Chunk-Länge, Bildanzahl.
- Retrieval-Performance auf Frage-Sample: mittlere Retrieval-Qualität, Anteil Fragen mit nicht-leerer Evidenz.

Ableitungslogik:
- Wenn Retrieval-Qualität niedrig oder Trefferquote schwach: `knowledge_top_k` und `knowledge_max_chars` eher erhöhen, `knowledge_min_score` moderat senken.
- Wenn Retrieval stabil gut und Evidenz präzise: eher kompaktere Retrieval-Parameter, um Rauschen/Kosten zu reduzieren.
- Entscheidungen stets konservativ und mit Blick auf Verlässlichkeit der fachlichen Bewertung treffen.

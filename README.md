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

## Verbesserter Analyse-Workflow

Die Pipeline arbeitet jetzt explizit in mehreren Schritten, passend zum gewünschten Ablauf:

1. **Datensatz-Clusterung nach Frageinhalt** (Fragetext + Antworten) vor der Modellanalyse.
2. **Bild-Clusterung** für Fragenbilder inkl. Ähnlichkeitsabgleich zu Bildern aus PDF-Dateien der Knowledge-Base.
3. **Robustes Retrieval (BM25-basiert + Diversität)** statt einfachem Token-Overlap für den Fachkontext.
4. **Vorläufige Themenerkennung** (`topic_initial`) nur aus Frage + Antwortoptionen.
5. **Inhaltliche Antwortprüfung** (`answer_review`) inkl. Plausibilitätsbewertung und Änderungsvorschlag unter Nutzung von Cluster-/Bildkontext.
6. **Unabhängige Verifikation** in Pass B (`verify_answer`) bei Triggern wie niedriger Confidence oder Wartungsverdacht.
7. **Finale Themenzuordnung** (`topic_final`) + **Frageabstraktion** (`question_abstraction.summary`).
8. **Abstraktions-Clusterung** nach Abschluss der Fragenanalyse.
9. **Optionaler Review-Pass** für schwierige Wartungsfälle mit separatem Modell inkl. Konflikt-Triggern.
10. **Finale Entscheidung + Flags** im Output:
   - `topicInitial.reasonDetailed`, `topicFinal.reasonDetailed` und ausführliche Begründungen in der Antwortprüfung
   - `answerPlausibility.finalCorrectIndices`
   - `answerPlausibility.finalAnswerConfidence` (0..1)
   - `answerPlausibility.aiDisagreesWithDataset` (AI-Antwort weicht vom Datensatz ab)
   - `answerPlausibility.changedInDataset` (Datensatz wurde tatsächlich aktualisiert)

Zusätzlich werden Fragen mit niedriger Sicherheit automatisch als wartungsbedürftig markiert.
Der Grenzwert ist per CLI konfigurierbar:

```bash
--low-conf-maintenance-threshold 0.65
```

## Ausführen

```bash
export OPENAI_API_KEY="..."
python classify_topics_merged_config_fixed.py \
  --input export.json \
  --topics topic-tree.json
```

Wenn `--output` nicht gesetzt ist, wird automatisch `<Input-Dateiname> AIannotated.json` im selben Ordner erzeugt.

## Lokale Benutzeroberfläche (UI)

Es gibt jetzt eine Streamlit-Oberfläche, in der du:
- alle wichtigen Einstellungen setzen kannst,
- den OpenAI-Key eingeben kannst,
- die Analyse per Button startest,
- und den Live-Status (z. B. Frage X/Total, aktueller Schritt, Checkpoints) siehst.

### Starten

```bash
pip install streamlit
streamlit run run_ui.py
```

Die UI ist anschließend lokal unter `http://localhost:8501` erreichbar.

## Wissensbasis aus ZIP (PDF/TXT/MD) nutzen

Für höhere Qualität kann ein Fach-Korpus (z. B. ZIP mit Vorlesungsfolien) direkt angebunden werden.
Die Pipeline extrahiert Text-Chunks, reduziert die Datenmenge per Retrieval (Top-K + Zeichenlimit)
und übergibt nur relevante Evidenz pro Frage an Pass A/Pass B.

### Ablauf
1. ZIP wird gelesen (PDF/TXT/MD unterstützt).
2. Inhalte werden in Chunks zerlegt und tokenbasiert indiziert.
3. Pro Frage werden die relevantesten Chunks gesucht.
4. Nur diese Chunks (`retrievedEvidence`) gehen in den Prompt.
5. Audit enthält Evidenz + Retrieval-Qualität.
   - Im Output werden Evidenzen kompakt als Quelle (PDF/Datei), Seite und Score gespeichert (ohne langen Chunk-Text).

Hinweis: Bilder in PDFs werden ohne OCR nicht in Text umgewandelt. Für bildlastige Folien sollte OCR vorgeschaltet werden.

### Beispielaufruf

```bash
python classify_topics_merged_config_fixed.py \
  --input export.json \
  --topics topic-tree.json \
  --output export.AIannotated.json \
  --knowledge-zip fach_materialien.zip \
  --knowledge-index fach_materialien.index.json \
  --knowledge-top-k 6 \
  --knowledge-max-chars 4000 \
  --knowledge-min-score 0.06
```

## Post-Processing auf vorhandenem Output erneut ausführen (ohne kompletten Neustart)

Wenn ein Lauf inhaltlich schon plausibel ist, aber `reviewPass` oder `reconstruction` teilweise fehlgeschlagen sind,
kannst du die optionalen Post-Processing-Schritte auf Basis eines bestehenden `...AIannotated.json` erneut laufen lassen.

Beispiel:

```bash
python classify_topics_merged_config_fixed.py \
  --input "Sample_Data/mibi_prac/output/export AIannotated.json" \
  --topics "Sample_Data/mibi_prac/topic-tree.json" \
  --output "Sample_Data/mibi_prac/output/export AIannotated.postprocessed.json" \
  --postprocess-only \
  --enable-review-pass \
  --enable-reconstruction-pass \
  --knowledge-zip "Sample_Data/mibi_prac/knowledge.zip" \
  --images-zip "Sample_Data/mibi_prac/images.zip"
```

Wichtige Flags:
- `--postprocess-only`: überspringt Pass A/B und nutzt vorhandenes `aiAudit` als Ausgangspunkt.
- `--force-rerun-review`: erzwingt Review-Neuberechnung auch wenn `reviewPass` schon existiert.
- `--force-rerun-reconstruction`: erzwingt Reconstruction-Neuberechnung auch wenn bereits vorhanden.

Hinweis: Für bestmögliche Qualität sollten dieselben Wissens-/Bildquellen wie im Ursprungslauf eingebunden werden.

Einzelfragen gezielt neu rechnen:

```bash
python classify_topics_merged_config_fixed.py \
  --input "Sample_Data/mibi_prac/output/export AIannotated.json" \
  --topics "Sample_Data/mibi_prac/topic-tree.json" \
  --output "Sample_Data/mibi_prac/output/export AIannotated.single-rerun.json" \
  --resume \
  --only-question-id "01f1-0751-0a3ca26c-ae00-79b8acb6e751"
```

Danach kannst du für alle Fragen nur das Postprocessing nachziehen:

```bash
python classify_topics_merged_config_fixed.py \
  --input "Sample_Data/mibi_prac/output/export AIannotated.single-rerun.json" \
  --topics "Sample_Data/mibi_prac/topic-tree.json" \
  --output "Sample_Data/mibi_prac/output/export AIannotated.postprocessed.json" \
  --postprocess-only \
  --enable-review-pass \
  --enable-reconstruction-pass
```

### Wichtige Optionen
- `--knowledge-zip`: ZIP-Datei mit Fachmaterialien.
- `--knowledge-index`: optionaler Cache der extrahierten Chunks (schneller bei Wiederholungsruns).
- `--knowledge-top-k`: Anzahl Chunks pro Frage.
- `--knowledge-max-chars`: harte Obergrenze für mitgeschickte Evidenz pro Frage.
- `--knowledge-min-score`: minimale Relevanzschwelle.
- `--knowledge-chunk-chars`: Chunk-Größe beim Parsing.

### Ohne Gold-Set trotzdem robust arbeiten
Falls kein Gold-Set möglich ist, nutze ein konservatives Betriebsmodell:
- Datensatzänderungen nur wenn Pass B zustimmt und die kombinierte Confidence hoch ist.
- Niedrige kombinierte Confidence automatisch als Wartungsfall markieren.
- Fälle mit AI/Datensatz-Abweichung priorisiert manuell prüfen.

## Fragebilder aus ZIP einbinden (images.zip)

Wenn Fragen bildbasiert sind, können die Bilder direkt mit an das Modell übergeben werden.
Standardmäßig nutzt die CLI `images.zip` (falls im Arbeitsverzeichnis vorhanden).

- Dateinamen werden anhand des Musters `img_<frage-id>_<nr>.<ext>` der Frage zugeordnet.
- Zusätzlich werden bestehende `imageFiles`-Referenzen aus dem Datensatz ausgewertet.
- Bild-Metadaten landen pro Frage unter `aiAudit.images` (`providedImageCount`, `missingExpectedImageRefs`, …).
- Fehlen erwartete Bilder, wird dies im Prompt explizit markiert, damit das Modell die Frage auf Wartungsbedarf setzen kann.

### Beispielaufruf mit Bildern

```bash
python classify_topics_merged_config_fixed.py \
  --input export.json \
  --topics topic-tree.json \
  --output export.AIannotated.json \
  --images-zip images.zip
```


## Nur Cluster erneut berechnen (PowerShell CLI)

Wenn du einen bestehenden Datensatz (z. B. `.../export AIannotated.json`) **ohne neue Modell-Calls** nur neu clustern willst, gibt es jetzt ein PowerShell-Skript mit absichtlich etwas lockereren Defaults:

- `text-cluster-similarity`: **0.12** (statt 0.15)
- `abstraction-cluster-similarity`: **0.18** (statt 0.22)

Beispiel:

```powershell
./rerun-clustering.ps1 `
  -Input "Sample_Data/mibi_prac/output/export AIannotated.json" `
  -Output "Sample_Data/mibi_prac/output/export AIannotated.reclustered.json" `
  -TextClusterSimilarity 0.12 `
  -AbstractionClusterSimilarity 0.18
```

Optional mit Bildern (für `questionImageClusterIds`):

```powershell
./rerun-clustering.ps1 `
  -Input "Sample_Data/mibi_prac/output/export AIannotated.json" `
  -Output "Sample_Data/mibi_prac/output/export AIannotated.reclustered.json" `
  -ImagesZip "Sample_Data/mibi_prac/images.zip"
```

Hinweis: Das Skript ruft intern `python -m ai_exam_analyzer.recluster_only` auf und aktualisiert nur `aiAudit.clusters` (plus `meta.clusteringRerun` im Container).
Optionales Hybrid-Design mit LLM (für Abstraction-Cluster-Qualität):
- `--enable-llm-abstraction-cluster-refinement`
- `--cluster-refinement-model o4-mini`
- `--cluster-refinement-max-clusters 30`
- `--cluster-refinement-min-cluster-size 2`
- `--cluster-refinement-merge-candidates 5`

Dabei bewertet ein LLM pro Cluster mögliche thematische Ausreißer (werden ausgelagert) und prüft sinnvolle Cluster-Merges mit ähnlichen Kandidatenclustern.

Dabei wird für **jede Frage** ein Status in der CLI ausgegeben (Start/Ende), damit Fehler und Fortschritt direkt sichtbar sind.


### Hybrid-Refinement-Run (PowerShell)

Für den neuen LLM-basierten Abstraction-Cluster-Refinement-Run gibt es zusätzlich:

```powershell
./run-hybrid-refinement.ps1 `
  -InputPath "Sample_Data/mibi_prac/output/export AIannotated.json" `
  -TopicsPath "Sample_Data/mibi_prac/topic-tree.json" `
  -Output "Sample_Data/mibi_prac/output/export AIannotated.hybrid-refined.json" `
  -ImagesZip "Sample_Data/mibi_prac/images.zip" `
  -KnowledgeZip "Sample_Data/mibi_prac/knowledge.zip" `
  -ClusterRefinementModel "o4-mini" `
  -ClusterRefinementMaxClusters 30 `
  -ClusterRefinementMinClusterSize 2 `
  -ClusterRefinementMergeCandidates 5
```

Das Skript startet intern einen `--postprocess-only` Lauf mit aktivem `--enable-llm-abstraction-cluster-refinement`.

## Optional: Clean-up des Output-Datensatzes

Mit `--cleanup-spec <datei.json>` kann der Output nach der Verarbeitung auf definierte Felder reduziert werden.
Die Spec ist eine Whitelist auf JSON-Ebene (nur gelistete Elemente bleiben erhalten).

Beispiel `cleanup-spec.json`:

```json
{
  "questions": {
    "*": {
      "id": true,
      "text": true,
      "answers": {
        "*": {
          "text": true,
          "isCorrect": true
        }
      },
      "correctIndices": true,
      "aiAudit": true
    }
  }
}
```

Anwendung:

```bash
python classify_topics_merged_config_fixed.py \
  --input export.json \
  --topics topic-tree.json \
  --output export.AIannotated.json \
  --cleanup-spec cleanup-spec.json
```

Hinweis: Die Bereinigung wirkt auf den geschriebenen Output (inkl. Checkpoints), nicht auf die interne Analyse-Logik.

## Qualitätsstrategie für höchste Ergebnisgüte

Kurzantwort: **Dein Workflow ist grundsätzlich gut**, aber für maximale Qualität sollte er um einen kontrollierten **Kontext-Layer (RAG)**, strengere **Verifikationsregeln** und ein **Kalibrierungs-/Review-Setup** erweitert werden.

### 1) Ist ein ZIP mit PDFs sinnvoll?
Ja, aber nicht als "blindes Mitschicken" in jeden Prompt, sondern als **indizierte Wissensbasis**:

1. ZIP entpacken.
2. PDF-Text strukturiert extrahieren (Seiten, Überschriften, Absätze).
3. In kleine, zitierfähige Chunks zerlegen.
4. Embeddings + Vektorsuche aufbauen.
5. Pro Frage nur die Top-K relevanten Chunks in den Prompt geben (mit Quelle/Seite).

So erhältst du deutlich bessere fachliche Konsistenz, weniger Halluzinationen und bessere Themenzuordnung.

### 2) Empfohlener Ziel-Workflow (High-Quality)

**Schritt A – Retrieval vor Pass A**
- Ermittele pro Frage relevante Kontextstellen aus dem Fachkorpus (Top-K + Mindest-Ähnlichkeit).
- Übergib diese als "evidence" in Pass A.

**Schritt B – Antwortentscheidung mit Evidenzpflicht**
- Modell muss je vorgeschlagener Antwort kurz begründen, **welche Evidenz** sie stützt.
- Fehlt belastbare Evidenz: Confidence absenken + Wartungsflag erhöhen.

**Schritt C – Unabhängige Verifikation (Pass B) mit Gegenprüfung**
- Pass B sieht Frage, Optionen, Datensatzlösung, Pass-A-Vorschlag und dieselbe Evidenz.
- Pass B muss aktiv nach Gegenargumenten suchen ("disconfirming evidence").

**Schritt D – Finale Themenzuordnung evidenzbasiert**
- Topic nicht nur aus Frageformulierung, sondern aus dem tatsächlich nötigen Fachwissen ableiten.
- Optional: hierarchisch klassifizieren (Supertopic -> Subtopic), jeweils mit eigener Confidence.

### 3) Modellwahl (praxisnah)
- **Pass A (breit, schnell):** robustes generalistisches Modell mit guter Kosten/Qualitäts-Balance.
- **Pass B (strenger Verifier):** reasoning-stärkeres Modell mit konservativen Regeln.
- Für strittige Fälle optional **Pass C (Tie-Breaker)** nur bei Konflikten oder sehr niedriger Confidence.

### 4) Confidence wirklich belastbar machen
Ein einzelner Modell-Confidence-Wert ist oft schlecht kalibriert. Besser ein kombinierter Score:

- `answer_conf_model` (Modellselbsteinschätzung)
- `retrieval_quality` (Ähnlichkeit/Abdeckung der gefundenen Quellen)
- `agreement_score` (Pass A vs Pass B)
- `option_margin` (Abstand beste vs zweitbeste Antwort)

Daraus einen Gesamtwert berechnen (z. B. gewichtetes Mittel). Niedriger Gesamtwert => "Wartungsbedürftig".

### 5) Konkrete Quality-Gates (empfohlen)
- **Auto-Change nur**, wenn:
  - Pass B zustimmt,
  - Evidenz vorhanden,
  - Gesamt-Confidence über Schwellwert,
  - kein harter Widerspruch in Quellen.
- **Human-Review-Pflicht**, wenn:
  - AI ≠ Datensatz,
  - Gesamt-Confidence niedrig,
  - Themenzuordnung instabil (Pass A/B unterschiedlich),
  - Frage ist mehrdeutig oder kontextabhängig.

### 6) Metriken für echte Qualitätssteuerung
Baue ein kleines Gold-Set (manuell geprüfte Fragen) und tracke:
- Antwortgenauigkeit (Top-1 / ggf. Multi-Label F1)
- Themenklassifikation (Accuracy + Confusion Matrix)
- Precision bei "AI hat Datensatz geändert"
- Anteil wartungsbedürftiger Fragen + Trefferquote dieser Markierung

Nur mit diesen Metriken kannst du sicher entscheiden, ob neue Prompts/Modelle wirklich besser sind.

### 7) Praktische Empfehlung für deinen nächsten Schritt
1. ZIP/PDF-Korpus als RAG-Quelle integrieren (klein anfangen, z. B. 2-3 Fächer).
2. Evidence-Felder in `aiAudit` speichern (Dokument, Seite, Chunk-ID).
3. Gesamt-Confidence aus mehreren Signalen berechnen.
4. Schwellenwerte mit einem Gold-Set kalibrieren.
5. Erst dann aggressive Auto-Änderungen zulassen.

Wenn du möchtest, kann ich im nächsten Schritt direkt eine konkrete **Datenstruktur für Evidence + Confidence-Komposition** und ein minimales **RAG-Interface** für deine bestehende Pipeline vorschlagen.

## Hinweise zu potenziellen Problemen (ohne Funktionsänderung)

Siehe `KNOWN_ISSUES.md`.


### Neue optionale Workflow-Parameter

```bash
--text-cluster-similarity 0.32 \
--abstraction-cluster-similarity 0.45 \
--enable-review-pass \
--review-model o4-mini \
--review-min-maintenance-severity 2 \
--topic-candidate-top-k 3 \
--topic-candidate-outside-force-passb-conf 0.92 \
--enable-repeat-reconstruction \
--repeat-min-similarity 0.72 \
--repeat-min-anchor-conf 0.82 \
--repeat-min-anchor-consensus 1 \
--repeat-min-match-ratio 0.60 \
--auto-apply-repeat-reconstruction \
--run-report workflow_report.json
```

`--topic-candidate-top-k` steuert, wie viele deterministische Topic-Kandidaten vor Pass A in den Payload aufgenommen werden.

`--topic-candidate-outside-force-passb-conf` erzwingt Pass B, wenn Pass A trotz Kandidatenfilter ein Topic außerhalb der Top-k vorschlägt und die Topic-Confidence unter der Schwelle liegt.

`--enable-repeat-reconstruction` aktiviert die Erkennung wiederkehrender Fragen über verschiedene Klausurjahre mit Vorschlägen zur Rekonstruktion schwacher Items.

`--repeat-min-similarity`, `--repeat-min-anchor-conf`, `--repeat-min-anchor-consensus` und `--repeat-min-match-ratio` steuern, wie streng Repeat-Cluster, Anchor-Konsens und Textüberlappung für Rekonstruktionsvorschläge bewertet werden.

`--auto-apply-repeat-reconstruction` erlaubt automatisches Übernehmen der Rekonstruktionsvorschläge (nur wenn Preprocessing-Gates Auto-Änderungen erlauben).

`--run-report` schreibt einen JSON-Laufbericht (u. a. Preprocessing-Gates, Candidate-Konflikte, Topic-Drift, Repeat-Rekonstruktionen, blockierte Auto-Changes, Pass-B/Review-Häufigkeiten, Maintenance-Grundverteilung) für die nachgelagerte Kalibrierung.


**Hinweis V5:**
- Retrieval verwendet jetzt BM25-Scoring mit Quell-Diversitätsbonus.
- Entscheidung über Antwortänderungen ist konservativer (Evidenz-/Qualitätsgates).
- Pass C kann zusätzlich bei Konflikten (Topic-Shift, Datensatzabweichung + geringe Gesamtconfidence) auslösen.

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

1. **Vorläufige Themenerkennung** (`topic_initial`) nur aus Frage + Antwortoptionen.
2. **Inhaltliche Antwortprüfung** (`answer_review`) inkl. Plausibilitätsbewertung und Änderungsvorschlag.
3. **Unabhängige Verifikation** in Pass B (`verify_answer`) bei Triggern wie niedriger Confidence oder Wartungsverdacht.
4. **Finale Themenzuordnung** (`topic_final`) nach Abschluss der inhaltlichen Prüfung.
5. **Finale Entscheidung + Flags** im Output:
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
  --topics topic-tree.json \
  --output export.AIannotated.json
```

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

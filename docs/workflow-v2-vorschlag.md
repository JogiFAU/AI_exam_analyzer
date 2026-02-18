# Workflow-V2-Vorschlag für AI Exam Analyzer

## Zielbild
Der Workflow soll zuverlässig drei Kernziele erfüllen:
1. Topic-Zuordnung aus `topic-tree.json`.
2. Plausibilitätsprüfung der hinterlegten Lösung(en).
3. Erkennung/Clusterung ähnlicher Fragen.

## Ausgangslage im aktuellen Stand
- Topic-Zuordnung und Plausibilitätsprüfung laufen bereits in einer 2-Pass-Architektur (Pass A + Verifier Pass B, optional Review Pass C).
- Deterministische Preprocessing-Checks markieren strukturelle Datenprobleme vor den LLM-Pässen.
- Clustering basiert auf tokenisierten Textmengen und Jaccard mit Union-Find, inzwischen mit Stopword-/Template-Filter und DF-Pruning.
- Antwortindizes werden über `answerIndex` (1-basiert) verarbeitet, mit Fallback-/Kompatibilitätslogik für ältere Datensätze.

## Empfohlener Ziel-Workflow (robust + skalierbar)

### Phase 0: Dataset-Profiling (vor jedem Run)
Erzeuge einen kurzen Profilbericht pro Fach (JSON + Markdown):
- Anzahl Fragen, Bildanteil, Anteil ohne `correctIndices`, Anteil extrem kurzer Fragen, Anteil unklarer Notiztexte.
- Verteilung von Antwortanzahl, Anteil Duplikate, Anteil fehlender Erklärung.
- Top-Token/Top-Entities je Fach.

**Zweck:** Schwellenwerte und Gates datengetrieben je Fach setzen statt global statisch.

### Phase 1: Deterministisches Preprocessing & Gate-Entscheidung
Baue aus den Regeln nicht nur `reasons`, sondern ein Gate-Objekt:
- `run_llm`: ja/nein
- `allow_auto_change`: ja/nein
- `force_manual_review`: ja/nein
- `quality_score`: 0..1

Reason-Klassen:
- **Hard blockers** (z. B. fehlende korrekte Antwortstruktur, kaputte Antwortoptionen)
- **Soft blockers** (z. B. sehr kurzer/generischer Text)
- **Context blockers** (z. B. Bild erwähnt, aber Asset fehlt)

**Wichtig:** Hard blocker sperren Auto-Korrektur immer; LLM darf optional nur für Topic/Abstraktion laufen.

### Phase 2: Topic-Zuordnung als „Candidate + Verify“ statt Single Shot
1. **Kandidatenbildung** (schnell, deterministisch):
   - TF-IDF/BM25 auf Subtopic-Beschreibungen + Aliasliste
   - Top-k Subtopics (z. B. 3)
2. **LLM-Entscheidung nur innerhalb Kandidatenmenge**:
   - reduziert Label-Drift,
   - erhöht Konsistenz zwischen Fächern,
   - senkt Kosten.
3. **Verifier (Pass B)** prüft nur bei niedriger Topic-Confidence oder Konflikt mit Signalschicht.

### Phase 3: Plausibilitätsprüfung mit explizitem Evidenzmodus
Unterscheide drei Modi je Frage:
- `evidence_strong` (KB + Kontext gut) → normale Plausibilitätsprüfung.
- `evidence_weak` → konservativ, keine Auto-Korrektur außer bei klaren Widersprüchen.
- `evidence_missing` (z. B. Bildfrage ohne Bild) → `cannot_judge` + Maintenance, aber keine Pseudo-Sicherheit.

Kombinierte Confidence:
- Komponenten nur gewichten, wenn die Komponente überhaupt verfügbar ist (bereits begonnen).
- Zusätzlich „coverage“-Signal einführen (Anteil relevanter Antwortoptionen, die evidenzseitig adressiert wurden).

### Phase 4: Ähnlichkeits-Clusterung als 2-stufiges Retrieval-Problem
Statt globalem Union-Find direkt:
1. **Blocking**:
   - primär innerhalb Subtopic,
   - plus harter Bild-Link (`imageFile`/`imageUrl` gleich) als Sonderfall.
2. **Kandidatenretrieval**:
   - TF-IDF/BM25 Top-k Kandidaten pro Frage.
3. **Re-Ranking**:
   - Embedding-Cosine oder Weighted Jaccard + Entity-Overlap.
4. **Merge-Gates**:
   - Mindestzahl seltener gemeinsamer Tokens/Entities,
   - Grenzfälle nicht mergen, sondern als „near-duplicate-candidate" markieren.

Damit verhindert man transitive Fehl-Merges durch Template-Fragen.

### Phase 5: Qualitätsmetriken ohne Goldlabels
Pro Run automatisch berechnen:
- Topic-Konsistenz: Pass-A vs Pass-B/Review Übereinstimmung.
- Änderungspräzision (Proxy): Anteil später zurückgenommener Auto-Changes.
- Cluster-Kohäsion/Purity/Bridge-Score.
- Maintenance-Rate nach Reason-Klasse je Fach.

Diese Metriken dienen als „Regler“, um Schwellen je Fach iterativ zu kalibrieren.

## Umsetzungsstatus & nächste Schritte (an neues Vorgehen angepasst)

### Bereits umgesetzt
- [x] **Phase 1 (teilweise):** strukturierte Preprocessing-Gates inkl. `runLlm`, `allowAutoChange`, `forceManualReview` und `qualityScore`.
- [x] **Phase 2 (weitgehend):** deterministische `topicCandidates` (Top-k) werden je Frage in den Payload gelegt; Candidate-Konflikte können Pass B zusätzlich triggern.
- [x] **Phase 5 (teilweise):** optionaler Run-Report (`--run-report`) mit Gate-/Pass-Metriken, Candidate-Konflikten, Topic-Drift und Maintenance-Grundverteilungen.

### Nächste Prioritäten (konkret)
1. **Repeat-Reconstruction (automatisierte Verbesserung) stabilisieren**
   - High-Quality-Anker über Jahre erkennen und auf Maintenance-Items übertragen.
   - Auto-Apply nur bei sauberem Text-Matching + erlaubten Gates, sonst Suggestion-only.

2. **Run-Report weiter kalibrieren**
   - Präzisions-Proxy für Repeat-Rekonstruktionen (später durch Review bestätigt?).
   - Anteil blockierter Auto-Changes je Reason-Klasse.

3. **Clustering V2 umsetzen**
   - Retrieval-Top-k + Re-Ranking + Merge-Gates statt reinem globalen Jaccard-Union-Find.
   - Cluster-Qualitätsflags (`cohesion`, `bridge_score`) direkt im Audit speichern.

## Bezug zu den Zukunftszielen
- **Rekonstruktion mangelhafter Fragen (Ziel 4):** wird erst belastbar, wenn Near-Duplicate-Erkennung sauber ist und Cluster-Purity stimmt.
- **Neue Fragen generieren (Ziel 5):** erst sinnvoll nach stabilen Topic- und Plausibilitätsmetriken; bis dahin niedrige Priorität beibehalten.

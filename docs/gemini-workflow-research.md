# Recherche: Sinnvoller Gemini-Workflow ab Input-JSON (AI Exam Analyzer)

## Ziel der Recherche
Auf Basis des aktuellen Projekts einen **Gemini-optimierten End-to-End-Workflow** definieren, der für folgende Aufgaben höhere Qualität anstrebt:
1. Fragen in Topics/Subtopics einsortieren,
2. Fragen inhaltlich prüfen/lösen,
3. ähnliche Fragen erkennen/clustern,
4. mangelhafte Fragen rekonstruieren.

Der aktuelle Workflow dient als Ausgangspunkt, wird aber dort bewusst verändert, wo Gemini-Vorteile genutzt oder Gemini-Schwächen kompensiert werden können.

---

## 1) Ausgangspunkt im aktuellen Code

Der Ist-Stand ist bereits mehrstufig und robust:
- Preprocessing + Gates vor LLM-Aufrufen.
- Pass A (Analyse/Klassifikation) + optionaler Pass B (Verifikation) + optionale Spezialpässe.
- KB-Retrieval auf Chunk-Basis mit BM25-ähnlicher Scoring-Logik.
- nachgelagerte Cluster-/Rekonstruktionsschritte.

Relevante Stellen im Code:
- Pipeline-Steuerung und per-Question Ablauf: `processor.py`.
- Pass-Prompts und strukturierte Ausgaben: `passes.py`.
- KB-Retrieval: `knowledge_base.py`.
- Provider-/Modellwahl und Defaults: `llm_clients.py`, `model_profiles.py`, `workflow_profiles.py`, `cli.py`, `ui.py`.

---

## 2) Gemini-spezifische Stärken/Schwächen für eure Aufgaben

## Stärken (nützlich für eure Domäne)
1. **Große Kontextfenster**: mehr Evidence-Chunks und mehr Frage-/Clusterkontext gleichzeitig nutzbar.
2. **Starke Long-Context-Synthese**: hilfreich für „verstreute Evidenz“ über mehrere PDF-Seiten/Chunks.
3. **Gute multimodale Verarbeitung**: bei bildlastigen Fragen/Bild-Referenzen vorteilhaft.

## Schwächen/Risiken (die man abfedern sollte)
1. **Konfidente, aber dünn belegte Antworten** bei schwacher Evidenz.
2. **Schema-/JSON-Robustheit** kann modell-/promptabhängig schwanken.
3. **Kosten/Latenz** steigen bei sehr langen Kontexten ohne striktes Budget-/Gate-Management.

## Konsequenz für Workflow-Design
Gemini sollte **nicht** nur als Drop-in-Modell genutzt werden, sondern mit:
- stärker evidenzzentrierter Orchestrierung,
- adaptiver Retrieval- und Verifikationslogik,
- konservativen Änderungen am Datensatz bei Evidenzunsicherheit.

---

## 3) Empfohlener Gemini-Workflow ab Input-JSON

## Phase A – Intake & Datensatzprofil (neu schärfen)
Input: `export.json` (Fragen + Antworten + Metadaten).

Empfehlung:
1. **Datensatzprofil vor Run** erzeugen (pro Fach):
   - Anteil Bildfragen, kurze/ambige Fragen, fehlende `correctIndices`, Antwortanzahl-Verteilung.
2. Daraus **run-spezifische Parameter** ableiten:
   - Retrieval-Budgets,
   - Trigger-Schwellen,
   - Aktivierung zusätzlicher Verifikationsschritte.

Warum für Gemini: Große Kontexte bringen nur dann Qualitätsgewinn, wenn Budget und Trigger fach-/datenabhängig gesetzt werden.

## Phase B – Preprocessing & harte Sicherheitsgates (ausbauen)
Beibehalten + erweitern:
1. Hard-Blocker (z. B. kaputte Antwortstruktur) -> keine Auto-Korrektur.
2. Soft-Blocker -> LLM erlaubt, aber mit konservativem Änderungsprofil.
3. **Gemini-spezifischer Guard**:
   - bei niedriger Retrieval-Qualität Auto-Changes sperren,
   - Verifikation erzwingen.

Warum: kompensiert Risiko überkonfidenter Korrekturen bei schwacher Evidenz.

## Phase C – Zweistufiges Evidence-Retrieval (zentral)
Empfehlung für Gemini:
1. **Stufe 1 (präzise)**: normales Retrieval mit konservativem Min-Score.
2. **Stufe 2 (adaptiv, nur bei Bedarf)**:
   - mehr `top_k`, mehr `max_chars`, leicht gesenkter `min_score`.
3. Bestes Ergebnis nach Retrieval-Qualität auswählen.
4. Retrieval-Strategie im Audit mitloggen (`single_pass` vs. `expanded_retry`).

Warum: nutzt Gemini-Long-Context gezielt statt pauschal immer große Prompts zu senden.

## Phase D – Topic-Entscheidung als Candidate-First + Evidence-Consensus
Empfehlung:
1. Candidate-Top-k deterministisch bilden (bereits vorhanden).
2. Gemini-Pass A soll Topic primär innerhalb Candidate-Set entscheiden.
3. Außerhalb Candidate-Set nur mit starker evidenzbasierter Begründung.
4. Bei Candidate-Konflikt + niedriger Topic-Confidence immer Verifikation.

Warum: stabilere Topic-Konsistenz und weniger Drift.

## Phase E – Inhaltsprüfung/Lösen mit evidenzgebundener Confidence
Empfehlung:
1. Gemini soll Antwortentscheidung chunk-übergreifend begründen.
2. Confidence darf nicht nur vom Modellscore abhängen, sondern von:
   - Retrieval-Qualität,
   - Evidenzabdeckung,
   - Pass-A/Pass-B-Konsens.
3. Bei unzureichender Evidenz: `cannotJudge`/Maintenance bevorzugen.

Warum: verhindert Scheinsicherheit.

## Phase F – Similarity/Clustering als Retrieval+Re-Ranking statt nur globaler Merge
Empfehlung:
1. Blocking (Topic/Jahr/Bildanker),
2. Kandidatenretrieval,
3. Re-Ranking (text + strukturelle + Bildsignale),
4. konservative Merge-Gates.

Gemini-Rolle:
- nur für Grenzfälle zur semantischen „tie-break“-Entscheidung einsetzen,
- nicht für jedes Clusterpaar (Kosten/Latency).

## Phase G – Rekonstruktion mangelhafter Fragen (Gemini als Redaktionsmodell)
Empfehlung:
1. Rekonstruktion nur mit explizitem Kontextpaket:
   - ähnliche Fragen,
   - gesicherte Evidence,
   - klare Constraints (nur eine korrekte Antwort etc.).
2. Ausgabe als „Vorschlag mit Evidenzanker“ statt stiller Überschreibung.
3. Auto-Apply nur bei hoher Evidenz + geringer Ambiguität.

Warum: maximiert Nutzen der Generierungsstärke von Gemini, minimiert Halluzinationsrisiko.

---

## 4) Konkrete Soll-Änderungen gegenüber einem OpenAI-nahen Workflow

1. **Provider-spezifische Orchestrierung** statt nur Provider-spezifischer API-Aufruf.
2. **Retrieval first-class**: adaptive Zweitstufe und Qualität als zentrale Steuergröße.
3. **Verifikation aggressiver bei schwacher Evidenz** (Gemini-Pfad).
4. **Auto-Change restriktiver** im Gemini-Pfad, wenn Evidenzqualität zu niedrig.
5. **Rekonstruktion stärker evidenzgebunden** mit klaren Freigaberegeln.

---

## 5) Priorisierte Umsetzung (empfohlen)

## Sprint 1 (hoher Impact, geringes Risiko)
- Retrieval-Qualität + Strategie vollständig im Audit/Run-Report ausweisen.
- Gemini-spezifische Preprocessing-Guards und Pass-B-Forcing final kalibrieren.
- Provider-spezifische Modelldefaults für alle Pässe konsistent halten.

## Sprint 2 (mittleres Risiko, hoher Qualitätsgewinn)
- Evidence-Coverage-Metrik ergänzen (wie viele Optionen/Claims durch Evidence gestützt sind).
- Candidate-Conflict-Handling + Topic-Drift-Metriken verschärfen.
- Rekonstruktions-Freigaberegeln mit Evidenzanker härten.

## Sprint 3 (optional, forschungsnah)
- Cluster-Re-Ranking um Embeddings erweitern.
- Fachspezifische dynamische Schwellen aus Verlaufsmessungen lernen.

---

## 6) Antwort auf die Kernfrage

Ja: Für eure Aufgaben gibt es **eine bessere Herangehensweise speziell für Gemini** als den bisherigen OpenAI-nahen Ablauf.

Der Schlüssel ist nicht nur „anderes Modell“, sondern eine **andere Steuerlogik**:
- evidenz- und retrievalgetrieben,
- adaptiv bei Unsicherheit,
- konservativ bei Datensatzänderungen,
- stärkerer Einsatz von Gemini dort, wo Long-Context und multimodale Synthese echten Mehrwert liefern.

So steigt die erwartete Ergebnisqualität besonders bei:
- schwierigen Topic-Abgrenzungen,
- fragmentierter Knowledge-Base-Evidenz,
- bildabhängigen Fragen,
- Rekonstruktion unvollständiger Altfragen.

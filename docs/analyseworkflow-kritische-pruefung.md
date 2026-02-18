# Analyseworkflow des AI Exam Analyzer (implementierungsnah)

Dieses Dokument beschreibt den **tatsächlich implementierten** Analyseworkflow in der aktuellen Codebasis, mit dem Ziel, ihn systematisch kritisch zu prüfen und gezielt zu verbessern.

> Fokus: exakte Abbildung der Ist-Logik (inkl. Schwellwerte, Trigger, Scores, Datenfluss).

## 1) Zielbild und Geltungsbereich

Die Pipeline analysiert pro Frage:
- fachliche Plausibilität der hinterlegten richtigen Antwort(en),
- ggf. Korrekturvorschlag,
- initiale und finale Themenzuordnung,
- Wartungsbedarf (Maintenance),
- Vertrauensmaße (Einzel- und kombinierte Confidence),
- Kontextbezüge (Textcluster, Bildcluster, Knowledge-Retrieval).

Die Analyse läuft standardmäßig als **2-Pass-Verfahren** (Pass A + konditional Pass B) mit optionalem **Review-Pass C**.

## 2) Modelle und Begründung der Modellwahl (Ist-Stand)

### 2.1 Standardmodelle

Per Default werden genutzt:
- **Pass A:** `gpt-4.1-mini`
- **Pass B:** `o4-mini`
- **Review-Pass:** `o4-mini` (nur wenn aktiviert)

### 2.2 Implementierte Begründungslogik

Die Modellaufteilung spiegelt eine Rollen-Trennung wider:
1. **Pass A als Erstanalysator** (kosten-/latenzorientiert, breite Erstanalyse inkl. Topic + Answer Review).
2. **Pass B als unabhängiger Verifier** (strenger, konservativer, reasoning-fokussiert durch `reasoning_effort=high`).
3. **Pass C als Eskalation** nur bei schwierigen/konflikthaften Fällen.

Diese Trennung ist in Prompts und Triggerlogik verankert:
- Pass A führt vollständige Erstbewertung aus.
- Pass B wird nur bei Risikoindikatoren gestartet.
- Review-Pass wird nur bei erhöhtem Wartungs-/Konfliktsignal gestartet.

### 2.3 Modell-spezifische API-Parameter

- Für Reasoning-Modelle (o-Serie / `gpt-5*`) wird **kein `temperature`** gesetzt.
- Für Reasoning-Modelle kann `reasoning.effort` gesetzt werden (für Pass B standardmäßig `high`).
- Structured Output erfolgt strikt über JSON-Schema (`strict: true`).

## 3) Input, Vorverarbeitung und Kontextaufbau

## 3.1 Input-Formate

Unterstützt werden:
- Datensatz als `[{question...}]` oder `{ "questions": [...] }`.
- Themenbaum (`superTopics/subtopics`) zur Topic-Key-Katalogisierung.
- Optional:
  - `images.zip` (Fragebilder),
  - `knowledge.zip` bzw. `knowledge-index` (Wissensbasis aus PDF/TXT/MD),
  - Cleanup-Spec (Output-Whitelist).

## 3.2 Payload pro Frage

Für jeden Eintrag wird ein Prompt-Payload erzeugt mit:
- `questionId`, `questionText`,
- `answers` (Index, ID, Text),
- `currentCorrectIndices`,
- `explanationText`,
- Bildindikatoren (`hasImages`, `imageRefs`, `imageUrls`).

Zusätzlich wird später angereichert um:
- `imageContext`,
- `questionClusterContext`,
- `imageClusterContext`,
- `knowledgeImageContext`,
- `retrievedEvidence` (falls Knowledge aktiv).

## 3.3 Datensatzweiter Kontext (vor der eigentlichen Modellanalyse)

Vor der Schleife über Einzel-Fragen wird ein globaler Kontext berechnet:

1. **Textcluster über Frageninhalt**
   - Tokenisierung (lowercase, alnum + Umlaute, Tokenlänge >= 3).
   - Ähnlichkeit = **Weighted Jaccard** (IDF-gewichtet) über Tokenmengen.
   - Clustering via Union-Find über Kandidatenpaare aus invertiertem Tokenindex.
   - Default-Schwelle: `text_cluster_similarity = 0.32`.

2. **Bildcluster über Fragebilder** (falls `images.zip` vorhanden)
   - Perzeptueller Hash je Bild.
   - Clusterbildung über Hamming-Distanz (`<= 8`).

3. **Knowledge-Bild-Matching** (falls zusätzlich Knowledge-Basis vorhanden)
   - Vergleich Fragebild-Hash vs. Knowledge-Bild-Hashes.
   - Treffer bei Hamming-Distanz `<= 10`.

## 3.4 Retrieval aus Wissensbasis (RAG-ähnlich)

Wenn Knowledge aktiviert ist:
1. Dokumente aus ZIP (PDF/TXT/MD) werden in Chunks zerlegt.
2. Pro Chunk werden Token, Termfrequenzen und Längen gehalten.
3. Für jede Frage wird Query aus `questionText + explanationText + answer texts` gebaut.
4. Ranking: BM25-artiger Score (`k1=1.4`, `b=0.72`).
5. Auswahl: Greedy mit Diversitätsbonus (`+0.12` bei neuer Quelle), begrenzt durch:
   - `top_k` (Default 6),
   - `min_score` (Default 0.06),
   - `max_chars` (Default 4000).
6. Retrieval-Qualität:
   - `retrieval_quality = 1 - exp(-0.35 * mean_score)` (auf 4 Nachkommastellen gerundet).

## 4) Kernalgorithmus pro Frage

## 4.1 Pass A (immer)

Pass A erhält Fragepayload + optionale Bilder. Zusätzlich kann das Preprocessing deterministische `topicCandidates` (Top-k) beilegen, um die Topic-Auswahl zu begrenzen.

Pass A muss im Schema liefern:
- `topic_initial` (nur aus Frage + Antworten),
- `answer_review` (Plausibilität + Änderungsvorschlag + Evidenz-IDs),
- `maintenance`,
- `topic_final`,
- `question_abstraction.summary`.

Wesentliche Prompt-Regeln:
- Bildkontext verpflichtend berücksichtigen, falls vorhanden.
- Bei fehlender/schwacher Evidenz konservative Confidence + ggf. Wartungsflag.
- `proposedCorrectIndices`/`verifiedCorrectIndices` verwenden den `answerIndex` der Antwortoptionen (1-basiert), nicht die Array-Position.
- Bei erwartetem aber fehlendem Bild: Maintenance setzen.

## 4.2 Triggerlogik für Pass B

Pass B wird gestartet, wenn **mindestens eine** Bedingung zutrifft:
1. Pass A empfiehlt Änderung (`recommendChange == true`), oder
2. Antwort-Confidence Pass A < `trigger_answer_conf` (Default 0.80), oder
3. Pass A markiert Maintenance, oder
4. `topic_initial.confidence` < `trigger_topic_conf` (Default 0.85), oder
5. `topic_final.confidence` < `trigger_topic_conf`.

## 4.3 Pass B (konditional)

Pass B ist unabhängige Verifikation und bekommt:
- komplette Frage + Kontext,
- Pass-A-Ergebnis.

Pass B liefert:
- `verify_answer` (inkl. `cannotJudge`, `agreeWithChange`, `verifiedCorrectIndices`, Confidence, Evidenz-IDs),
- `maintenance`,
- `topic_final`.

Maintenance-Reasons aus Pass A und Pass B werden dedupliziert zusammengeführt.

## 4.4 Änderung der Datensatzlösung (nur nach Regelwerk)

Eine Änderung der `correctIndices` erfolgt **nur**, wenn alle Bedingungen erfüllt sind:
1. `cannotJudge == false`,
2. `agreeWithChange == true`,
3. `verifiedCorrectIndices` nicht leer und ungleich aktuellem Stand,
4. `confidence_b >= apply_change_min_conf_b` (Default 0.80),
5. nicht gleichzeitig `evidence_count <= 0` und `retrieval_quality < 0.08`.

Wenn Bedingung verletzt ist, bleibt Datensatz unverändert.

## 4.5 Kombinierte Confidence (kalibrierte Heuristik)

Nach Pass A/B wird eine kombinierte Confidence berechnet:

`score = 0.34*answer_conf + 0.24*topic_conf + 0.20*retrieval_quality + 0.14*agreement + 0.08*evidence_prior`

mit:
- `agreement = 1.0` (Verifier stimmt zu), `0.45` (kein Verifier), `0.2` (Verifier stimmt nicht zu),
- `evidence_prior = 1.0` (>=3 Evidenzen), `0.8` (2), `0.55` (1), `0.35` (0).

Score wird auf `[0,1]` begrenzt und auf 4 Nachkommastellen gerundet.

## 4.6 Low-Confidence-Maintenance-Regel

Unabhängig von bisherigen Flags wird `needsMaintenance=true`, wenn einer dieser Werte unter Schwellwert liegt (Default 0.65):
- finale Antwort-Confidence,
- finale Topic-Confidence,
- kombinierte Confidence.

Dann zusätzlich:
- `severity >= 2`,
- Reason-Ergänzung: `low_confidence_answer_or_topic_or_combined`.

## 4.7 Optionaler Review-Pass C

Nur aktiv, wenn `enable_review_pass=true`.

Run-Trigger (mindestens einer):
1. Maintenance aktiv und `severity >= review_min_maintenance_severity` (Default 2),
2. AI widerspricht Datensatz und `final_combined_confidence < 0.85`,
3. Topic-Wechsel zwischen Pass A und finalem Topic,
4. `final_combined_confidence < max(0.45, low_conf_threshold - 0.1)`.

Review kann finale Indizes und finales Topic überschreiben und manuelles Review empfehlen.

## 5) Output-Logik

## 5.1 Audit-Struktur pro Frage (`aiAudit`)

Persistiert werden u. a.:
- Pipeline-Version,
- Modellnamen je Pass,
- Topic initial/final inkl. kurzer und detaillierter Gründe,
- Answer-Plausibility inkl. finaler Indizes, Confidence, RetrievalQuality, Evidenz-Metadaten,
- Verifikationsdetails (Pass B),
- Maintenance-Entscheid,
- Frageabstraktion,
- Cluster-IDs.

## 5.2 Top-Level-Komfortfelder (optional)

Bei `write_top_level=true` werden zusätzliche Felder gesetzt:
- `aiSuperTopic`, `aiSubtopic`, `aiTopicConfidence`,
- `aiNeedsMaintenance`, `aiMaintenanceSeverity`, `aiMaintenanceReasons`.

## 5.3 Abstraktions-Clusterung nach Abschluss

Nach Verarbeitung aller Fragen:
- Clustering über `question_abstraction.summary` (fallback: `questionText`),
- gleiche Token-/Weighted-Jaccard-/Union-Find-Methodik mit Kandidatenretrieval (Top-k) und Rare-Token-Merge-Gates,
- Default-Schwelle: `abstraction_cluster_similarity = 0.45`.

Cluster-ID wird in `aiAudit.clusters.abstractionClusterId` gespeichert.

## 6) Kritische Prüfpunkte für Verbesserungen

Für eine fundierte Weiterentwicklung sollten insbesondere diese Punkte auditiert werden:

1. **Kalibrierung der Schwellenwerte**
   - Trigger 0.80/0.85/0.65 und Change-Minimum 0.80 sind heuristisch.
   - Empfehlung: gegen Gold-Set/Review-Set empirisch neu kalibrieren.

2. **Confidence-Formel**
   - Gewichte (0.34/0.24/0.20/0.14/0.08) sind fest kodiert.
   - Empfehlung: Metrik-gesteuertes Tuning (z. B. Brier Score, Precision@Change).

3. **Retrieval-Bias durch Chunking und Min-Score**
   - PDF-Extraktion ohne OCR kann bildlastige Inhalte verlieren.
   - Empfehlung: OCR-Pfad und Quellgewichtung prüfen.

4. **Bildhash-basierte Ähnlichkeit**
   - pHash + Hamming kann bei Diagrammvarianten empfindlich sein.
   - Empfehlung: robustere Vision-Embeddings ergänzen.

5. **Pass-B-Triggerbreite**
   - Derzeit relativ aggressiv (mehrere OR-Bedingungen).
   - Empfehlung: Kosten-Qualitäts-Analyse für Triggerverfeinerung.

6. **Review-Pass Governance**
   - Review kann finale Felder überschreiben.
   - Empfehlung: optionales „no auto-override“-Profil für konservativen Betrieb.

## 7) Kurzreferenz: Default-Parameter

- `passA_model = gpt-4.1-mini`
- `passB_model = o4-mini`
- `passB_reasoning_effort = high`
- `trigger_answer_conf = 0.80`
- `trigger_topic_conf = 0.85`
- `apply_change_min_conf_b = 0.80`
- `low_conf_maintenance_threshold = 0.65`
- `knowledge_top_k = 6`
- `knowledge_max_chars = 4000`
- `knowledge_min_score = 0.06`
- `text_cluster_similarity = 0.32`
- `abstraction_cluster_similarity = 0.45`
- `enable_review_pass = false`
- `review_model = o4-mini`
- `review_min_maintenance_severity = 2`

## 8) Fazit

Der implementierte Algorithmus ist ein **mehrstufiges, evidenzbasiertes Prüfverfahren** mit klarer Trennung von Erstanalyse und Verifikation.
Die aktuelle Logik priorisiert Vorsicht (Maintenance/Verifier/Confidence-Kombination), bleibt aber in zentralen Parametern heuristisch.
Damit ist sie gut für produktiven Betrieb mit Human-Review-Anbindung geeignet, benötigt für maximale Qualität jedoch eine datengetriebene Nachkalibrierung der Schwellen, Gewichte und Trigger.

## 9) Ergebnis der Implementierungsprüfung (Code-Check)

Die Codebasis wurde darauf geprüft, ob der dokumentierte Workflow praktisch ausführbar ist und ob UI-Parameter bis in die interne Verarbeitung durchgereicht werden.

### 9.1 Strukturelle Befunde

1. **Output-Ordner in der UI war zuvor nur teilweise wirksam**
   - Vor der Korrektur wurde der automatisch abgeleitete Output-Pfad aus dem Input-Pfad erzeugt.
   - Dadurch konnte die Auswahl „Ausgabeordner“ wirkungslos sein, wenn kein expliziter Output-Dateiname gesetzt wurde.
   - **Status:** behoben (Output-Autopfad berücksichtigt jetzt den gewählten Ausgabeordner).

2. **`knowledge_subject_hint` war zuvor funktional wirkungslos**
   - In der ZIP-Ingestion wurde zwar `subject_hint` tokenisiert, aber bei fehlender Übereinstimmung nur `pass` ausgeführt.
   - Damit hatte die UI-Einstellung „Subject Hint“ keinen Effekt auf den Index.
   - **Status:** behoben (bei gesetztem Subject Hint werden nur Dateien mit Token-Überlappung indexiert).

### 9.2 UI → interne Settings (Wirkungsprüfung)

Folgende UI-Parameter wirken auf die interne Pipeline:
- Pass-A/B Modell + Temperatur/Reasoning Effort,
- Trigger- und Change-Schwellen,
- Low-Confidence-Maintenance,
- Knowledge-Parameter (`top_k`, `max_chars`, `min_score`, `chunk_chars`),
- Clustering-Schwellen,
- Review-Pass-Schalter und Schweregrad,
- Debug/Top-Level-Write,
- Eingabe-/Ausgabe-/ZIP-/Index-Pfade.

Nach den beiden Korrekturen sind auch die zuvor kritischen Punkte
**Output-Folder-Ableitung** und **Subject-Hint-Wirkung** technisch wirksam.

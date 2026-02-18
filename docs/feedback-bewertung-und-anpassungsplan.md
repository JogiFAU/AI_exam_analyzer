# Bewertung des Feedbacks & Anpassungsplan

## Kurzfazit
Das Feedback ist **inhaltlich sehr treffend** und passt gut zur aktuellen Pipeline-Architektur:

- Die aktuelle 2-Pass-Logik ist konservativ und damit für "Probleme markieren" besser geeignet als für automatische Korrekturen.
- Die größten Risiken liegen nicht im Pass-A/Pass-B-Prinzip, sondern in
  1) der Datenqualität (fehlende Assets/Indices, sehr kurze Fragen) und
  2) der derzeit einfachen Text-Clusterlogik (Token-Set + Jaccard + Union-Find).

## Gegencheck an `sample_data/export.json`
Eine schnelle Reproduktionsauswertung auf den mitgelieferten Beispieldaten bestätigt die Stoßrichtung des Feedbacks:

- 155 Fragen insgesamt.
- Nur 17 Fragen mit `explanationText`.
- 23 Fragen mit Bildreferenzen (`imageUrls`/`imageFiles`).
- 7 Fragen ohne `correctIndices`.
- 6 Fragen mit mindestens einer Antwortoption `?`.
- Viele sehr kurze Fragen (`<=3` Wörter), je nach Tokenisierungsheuristik im Bereich ca. 18–20.
- Zusätzlich: Fragen mit Bildhinweis im Text ohne Bild-Asset (heuristikabhängig; hier 13 gefunden, abhängig von Regex/Sprachvarianten).

Damit ist die Kernaussage robust: **Maintenance-Flags sind sinnvoll und erwartbar; vollautomatische Korrektur bleibt limitiert.**

## Bewertung der einzelnen Feedbackpunkte

### 1) Eignung für Plausibilitätsprüfung
**Bewertung: Zustimmung.**
Die Pipeline ist richtig auf konservative Entscheidungen ausgelegt (Pass-B-Verifikation, Change-Gates, Low-Confidence-Maintenance).

**Risiko:** Bei schwachem oder deaktiviertem Retrieval wird `finalCombinedConfidence` systematisch gedrückt, weil `retrieval_quality` und `evidence_prior` fix in die Kombinationsformel eingehen.

**Empfehlung:** Kombinierte Confidence retrieval-aware machen:
- wenn Knowledge deaktiviert: Retrieval-Anteil neutralisieren (Gewichte re-normalisieren oder neutrale Defaults),
- wenn Knowledge aktiviert, aber 0 Treffer: weiter konservativ bleiben.

### 2) Eignung für thematische Gruppierung
**Bewertung: Teilweise Zustimmung (kritischer Punkt).**
Die aktuelle Clusterbildung ist schnell, aber anfällig für Template-Überlappungen ("Was stimmt", "Welche Aussage", ...), besonders bei kurzen Fragen.

**Konsequenz:** Falsch positive Kanten können über Union-Find transitiv große, thematisch heterogene Cluster erzeugen.

**Empfehlung (kurzfristig):**
- Stopwords + Template-Tokens entfernen,
- zusätzlich `explanationText` in den Clustering-Text aufnehmen,
- Cluster-Kontext nur nutzen, wenn Cluster kohäsiv genug ist.

### 3) Deterministische Problemfinder vor LLM
**Bewertung: Sehr hohe Priorität, volle Zustimmung.**
Die vorgeschlagenen Checks sind mit hohem ROI und niedriger Komplexität umsetzbar.

Empfohlene strukturierte Reasons:
- `missing_correct_indices`
- `invalid_answer_option`
- `missing_required_image_asset`
- `insufficient_question_context`
- `non_exam_question_or_uncertain_source`

Vorteil: Statt pauschal "low confidence" entstehen klar handlungsleitende Maintenance-Gründe.

## Priorisierter Umsetzungsplan (ROI-Orientierung)

### Phase 1 (schnell, hoher Nutzen)
1. **Deterministische Qualitätschecks vor Pass A** einbauen und als Maintenance-Reasons persistieren.
2. **Confidence-Berechnung retrieval-aware** machen (Knowledge aus/offline ≠ automatisch niedrige Gesamtconfidence).
3. **Clustering-Text erweitern** auf `questionText + answers[].text + explanationText`.

### Phase 2 (Clustering robust machen)
4. Tokenfilter erweitern: Stopwords, Template-Phrasen, sehr häufige Tokens (DF-Cutoff).
5. Similarity von Set-Jaccard auf IDF-gewichtete Variante umstellen (Weighted Jaccard oder TF-IDF Cosine).
6. Merge-Gates ergänzen, um Bridge-Ketten in Union-Find zu bremsen.

### Phase 3 (Taxonomie und Qualitätssicherung)
7. Topic-Katalog strikt als Blocking-Key nutzen (primär innerhalb Subtopic clustern).
8. Cluster-Qualität automatisch scorieren (cohesion/purity/bridge-score) und unsichere Cluster markieren.

## Entscheidungsvorlage
Wenn nur 1–2 Maßnahmen sofort gestartet werden sollen:

1. **Deterministische Problemfinder** (stabilste Verbesserung für "Probleme anmerken").
2. **Stopword/Template-Filter + IDF-Gewichtung im Clustering** (größter Hebel gegen thematisch falsche Cluster).

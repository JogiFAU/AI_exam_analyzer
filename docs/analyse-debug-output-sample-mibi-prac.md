# Analyse: `Sample_Data/mibi_prac/output/export AIannotated.json` (Debug-Log-Modus)

## Kurzfazit
Der Lauf ist **größtenteils erfolgreich**, aber nicht vollständig stabil:
- 154 von 155 Fragen wurden mit `aiAudit.status = "completed"` verarbeitet.
- 1 Frage endete mit `aiAudit.status = "error"` (unvollständige API-Antwort).
- Zusätzlich traten bei einzelnen Teil-Pässen häufig partielle Fehler auf (v. a. `reconstruction` und `reviewPass`).

## Auffälligkeiten im Datensatz

### 1) Ein echter Pipeline-Fehler auf Fragebene
- **1 Datensatz** mit Gesamtstatus `error`.
- Betroffene Frage-ID: `01f1-0751-0a3ca26c-ae00-79b8acb6e751`.
- Fehlertext im Audit: `Response not completed: incomplete`.

### 2) Partielle Fehler in Teil-Pässen trotz `completed`
- `reconstruction.error` kommt **98-mal** vor (davon 97-mal trotz Gesamtstatus `completed`).
- `reviewPass`:
  - **35** erfolgreich,
  - **25** mit Fehler,
  - **95** gar nicht vorhanden.
- `answerPlausibility` und `topicFinal` sind fast immer vorhanden (jeweils nur beim einen globalen Fehlerfall fehlend).

### 3) Maintenance-Flags wirken konsistent, aber mit Interpretationsbedarf
- Verteilung `aiNeedsMaintenance`:
  - `False`: 97
  - `True`: 57
  - `None`: 1 (Fehlerfall)
- Verteilung `aiMaintenanceSeverity`:
  - `1`: 97
  - `2`: 31
  - `3`: 26
  - `None`: 1
- Hinweis: Es gibt wenige Fälle (`7`), in denen `aiNeedsMaintenance = False` aber trotzdem Gründe (`aiMaintenanceReasons`) vorliegen. Das ist nicht zwingend falsch (kann als Hinweis statt Änderungsbedarf gemeint sein), sollte aber semantisch klar dokumentiert werden.

### 4) Technischer Portabilitäts-Hinweis
- In den Debug-Daten steht ein lokaler Windows-Pfad (`imageZipPath`), z. B. `C:/Users/.../images.zip`. Das ist für Debugging ok, aber nicht portabel für Team-/CI-Kontexte.

## Einschätzung: Sind Anpassungen notwendig?
Ja, es sind **gezielte Robustheits-Anpassungen** sinnvoll:

1. **Retries + Backoff für Responses-Calls**
   - Ziel: `incomplete`-Antworten besser abfangen statt sofort als Fehler zu markieren.

2. **Striktes `status == completed` entschärfen**
   - Aktuell führt jeder andere Zustand direkt in Fehlerpfade.
   - Besser: definierte Behandlung von Zwischen-/Sonderzuständen + ggf. Retry.

3. **Teil-Pässe entkoppelt und transparent behandeln**
   - `reconstruction`/`reviewPass` sollten bei Fehlern klar als „optional fehlgeschlagen“ markiert werden,
     ohne den Gesamtdurchlauf unnötig zu destabilisieren.
   - Gleichzeitig sollten diese Fehlschläge zentral aggregiert (Metriken) gemeldet werden.

4. **Semantik für Maintenance normalisieren**
   - Entscheiden, ob `aiMaintenanceReasons` bei `aiNeedsMaintenance = False` erlaubt ist.
   - Falls ja: Feldname/Definition ggf. präzisieren (z. B. „Hinweise“ vs. „Pflicht-Anpassungen“).

5. **Debug-Pfade optional anonymisieren/normalisieren**
   - Für reproduzierbare Exporte besser relative oder neutralisierte Pfade speichern.

## Priorisierung (empfohlen)
1. **P1:** Retry/Backoff + Status-Handling für API-Antworten.
2. **P1:** Stabilisierung `reconstruction`-Pass (da hohe Fehlerquote).
3. **P2:** Stabilisierung/Transparenz `reviewPass` (fehlend/Fehler klar unterscheiden).
4. **P3:** Maintenance-Semantik und Debug-Portabilität konsolidieren.

## Weitere Unstimmigkeiten / Qualitäts-Signale

### 5) Optional-Pässe sind stark asymmetrisch erfolgreich
- `reconstruction` ist nur bei **57** Fragen erfolgreich und bei **98** Fragen im Fehlerzustand.
- `reviewPass` ist bei **95** Fragen gar nicht vorhanden, bei **25** fehlerhaft und nur bei **35** erfolgreich.
- Das spricht für ein robustes Kern-Processing, aber fragile optionale Post-Processing-Pässe.

### 6) Confidence-/Quelle-Verteilung wirkt unausgewogen
- `finalAnswerConfidenceSource` stammt fast immer aus `passA` (**152/155**), nur in **2** Fällen aus `passB`.
- `topicFinal.source` ist ebenfalls stark auf frühe Schritte konzentriert (`passA`: **118**, `review`: **35**, `passB`: **1**).
- Das ist nicht zwingend ein Bug, kann aber auf begrenzten Zusatznutzen der späteren Korrektur-Pässe hindeuten.

### 7) Retrieval-Qualitätswert wirkt wenig trennscharf
- `retrievalQuality` liegt für alle Fragen sehr hoch im engen Bereich (**0.9319–1.0**).
- Falls dieser Wert als Entscheidungsgrundlage genutzt wird, sollte geprüft werden, ob die Metrik ausreichend zwischen guten/schwachen Retrieval-Treffern differenziert.

### 8) Einzelfall ohne finale Antwort-Indizes
- Es gibt genau **einen** Eintrag ohne `finalAiCorrectIndices` (korrespondierend zum globalen Fehlerfall).
- Für Downstream-Verarbeitung ist das ok, sofern `status=error` sauber abgefangen wird; sonst sollte ein expliziter Fallbackwert gesetzt werden.

### 9) Positiver Konsistenzcheck
- Keine doppelten Frage-IDs gefunden.
- Top-Level-Felder (`aiSuperTopic`, `aiSubtopic`, `aiTopicConfidence`, Maintenance-Felder) sind konsistent zu den jeweiligen `aiAudit`-Feldern.
- `correctIndices` und `correctAnswers` sind untereinander konsistent.

## Ergänzte Handlungsempfehlungen
6. **Monitoring für optionale Pass-Erfolgsraten einführen**
   - Separate KPIs für `reviewPass` und `reconstruction` (success/error/missing) pro Lauf.
   - Zielwerte definieren und bei Regression automatisch warnen.

7. **Nutzwertanalyse von Pass-B/Review quantifizieren**
   - Metriken: Wie oft ändern sich `finalCorrectIndices`/`topicFinal` tatsächlich und mit welchem Qualitätsgewinn?
   - Falls selten wirksam, Prompt-/Trigger-Logik gezielt nachschärfen.

8. **Retrieval-Quality-Metrik kalibrieren**
   - Prüfen, ob die aktuelle Skala zu eng ist.
   - Gegebenenfalls zusätzliche Signale ergänzen (z. B. Evidenzabdeckung, Widerspruchssignale, Quellenvielfalt).

# Gemini-Migration (v2) – Vorgehen ohne Verlust des aktuellen Stands

## 1) GitHub-Strategie: stabilen Stand behalten und parallel v2 entwickeln

Empfehlung: **GitFlow-light mit einem langlebigen Feature-Branch**.

1. `main` bleibt immer lauffähig (OpenAI-Version, Bugfixes nur per PR).
2. Für die neue Variante einen eigenen Branch anlegen, z. B. `feature/gemini-v2`.
3. In kurzen Zyklen arbeiten und regelmäßig mit `main` synchronisieren:
   - `git checkout main && git pull`
   - `git checkout feature/gemini-v2 && git rebase main` (oder `merge`, wenn Rebase nicht gewünscht ist)
4. Optional für größere Änderungen: Zwischen-Branches wie
   - `feature/provider-abstraction`
   - `feature/gemini-client`
   - `feature/gemini-ui-config`
5. Erst mergen, wenn beide Bedingungen erfüllt sind:
   - Regressionstests gegen OpenAI-Pfad bleiben grün.
   - Gemini-Pfad liefert dieselben Pflichtfelder in den JSON-Strukturen.

Damit bleibt der aktuelle Produktionsstand unangetastet, während v2 isoliert und reviewbar entwickelt wird.

---

## 2) Konkrete Code-Änderungen für Gemini

### A) API-Anbindung entkoppeln (wichtigster Schritt)

Ist-Zustand:
- API-Client ist OpenAI-spezifisch (`ai_exam_analyzer/openai_client.py`).
- `process_questions` instanziiert direkt `OpenAI()` in `processor.py`.
- CLI/UI validieren nur `OPENAI_API_KEY`.

Soll-Zustand:
1. Neues Interface, z. B. `ai_exam_analyzer/llm_client.py`:
   - `call_json_schema(...)` als provider-neutrale Signatur.
2. Provider-Implementierungen:
   - `providers/openai_client.py` (bestehende Logik übernehmen)
   - `providers/gemini_client.py` (Gemini-Implementierung)
3. Factory/Resolver, z. B. `build_llm_client(provider, api_key, model, ...)`.
4. `passes.py` und `processor.py` nur noch gegen das Interface, nicht gegen OpenAI direkt.

Nutzen: Der Rest der Pipeline (Schemas, Decision Logic, Topic-/Cluster-Logik) bleibt weitgehend unverändert.

### B) Konfiguration erweitern

`config.py`, `cli.py`, `ui.py` um folgende Felder erweitern:
- `LLM_PROVIDER` (z. B. `openai|gemini`)
- `GEMINI_API_KEY` (alternativ gemeinsames `LLM_API_KEY`)
- optionale provider-spezifische Defaults (Modelnamen, Retries, Tokenlimits)

### C) Structured-Output/Schema-Handling für Gemini

Der zentrale technische Punkt ist, dass dieselben Ergebnisobjekte entstehen wie heute.

- Die bestehenden Schemata in `passes.py` sollen **führend** bleiben.
- Gemini-Responses müssen auf exakt diese Dict-Strukturen normalisiert werden.
- Bei teilweiser Inkompatibilität: robustes Parsing + Validierung + Retry + Fallback-Strategie.

### D) Fehlerbehandlung und Retries provider-unabhängig machen

Die Retry-Logik aus der aktuellen OpenAI-Implementierung sollte als gemeinsame Utility nutzbar sein,
weil ähnliche Probleme (Timeouts, unvollständige Antworten, temporäre API-Fehler) bei beiden Providern auftreten.

### E) Tests absichern

Mindestens:
1. Contract-Tests: OpenAI und Gemini liefern dieselbe Zielstruktur pro Pass.
2. Snapshot-/Golden-Tests auf kleinen Sample-Daten (`Sample_Data/...`).
3. Regression: OpenAI-Flow darf durch Gemini-Einführung nicht brechen.

---

## 3) Robustheit und Risiko-Einschätzung

## Kurzfazit

**Robust umsetzbar ohne kompletten Neuaufbau**, wenn die Provider-Abstraktion sauber als erste Etappe kommt.

## Warum kein vollständiger Neuaufbau nötig ist

Die fachliche Logik ist bereits gut getrennt:
- Pipeline/Pässe/Entscheidungslogik sind unabhängig von einem konkreten Provider.
- Der größte Kopplungspunkt ist die direkte OpenAI-Client-Erzeugung plus OpenAI-spezifischer Call-Wrapper.

Das ist eine **mittelgroße Refaktorierung**, kein Architektur-Neustart.

## Reale Risiken

1. **Schema-/Output-Abweichungen** zwischen Providern
   - Risiko: Parsen schlägt sporadisch fehl.
   - Gegenmaßnahme: strenge Validierung, normalisierte Mapper, deterministische Retries.

2. **Parameter-Unterschiede** (temperature, token limits, reasoning options)
   - Risiko: Laufzeitfehler bei nicht unterstützten Parametern.
   - Gegenmaßnahme: provider-spezifische Parametermatrix und defensive Defaults.

3. **Qualitätsdrift in Bewertungen**
   - Risiko: andere Modellcharakteristik verändert Pass-Entscheidungen.
   - Gegenmaßnahme: Vergleichsläufe auf denselben Sample-Datensätzen + Schwellwerte ggf. feinjustieren.

## Empfehlung für Umsetzung in Phasen

1. Provider-Interface + OpenAI-Adapter (funktional identisch zu heute).
2. Gemini-Adapter mit minimaler Funktionsmenge (ein Pass-End-to-End).
3. Alle Pässe auf Gemini erweitern.
4. UI/CLI-Umschalter finalisieren.
5. Test-/Benchmark-Runde und erst dann Merge.

So bleibt das Projekt kontinuierlich lauffähig, und das Migrationsrisiko bleibt kontrollierbar.

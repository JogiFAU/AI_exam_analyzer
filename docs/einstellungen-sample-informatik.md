# Empfohlene Einstellungen – Sample_Data/informatik

Diese Konfiguration ist für `Sample_Data/informatik/export.json` optimiert.

## Datensatzprofil (kurz)
- 255 Fragen, 7 Jahrgänge (2019–2025)
- Keine fehlenden `correctIndices`
- 29 Hinweise auf unsichere/Legacy-Formulierungen (u. a. "Altfrage")
- Keine expliziten `imageFiles`/`imageUrls` im Export

## Empfohlene Parameter

| Bereich | Einstellung | Wert |
|---|---|---|
| Input | `--input` | `Sample_Data/informatik/export.json` |
| Topics | `--topics` | `Sample_Data/informatik/topic-tree.json` |
| Bilder | `--images-zip` | `Sample_Data/informatik/images.zip` |
| Knowledge | `--knowledge-zip` | `Sample_Data/informatik/knowledge.zip` |
| Knowledge | `--knowledge-top-k` | `8` |
| Knowledge | `--knowledge-max-chars` | `5000` |
| Knowledge | `--knowledge-min-score` | `0.05` |
| Core | `--trigger-answer-conf` | `0.82` |
| Core | `--trigger-topic-conf` | `0.86` |
| Core | `--low-conf-maintenance-threshold` | `0.68` |
| Optional | `--enable-review-pass` | aktiv |
| Optional | `--review-model` | `o4-mini` |
| Optional | `--review-min-maintenance-severity` | `2` |
| Optional | `--enable-repeat-reconstruction` | aktiv |
| Optional | `--auto-apply-repeat-reconstruction` | aktiv (nur Audit-Suggestion) |
| Optional | `--repeat-min-similarity` | `0.74` |
| Optional | `--repeat-min-anchor-conf` | `0.84` |
| Optional | `--repeat-min-anchor-consensus` | `2` |
| Optional | `--repeat-min-match-ratio` | `0.65` |
| Optional | `--enable-reconstruction-pass` | aktiv |
| Optional | `--reconstruction-model` | `o4-mini` |
| Optional | `--enable-explainer-pass` | aktiv |
| Optional | `--explainer-model` | `gpt-4.1-mini` |
| Logging | `--run-report` | `Sample_Data/informatik/run-report.json` |

## Begründung
1. **Review/Repeat/Reconstruction/Explainer aktiv**: Wegen vieler Jahrgänge und erkennbarer Legacy-Hinweise lohnt sich die mehrstufige Nachprüfung.
2. **Knowledge etwas breiter (`top-k=8`)**: Informatikfragen profitieren meist von zusätzlichem Konzeptkontext aus PDFs/TXT.
3. **Leicht strengere Confidence-Schwellen**: Der Datensatz ist grundsätzlich stabil (keine fehlenden Lösungen), daher kann die Pipeline konservativer entscheiden.
4. **Repeat-Konsens = 2**: Vermeidet, dass einzelne Ausreißer als zu starke Vorlage dienen.

## Beispielaufruf

```bash
python classify_topics_merged_config_fixed.py \
  --input Sample_Data/informatik/export.json \
  --topics Sample_Data/informatik/topic-tree.json \
  --output Sample_Data/informatik/export.AIannotated.json \
  --images-zip Sample_Data/informatik/images.zip \
  --knowledge-zip Sample_Data/informatik/knowledge.zip \
  --knowledge-top-k 8 \
  --knowledge-max-chars 5000 \
  --knowledge-min-score 0.05 \
  --trigger-answer-conf 0.82 \
  --trigger-topic-conf 0.86 \
  --low-conf-maintenance-threshold 0.68 \
  --enable-review-pass \
  --review-model o4-mini \
  --review-min-maintenance-severity 2 \
  --enable-repeat-reconstruction \
  --auto-apply-repeat-reconstruction \
  --repeat-min-similarity 0.74 \
  --repeat-min-anchor-conf 0.84 \
  --repeat-min-anchor-consensus 2 \
  --repeat-min-match-ratio 0.65 \
  --enable-reconstruction-pass \
  --reconstruction-model o4-mini \
  --enable-explainer-pass \
  --explainer-model gpt-4.1-mini \
  --run-report Sample_Data/informatik/run-report.json
```

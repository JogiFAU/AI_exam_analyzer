# Empfohlene Einstellungen – Sample_Data/mibi_prac

Diese Konfiguration ist für `Sample_Data/mibi_prac/export.json` optimiert.

## Datensatzprofil (kurz)
- 155 Fragen, 5 Jahrgänge (2021–2025)
- 7 Fragen ohne `correctIndices`
- Mehrere sehr kurze Fragen (<=3 Wörter)
- Keine expliziten `imageFiles`/`imageUrls` im Export

## Empfohlene Parameter

| Bereich | Einstellung | Wert |
|---|---|---|
| Input | `--input` | `Sample_Data/mibi_prac/export.json` |
| Topics | `--topics` | `Sample_Data/mibi_prac/topic-tree.json` |
| Bilder | `--images-zip` | `Sample_Data/mibi_prac/images.zip` |
| Knowledge | `--knowledge-zip` | `Sample_Data/mibi_prac/knowledge.zip` |
| Knowledge | `--knowledge-top-k` | `10` |
| Knowledge | `--knowledge-max-chars` | `5500` |
| Knowledge | `--knowledge-min-score` | `0.045` |
| Core | `--trigger-answer-conf` | `0.78` |
| Core | `--trigger-topic-conf` | `0.83` |
| Core | `--low-conf-maintenance-threshold` | `0.64` |
| Optional | `--enable-review-pass` | aktiv |
| Optional | `--review-model` | `o4-mini` |
| Optional | `--review-min-maintenance-severity` | `2` |
| Optional | `--enable-repeat-reconstruction` | aktiv |
| Optional | `--auto-apply-repeat-reconstruction` | aktiv (nur Audit-Suggestion) |
| Optional | `--repeat-min-similarity` | `0.70` |
| Optional | `--repeat-min-anchor-conf` | `0.80` |
| Optional | `--repeat-min-anchor-consensus` | `1` |
| Optional | `--repeat-min-match-ratio` | `0.60` |
| Optional | `--enable-reconstruction-pass` | aktiv |
| Optional | `--reconstruction-model` | `o4-mini` |
| Optional | `--enable-explainer-pass` | aktiv |
| Optional | `--explainer-model` | `gpt-4.1-mini` |
| Logging | `--run-report` | `Sample_Data/mibi_prac/run-report.json` |

## Begründung
1. **Mehr Korrektur-/Rekonstruktionsbedarf**: Fehlende `correctIndices` und kurze Fragen sprechen für stärkere Kontextnutzung.
2. **Knowledge stärker priorisieren (`top-k=10`, niedrigeres `min-score`)**: Damit auch schwächer formulierte Fragen genügend Evidenz bekommen.
3. **Etwas niedrigere Trigger-Schwellen**: Hilft, mehr schwierige Fälle in Pass B/C zu ziehen statt früh festzuschreiben.
4. **Repeat-Parameter toleranter**: Bei kleineren/uneinheitlicheren Formulierungen erhöht das die Chance, Altfragen-Bezüge trotzdem zu erkennen.

## Beispielaufruf

```bash
python classify_topics_merged_config_fixed.py \
  --input Sample_Data/mibi_prac/export.json \
  --topics Sample_Data/mibi_prac/topic-tree.json \
  --output Sample_Data/mibi_prac/export.AIannotated.json \
  --images-zip Sample_Data/mibi_prac/images.zip \
  --knowledge-zip Sample_Data/mibi_prac/knowledge.zip \
  --knowledge-top-k 10 \
  --knowledge-max-chars 5500 \
  --knowledge-min-score 0.045 \
  --trigger-answer-conf 0.78 \
  --trigger-topic-conf 0.83 \
  --low-conf-maintenance-threshold 0.64 \
  --enable-review-pass \
  --review-model o4-mini \
  --review-min-maintenance-severity 2 \
  --enable-repeat-reconstruction \
  --auto-apply-repeat-reconstruction \
  --repeat-min-similarity 0.70 \
  --repeat-min-anchor-conf 0.80 \
  --repeat-min-anchor-consensus 1 \
  --repeat-min-match-ratio 0.60 \
  --enable-reconstruction-pass \
  --reconstruction-model o4-mini \
  --enable-explainer-pass \
  --explainer-model gpt-4.1-mini \
  --run-report Sample_Data/mibi_prac/run-report.json
```

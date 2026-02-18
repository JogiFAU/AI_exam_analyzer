# Repair-Run Analyse (mibi_prac)

- Datensatz: `Sample_Data/mibi_prac/output/export AIannotated.json`
- Fragen gesamt: **155**
- `aiAudit.status`: {'completed': 155}

## Pass-/Run-Status
- Verification vorhanden: 155 (fehlend: 0)
- Verification `ran=true`: 132; `ran=false`: 23
- Verification mit Error: **129**; ohne Error: 26
- ReviewPass OK: 36; ReviewPass Error: **25**; ReviewPass `null` (nicht ausgeführt): **94**
- PassB (`_debug.passB_raw`) vorhanden (lief): **2**; `null` (nicht gelaufen): **152**; fehlend: 1

## Fehlerbilder
- `Response not completed: incomplete`: **154** Vorkommen

## Cluster-Verteilung
- Content-Cluster (`questionContentClusterId`): **131** eindeutige Cluster bei 155 Fragen
- Abstraction-Cluster (`abstractionClusterId`): **134** eindeutige Cluster bei 155 Fragen
- Fragen mit unterschiedlichem Content- vs. Abstraction-Cluster: **110** (71.0%)

### Größe der Cluster (Anzahl Fragen pro Cluster)
- Content-Cluster Größenverteilung (Clustergröße -> Anzahl solcher Cluster):
  - 1 -> 113
  - 2 -> 15
  - 3 -> 1
  - 4 -> 1
  - 5 -> 1
- Abstraction-Cluster Größenverteilung (Clustergröße -> Anzahl solcher Cluster):
  - 1 -> 118
  - 2 -> 11
  - 3 -> 5

### Größte Cluster
- Top 10 Content-Cluster (Cluster-ID -> Fragen):
  - 62 -> 5
  - 2 -> 4
  - 34 -> 3
  - 13 -> 2
  - 20 -> 2
  - 25 -> 2
  - 26 -> 2
  - 29 -> 2
  - 30 -> 2
  - 35 -> 2
- Top 10 Abstraction-Cluster (Cluster-ID -> Fragen):
  - 33 -> 3
  - 36 -> 3
  - 55 -> 3
  - 84 -> 3
  - 91 -> 3
  - 22 -> 2
  - 23 -> 2
  - 25 -> 2
  - 27 -> 2
  - 30 -> 2

### Image-Cluster
- Fragen ohne Bild-Cluster: **132**
- Fragen mit genau 1 Bild-Cluster: **23**
- Eindeutige Bild-Cluster insgesamt: **22**
- Top Bild-Cluster (Cluster-ID -> Fragen):
  - img-cluster-10 -> 2
  - img-cluster-1 -> 1
  - img-cluster-2 -> 1
  - img-cluster-3 -> 1
  - img-cluster-4 -> 1
  - img-cluster-5 -> 1
  - img-cluster-6 -> 1
  - img-cluster-7 -> 1
  - img-cluster-8 -> 1
  - img-cluster-9 -> 1

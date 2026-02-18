param(
    [Parameter(Mandatory=$true)]
    [string]$Input,

    [Parameter(Mandatory=$false)]
    [string]$Output = "",

    [Parameter(Mandatory=$false)]
    [double]$TextClusterSimilarity = 0.12,

    [Parameter(Mandatory=$false)]
    [double]$AbstractionClusterSimilarity = 0.18,

    [Parameter(Mandatory=$false)]
    [string]$ImagesZip = "",

    [Parameter(Mandatory=$false)]
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$args = @(
    "-m", "ai_exam_analyzer.recluster_only",
    "--input", $Input,
    "--text-cluster-similarity", $TextClusterSimilarity,
    "--abstraction-cluster-similarity", $AbstractionClusterSimilarity
)

if ($Output -ne "") {
    $args += @("--output", $Output)
}

if ($ImagesZip -ne "") {
    $args += @("--images-zip", $ImagesZip)
}

Write-Host "Starte Reclustering mit weniger strengen Kriterien..."
Write-Host "$PythonExe $($args -join ' ')"

& $PythonExe @args

if ($LASTEXITCODE -ne 0) {
    throw "Reclustering fehlgeschlagen mit Exit-Code $LASTEXITCODE"
}

Write-Host "Fertig."

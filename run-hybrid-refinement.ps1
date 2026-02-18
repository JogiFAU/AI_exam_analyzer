param(
    [Parameter(Mandatory=$true)]
    [string]$InputPath,

    [Parameter(Mandatory=$true)]
    [string]$TopicsPath,

    [Parameter(Mandatory=$false)]
    [string]$Output = "",

    [Parameter(Mandatory=$false)]
    [string]$ImagesZip = "",

    [Parameter(Mandatory=$false)]
    [string]$KnowledgeZip = "",

    [Parameter(Mandatory=$false)]
    [string]$KnowledgeIndex = "",

    [Parameter(Mandatory=$false)]
    [switch]$EnableReviewPass,

    [Parameter(Mandatory=$false)]
    [switch]$EnableReconstructionPass,

    [Parameter(Mandatory=$false)]
    [switch]$ForceRerunReview,

    [Parameter(Mandatory=$false)]
    [switch]$ForceRerunReconstruction,

    [Parameter(Mandatory=$false)]
    [string]$ClusterRefinementModel = "o4-mini",

    [Parameter(Mandatory=$false)]
    [int]$ClusterRefinementMaxClusters = 30,

    [Parameter(Mandatory=$false)]
    [int]$ClusterRefinementMinClusterSize = 2,

    [Parameter(Mandatory=$false)]
    [int]$ClusterRefinementMergeCandidates = 5,

    [Parameter(Mandatory=$false)]
    [double]$TextClusterSimilarity = 0.12,

    [Parameter(Mandatory=$false)]
    [double]$AbstractionClusterSimilarity = 0.18,

    [Parameter(Mandatory=$false)]
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

function Has-Text([string]$Value) {
    return -not [string]::IsNullOrWhiteSpace($Value)
}

if ($null -eq $InputPath) { $InputPath = "" }
if ($null -eq $TopicsPath) { $TopicsPath = "" }
if ($null -eq $Output) { $Output = "" }
if ($null -eq $ImagesZip) { $ImagesZip = "" }
if ($null -eq $KnowledgeZip) { $KnowledgeZip = "" }
if ($null -eq $KnowledgeIndex) { $KnowledgeIndex = "" }
if ($null -eq $PythonExe) { $PythonExe = "python" }

$InputPath = $InputPath.Trim()
$TopicsPath = $TopicsPath.Trim()
$Output = $Output.Trim()
$ImagesZip = $ImagesZip.Trim()
$KnowledgeZip = $KnowledgeZip.Trim()
$KnowledgeIndex = $KnowledgeIndex.Trim()
$PythonExe = $PythonExe.Trim()

if (-not (Has-Text $InputPath)) {
    throw "Input darf nicht leer sein."
}
if (-not (Has-Text $TopicsPath)) {
    throw "Topics darf nicht leer sein."
}

if (-not (Test-Path -LiteralPath $InputPath)) {
    throw "Input-Datei nicht gefunden: $InputPath"
}
if (-not (Test-Path -LiteralPath $TopicsPath)) {
    throw "Topic-Datei nicht gefunden: $TopicsPath"
}
if ((Has-Text $ImagesZip) -and -not (Test-Path -LiteralPath $ImagesZip)) {
    throw "ImagesZip-Datei nicht gefunden: $ImagesZip"
}
if ((Has-Text $KnowledgeZip) -and -not (Test-Path -LiteralPath $KnowledgeZip)) {
    throw "KnowledgeZip-Datei nicht gefunden: $KnowledgeZip"
}

$args = @(
    "classify_topics_merged_config_fixed.py",
    "--input", $InputPath,
    "--topics", $TopicsPath,
    "--postprocess-only",
    "--enable-llm-abstraction-cluster-refinement",
    "--cluster-refinement-model", $ClusterRefinementModel,
    "--cluster-refinement-max-clusters", $ClusterRefinementMaxClusters,
    "--cluster-refinement-min-cluster-size", $ClusterRefinementMinClusterSize,
    "--cluster-refinement-merge-candidates", $ClusterRefinementMergeCandidates,
    "--text-cluster-similarity", $TextClusterSimilarity,
    "--abstraction-cluster-similarity", $AbstractionClusterSimilarity
)

if (Has-Text $Output) {
    $args += @("--output", $Output)
}
if (Has-Text $ImagesZip) {
    $args += @("--images-zip", $ImagesZip)
}
if (Has-Text $KnowledgeZip) {
    $args += @("--knowledge-zip", $KnowledgeZip)
}
if (Has-Text $KnowledgeIndex) {
    $args += @("--knowledge-index", $KnowledgeIndex)
}
if ($EnableReviewPass) {
    $args += "--enable-review-pass"
}
if ($EnableReconstructionPass) {
    $args += "--enable-reconstruction-pass"
}
if ($ForceRerunReview) {
    $args += "--force-rerun-review"
}
if ($ForceRerunReconstruction) {
    $args += "--force-rerun-reconstruction"
}

Write-Host "Starte Hybrid-Refinement-Run (Postprocess-Only + LLM Cluster Refinement)..." -ForegroundColor Cyan
Write-Host "$PythonExe $($args -join ' ')" -ForegroundColor DarkGray

& $PythonExe @args

if ($LASTEXITCODE -ne 0) {
    throw "Hybrid-Refinement-Run fehlgeschlagen (Exit-Code $LASTEXITCODE)"
}

Write-Host "Fertig." -ForegroundColor Green

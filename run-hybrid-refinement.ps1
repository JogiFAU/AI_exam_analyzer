param(
    [Parameter(Mandatory=$true)]
    [string]$Input,

    [Parameter(Mandatory=$true)]
    [string]$Topics,

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

if (-not (Test-Path -LiteralPath $Input)) {
    throw "Input-Datei nicht gefunden: $Input"
}
if (-not (Test-Path -LiteralPath $Topics)) {
    throw "Topic-Datei nicht gefunden: $Topics"
}
if ($ImagesZip -ne "" -and -not (Test-Path -LiteralPath $ImagesZip)) {
    throw "ImagesZip-Datei nicht gefunden: $ImagesZip"
}
if ($KnowledgeZip -ne "" -and -not (Test-Path -LiteralPath $KnowledgeZip)) {
    throw "KnowledgeZip-Datei nicht gefunden: $KnowledgeZip"
}

$args = @(
    "classify_topics_merged_config_fixed.py",
    "--input", $Input,
    "--topics", $Topics,
    "--postprocess-only",
    "--enable-llm-abstraction-cluster-refinement",
    "--cluster-refinement-model", $ClusterRefinementModel,
    "--cluster-refinement-max-clusters", $ClusterRefinementMaxClusters,
    "--cluster-refinement-min-cluster-size", $ClusterRefinementMinClusterSize,
    "--cluster-refinement-merge-candidates", $ClusterRefinementMergeCandidates,
    "--text-cluster-similarity", $TextClusterSimilarity,
    "--abstraction-cluster-similarity", $AbstractionClusterSimilarity
)

if ($Output -ne "") {
    $args += @("--output", $Output)
}
if ($ImagesZip -ne "") {
    $args += @("--images-zip", $ImagesZip)
}
if ($KnowledgeZip -ne "") {
    $args += @("--knowledge-zip", $KnowledgeZip)
}
if ($KnowledgeIndex -ne "") {
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

Param(
  [switch]$CopyToRoot = $true
)

$ErrorActionPreference = 'Stop'

Function Log($msg){ Write-Host "[build] $msg" }

try {
  Log "Ingesting metrics..."
  if (Get-Command python -ErrorAction SilentlyContinue) {
    python scripts/ingest_metrics.py | Out-Null
  } else {
    Log "python not found; skipping ingestion"
  }

  # Ensure figures are available where LaTeX expects them
  if (-not (Test-Path paper\figures)) { New-Item -ItemType Directory paper\figures | Out-Null }
  $copies = @(
    @{src='figures\phase2_collapse_panels.png'; dst='paper\figures\phase2_collapse_panels.png'},
    @{src='phase3-out\phase3_alpha_vs_invD.png'; dst='paper\figures\phase3_alpha_vs_invD.png'},
    @{src='phase8-out\figures\phase8_var_scaling.png'; dst='paper\figures\phase8_var_scaling.png'},
    @{src='phase9-plus-haar-extend\figures\phase9_varY_vs_D_haar.png'; dst='paper\figures\phase9_varY_scaling.png'}
  )
  foreach($c in $copies){ if(Test-Path $c.src){ Copy-Item $c.src $c.dst -Force } }

  # Build LaTeX (pdflatex -> bibtex -> pdflatex x2)
  Push-Location paper
  $log = Join-Path (Get-Location) 'build.log'
  $job = 'main'
  # If main.pdf is locked, compile to an alternate jobname
  try {
    if (Test-Path main.pdf) { Remove-Item main.pdf -Force }
  } catch { $job = 'main_build' }
  Log "Using jobname: $job"
  Log "Running pdflatex (1/3)..."
  & pdflatex -interaction=nonstopmode -jobname=$job main.tex *> $log
  Log "Running bibtex..."
  & bibtex $job >> $log 2>&1
  Log "Running pdflatex (2/3)..."
  & pdflatex -interaction=nonstopmode -jobname=$job main.tex >> $log 2>&1
  Log "Running pdflatex (3/3)..."
  & pdflatex -interaction=nonstopmode -jobname=$job main.tex >> $log 2>&1
  Pop-Location

  # Verify PDF (use jobname)
  $outPdf = Join-Path 'paper' ("$job.pdf")
  $pdf = Get-Item $outPdf -ErrorAction SilentlyContinue
  if (-not $pdf) { throw "paper/main.pdf not found; see paper/build.log" }
  if ($pdf.Length -lt 1024) { throw "paper/main.pdf is too small ($($pdf.Length) bytes); see paper/build.log" }

  if ($CopyToRoot) {
    $dest = 'Curvature-Information-Principle.pdf'
    if (Test-Path $dest) {
      try { Copy-Item $outPdf $dest -Force } catch { Start-Sleep -Milliseconds 300; Copy-Item $outPdf $dest -Force }
    } else {
      Copy-Item $outPdf $dest -Force
    }
    Log "Copied to $dest ($((Get-Item $dest).Length) bytes)."
  }

  Log "Done."
} catch {
  Write-Host "[build] ERROR: $($_.Exception.Message)" -ForegroundColor Red
  Write-Host "[build] See paper/build.log for details." -ForegroundColor Yellow
  exit 1
}

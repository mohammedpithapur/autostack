# AutoStack llama.cpp Server Launcher
# This script helps start llama-server with proper memory management

Write-Host "═══ AutoStack llama.cpp Server Setup ═══`n" -ForegroundColor Cyan

# Check available models
$models_dir = "$env:USERPROFILE\AppData\Local\llama.cpp\models"
Write-Host "Checking for GGUF models in common locations...`n"

# Search for GGUF files
$gguf_files = @()

# Common download locations
$search_paths = @(
    "$env:USERPROFILE\Downloads",
    "$env:USERPROFILE\Documents\models",
    "C:\models",
    "$PSScriptRoot\models"
)

foreach ($path in $search_paths) {
    if (Test-Path $path) {
        $gguf_files += Get-ChildItem -Path $path -Filter "*.gguf" -ErrorAction SilentlyContinue
    }
}

if ($gguf_files.Count -eq 0) {
    Write-Host "❌ No GGUF models found!`n" -ForegroundColor Red
    Write-Host "To use AutoStack with local models, you need a GGUF model file.`n"
    Write-Host "Recommended models for 7.6GB RAM:`n" -ForegroundColor Yellow
    Write-Host "1. TinyLlama (0.6 GB)`n   Download: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`n   File: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    Write-Host "`n2. Phi-3.5 Mini (1 GB)`n   Download: https://huggingface.co/microsoft/Phi-3.5-mini-instruct-GGUF"
    Write-Host "`n3. Neural Chat (4 GB)`n   Download: https://huggingface.co/TheBloke/neural-chat-7b-v3-3-GGUF`n"
    Write-Host "Once downloaded, save the model file and run this script again.`n"
    pause
    exit 1
}

Write-Host "Found GGUF models:`n"
for ($i = 0; $i -lt $gguf_files.Count; $i++) {
    $size_mb = [math]::Round($gguf_files[$i].Length / 1MB, 2)
    Write-Host "$($i+1). $($gguf_files[$i].Name) ($size_mb MB)"
}

Write-Host "`n"
$selection = Read-Host "Select model to load (enter number)"

try {
    $model_index = [int]$selection - 1
    if ($model_index -lt 0 -or $model_index -ge $gguf_files.Count) {
        Write-Host "Invalid selection" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Invalid input" -ForegroundColor Red
    exit 1
}

$selected_model = $gguf_files[$model_index]
$model_path = $selected_model.FullName

Write-Host "`n✓ Selected: $($selected_model.Name)`n" -ForegroundColor Green

# Determine GPU layers based on model size
$model_size_mb = $selected_model.Length / 1MB
if ($model_size_mb -lt 1000) {
    $ngl = 20  # More aggressive for small models
} elseif ($model_size_mb -lt 2000) {
    $ngl = 15
} else {
    $ngl = 10
}

Write-Host "GPU Configuration:" -ForegroundColor Cyan
Write-Host "  Model Size: $([math]::Round($model_size_mb, 2)) MB"
Write-Host "  GPU Layers: $ngl`n"

Write-Host "Starting llama-server...`n" -ForegroundColor Cyan
Write-Host "Command: llama-server -m `"$model_path`" -ngl $ngl`n"

# Start llama-server
& llama-server -m "$model_path" -ngl $ngl

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ llama-server failed to start`n" -ForegroundColor Red
    Write-Host "Troubleshooting:`n"
    Write-Host "1. Verify llama-server is installed: Run 'llama-server --version' in a new terminal"
    Write-Host "2. Try with fewer GPU layers: llama-server -m `"$model_path`" -ngl 5"
    Write-Host "3. Check your system has enough RAM for the model"
}

# 027 MPDD 补齐实验脚本（种子 41, 53）
# 等待 050 VisA 实验完成后运行

$PYTHON = "python"
$SCRIPT = "C:\Users\PC\Documents\Codex\实验\06_主线项目/027_自适应稀疏注意力与准确率效率前沿\02_代码\evaluate_mpdd_external.py"
$DATA = "D:\data数据集\MPDD"
$OUT = "C:\Users\PC\Documents\Codex\实验\06_主线项目\027_自适应稀疏注意力与准确率效率前沿\05_运行记录\一区候选_027_mpdd_multiseed"

$CATEGORIES = @("bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes")
$SEEDS = @(41, 53)

# Create output directory
if (-not (Test-Path $OUT)) {
    New-Item -ItemType Directory -Path $OUT -Force | Out-Null
}

Write-Output "=== 027 MPDD 补齐实验 ==="
Write-Output "数据目录: $DATA"
Write-Output "输出目录: $OUT"
Write-Output "类别: $($CATEGORIES -join ', ')"
Write-Output "种子: $($SEEDS -join ', ')"
Write-Output ""

foreach ($seed in $SEEDS) {
    $seedDir = Join-Path $OUT "seed$seed"
    if (-not (Test-Path $seedDir)) {
        New-Item -ItemType Directory -Path $seedDir -Force | Out-Null
    }
    
    $resultFile = Join-Path $seedDir "results.json"
    if (Test-Path $resultFile) {
        Write-Output "SKIP seed=$seed (already exists)"
        continue
    }
    
    Write-Output "Running seed=$seed..."
    & $PYTHON $SCRIPT --data-root $DATA --categories $CATEGORIES --output-root $seedDir --seed $seed --device cuda
    
    if (Test-Path $resultFile) {
        Write-Output "  seed=$seed completed"
    } else {
        Write-Output "  seed=$seed failed"
    }
}

Write-Output ""
Write-Output "=== MPDD 补齐实验完成 ==="

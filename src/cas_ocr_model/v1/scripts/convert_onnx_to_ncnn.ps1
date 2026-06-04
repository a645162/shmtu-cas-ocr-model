# ONNX -> NCNN 转换脚本 (Windows)
# Usage: powershell -File convert_onnx_to_ncnn.ps1

$toolPath = ".\3rdparty\ncnn\bin"
$pthDirPath = ".\workdir\Models"

function ConvertModelToNCNN($modelName) {
    Write-Host "Converting $modelName to NCNN format"
    & "$toolPath\onnx2ncnn.exe" "$pthDirPath\$modelName.onnx" "$pthDirPath\$modelName.fp32.param" "$pthDirPath\$modelName.fp32.bin"
    & "$toolPath\ncnnoptimize.exe" "$pthDirPath\$modelName.fp32.param" "$pthDirPath\$modelName.fp32.bin" "$pthDirPath\$modelName.fp16.param" "$pthDirPath\$modelName.fp16.bin" 1
}

ConvertModelToNCNN "resnet18_equal_symbol_latest"
ConvertModelToNCNN "resnet18_operator_latest"
ConvertModelToNCNN "resnet34_digit_latest"

# Convert ONNX model to NCNN format
# Usage: powershell -File convert_onnx_to_ncnn.ps1

function ConvertModelToNCNN {
    param (
        [string]$modelName
    )

    $toolPath = ".\3rdparty\ncnn\bin"
    $pthDirPath = ".\workdir\Models"

    Write-Host "Converting $modelName to NCNN format"

    & "$toolPath\onnx2ncnn.exe" `
        "$pthDirPath\$modelName.onnx" `
        "$pthDirPath\$modelName.fp32.param" `
        "$pthDirPath\$modelName.fp32.bin"

    # Optimize quantization, 1: fp16, 0: fp32
    & "$toolPath\ncnnoptimize.exe" `
        "$pthDirPath\$modelName.fp32.param" `
        "$pthDirPath\$modelName.fp32.bin" `
        "$pthDirPath\$modelName.fp16.param" `
        "$pthDirPath\$modelName.fp16.bin" `
        1
}

ConvertModelToNCNN -modelName "resnet18_equal_symbol_latest"
ConvertModelToNCNN -modelName "resnet18_operator_latest"
ConvertModelToNCNN -modelName "resnet34_digit_latest"

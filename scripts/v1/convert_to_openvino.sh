dir_path="./workdir/Models"
file_name="resnet34_digit_latest"

mo \
    --input_model "$dir_path/$file_name.onnx" \
    --input_shape "[1,3,224,224]"\
    --output_dir "$dir_path/$file_name"

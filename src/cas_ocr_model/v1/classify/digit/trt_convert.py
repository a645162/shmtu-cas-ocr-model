import argparse
import tensorrt as trt


def ONNX2TRT(args, calib=None):
    ''' convert onnx to tensorrt engine, use mode of ['fp32', 'fp16', 'int8']
    :return: trt engine
    '''

    assert args.mode.lower() in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8']"

    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(G_LOGGER) as builder, builder.create_network() as network, \
            trt.OnnxParser(network, G_LOGGER) as parser:

        builder.max_batch_size = args.batch_size
        builder.max_workspace_size = 1 << 30
        if args.mode.lower() == 'int8':
            assert (builder.platform_has_fast_int8 == True), "not support int8"
            builder.int8_mode = True
            builder.int8_calibrator = calib
        elif args.mode.lower() == 'fp16':
            assert (builder.platform_has_fast_fp16 == True), "not support fp16"
            builder.fp16_mode = True

        print('Loading ONNX file from path {}...'.format(args.onnx_file_path))
        with open(args.onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        print('Completed parsing of ONNX file')

        print('Building an engine from file {}; this may take a while...'.format(args.onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print("Created engine success! ")

        # 保存计划文件
        print('Saving TRT engine file to path {}...'.format(args.engine_file_path))
        with open(args.engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print('Engine file has already saved to {}!'.format(args.engine_file_path))
        return engine


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT engine")
    parser.add_argument("--onnx_file_path", type=str, required=True, help="Path to the ONNX file")
    parser.add_argument("--engine_file_path", type=str, required=True, help="Path to save the TensorRT engine")
    parser.add_argument("--mode", type=str, default="fp32", choices=['fp32', 'fp16', 'int8'],
                        help="Mode for TensorRT engine (fp32, fp16, int8)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the TensorRT engine")
    args = parser.parse_args()

    # 如果使用 int8 模式，你需要提供一个校准器
    calib = None
    if args.mode.lower() == 'int8':
        # 这里只是一个示例，你需要实现自己的校准器
        class DummyCalibrator(trt.IInt8Calibrator):
            def __init__(self):
                self.cache = {}

            def get_batch_size(self):
                return args.batch_size

            def get_calibration_data(self, names):
                if names not in self.cache:
                    # 这里你需要加载你的校准数据
                    # 假设你有一个函数 load_calibration_data(names) 来加载数据
                    self.cache[names] = load_calibration_data(names)
                return self.cache[names]

        calib = DummyCalibrator()

        # 转换 ONNX 到 TensorRT 引擎
    engine = ONNX2TRT(args, calib=calib)

    # 在这里，你可以使用引擎进行推理或其他操作
    # ...


if __name__ == "__main__":
    main()

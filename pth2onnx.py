import torch
import torch.onnx

def pth_to_onnx(input, checkpoint, onnx_path, input_names,output_names):
    if not onnx_path.endswith('.onnx'):
        print('Warning!')
        return 0
    model = torch.load('saved_models/dehaze_net_epoch_17.pth', map_location=torch.device('cpu'))
    # #指定模型的输入，以及onnx的输出路径
    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names,
                      output_names=output_names)  # 指定模型的输入，以及onnx的输出路径


if __name__ == '__main__':
    checkpoint = './saved_models/dehaze_net_epoch_17.pth'
    onnx_path = './dehaze_net.onnx'
    input = torch.randn(1, 3, 450, 600)
    input_names = ['input']
    output_names = ['output']
    pth_to_onnx(input, checkpoint, onnx_path, input_names, output_names)
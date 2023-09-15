import cv2
from rknn.api import RKNN

QUANTIZE_ON = True

IMG_SIZE = 544

def main():
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588")
    print('done')

    model = 'res101_kaggle_3250'
    if QUANTIZE_ON is True:
        model_out = model + "_quant"
    else:
        model_out = model
    opt_input = f"yolact/models/{model}.onnx" 
    opt_ouput = f"yolact/models/{model_out}.rknn" 
    dataset = "dataset/dataset.txt"
    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=opt_input)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=dataset)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(opt_ouput)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # __TEST____
    # Set inputs
    img = cv2.imread('./dataset/33.jpg')
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    outputs = rknn.inference(inputs=input_img)
    print(outputs)
    print('done')
    rknn.release()

if __name__ == '__main__':
    main()

1.首先使用pytoch自带的onnx转化成onnx

```python 
    torch.onnx.export(spline_model,
                      torch.zeros(size=input_size, device=device, dtype=torch.float32),
                      onnx_name,
                      # export_params=False,
                      dynamic_axes={'input_img': {0: 'h', 1: 'w'}}, #指定动态轴
                      input_names=['input_img'],
                      output_names=['r', 'g', 'b'],
                      opset_version=11)
```

2. 在使用tnn的 onnx，推进使用官方构建好的镜像, -in input_img 虽然指定了大小，但是onnx构建的时使用的是动态输入，所以不影响，参见code
``` shell
docker pull ccr.ccs.tencentyun.com/qcloud/tnn-convert

docker run -it -v /mnt/sda1/wokspace/ImageCureEnhance/inference/model:/data ccr.ccs.tencentyun.com/qcloud/tnn-convert:latest  python3 ./converter.py onnx2tnn /data/spline_att_model_13xy.onnx -v v1.0 -in input_img:1,3,750,500

```

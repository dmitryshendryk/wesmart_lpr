# wesmart_lpr


## Mask RCNN

cd /models/mask_rcnn

### Train 

```python
python main.py
```

### Evaluate

```python
python evaluate.py
```

### Performance 

```python
python performance.py
```

## CenterMask

cd /models/centermask

### Train

```python
python train_net.py --config-file "../../config/centermask_V_57_eSE_FPN_ms_3x.yaml"
```

### Performance 

```python
python performance.py
```


### TRT

```
./trtexec --onnx=/home/wesmart/Documents/apps/wesmart_lpr/models/segmentation_unet/super_resolution.onnx --batch=1 --saveEngine=unet_segemntation.trt --workspace=1024 --verbose=True --optShapes=input:1x3x512x288
```


### GStreamer


#### One thread 
```
gst-launch-1.0  filesrc location=/home/wesmart/Documents/apps/wesmart_lpr/tensorrt/main_video.mp4 ! qtdemux name=d d.video_0 ! queue ! h264parse ! avdec_h264 ! videoconvert ! autovideosin
```


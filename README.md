# Dicari Orang Hilang (DOH)
Lost and found of people medium



## Aktifkan virtualenv
Dir: /Users/thomas/Documents/Volantis/Project/mins_like_ml
$ source mins_like_ml2/bin/activate

## Aktifkan server
Pergi ke directory dimana ada MODEL_STORE berada. Di MODEL_STORE tersebut pastikan ada model misal: onnx-arcface2.mar

Dir: /Users/thomas/Documents/Volantis/Project/mins_like_ml/arcface/model-zoo-old/models/face_recognition

$ mxnet-model-server --start --model-store MODEL_STORE --models arcface=onnx-arcface2.mar

## Predict
Peri ke direktori di mana gambar yang mau di predict berada, misal mau predict dengan c.jpg dan c1.jpg

Dir: /Users/thomas/Documents/Volantis/Project/mins_like_ml/arcface/model-zoo-old/models/face_recognition/testing

$ curl -X POST http://127.0.0.1:8080/predictions/arcface -F "img1=@c.jpg" -F "img2=@c2.jpg"

Hasil:
{
  "Distance": 1.372612714767456,
  "Similarity": 0.05796707421541214
}

## Predict with python code
See predict_code.py 
In 
Dir: /Users/thomas/Documents/Volantis/Project/mins_like_ml/arcface/model-zoo-old/models/face_recognition/testing

## Stop server
$ mxnet-model-server --stop

*Catatan:
Jalankan server, kemudian predict dengan predict_code.py

## ARCFACE
https://github.com/pangyupo/mxnet_mtcnn_face_detection
env======
Dir: /Users/thomas/Documents/Volantis/Project/mins_like_ml
source mins_like_ml2/bin/activate
===============
Got to
$ conda install python=3.6.7
$ cd /Users/thomas/Documents/Volantis/Project/mins_like_ml/arcface/model-zoo-old/models/face_recognition/ArcFace
And 
$ python arcface_inference.py
TODO! His is good starting point

Serve a model
https://github.com/awslabs/mxnet-model-server#serve-a-model
Model
https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md#arcface-resnet100_onnx
Tutorial 
https://github.com/lupesko/model-zoo-old/blob/master/models/face_recognition/ArcFace/arcface_inference.ipynb

## MXNET
Run
$ mxnet-model-server --start --models arcface=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-arcface-resnet100.mar
Stop 
$ mxnet-model-server --stop


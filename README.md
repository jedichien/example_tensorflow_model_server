# Keras-based model export to tensorflow-model-server
Tensorflow model server communicate with clients by using grpc which is transport protocol similar to rpc. <br/>
You may wonder why should we install two kind of package instead of only one. Because In client side we have to package our message as `.proto` format which is grpc criteria and there is several packages file including then you won't to package them foreach I guess.<br/><br/>
In this sample, we will demonstrate how to export [this] model to tensorflow-model-server, then simple send a request  from client.

## Requirement
1. grpc
    `pip install grpcio`
2. tensorflow-model-server
    `pip install tensorflow-serving-api`
2. tensorflow-serving-dependencies
    ```
    sudo apt-get update && sudo apt-get install -y \
        build-essential \
        curl \
        libcurl3-dev \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev
    ```
3. tensorflow-serving-api
    `pip install tensorflow-serving-api`

## Export model
Please clone [this] repo and do the following 9 steps.
1. clone ssd repo `git clone https://github.com/jedichien/ssd_keras.git`
2. download weights (check repo discription) to this repo root.
3. make two directory named `/utils` and `/model` respectively
4. copy `ssd.py` and `ssd_layers.py` to `/model`
5. copy `ssd_utils.py` to `/utils`
6. `python export_model.py`
7. if success will see directory `/tmp/ssd/1`.
8. `python export_model.py`
9. `tensorflow_model_server --port=9000 --model_name=ssd --model_base_path=/tmp/ssd/`

## Make a request
`python make_request.py`
Result:
```json
[
 {
  "ymax": 0.9189264870315439,
  "label": "Motorbike",
  "label_id": 13,
  "score": 0.9999115467071533,
  "xmax": 0.7904480306583859,
  "xmin": 0.043379833733861395,
  "ymin": 0.30832826941576985
 },
 {
  "ymax": 0.545742948895173,
  "label": "Person",
  "label_id": 14,
  "score": 0.9823418259620667,
  "xmax": 0.5728003862966172,
  "xmin": 0.3283060760673855,
  "ymin": 0.10569146035108948
 }
]
```
[this]: <https://github.com/jedichien/ssd_keras>

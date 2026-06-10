```python
import sys
print(sys.version)
```

    3.12.4 | packaged by Anaconda, Inc. | (main, Jun 18 2024, 15:03:56) [MSC v.1929 64 bit (AMD64)]
    


```python
import tarfile
from pathlib import Path

import numpy as np
import tensorflow as tf

# TensorFlow 官方花卉数据集。第一次运行时会自动下载，之后会复用本地缓存。
FLOWER_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

print("TensorFlow 版本:", tf.__version__)

```

    TensorFlow 版本: 2.21.0
    


```python
# 数据目录配置：
# - DATA_DIR = None：自动下载并使用 TensorFlow 官方 flowers 数据集。
# - DATA_DIR = r"D:\path\to\my_images"：使用你自己的图片分类目录。
#
# 自定义图片目录需要按类别分文件夹，例如：
# my_images/
#   daisy/
#     1.jpg
#   roses/
#     2.jpg
DATA_DIR = None

# 导出目录。训练完成后会在这里生成 model.tflite、labels.txt 和 flower_classifier.keras。
EXPORT_DIR = "exported_flower_model"

# 训练参数。教程演示可以先用 3 到 5 个 epoch；如果使用自己的数据，可以适当增加。
EPOCHS = 5
BATCH_SIZE = 32
IMAGE_SIZE = 224
LEARNING_RATE = 1e-3

# TFLite 量化方式：
# - "dynamic"：默认推荐，模型更小，通常最容易成功。
# - "float16"：适合部分支持 float16 的设备。
# - "int8"：体积更小，但需要代表性数据集，转换要求更严格。
# - "none"：不量化，保留浮点模型。
QUANTIZATION = "dynamic"

# 固定随机种子，方便训练/验证划分尽量可复现。
SEED = 123

```


```python
def load_flower_datasets(data_dir, image_size, batch_size, seed):
    # 如果没有传入自定义数据目录，就下载 TensorFlow 官方 flower_photos 数据集。
    if data_dir is None:
        archive_path = tf.keras.utils.get_file(
            "flower_photos.tgz",
            FLOWER_URL,
            extract=False,
        )
        archive_path = Path(archive_path)
        candidates = [
            archive_path.parent / "flower_photos",
            archive_path.parent / "flower_photos_extracted" / "flower_photos",
        ]
        data_dir = next((path for path in candidates if path.exists()), None)
        if data_dir is None:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(archive_path.parent / "flower_photos_extracted")
            data_dir = archive_path.parent / "flower_photos_extracted" / "flower_photos"
    else:
        data_dir = Path(data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
    )
    class_names = train_ds.class_names

    val_batches = int(tf.data.experimental.cardinality(val_ds).numpy())
    test_ds = val_ds.take(val_batches // 2)
    val_ds = val_ds.skip(val_batches // 2)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names

```


```python
# 加载数据集并查看类别名称。
train_ds, val_ds, test_ds, class_names = load_flower_datasets(
    DATA_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
)

print("类别数量:", len(class_names))
print("类别名称:", class_names)

```

    Found 3670 files belonging to 5 classes.
    Using 2936 files for training.
    Found 3670 files belonging to 5 classes.
    Using 734 files for validation.
    类别数量: 5
    类别名称: ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    


```python
def build_model(num_classes, image_size, learning_rate):
    # 输入图片尺寸固定为 IMAGE_SIZE x IMAGE_SIZE x 3。
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")

    # MobileNetV2 有自己的预处理方式，这里把像素值转换到模型期望的范围。
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # include_top=False 表示不要 ImageNet 原始的 1000 类分类头，只保留特征提取部分。
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    # 冻结预训练模型参数，只训练后面的 Dense 分类层。
    base_model.trainable = False
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)

    # 输出维度等于类别数量，softmax 输出每个类别的概率。
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model

```


```python
# 创建模型并打印结构。第一次运行会下载 MobileNetV2 的 ImageNet 预训练权重。
model = build_model(len(class_names), IMAGE_SIZE, LEARNING_RATE)
model.summary()

```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
    [1m9406464/9406464[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 1us/step
    WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ image (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ true_divide (<span style="color: #0087ff; text-decoration-color: #0087ff">TrueDivide</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ subtract (<span style="color: #0087ff; text-decoration-color: #0087ff">Subtract</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ mobilenetv2_1.00_224            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1280</span>)           │     <span style="color: #00af00; text-decoration-color: #00af00">2,257,984</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)                    │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1280</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ predictions (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">5</span>)              │         <span style="color: #00af00; text-decoration-color: #00af00">6,405</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,264,389</span> (8.64 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">6,405</span> (25.02 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,257,984</span> (8.61 MB)
</pre>




```python
# 开始训练。history 中会保存每个 epoch 的 loss、accuracy、val_loss、val_accuracy。
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

```

    Epoch 1/5
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m25s[0m 239ms/step - accuracy: 0.6921 - loss: 0.8275 - val_accuracy: 0.8508 - val_loss: 0.4382
    Epoch 2/5
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 225ms/step - accuracy: 0.8580 - loss: 0.4097 - val_accuracy: 0.8717 - val_loss: 0.3825
    Epoch 3/5
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 226ms/step - accuracy: 0.8856 - loss: 0.3356 - val_accuracy: 0.8874 - val_loss: 0.3119
    Epoch 4/5
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 223ms/step - accuracy: 0.8978 - loss: 0.2933 - val_accuracy: 0.8901 - val_loss: 0.3037
    Epoch 5/5
    [1m92/92[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 225ms/step - accuracy: 0.9172 - loss: 0.2569 - val_accuracy: 0.9058 - val_loss: 0.2862
    


```python
# 使用测试集评估模型。测试集没有参与训练，用于更客观地观察最终效果。
loss, accuracy = model.evaluate(test_ds)
print(f"test_loss={loss:.4f}, test_accuracy={accuracy:.4f}")

```

    [1m11/11[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 210ms/step - accuracy: 0.8778 - loss: 0.3196
    test_loss=0.3196, test_accuracy=0.8778
    


```python
def convert_to_tflite(model, quantization, representative_ds):
    # 从 Keras 模型创建 TFLite 转换器。
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == "dynamic":
        # 动态范围量化：最常用、最容易成功的压缩方式。
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == "float16":
        # float16 量化：权重使用半精度浮点数，适合部分移动端/GPU 场景。
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "int8":
        # int8 全整数量化：体积更小，但需要代表性数据集校准输入分布。
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative_data_gen():
            for images, _ in representative_ds.take(100):
                for image in images:
                    yield [tf.expand_dims(tf.cast(image, tf.float32), 0)]

        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    elif quantization != "none":
        raise ValueError(f"Unsupported quantization mode: {quantization}")

    return converter.convert()

```


```python
# 创建导出目录。
export_dir = Path(EXPORT_DIR)
export_dir.mkdir(parents=True, exist_ok=True)

# 保存标签文件。部署时需要 labels.txt 把模型输出编号映射回类别名称。
labels_path = export_dir / "labels.txt"
labels_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")

# 保存 Keras 原始模型，便于以后继续训练或重新转换。
keras_path = export_dir / "flower_classifier.keras"
model.save(keras_path)

# 转换并保存 TFLite 模型。
tflite_model = convert_to_tflite(model, QUANTIZATION, train_ds)
tflite_path = export_dir / "model.tflite"
tflite_path.write_bytes(tflite_model)

print(f"已保存 Keras 模型: {keras_path}")
print(f"已保存 TFLite 模型: {tflite_path}")
print(f"已保存标签文件: {labels_path}")

```

    INFO:tensorflow:Assets written to: C:\Users\Feng\AppData\Local\Temp\tmpz8qvkani\assets
    

    INFO:tensorflow:Assets written to: C:\Users\Feng\AppData\Local\Temp\tmpz8qvkani\assets
    

    Saved artifact at 'C:\Users\Feng\AppData\Local\Temp\tmpz8qvkani'. The following endpoints are available:
    
    * Endpoint 'serve'
      args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name='image')
    Output Type:
      TensorSpec(shape=(None, 5), dtype=tf.float32, name=None)
    Captures:
      1875116036304: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875147924176: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875147930512: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875147930704: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875147931088: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875116037264: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151419344: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151419536: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875147930896: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875147930128: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151418192: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151420880: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151421264: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151417232: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151419728: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151416656: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151416272: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151414352: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151417616: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151416080: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151420496: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151414544: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151414736: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151414928: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151421072: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151415696: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151415888: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151415504: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151415120: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151413008: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151413776: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151413200: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151413392: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151413968: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151412048: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151413584: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151411472: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151412624: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151416464: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151405904: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151406288: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151406480: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151405712: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151406096: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151406864: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151407248: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151407824: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151406672: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151407440: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151408208: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151405520: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151408016: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151407632: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151405136: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151408976: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151418384: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151408400: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151408784: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151405328: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151409936: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151408592: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151409552: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151409360: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151407056: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151410128: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151410320: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151410512: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151409744: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151411664: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151411088: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151409168: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151410896: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151411280: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151410704: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875151418768: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160400912: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160399952: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160400144: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160400528: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160401872: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160401296: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160401488: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160401680: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160401104: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160402832: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160402256: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160402448: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160402640: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160400720: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160403792: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160403216: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160403408: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160403600: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160400336: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160404752: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160404176: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160404368: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160404560: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160402064: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160405712: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160405136: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160405328: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160405520: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160403024: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160406672: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160406096: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160406288: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160406480: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160403984: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160407632: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160407056: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160407248: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160407440: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160404944: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160408592: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160408016: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160408208: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160408400: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160405904: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160409552: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160408976: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160409168: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160409360: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160406864: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160410512: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160409936: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160410128: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160410320: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160407824: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160411472: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160410896: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160411088: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160411280: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160408784: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160412432: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160411856: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160412048: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160412240: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160409744: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160413392: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160412816: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160413008: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160413200: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160410704: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160414352: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160413776: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160413968: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160414160: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160411664: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160415312: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160414736: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160414928: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160414544: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160416080: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160415504: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160413584: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160415696: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160415888: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160415120: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875160412624: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150520592: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150521744: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150521552: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150521168: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150522320: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150520784: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150521936: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150522128: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150520400: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150523280: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150522704: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150522896: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150523088: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150520976: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150524240: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150523664: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150523856: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150524048: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150521360: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150525200: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150524624: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150524816: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150525008: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150522512: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150526160: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150525584: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150525776: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150525968: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150523472: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150527120: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150526544: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150526736: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150526928: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150524432: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150528080: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150527504: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150527696: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150527888: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150525392: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150529040: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150528464: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150528656: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150528848: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150526352: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150530000: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150529424: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150529616: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150529808: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150527312: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150530960: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150530384: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150530576: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150530768: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150528272: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150531920: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150531344: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150531536: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150531728: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150529232: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150532880: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150532304: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150532496: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150532688: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150530192: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150533840: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150533264: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150533456: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150533648: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150531152: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150534800: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150534224: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150534416: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150534608: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150532112: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150535760: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150535184: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150535376: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150534992: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150536528: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150535952: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150534032: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150536144: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150536336: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150535568: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875150533072: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875149275408: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875149276560: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875149276368: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875149275984: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875149277136: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875149275600: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875149276752: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875149276944: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875149275216: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875149278096: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875149277520: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875149277712: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875149277904: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875149275792: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875149279056: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1874878891216: TensorSpec(shape=(), dtype=tf.resource, name=None)
      1875149279632: TensorSpec(shape=(), dtype=tf.resource, name=None)
    已保存 Keras 模型: exported_flower_model\flower_classifier.keras
    已保存 TFLite 模型: exported_flower_model\model.tflite
    已保存标签文件: exported_flower_model\labels.txt
    


```python
def smoke_test_tflite(tflite_path, test_ds, class_names):
    # 加载 TFLite 模型并分配张量内存。
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # 从测试集中取 8 张图片做快速推理。
    images, labels = next(iter(test_ds.unbatch().batch(8)))
    input_data = tf.cast(images, input_details["dtype"]).numpy()

    # 如果模型是 uint8 输入，需要按照量化参数把图片转换到对应范围。
    if input_details["dtype"] == np.uint8:
        scale, zero_point = input_details["quantization"]
        if scale:
            input_data = images.numpy() / scale + zero_point
            input_data = np.clip(input_data, 0, 255).astype(np.uint8)

    predictions = []
    for image in input_data:
        interpreter.set_tensor(input_details["index"], np.expand_dims(image, 0))
        interpreter.invoke()
        predictions.append(interpreter.get_tensor(output_details["index"])[0])

    predicted_ids = np.argmax(np.asarray(predictions), axis=1)
    for expected, predicted in zip(labels.numpy()[:5], predicted_ids[:5]):
        print(f"真实类别={class_names[expected]}, 预测类别={class_names[predicted]}")

```


```python
# 运行 TFLite 快速测试。
smoke_test_tflite(tflite_path, test_ds, class_names)

```

    D:\anaconda3\Lib\site-packages\tensorflow\lite\python\interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
        TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
        See the [migration guide](https://ai.google.dev/edge/litert/migration)
        for details.
        
      warnings.warn(_INTERPRETER_DELETION_WARNING)
    

    真实类别=daisy, 预测类别=daisy
    真实类别=tulips, 预测类别=tulips
    真实类别=sunflowers, 预测类别=sunflowers
    真实类别=daisy, 预测类别=daisy
    真实类别=sunflowers, 预测类别=sunflowers
    


```python

```

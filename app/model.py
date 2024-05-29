import tensorflow as tf
from tensorflow.keras.models import load_model
import os

def mard_loss(y_true, y_pred):
    epsilon = tf.keras.backend.epsilon()
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    relative_difference = tf.abs((y_true - y_pred) / (y_true + epsilon))
    mard = tf.reduce_mean(relative_difference)
    return mard

def predict(ppg, ppg_info):
    # 確認文件路徑
    model_path = os.path.abspath('app/model/model_fold_1.keras')

    # 使用加載模型
    CNN_model = load_model(model_path, custom_objects={'mard_loss': mard_loss})

    pred = CNN_model.predict([ppg, ppg_info])

    return pred
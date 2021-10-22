import tensorflow as tf
from NN_config import config

#create model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units = config.model_params.dense_size.value, activation=config.model_params.activation_function.value))
model.add(tf.keras.layers.Dropout(rate = config.model_params.dropout_size.value))
model.add(tf.keras.layers.Dense(units = config.model_params.dense_size.value, activation=config.model_params.activation_function.value))
model.add(tf.keras.layers.Dropout(rate = config.model_params.dropout_size.value))
model.add(tf.keras.layers.Dense(units=config.model_params.labels.value, activation=config.model_params.final_layer_activation.value))

#compile model
model.compile(optimizer=config.compiler_params.optimizer.value, loss=config.compiler_params.loss.value, metrics=["accuracy"])
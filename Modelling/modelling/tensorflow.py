from Modelling import *


class TensorFlow_Modelling:
    def train(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        name="BaseLine",
        model="https://tfhub.dev/google/nnlm-en-dim50/2",
        trainable=True,
        tl_model_output=16,
        output=1,
        activation="relu",
        optimizer="adam",
        criterion=tf.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name="accuracy")],
    ):
        wandb.init(project=PROJECT_NAME, name=name, sync_tensorboard=True)
        hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=trainable)
        model = tf.keras.Sequential()
        model.add(hub_layer)
        model.add(tf.keras.layers.Dense(tl_model_output, activation=activation))
        model.add(tf.keras.layers.Dense(output))
        model.compile(
            optimizer=optimizer,
            loss=criterion,
            metrics=metrics,
        )
        history = model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1,
        )
        wandb.log({
            "Accuracy":model.evaluate(X_train, y_train)[0],
            "Loss":model.evaluate(X_train, y_train)[1],
             "Val Accuracy":model.evaluate(X_test, y_test)[0],
            "Val Loss":model.evaluate(X_test, y_test)[1],
        })
        return model

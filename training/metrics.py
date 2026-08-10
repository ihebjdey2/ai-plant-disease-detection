from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="AgriDiagnose")
class MacroF1(tf.keras.metrics.Metric):
    """Epoch-level macro F1 computed from one accumulated confusion matrix."""

    def __init__(self, num_classes: int = 39, name: str = "macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        if num_classes <= 1:
            raise ValueError("MacroF1 requires at least two classes.")
        self.num_classes = int(num_classes)
        self.confusion_matrix = self.add_weight(
            name="confusion_matrix",
            shape=(self.num_classes, self.num_classes),
            initializer="zeros",
            dtype=self.dtype,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        labels = tf.cast(tf.reshape(y_true, (-1,)), tf.int32)
        predictions = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        predictions = tf.reshape(predictions, (-1,))
        weights = None
        if sample_weight is not None:
            weights = tf.cast(tf.reshape(sample_weight, (-1,)), self.dtype)
        matrix = tf.math.confusion_matrix(
            labels,
            predictions,
            num_classes=self.num_classes,
            weights=weights,
            dtype=self.dtype,
        )
        self.confusion_matrix.assign_add(matrix)

    def result(self):
        true_positive = tf.linalg.diag_part(self.confusion_matrix)
        actual = tf.reduce_sum(self.confusion_matrix, axis=1)
        predicted = tf.reduce_sum(self.confusion_matrix, axis=0)
        precision = tf.math.divide_no_nan(true_positive, predicted)
        recall = tf.math.divide_no_nan(true_positive, actual)
        f1 = tf.math.divide_no_nan(2.0 * precision * recall, precision + recall)
        return tf.reduce_mean(f1)

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))

    def get_config(self):
        return {**super().get_config(), "num_classes": self.num_classes}

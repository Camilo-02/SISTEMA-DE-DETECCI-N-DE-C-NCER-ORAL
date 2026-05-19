from pathlib import Path
import io
import base64

from django.shortcuts import render
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def main(request):
    return render(request, 'index.html', context={})


def load_image_dataset(dataset_dir, image_size=(128, 128), batch_size=16):
    # Cargar ambos datasets ANTES de aplicar transformaciones
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        labels='inferred',
        label_mode='binary',
        validation_split=0.20,
        subset='training',
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        labels='inferred',
        label_mode='binary',
        validation_split=0.20,
        subset='validation',
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
    )

    # Guardar class_names ANTES de cache/prefetch
    class_names = train_ds.class_names
    
    # Ahora sí aplicar optimizaciones
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(input_shape=(128, 128, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


def encode_confusion_matrix(cm, class_names):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Matriz de Confusión')
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150)
    plt.close(fig)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return image_base64


def format_report(report_dict):
    rows = []
    for label, values in report_dict.items():
        if label == 'accuracy':
            rows.append(
                {
                    'label': label,
                    'precision': round(values * 100, 2),
                    'recall': round(values * 100, 2),
                    'f1_score': round(values * 100, 2),
                    'support': '',
                }
            )
        elif isinstance(values, dict):
            rows.append(
                {
                    'label': label,
                    'precision': round(values.get('precision', 0) * 100, 2),
                    'recall': round(values.get('recall', 0) * 100, 2),
                    'f1_score': round(values.get('f1-score', 0) * 100, 2),
                    'support': int(values.get('support', 0)),
                }
            )
    return rows


def prediccion(request):
    dataset_dir = Path(__file__).resolve().parent.parent / 'dataset'

    # Cargar datasets
    train_ds, val_ds, class_names = load_image_dataset(dataset_dir)
    model = build_model()

    # Entrenar
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
            )
        ],
    )

    # Obtener predicciones en validación
    y_true = np.concatenate([labels.numpy() for _, labels in val_ds], axis=0)
    y_pred_prob = model.predict(val_ds, verbose=0)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    # Calcular métricas globales
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100

    # Reporte por clase
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Matriz de confusión
    confusion = confusion_matrix(y_true, y_pred)
    confusion_image = encode_confusion_matrix(confusion, class_names)

    # Preparar contexto
    context = {
        'class_names': class_names,
        'metrics': {
            'precision': round(precision, 2),
            'recall': round(recall, 2),
            'f1_score': round(f1, 2),
            'train_accuracy': round(history.history['accuracy'][-1] * 100, 2),
            'val_accuracy': round(history.history['val_accuracy'][-1] * 100, 2),
        },
        'report_rows': format_report(report_dict),
        'confusion_image': confusion_image,
        'report_text': classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        ),
    }

    return render(request, 'index.html', context=context)

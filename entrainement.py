import tensorflow as tf

TAILLE = (224, 224)
LOT = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=TAILLE,
    batch_size=LOT,
    label_mode="binary",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=TAILLE,
    batch_size=LOT,
    label_mode="binary",
)

print("Classes :", train_ds.class_names)
import numpy as np

# 1) Poids de classes (compense le déséquilibre fertile / non fertile)
y_train = np.concatenate([y.numpy() for _, y in train_ds]).flatten()
n0 = int((y_train == 0).sum())
n1 = int((y_train == 1).sum())
total = len(y_train)
poids_classes = {0: total / (2 * n0), 1: total / (2 * n1)}
print("Poids des classes :", poids_classes)

# 2) Pixels entre 0 et 1 (comme dans la vue du prof : image / 255.0)
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

# 3) Modèle : MobileNetV2 pré-entraîné + notre propre tête de classification
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base.trainable = False  # on garde les connaissances déjà apprises

entree = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.layers.RandomFlip("horizontal")(entree)
x = tf.keras.layers.RandomRotation(0.1)(x)
x = tf.keras.layers.RandomZoom(0.1)(x)
x = tf.keras.layers.Rescaling(2.0, offset=-1.0)(x)  # [0,1] -> [-1,1] attendu par MobileNetV2
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
sortie = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(entree, sortie)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()
historique = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=poids_classes,
)
# Évaluation finale
perte, precision = model.evaluate(val_ds, verbose=0)
print(f"Précision finale sur la validation : {precision * 100:.1f} %")

# Matrice de confusion (fertile = 0, non fertile = 1)
y_vrai, y_pred = [], []
for images, labels in val_ds:
    p = model.predict(images, verbose=0).flatten()
    y_pred.extend((p > 0.5).astype(int))
    y_vrai.extend(labels.numpy().flatten().astype(int))
y_vrai = np.array(y_vrai)
y_pred = np.array(y_pred)

print("Fertile correctement reconnu       :", int(((y_vrai == 0) & (y_pred == 0)).sum()))
print("Fertile pris pour non fertile      :", int(((y_vrai == 0) & (y_pred == 1)).sum()))
print("Non fertile pris pour fertile      :", int(((y_vrai == 1) & (y_pred == 0)).sum()))
print("Non fertile correctement reconnu   :", int(((y_vrai == 1) & (y_pred == 1)).sum()))

# Sauvegarde exigée par le sujet
model.save("modele_fertilite_sol.h5")
print("Modèle sauvegardé : modele_fertilite_sol.h5")

# Matrice de confusion (fertile = 0, non fertile = 1)
y_vrai, y_pred = [], []
for images, labels in val_ds:
    p = model.predict(images, verbose=0).flatten()
    y_pred.extend((p > 0.5).astype(int))
    y_vrai.extend(labels.numpy().flatten().astype(int))
y_vrai = np.array(y_vrai)
y_pred = np.array(y_pred)

print("Fertile correctement reconnu       :", int(((y_vrai == 0) & (y_pred == 0)).sum()))
print("Fertile pris pour non fertile      :", int(((y_vrai == 0) & (y_pred == 1)).sum()))
print("Non fertile pris pour fertile      :", int(((y_vrai == 1) & (y_pred == 0)).sum()))
print("Non fertile correctement reconnu   :", int(((y_vrai == 1) & (y_pred == 1)).sum()))

# Sauvegarde exigée par le sujet
model.save("modele_fertilite_sol.h5")
print("Modèle sauvegardé : modele_fertilite_sol.h5")
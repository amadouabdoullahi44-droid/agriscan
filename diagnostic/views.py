import os
import numpy as np
import tensorflow as tf
from PIL import Image
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.preprocessing import image

# Chargement du modèle à la première analyse, puis conservé en mémoire
CHEMIN_MODELE = os.path.join(settings.BASE_DIR, 'modele_fertilite_sol.h5')
_model = None
def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(CHEMIN_MODELE, compile=False)
    return _model

EXTENSIONS_VALIDES = ('.jpg', '.jpeg', '.png')

CONSEIL_FERTILE_SOMBRE = (
    "Votre sol est très sombre, signe habituel d'une bonne richesse en matière "
    "organique. Pour la préserver, ajoutez du compost régulièrement, évitez de "
    "laisser la terre nue et alternez les cultures."
)
CONSEIL_FERTILE = (
    "Votre sol semble fertile. Pour le garder en bon état, apportez du compost ou "
    "du fumier avant les semis, gardez le sol couvert (paillage) et alternez les "
    "cultures d'une saison à l'autre."
)
CONSEIL_SABLEUX = (
    "Votre sol est clair et sableux : il retient mal l'eau et les éléments "
    "nutritifs. Ajoutez beaucoup de matière organique (compost, fumier), paillez "
    "pour garder l'humidité et arrosez peu mais souvent. Choisissez des cultures "
    "résistantes à la sécheresse comme le mil, le sorgho ou le niébé."
)
CONSEIL_ROUGE = (
    "Votre sol est rouge, souvent de type latéritique : il est fréquemment pauvre "
    "en matière organique et s'érode vite sous la pluie. Ajoutez du compost et des "
    "résidus de récolte, couvrez la terre pour la protéger, et commencez par des "
    "cultures peu exigeantes comme le niébé ou l'arachide."
)
CONSEIL_NON_FERTILE = (
    "Votre sol semble pauvre ou sec. Ajoutez du compost, du fumier ou des résidus "
    "de récolte, couvrez le sol (paillage) pour garder l'humidité et choisissez des "
    "cultures peu exigeantes comme le mil, le sorgho ou le niébé."
)


def choisir_conseil(est_fertile, img_array):
    # Couleur moyenne de la photo (valeurs entre 0 et 1)
    r, g, b = img_array[0].mean(axis=(0, 1))
    luminosite = 0.299 * r + 0.587 * g + 0.114 * b
    rapport_vert_rouge = g / (r + 1e-6)

    if est_fertile:
        return CONSEIL_FERTILE_SOMBRE if luminosite < 0.20 else CONSEIL_FERTILE
    if rapport_vert_rouge < 0.65 and luminosite < 0.65:
        return CONSEIL_ROUGE
    if luminosite > 0.55:
        return CONSEIL_SABLEUX
    return CONSEIL_NON_FERTILE

def analyser_sol(request):
    contexte = {}

    if request.method == 'POST':
        photo = request.FILES.get('photo_sol')

        # Erreur 1 : formulaire envoyé sans photo
        if not photo:
            contexte['erreur'] = ("Aucune photo sélectionnée. Prenez ou choisissez "
                                  "une photo du sol avant de lancer l'analyse.")
            return render(request, 'diagnostic/index.html', contexte)

        # Erreur 2 : mauvais format de fichier
        extension = os.path.splitext(photo.name)[1].lower()
        if extension not in EXTENSIONS_VALIDES:
            contexte['erreur'] = ("Format de fichier non valide. Utilisez une image "
                                  "JPG ou PNG.")
            return render(request, 'diagnostic/index.html', contexte)

        # Erreur 3 : fichier qui n'est pas réellement une image
        try:
            Image.open(photo).verify()
            photo.seek(0)
        except Exception:
            contexte['erreur'] = ("Ce fichier n'est pas une image lisible. "
                                  "Choisissez une autre photo.")
            return render(request, 'diagnostic/index.html', contexte)

        # Sauvegarde temporaire dans MEDIA_ROOT
        fs = FileSystemStorage()
        nom_fichier = fs.save(photo.name, photo)
        chemin_complet = fs.path(nom_fichier)

        # Prétraitement : 224x224 et pixels entre 0 et 1
        img = image.load_img(chemin_complet, target_size=(224, 224))
        img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

        # Prédiction (proche de 0 = fertile, proche de 1 = non fertile)
        prediction = float(get_model().predict(img_array, verbose=0)[0][0])
        est_fertile = prediction < 0.5
        confiance = round((1 - prediction if est_fertile else prediction) * 100, 1)

        contexte = {
            'resultat': True,
            'est_fertile': est_fertile,
            'statut': "Sol Fertile" if est_fertile else "Sol Non Fertile",
            'confiance': confiance,
            'image_url': fs.url(nom_fichier),
            'incertain': confiance < 70, 
            'conseil':   choisir_conseil(est_fertile, img_array),
        }

    return render(request, 'diagnostic/index.html', contexte)
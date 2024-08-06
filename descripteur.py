# Imports 
#scikit-image
from skimage.feature import graycomatrix,graycoprops

#import biydesc
from BiT import bio_taxo
# Import Haralick
import mahotas.features as features

# Import cv2
import cv2


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Impossible de charger l'image {image_path}")
        return None
    
    if len(image.shape) == 2:
        # L'image est déjà en 2D (niveaux de gris)
        return image
    elif len(image.shape) == 3:
        # Convertir l'image en niveaux de gris
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image
    else:
        print(f"L'image {image_path} n'est pas au bon format.")
        return None


def glcm(image):
    glcm_matrix = graycomatrix(image, [2], [0], None, symmetric=True, normed=True)
    diss = graycoprops(glcm_matrix, 'dissimilarity')[0, 0]
    con = graycoprops(glcm_matrix, 'contrast')[0, 0]
    corr = graycoprops(glcm_matrix, 'correlation')[0, 0]
    ener = graycoprops(glcm_matrix, 'energy')[0, 0]
    homo = graycoprops(glcm_matrix, 'homogeneity')[0, 0]
    return [diss, con, corr, ener, homo]



# def glcm(image):
#     data=cv2.imread(image,0)#auto en noir blanc
#     glcm=graycomatrix(data,[2],[0],None,symmetric=True,normed=True)
#     diss=graycoprops(glcm,'dissimilarity')[0,0]
#     con=graycoprops(glcm,'contrast')[0,0]
#     corr=graycoprops(glcm,'correlation')[0,0]
#     ener=graycoprops(glcm,'energy')[0,0]
#     homo=graycoprops(glcm,'homogeneity')[0,0]
#     return [diss,con,corr,ener,homo]

def haralick(image):
    #data=cv2.imread(image)
    return features.haralick(image).mean(0).tolist()

def Bitdesc(image):
    # data=cv2.imread(image,0)#auto en noir blanc

    return bio_taxo(image)

def glcm_bitdesc(image):
    return glcm(image)+Bitdesc(image)

def glcm_haralick(image):
    return glcm(image)+haralick(image)

def haralick_bitdesc(image):
    return haralick(image)+Bitdesc(image)
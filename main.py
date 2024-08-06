


import os
import pandas as pd
from descripteur import glcm, Bitdesc, glcm_bitdesc, haralick, glcm_haralick, haralick_bitdesc,preprocess_image

descriptors = [glcm, Bitdesc, glcm_bitdesc, haralick, glcm_haralick, haralick_bitdesc]
descriptor_names = ['glcm', 'Bitdesc', 'glcm_bitdesc', 'haralick', 'glcm_haralick', 'haralick_bitdesc']
folder_names = ['Covid', 'Outex24', 'KTH-TIPS2-a', 'Glaucoma', 'Satelite_dataset']




def process_folder(folder_path, descriptor):
    dataset = []
    for root, dirs, _ in os.walk(folder_path):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            for image_name in os.listdir(subdir_path):
                if image_name.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
                    image_path = os.path.join(subdir_path, image_name)
                    im=preprocess_image(image_path)
                    features = descriptor(im)
                    features.append(subdir)
                    dataset.append(features)
    return dataset



def save_to_csv(dataset, descriptor_name, folder_name):
    output_csv = f'../Prj_Csv/{folder_name}_{descriptor_name}.csv'
    df = pd.DataFrame(dataset)
    df.to_csv(output_csv, index=False, header=False)
    print(f'Dataset saved to {output_csv}')

def main():
    for folder_name in folder_names:
        folder_path = f'../datasets/{folder_name}'
        for descriptor, descriptor_name in zip(descriptors, descriptor_names):
            dataset = process_folder(folder_path, descriptor)
            save_to_csv(dataset, descriptor_name, folder_name)

if __name__ == '__main__':
    main()

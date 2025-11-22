from disease_classifier import PlantDiseaseClassifier

dataset_path = r"C:\Users\vinay\Projectz\PlantDiseasePred&Cred\archive\New Plant Diseases Dataset(Augmented)"

clf = PlantDiseaseClassifier()
train_gen, val_gen = clf.prepare_data(dataset_path)
clf.build_model()
clf.train(train_gen, val_gen, epochs=15)
clf.save_model()
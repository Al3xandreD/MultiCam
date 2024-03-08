::Faire tourner avec la webcam
python detect.py --source "Fireman\1_raw_train_dataset" --weights "yolov7.pt"  --save-txt --project "Fireman\2_labeled_train_dataset"

:: Pour directement mettre les résultats dans dossier "2_labeled_train_dataset" ET avec ecrasement ajouter options : --name "" --exist-ok

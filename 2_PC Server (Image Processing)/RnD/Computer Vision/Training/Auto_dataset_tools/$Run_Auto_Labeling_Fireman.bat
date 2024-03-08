:: Pour directement mettre les résultats dans dossier "2_labeled_train_dataset" ET avec ecrasement ajouter options : --name "" --exist-ok

python auto_labeling.py --source "..\Fireman\1_mixed_train_dataset" --weights "yolov7.pt"  --save-txt --project "..\Fireman\2_labeled_train_dataset" --name "" --exist-ok


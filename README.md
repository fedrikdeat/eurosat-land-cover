# EuroSAT: Sun'iy joldas suwretlerinde jer qoplamın klassifikaciyalaw

Sentinel-2 suwretleri (EuroSAT, 10 klass, 27 000 suwret) tiykarında jer qoplamın anıqlaytuǵın model.
Model **MobileNetV2** (transfer learning + fine-tuning) tiykarında Google Colab-ta oqıtıladı.

**Dataset:** https://www.kaggle.com/datasets/apollo2506/eurosat-dataset

## Klasslar
AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

## Qalay isletiledi
1. `EuroSAT_land_cover_classification.ipynb` faylın Google Colab-ta ashıń.
2. `Runtime -> Change runtime type -> T4 GPU`.
3. Cell-lerdi ústinen tómenge qaray iske túsiriń.
4. 7-bólimde óz suwretińizdi júkleseńiz, model analiz etedi
   (bir klass boljawı, top-5 itimallıq, plitka rejimi menen jer qoplamı kartası hám klasslar úlesi).

## Fayllar
| Fayl | Mazmunı |
|------|---------|
| `EuroSAT_land_cover_classification.ipynb` | Tiykarǵı notebook (oqıtıw, baha, suwret analizi) |
| `eurosat_model.keras` | Oqıtılǵan model |
| `class_names.json` | Klass atamaları |
| `confusion_matrix.png`, `training_curves.png`, `sample_images.png`, `analysis_example.png` | Nátiyje grafikleri |
| `predict.py` | Modelden paydalanıp kompyuterde suwretti analiz etiw |
| `requirements.txt` | Kerekli kitapxanalar |

## Kompyuterde paydalanıw
```bash
pip install -r requirements.txt
python predict.py photo.jpg
```

## Nátiyjeler
Test accuracy: **XX.X %**

## Eskertiw
EuroSAT suwretleri 64x64 piksel (~10 m/piksel). Modeldiń eń jaqsı islewi usı masshtabtaǵı Sentinel-2 suwretlerinde.

# HA: The Lens Problem

Everything you need is in this repository; the background story and the tasks are at the top of `notebook.ipynb`.

| Item | Content |
|---|---|
| `notebook.ipynb` | template notebook to fill in and submit |
| `model.py` | the classifier class (`ResNetClassifier`) and `load_classifier` |
| `classifier.pt` | the pre-trained classifier weights |
| `experiments.zip` | 100 images per folder: `old_experiments/` (experiments 1-4, original lens), `current_experiment/` (experiment 5, new lens), `followup_experiments/` (experiment 5 again, original lens) |

The classifier takes single-channel images as a float tensor `[N, 1, H, W]` with pixels in `[0, 1]`
(resizing and normalisation happen inside the model):

```python
from model import load_classifier

model = load_classifier("classifier.pt")
model.is_mitotic(x)       # [N] labels: 1 = dividing (mitotic), 0 = non-dividing
model.get_logits(x)       # [N, 2] class scores
model.get_embeddings(x)   # [N, D] features before the classification head
```

## Running in Google Colab

1. Upload and open the notebook in Google Colab
2. Optionally enable a GPU: **Runtime → Change runtime type → T4 GPU** (the CPU is enough for inference).
3. Run the cells in order. The setup cells install the packages (preinstalled in Colab, so this is quick) and
   `git clone` this repository, which brings `model.py`, `classifier.pt` and the images into the Colab session.
4. Save your work: **File → Save a copy in Drive**, and download it with **File → Download → .ipynb** for submission.

## Submission

`notebook.ipynb` (executed top to bottom, plots inline).
Do **not** submit `classifier.pt` or any images.

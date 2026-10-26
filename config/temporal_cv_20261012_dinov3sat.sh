# Temporal CV on gattaca2 with a DINOv3 ViT-L/16 segmenter (frozen backbone,
# trained decoder; scripts/train_planet_image_dinov3.py) instead of the Mask
# R-CNN, using the satellite-pretrained (SAT-493M) backbone. Same TOA chips, folds, epoch grid, labels and
# cloud handling as config/temporal_cv_20261012.sh, so the runs compare
# one-to-one through eval_temporal_cv.py. See
# docs/planet_temporal_generalization.md ("DINOv3 backbone"). The backbone is
# gated on Hugging Face; the weights are synced from Earth03
# (flower/hf_hub, plain files) to /scratch/tree-monitoring/hf_hub and loaded
# offline from there, so no login is needed on the cluster. CV only: the
# apply settings are unset.

source config/temporal_cv_20261012.sh
export TEMPORAL_RUNS=$HOME/Documents/tree-flowering/202610_temporal_dinov3sat
export TEMPORAL_TRAIN_SCRIPT=scripts/train_planet_image_dinov3.py
export TEMPORAL_TRAIN_ARGS="--ocm-masks --min-instance-size 64 --backbone facebook/dinov3-vitl16-pretrain-sat493m"
export HF_HUB_CACHE=/scratch/tree-monitoring/hf_hub HF_HUB_OFFLINE=1
unset APPLY_PLANETDIR APPLY_OCMDIR APPLY_COREG_JSON APPLY_CROWNS APPLY_OUT

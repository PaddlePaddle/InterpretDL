# Files within this path, contain the algorithms that evaluate
# the interpretability of deep models.

# The concept is that 
# (1) a trustworthy interpreter gives the explanation results;
# (2) the explanations reveal the model's interpretability;
# (3) if the explanations align with human understandings, then the model is more interpretable.

from .localization import PointGame, PointGameSegmentation

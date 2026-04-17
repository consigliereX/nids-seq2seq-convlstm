import numpy as np
from src.utils.metrics import confusion_to_fpr


def test_confusion_to_fpr_simple():
    # 2-class confusion matrix
    # true0: 8 correct, 2 predicted as 1
    # true1: 1 predicted as 0, 9 correct
    cm = np.array([[8, 2],
                   [1, 9]], dtype=np.int64)

    fpr = confusion_to_fpr(cm)

    # For class0: FP = cm[1,0]=1, TN = cm[1,1]=9 -> 1/(1+9)=0.1
    assert abs(fpr[0] - 0.1) < 1e-9

    # For class1: FP = cm[0,1]=2, TN = cm[0,0]=8 -> 2/(2+8)=0.2
    assert abs(fpr[1] - 0.2) < 1e-9

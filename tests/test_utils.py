import numpy as np
import pandas as pd

from guidedsum.utils import train_val_test_split


def test_train_val_test_split():
    df = pd.DataFrame(np.random.rand(10, 5), columns=list("ABCDE"))
    df_train, df_valid, df_test = train_val_test_split(
        df, frac_train=0.6, frac_val=0.1, frac_test=0.3
    )

    assert df_train.shape == (6, 5)
    assert df_valid.shape == (1, 5)
    assert df_test.shape == (3, 5)

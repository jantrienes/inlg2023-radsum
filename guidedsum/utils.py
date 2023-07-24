import pandas as pd
from IPython.display import HTML, display
from sklearn.model_selection import train_test_split


def train_val_test_split(
    df,
    frac_train=0.6,
    frac_val=0.15,
    frac_test=0.25,
    random_state=None,
):
    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError(
            f"fractions {frac_train}, {frac_val}, {frac_test} do not add up to 1.0"
        )

    df_train, df_temp = train_test_split(
        df, test_size=(1.0 - frac_train), random_state=random_state
    )

    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=relative_frac_test,
        random_state=random_state,
    )

    assert len(df) == len(df_train) + len(df_val) + len(df_test)
    return df_train, df_val, df_test


def display_sample(df, N=10, random_state=42, left_align=True):
    """Print a sample of a DataFrame without limits on column width.

    Strings are not truncated and line breaks are rendered.

    Parameters
    ----------
    N : int
        Number of rows to display. If None or larger than df, it displays all rows.
    """
    with pd.option_context("display.max_colwidth", None):
        if not N or N > len(df):
            sample = df
        else:
            sample = df.sample(N, random_state=random_state)

        styler = sample.style.set_properties(**{"white-space": "pre-wrap"})
        if left_align:
            styler = styler.set_properties(**{"text-align": "left"}).set_table_styles(
                [dict(selector="th", props=[("text-align", "left")])]
            )

        html = styler.render()
        display(HTML(html))

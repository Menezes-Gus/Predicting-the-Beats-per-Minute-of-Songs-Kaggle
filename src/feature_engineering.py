import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import *


def create_vocal_instrumental_ratio(df) -> pd.DataFrame:
    """
    Creates a Vocal/Instrumental ratio, based on the assumption that instrumental 
    songs may be more constant and have lower BPM, while songs with vocals may have 
    more changes in the structure to acomodate the singing, hence, possibly higher BPM

    Parameters
    ----------
    df : pd.DataFrame
        Input Dataframe.
    Return
    -------
    df : pd.DataFrame
        DataFrame with vocal_instrumental_ratio added
    """
    df['vocal_instrumental_ratio'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-5)
    return df

def create_energy_rhythm_interaction(df) -> pd.DataFrame:
    """
    Creates an Energy and Rhythm interaction, based on the assumption that both 
    energy and rhythmscore may be present in songs with higher BPM, so, when combined in a multiplication, 
    their combined value should be high enough to distance itself apart from slower songs

    Parameters
    ----------
    df : pd.DataFrame
        Input Dataframe.
    Return
    -------
    df : pd.DataFrame
        DataFrame with energy_rhythm_interaction added
    """
    df['energy_rhythm_interaction'] = df['Energy'] * df['RhythmScore']
    return df

def create_moodscore_bins(df) -> pd.DataFrame:
    """
    Creates MoodScore bins, based on the hypotesis that, because it measures something akin to an emotion, 
    probably works better as a range

    Parameters
    ----------
    df : pd.DataFrame
        Input Dataframe.
    Return
    -------
    df : pd.DataFrame
        DataFrame with MoodScore_bins added
    """
    df['MoodScore_bins'] = pd.qcut(df["MoodScore"], q=3, labels=['low','medium','high'])
    return df
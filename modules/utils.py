import smart_open as smart
from datetime import datetime

from config import boto_session


def push_sm_csv(loc, df, **kwargs):
    """
    Push a csv to S3. Just a wrapper for what was quite a verbose code chunk (also
    a function that is quite general in its utility).

    Parameters
    ----------
    loc : s3path.PureS3Path
        Where are we uploading to
    df : pd.DataFrame
        What are we uploading
    kwargs :
        to pass to pd.to_csv().

    Returns
    -------
    None

    """
    with smart.open(loc.as_uri(), 'w', transport_params={'session': boto_session}) as f:
        df.to_csv(f, **kwargs)


def get_now():
    return datetime.today().strftime('%Y%m%d%H%M%S')

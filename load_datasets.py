import pandas as pd

def load_oil(comp_dir):
    oil = pd.read_csv(
        comp_dir / "oil.csv",
        dtype={
            'dcoilwtico': 'float64',
        },
        parse_dates=['date'],
        infer_datetime_format=True,
    )
    oil = oil.set_index('date').to_period('D')
    oil = oil.rename(columns={'dcoilwtico':'price'})
    return oil


def load_holidays(comp_dir):
    holidays_events = pd.read_csv(
        comp_dir / "holidays_events.csv",
        dtype={
            'type': 'category',
            'locale': 'category',
            'locale_name': 'category',
            'description': 'category',
            'transferred': 'bool',
        },
        parse_dates=['date'],
        infer_datetime_format=True,
    )
    holidays_events = holidays_events.set_index('date').to_period('D')
    return holidays_events

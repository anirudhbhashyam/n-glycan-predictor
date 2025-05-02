import polars as pl

import os

def main() -> int:
    data_path = Path(os.path.join(os.path.dirname(__file__), *([os.pardir]*2), "data")).resolve()
    df = pl.read_csv(data_path / "data.csv")
    print(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
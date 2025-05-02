import polars as pl

def main() -> int:
    df = pl.read_csv("data/train.csv")
    print(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
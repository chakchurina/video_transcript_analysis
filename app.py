from app.preprocessing import DataProcessor
from config.config import DATA_PATH, FILE_NAMES, FILE


if __name__ == "__main__":
    processor = DataProcessor(DATA_PATH, FILE_NAMES, FILE)
    df = processor.prepare_data()

    print("\n".join(df['sentence']))
    print(df.head())
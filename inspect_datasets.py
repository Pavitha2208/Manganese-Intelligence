
from pathlib import Path
import pandas as pd

downloads = Path.home() / "Downloads"

targets = [
    downloads / "SIH26009_Six_Synthetic_Training_Datasets (2)",
    downloads / "Manganese_Project_Dataset_Starter_Bundle (1)",
    downloads / "ds140-manganese-2022.xlsx",
]

for target in targets:
    print(f"\n{'=' * 60}\n{target.name}\n{'=' * 60}")

    if not target.exists():
        print("Not found")
        continue

    files = (
        [target] if target.is_file()
        else sorted(target.glob("*"))
    )

    for file in files:
        if file.suffix.lower() not in {".xlsx", ".xls", ".csv"}:
            continue

        try:
            if file.suffix.lower() == ".csv":
                df = pd.read_csv(file, nrows=3)
                print(f"\nFile: {file.name}")
                print("Columns:", df.columns.tolist())
                print(df.head(2).to_string(index=False))
            else:
                excel = pd.ExcelFile(file)
                print(f"\nFile: {file.name}")
                print("Sheets:", excel.sheet_names)

                for sheet in excel.sheet_names:
                    df = pd.read_excel(
                        file, sheet_name=sheet, nrows=3
                    )
                    print(f"Sheet: {sheet}")
                    print("Columns:", df.columns.tolist())
                    print(df.head(2).to_string(index=False))

        except Exception as e:
            print(f"Could not read {file.name}: {e}")

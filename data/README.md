# Datasets (`data/`)

Raw and cleaned `.csv` datasets are excluded from Git via `.gitignore` (per Data Science Git best practices).  
Keep the cleaned CSV datasets in this folder locally when running the notebooks or Streamlit dashboard.

## Dataset Source

- **Kaggle Vehicle Dataset (CarDekho):** https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho

## Local Files & Cleaned Schema

| File | Rows | Cols | Key Columns |
| :--- | ---: | ---: | :--- |
| `Car details v3.csv` | 6,908 | 14 | `name`, `year`, `selling_price`, `km_driven`, `fuel`, `transmission`, `owner`, `mileage`, `engine`, `max_power`, `torque`, `seats`, `mileage_unit` |
| `car details v4.csv` | 2,059 | 32 | `make`, `model`, `price`, `year`, `kilometer`, `fuel_type`, `transmission`, `owner`, `engine_cc`, `max_power_bhp`, `max_torque_nm`, `seating_capacity` |
| `CAR DETAILS FROM CAR DEKHO.csv` | 3,577 | 8 | `name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner` |
| `car data.csv` | 299 | 9 | `Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Kms_Driven`, `Fuel_Type`, `Seller_Type`, `Transmission`, `Owner` |

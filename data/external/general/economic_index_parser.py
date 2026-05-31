import os

import pandas as pd


def parse_and_calculate_yearly_averages(
    input_file: str, output_file: str = "yearly_averages.csv"
) -> pd.DataFrame:
    """
    Parse CSV data and calculate yearly averages.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file

    Returns:
        DataFrame with yearly averages
    """

    try:
        # Read the CSV file
        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file)

        # Display basic info about the data
        print(f"Loaded {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        print(
            f"Date range: {df['observation_date'].min()} to {df['observation_date'].max()}"
        )

        # Convert observation_date to datetime
        df["observation_date"] = pd.to_datetime(df["observation_date"])

        # Extract year from the date
        df["year"] = df["observation_date"].dt.year

        # Get the value column name (assuming it's the second column)
        value_column = df.columns[1]  # DRCARC1Q027SBEA in this case

        print(f"Processing value column: {value_column}")

        # Calculate yearly averages
        yearly_averages = (
            df.groupby("year")[value_column]
            .agg(
                [
                    "mean",
                ]
            )
            .reset_index()
        )

        # Rename columns for clarity
        yearly_averages.columns = [
            "year",
            "average_value",
        ]

        # Round the values for better readability
        yearly_averages["average_value"] = yearly_averages["average_value"].round(3)

        # Create a simple output with just year and average
        simple_output = yearly_averages[["year", "average_value"]].copy()

        # Display results
        print("\nYearly Averages Summary:")
        print("=" * 50)
        for _, row in simple_output.iterrows():
            print(f"{int(row['year'])}: {row['average_value']:.3f}")

        # Save to CSV
        simple_output.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

        # Also save detailed statistics to a separate file
        detailed_file = output_file.replace(".csv", "_detailed.csv")
        yearly_averages.to_csv(detailed_file, index=False)
        print(f"Detailed statistics saved to {detailed_file}")

        return simple_output

    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing file: {e}")
        return pd.DataFrame()


def analyze_trends(df: pd.DataFrame) -> None:
    """
    Analyze trends in the yearly average data.

    Args:
        df: DataFrame with yearly averages
    """
    if df.empty:
        return

    print("\n=== Trend Analysis ===")
    print(f"Total years: {len(df)}")

    # Calculate year-over-year changes
    df_sorted = df.sort_values("year").copy()
    df_sorted["yoy_change"] = df_sorted["average_value"].pct_change() * 100
    df_sorted["absolute_change"] = df_sorted["average_value"].diff()

    # Find years with significant changes
    avg_change = df_sorted["yoy_change"].mean()
    print(f"Average year-over-year change: {avg_change:.2f}%")

    # Find highest and lowest values
    max_year = df_sorted.loc[df_sorted["average_value"].idxmax()]
    min_year = df_sorted.loc[df_sorted["average_value"].idxmin()]

    print(f"Highest average: {max_year['year']:.0f} ({max_year['average_value']:.3f})")
    print(f"Lowest average: {min_year['year']:.0f} ({min_year['average_value']:.3f})")

    # Find largest increases and decreases
    largest_increase = df_sorted.loc[df_sorted["yoy_change"].idxmax()]
    largest_decrease = df_sorted.loc[df_sorted["yoy_change"].idxmin()]

    if not pd.isna(largest_increase["yoy_change"]):
        print(
            f"Largest YoY increase: {largest_increase['year']:.0f} ({largest_increase['yoy_change']:.2f}%)"
        )

    if not pd.isna(largest_decrease["yoy_change"]):
        print(
            f"Largest YoY decrease: {largest_decrease['year']:.0f} ({largest_decrease['yoy_change']:.2f}%)"
        )


def main():
    """Main function to run the yearly average calculator."""

    # Default input file name
    input_file = "economic_index.csv"
    output_file = "economic_index_yearly_averages.csv"

    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Please provide a valid path.")
        return

    # Process the data
    result_df = parse_and_calculate_yearly_averages(input_file, output_file)

    if not result_df.empty:
        # Analyze trends
        analyze_trends(result_df)

        print("\n=== Summary ===")
        print(f"Successfully processed {len(result_df)} years of data")
        print(f"Output saved to: {output_file}")
        print(
            f"Detailed statistics saved to: {output_file.replace('.csv', '_detailed.csv')}"
        )
    else:
        print("Failed to process data.")


if __name__ == "__main__":
    main()

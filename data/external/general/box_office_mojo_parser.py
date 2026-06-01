import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import Dict


def scrape_boxoffice_data(
    url: str = "https://www.boxofficemojo.com/year/?area=XWW&grossesOption=totalGrosses",
) -> Dict[int, str]:
    """
    Scrape Box Office Mojo yearly data and extract Total Gross for each year.

    Args:
        url: The Box Office Mojo URL to scrape

    Returns:
        Dictionary with year as key and total gross as value
    """

    # Set headers to mimic a real browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # Make the request
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse the HTML
        soup = BeautifulSoup(response.content, "html.parser")

        # Find the data table
        table = soup.find("table", class_="mojo-body-table")

        if not table:
            raise ValueError("Could not find the data table on the page")

        # Extract data from table rows
        data = {}

        # Find all table rows (skip header)
        rows = (
            table.find("tbody").find_all("tr")
            if table.find("tbody")
            else table.find_all("tr")[1:]
        )

        for row in rows:
            cells = row.find_all("td")

            if len(cells) >= 2:  # Ensure we have at least year and total gross columns
                # Extract year
                year_cell = cells[0]
                year_link = year_cell.find("a")
                if year_link:
                    year_text = year_link.get_text().strip()
                    try:
                        year = int(year_text)
                    except ValueError:
                        continue
                else:
                    continue

                # Extract total gross
                gross_cell = cells[1]
                gross_text = gross_cell.get_text().strip()

                data[year] = gross_text  # Keep original formatting for display

        return data

    except requests.RequestException as e:
        print(f"Error making request: {e}")
        return {}
    except Exception as e:
        print(f"Error parsing data: {e}")
        return {}


def convert_gross_to_float(gross_str: str) -> float:
    """
    Convert gross string to float for numerical operations.

    Args:
        gross_str: String like "$24,871,532,862"

    Returns:
        Float value
    """
    cleaned = gross_str.replace("$", "").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def format_currency(amount: float) -> str:
    """Format float as currency string."""
    return f"${amount:,.0f}"


def analyze_boxoffice_trends(data: Dict[int, str]) -> None:
    """
    Analyze trends in the box office data.

    Args:
        data: Dictionary with year as key and gross as value
    """
    if not data:
        print("No data to analyze")
        return

    # Convert to list of tuples and sort by year
    sorted_data = sorted(
        [(year, convert_gross_to_float(gross)) for year, gross in data.items()]
    )

    print("\n=== Box Office Analysis ===")
    print(f"Years covered: {sorted_data[0][0]} - {sorted_data[-1][0]}")
    print(f"Total years: {len(sorted_data)}")

    # Find highest and lowest grossing years
    max_year, max_gross = max(sorted_data, key=lambda x: x[1])
    min_year, min_gross = min(sorted_data, key=lambda x: x[1])

    print(f"\nHighest grossing year: {max_year} ({format_currency(max_gross)})")
    print(f"Lowest grossing year: {min_year} ({format_currency(min_gross)})")

    # Calculate average
    avg_gross = sum(gross for _, gross in sorted_data) / len(sorted_data)
    print(f"Average yearly gross: {format_currency(avg_gross)}")


def save_to_csv(data: Dict[int, str], filename: str = "boxoffice_data.csv") -> None:
    """
    Save the data to a CSV file.

    Args:
        data: Dictionary with year as key and gross as value
        filename: Output CSV filename
    """
    # Convert to DataFrame
    df_data = []
    for year, gross in sorted(data.items()):
        df_data.append(
            {
                "Year": year,
                "Total_Gross": gross,
                "Total_Gross_Numeric": convert_gross_to_float(gross),
            }
        )

    df = pd.DataFrame(df_data)
    df.to_csv(filename, index=False)
    print(f"\nData saved to {filename}")


def main():
    """Main function to run the scraper."""
    print("Scraping Box Office Mojo data...")

    # Scrape the data
    data = scrape_boxoffice_data()

    if not data:
        print(
            "Failed to scrape data. Please check the website structure or your internet connection."
        )
        return

    # Display the results
    print(f"\nSuccessfully extracted data for {len(data)} years:")
    print("-" * 50)

    # Sort by year and display
    for year in sorted(data.keys(), reverse=True):
        print(f"{year}: {data[year]}")

    # Analyze trends
    analyze_boxoffice_trends(data)

    # Save to CSV
    save_to_csv(data)

    print("\nScraping completed successfully!")


if __name__ == "__main__":
    main()

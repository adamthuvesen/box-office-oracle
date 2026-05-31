import pandas as pd
import re
import random
import plotly.express as px

def assign_franchise_rating(row, major_franchise_keywords):
    # If franchise is already a number, return it as-is
    try:
        existing_rating = float(row['franchise_rating'])
        if not pd.isna(existing_rating):
            return int(existing_rating)
    except (ValueError, TypeError):
        pass
    
    if not row['franchise_rating']:
        return 0

    is_major_by_gross = row['worldwide_gross'] >= 300_000_000
    is_major_by_keyword = False
    title_lower = row['title'].lower()
    
    for keyword in major_franchise_keywords:
        if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', title_lower):
            is_major_by_keyword = True
            break
            
    if is_major_by_gross and is_major_by_keyword:
        return 2
    else:
        return 1

def generate_normal_int(min_val, max_val):
    mu = (min_val + max_val) / 2
    sigma = (max_val - min_val) / 3.5
    
    if sigma == 0:
        return round(mu)
        
    while True:
        value = random.gauss(mu, sigma)
        int_value = round(value)
        if min_val <= int_value <= max_val:
            return int_value

def generate_social_buzz(worldwide_gross):
    if worldwide_gross < 50_000_000:
        return generate_normal_int(1, 4)
    elif worldwide_gross < 250_000_000:
        return generate_normal_int(2, 6)
    elif worldwide_gross < 500_000_000:
        return generate_normal_int(3, 7)
    elif worldwide_gross < 1_000_000_000:
        return generate_normal_int(5, 9)
    elif worldwide_gross < 1_200_000_000:
        return generate_normal_int(9, 10)
    else:
        return generate_normal_int(10, 10)

def generate_ad_budget(production_budget):
    if production_budget < 10_000_000:
        multiplier = random.uniform(0.20, 0.35)
    elif production_budget < 50_000_000:
        multiplier = random.uniform(0.30, 0.40)
    elif production_budget < 100_000_000:
        multiplier = random.uniform(0.40, 0.50)
    elif production_budget < 200_000_000:
        multiplier = random.uniform(0.45, 0.55)
    else:
        multiplier = random.uniform(0.5, 0.70)
    
    return round(production_budget * multiplier)

def main():
    INPUT_FILE = "data/external/tmdb/enriched_movies_2020_2024.csv"
    OUTPUT_FILE = "data/external/tmdb/enriched_movies_2020_2024_final.csv"
    CHART_FILE = "data/external/tmdb/social_buzz_distribution_2020_2024.html"

    major_franchise_keywords = [
        # Marvel Cinematic Universe & Marvel properties
        "Star Wars", "Harry Potter", "Fantastic Beasts", "Avengers",
        "Spider-Man", "Iron Man", "Captain America", "Thor",
        "Guardians of the Galaxy", "Black Panther", "Doctor Strange",
        "Ant-Man", "Captain Marvel", "Black Widow", "Shang-Chi", "Eternals",
        "The Marvels", "The Incredible Hulk", "Deadpool", "Wolverine", "Logan",
        "X-Men", "Venom", "Morbius", "Silver Surfer", "Fantastic Four",
        
        # DC Extended Universe & DC properties
        "The Dark Knight", "Batman", "Joker", "Superman", "Man of Steel",
        "Wonder Woman", "Aquaman", "The Flash", "Shazam!", "Suicide Squad",
        "Justice League", "Birds of Prey", "Black Adam", "Blue Beetle",
        "Green Lantern", "Catwoman", "The Batman", "Batgirl",
        
        # Fantasy & Adventure franchises
        "The Lord of the Rings", "The Hobbit", "The Chronicles of Narnia",
        "Pirates of the Caribbean", "Indiana Jones", "The Matrix",
        "Avatar", "Dune", "John Carter", "Warcraft", "Mortal Kombat",
        "Tomb Raider", "Resident Evil", "Silent Hill", "Assassin's Creed",
        
        # Action franchises
        "Jurassic Park", "Jurassic World", "Mission: Impossible", 
        "Fast & Furious", "Hobbs & Shaw", "F9", "Fast X", "2 Fast 2 Furious",
        "Tokyo Drift", "Fast Five", "Fast & Furious 6", "Furious 7",
        "The Fate of the Furious", "Transformers", "Bumblebee",
        "Top Gun", "Die Hard", "Rambo", "The Expendables",
        "Taken", "John Wick", "Kingsman", "Bad Boys", "Terminator",
        "Alien", "Predator", "RoboCop", "Total Recall",
        
        # Horror franchises
        "It", "The Conjuring", "Annabelle", "The Nun", "Insidious",
        "Paranormal Activity", "Saw", "Scream", "Halloween", "Friday the 13th",
        "A Nightmare on Elm Street", "Child's Play", "The Ring", "The Grudge",
        "Final Destination", "Scary Movie", "The Purge", "Sinister",
        
        # Comedy franchises
        "The Hangover", "Ted", "22 Jump Street", "Anchorman", "Zoolander",
        "Meet the Parents", "American Pie", "Scary Movie", "The Grown Ups",
        "Night at the Museum", "Alvin and the Chipmunks", "The Smurfs",
        
        # Romance & Drama franchises
        "Fifty Shades", "Twilight", "The Notebook", "Magic Mike",
        "Mamma Mia!", "Bridget Jones", "Sex and the City",
        
        # Animated franchises
        "Frozen", "The Lion King", "The Super Mario Bros.", "Sonic the Hedgehog",
        "Minions", "Despicable Me", "Toy Story", "Incredibles", "Cars",
        "Finding Nemo", "Finding Dory", "Monsters", "Ice Age", "Shrek",
        "How to Train Your Dragon", "Madagascar", "Kung Fu Panda",
        "Rio", "The Secret Life of Pets", "Sing", "Hotel Transylvania",
        "Wreck-It Ralph", "Inside Out", "Zootopia", "Moana", "Coco",
        "Onward", "Soul", "Luca", "Turning Red", "Lightyear",
        "The Boss Baby", "Trolls", "The Croods", "Turbo", "Epic",
        "SpongeBob", "The Simpsons", "Family Guy", "South Park",
        
        # Monster & Kaiju franchises
        "Godzilla", "Kong", "Pacific Rim", "Cloverfield", "The Mummy",
        "Van Helsing", "Underworld", "Blade", "Ghost Rider",
        
        # Spy & Espionage franchises
        "James Bond", "Skyfall", "Spectre", "Casino Royale", "No Time to Die",
        "Quantum of Solace", "Goldeneye", "Tomorrow Never Dies",
        "The World Is Not Enough", "Die Another Day", "The Bourne Identity",
        "The Bourne Supremacy", "The Bourne Ultimatum", "The Bourne Legacy",
        "Jason Bourne", "xXx", "Men in Black", "Spy Kids",
        
        # Adventure & Family franchises
        "Jumanji", "National Treasure", "The Princess Diaries",
        "Cheaper by the Dozen", "The Santa Clause", "Home Alone",
        "Stuart Little", "Garfield", "Dr. Seuss", "The Cat in the Hat",
        "Horton Hears a Who", "The Grinch", "The Lorax",
        
        # Teen & Young Adult franchises
        "Hunger Games", "Divergent", "The Maze Runner", "Percy Jackson",
        "Mortal Instruments", "Vampire Academy", "The Host",
        "Beautiful Creatures", "Eragon", "The Golden Compass",
        
        # War & Historical franchises
        "300", "Gladiator", "Troy", "Alexander", "Kingdom of Heaven",
        "Master and Commander", "Black Hawk Down", "We Were Soldiers",
        "Saving Private Ryan", "Band of Brothers", "The Pacific",
        
        # Crime & Thriller franchises
        "Ocean's Eleven", "Ocean's Twelve", "Ocean's Thirteen", "Ocean's 8",
        "Now You See Me", "The Italian Job", "Gone in 60 Seconds",
        "Rush Hour", "Lethal Weapon", "Beverly Hills Cop",
        
        # Western franchises
        "The Magnificent Seven", "Django", "The Lone Ranger",
        "True Grit", "3:10 to Yuma", "Tombstone",
        
        # Planet of the Apes franchise
        "Planet of the Apes", "Rise of the Planet of the Apes",
        "Dawn of the Planet of the Apes", "War for the Planet of the Apes",
        
        # Star Trek franchise
        "Star Trek", "Into Darkness", "Beyond", "The Motion Picture",
        "The Wrath of Khan", "The Search for Spock", "The Voyage Home",
        "The Final Frontier", "The Undiscovered Country", "Generations",
        "First Contact", "Insurrection", "Nemesis",
        
        # Sherlock Holmes franchise
        "Sherlock Holmes", "A Game of Shadows", "Enola Holmes",
        
        # Miscellaneous franchises
        "Step Up", "The Fast and the Furious", "Need for Speed",
        "G.I. Joe", "Battleship", "The A-Team", "S.W.A.T.",
        "Charlie's Angels", "Lara Croft", "The Mummy Returns",
        "Scorpion King", "Universal Soldier", "Demolition Man",
        "Judge Dredd", "Dredd", "The Running Man", "Minority Report",
        "I, Robot", "Eagle Eye", "Surrogates", "In Time",
        "Elysium", "Chappie", "District 9", "Prometheus",
        "Blade Runner", "Ghost in the Shell", "Alita", "Ready Player One"
    ]

    try:
        print(f"Reading data from '{INPUT_FILE}'...")
        df = pd.read_csv(INPUT_FILE)

        required_cols = ['title', 'franchise_rating', 'worldwide_gross', 'production_budget']
        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            raise ValueError(f"Input file is missing required columns: {missing}")

        print("Step 0: Cleaning 'unknown' values...")
        # Replace all "unknown" values with NaN across all columns
        unknown_count = (df == 'unknown').sum().sum()
        if unknown_count > 0:
            print(f"  Found {unknown_count} 'unknown' values - replacing with empty values...")
            df = df.replace('unknown', pd.NA)
        else:
            print("  No 'unknown' values found, skipping...")

        print("Step 1: Filling missing franchise ratings...")
        missing_count = df['franchise_rating'].isna().sum()
        if missing_count > 0:
            print(f"  Found {missing_count} rows missing franchise_rating - filling them...")
            mask = df['franchise_rating'].isna()
            df.loc[mask, 'franchise_rating'] = df[mask].apply(
                assign_franchise_rating,
                args=(major_franchise_keywords,),
                axis=1
            )
        else:
            print("  All franchise_rating values already set, skipping...")

        print("Step 2: Filling missing social media buzz scores...")
        missing_count = df['social_media_buzz'].isna().sum()
        if missing_count > 0:
            print(f"  Found {missing_count} rows missing social_media_buzz - filling them...")
            mask = df['social_media_buzz'].isna()
            df.loc[mask, 'social_media_buzz'] = df.loc[mask, 'worldwide_gross'].apply(generate_social_buzz)
        else:
            print("  All social_media_buzz values already set, skipping...")
        
        print("Step 3: Filling missing advertising budgets...")
        missing_count = df['ad_budget'].isna().sum()
        if missing_count > 0:
            print(f"  Found {missing_count} rows missing ad_budget - filling them...")
            mask = df['ad_budget'].isna()
            df.loc[mask, 'ad_budget'] = df.loc[mask, 'production_budget'].apply(generate_ad_budget)
        else:
            print("  All ad_budget values already set, skipping...")
        
        print("Step 4: Saving final enriched data to '{OUTPUT_FILE}'...")
        df.to_csv(OUTPUT_FILE, index=False)

        print("Step 5: Saving buzz distribution chart to '{CHART_FILE}'...")
        fig = px.histogram(
            df,
            x='social_media_buzz',
            nbins=10,
            title='Distribution of Social Media Buzz Scores',
            labels={'social_media_buzz': 'Social Media Buzz Score (1-10)'},
            template='plotly_dark'
        )
        fig.update_layout(bargap=0.1)
        fig.write_html(CHART_FILE)
        
        print("\nProcessing complete.")

    except FileNotFoundError:
        print(f"Error: The input file '{INPUT_FILE}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
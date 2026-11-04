import csv
import re
import sys
import time
from urllib.parse import quote

import pandas as pd
import requests

# Defining the MediaWiki API endpoint URL for the Ukrainian-language Wikipedia as a constant
URL_FOR_UKRAINIAN_WIKI = 'https://uk.wikipedia.org/w/api.php'
# Defining the URL of the Wikidata API endpoint for retrieving structured data as a constant
URL_FOR_WIKIDATA_API = 'https://www.wikidata.org/w/api.php'
# Defining the User-Agent header containing contact details is defined in a dictionary to comply with Wikipedia’s API Terms of Use, 
HEADERS = {'User-Agent': 'UkrainianToponymsInfo/1.0 (https://github.com/katsufu)'}

# Defining Session object for Speeding up and stabilising communication by reusing TCP connections (Keep-Alive)
session = requests.Session()
# Registering a common User-Agent header for all generated Session objects in one go 
# (this will be automatically included in subsequent communications)
session.headers.update(HEADERS)

# Defining the list for Ukrainian 24 oblasts
UKRAINIAN_OBLASTS = [
    "Вінницька область", "Волинська область", "Дніпропетровська область",
    "Донецька область", "Житомирська область", "Закарпатська область",
    "Запорізька область", "Івано-Франківська область", "Київська область",
    "Кіровоградська область", "Луганська область", "Львівська область",
    "Миколаївська область", "Одеська область", "Полтавська область",
    "Рівненська область", "Сумська область", "Тернопільська область",
    "Харківська область", "Херсонська область", "Хмельницька область",
    "Черкаська область", "Чернівецька область", "Чернігівська область"
]

# Defining stems for above-defined UKRAINIAN_OBLASTS without suffix to be cased
OBLAST_STEM_PAIRS = [(region, region.split()[0][:-1].lower()) for region in UKRAINIAN_OBLASTS]

# Defining a dictionary for major cities, regional capitals and major cities in Crimea that only have their own city name category
# (e.g. Категорія:Вінниця)
EXACT_CITY_TO_OBLAST = {
    "київ": "Київ",
    "севастополь": "Севастополь",
    "сімферополь": "Автономна Республіка Крим",
    "керч": "Автономна Республіка Крим",
    "євпаторія": "Автономна Республіка Крим",
    "ялта": "Автономна Республіка Крим",
    "феодосія": "Автономна Республіка Крим",
    "джанкой": "Автономна Республіка Крим",
    "алушта": "Автономна Республіка Крим",
    "бахчисарай": "Автономна Республіка Крим",
    "красноперекопськ": "Автономна Республіка Крим",
    "яни капу": "Автономна Республіка Крим",
    "армянськ": "Автономна Республіка Крим",
    "судак (місто)": "Автономна Республіка Крим",
    "вінниця": "Вінницька область",
    "луцьк": "Волинська область",
    "дніпро (місто)": "Дніпропетровська область",
    "донецьк": "Донецька область",
    "житомир": "Житомирська область",
    "ужгород": "Закарпатська область",
    "запоріжжя": "Запорізька область",
    "івано-франківськ": "Івано-Франківська область",
    "кропивницький": "Кіровоградська область",
    "луганськ": "Луганська область",
    "львів": "Львівська область",
    "миколаїв": "Миколаївська область",
    "одеса": "Одеська область",
    "полтава": "Полтавська область",
    "рівне": "Рівненська область",
    "суми": "Сумська область",
    "тернопіль": "Тернопільська область",
    "харків": "Харківська область",
    "херсон": "Херсонська область",
    "хмельницький": "Хмельницька область",
    "черкаси": "Черкаська область",
    "чернівці": "Чернівецька область",
    "чернігів": "Чернігівська область",
    # Added Тульчин and Новодністровськ so they are directly classified as 'місто'
    "тульчин": "Вінницька область",
    "новодністровськ": "Чернівецька область"
}

# Defining regular expression 1: 
# A pattern that immediately excludes items, if they appear within parentheses (...) in the article title
# (such as foreign countries, stations, airports, neighbourhoods and natural features)
RE_INVALID_TITLE_QUALIFIER = re.compile(
    # Search for the following list of prohibited words, which begin immediately after an opening round bracket or at the start of a word within a bracket
    r'\((?:[^)]*\b)?('
    # Foreign countries except of Ukraine (Russia, Belarus, Poland, Kazakhstan, Moldova, Romania, Hungary, Slovakia, Bulgaria, Germany, USA, Canada)
    r'росія|білорусь|польща|казахстан|молдова|румунія|угорщина|словаччина|болгарія|німеччина|сша|канада|'
    # Foreign administrative units (gmina, voivodeship and county in Poland; county in Romania; region and district in Russia)
    r'гміна|воєводство|повіт|жудець|край|округ\s+росії|'
    # Transport infrastructure (railway stations, stops, platforms, signal boxes, airports, aerodromes)
    r'станція|зупинний\s+пункт|платформа|роз\'їзд|колія|аеропорт|летовище|'
    # Sports clubs and vessels (football clubs, sports clubs, ships, submarines)
    r'футбольний\s+клуб|спортклуб|баскетбольний|хокейний|корабель|судно|підводний|'
    # Sanatorium and nature reserves (sanatoriums, resorts, parks, nature reserves, Urochishche, forestry offices)
    r'санаторій|курорт|парк|дендропарк|заказник|урочище|заповідник|лісництво|'
    # Natural landforms (rivers, tributaries, lakes, reservoirs, hills, mountain ranges, islands, sandbars, bays)
    r'річка|притока|озеро|водосховище|гора|хребет|острів|коса|затока|'
    # Historical Regimes, Concepts, Personal Names and Military Units (People’s Republic of Ukraine, explanations of place names, surnames, regiments, battalions)
    r'унр|гетьманат|назва|топонім|прізвище|рід|герб|полк|батальйон|'
    # Media and works (periodicals, newspapers, magazines, novels, poetry, films, television dramas)
    r'часопис|газета|журнал|роман|поема|фільм|серіал|'
    # Archaeological sites, historic towns and urban districts (archaeological sites, castle ruins, ancient burial mounds, Cossack towns, urban districts)
    r'археологічна\s+пам\'ятка|городище|курган|стоянка|могильник|козацьке\s+місто|місцевість|район\s+міста'
    # From the end of a word to the closing round bracket (with the case-insensitive flag set)
    r')(?:\b[^)]*)?\)',
    re.IGNORECASE
)

# Defining regular expression 2: 
# Strictly exclude only categories for abolished/abandoned settlements and non-settlement infrastructure/landforms (removed 'місцевості' so suburban villages like Розсошенці and Старі Кодаки are not excluded)
RE_ABOLISHED_OR_NATURE_CATS = re.compile(
    # Patterns where ‘Former settlements, defunct settlements, abolished settlements, submerged settlements, villages subject to compulsory relocation and former villages’ appear immediately after ‘Category:’
    r'категорія:(колишні\s+населені\s+пункти|зниклі\s+населені\s+пункти|ліквідовані\s+населені\s+пункти|'
    r'затоплені\s+населені\s+пункти|виселені\s+села|колишні\s+села\b|'
    # Or categories such as rivers, lakes, mountains, islands, railway stations, airports and football clubs
    r'річки|озера|гори|острови|'
    r'залізничні\s+станції|зупинні\s+пункти|аеропорти|футбольні\s+клуби)',
    re.IGNORECASE
)

# Defining regular expression 3: 
# A pattern to exclude articles that explicitly state in the body text that the village has been ‘removed from the administrative register (abandoned village)’
RE_DEREGISTERED_TEXT = re.compile(
    # Matches ‘removed from the registeration’ or ‘deleted from the registeration’
    r'\b(знят[оеіий]\s+з\s+обліку|виключен[оеіий]\s+з\s+облікових\s+даних)\b',
    re.IGNORECASE
)

# Defining regular expression 4:
# A pattern that excludes cases where the term immediately following the dash in the first sentence (the definition of the subject) is ‘abandoned village’ or ‘non-settlement’
RE_NON_SETTLEMENT_SUBJECT = re.compile(
    # Strictly match only ‘former, defunct, abolished, submerged, evacuated, non-existent’ + ‘settlement, village, town, city’ (removed 'давнє/стародавнє' and 'містечко/колонія' which apply to active historic settlements)
    r'^\s*(?:це\s+)?(колишн[ійє]|зникл[ийеі]|ліквідован[ийеі]|затоплен[ийеі]|виселен[ийеі]|неіснуюч[ийеі])\s+'
    r'(?:[а-яіїєґ\'\u0301]+\s+)?(населений\s+пункт|село|селище(?!\s+міського\s+типу)|місто(?!\s+(республіканського|обласного|районного|державного))|хутір)\b'
    # Or where the subject is ‘nature reserves, Urochishche, rivers, lakes, mountains, islands, railway stations, airports, clubs, archaeological sites, Cossack towns or city districts’, allowing combining accent marks (\u0301)
    r'|^\s*(?:це\s+)?(?:[а-яіїєґ\'\u0301]+\s+){0,2}(заповідне\s+урочище|урочище|заказник|пам\'ятка\s+природи|лісництво|річка|озеро|гора|острів|'
    r'залізнична\s+станція|станція|зупинний\s+пункт|аеропорт|футбольний\s+клуб|городище|курган|козацьке\s+місто|козацьке\s+поселення|'
    r'місцевість|історична\s+місцевість|житловий\s+масив|мікрорайон|район\s+міста|частина\s+міста)\b',
    re.IGNORECASE
)

# Defining regular expression 5:
# A pattern for determining whether the subject immediately following the dash in the first sentence is ‘currently existing settlements (cities, capital cities, urban-type settlements, towns and villages)’
RE_VALID_SETTLEMENT_SUBJECT = re.compile(
    # A pattern where ‘city / capital / urban-type settlement / smt / settlement / village’ follows 0–8 words (removed generic 'населений пункт' to exclude general concept pages like Слобода)
    r'^\s*(?:це\s+)?(?:[а-яіїєґ0-9\'\-\u0301,]+\s+){0,8}\b(місто|столиця|селище\s+міського\s+типу|смт|селище|село)\b',
    re.IGNORECASE
)


# Defining the function to unify different apostrophe characters in Ukrainian
def normalize_apostrophe(text):
    """
    Replace all four types of apostrophes found in Ukrainian text (' U+0027, ’ U+2019, ʼ U+02BC, ` U+0060)
    with the standard half-width apostrophe "'" and return the result.
    """
    # Convert the input value to a string, use a regular expression to replace all types of apostrophes with "'", and return the result
    return re.sub(r"['’ʼ`]", "'", str(text))

# Defining the function that determines whether a searched place name matches (or matches in brackets) a Wikipedia article title
def is_matching_toponym_title(candidate_title, base_name):
    """
    Having taken into account the differences between apostrophes and upper- and lower-case letters, the system determines whether the title of the candidate page
    exactly matches the original place name, or whether it is in the format ‘Place name (Additional information)’.
    """
    # Normalize the use of apostrophes in the candidate titles, and convert them to lower case, removing any leading or trailing spaces
    normalized_candidate = normalize_apostrophe(candidate_title).strip().lower()
    # Normalize the use of apostrophes in place names and convert them to lower case, removing any leading or trailing spaces
    normalized_base_name = normalize_apostrophe(base_name).strip().lower()
    # Returns ‘True’ only if there is an exact match, or if the string begins with ‘reference place name + half-width space + opening round bracket’
    return normalized_candidate == normalized_base_name or normalized_candidate.startswith(f'{normalized_base_name} (')

# Defining the function that completely removes parentheses (...) and square brackets [...] and their contents from a string, including any nested brackets
def remove_parentheses(text):
    """
    When extracting definitions from the first sentence, remove all parentheses containing pronunciation notations or former names immediately following place names
    (e.g. (MFA: [...] (listen); until 1944 — ...)) in order from the innermost pair, so that they do not interfere with the use of dashes or full stops to separate the text.
    Also removes square brackets [...] (such as standalone IPA transcriptions) for clean sentence splitting.
    """
    # Initialise the variable that holds the original string to `None`
    previous = None
    # Assign the current string to the variable `current`
    current = text
    # Repeat the loop until the result of the bracket replacement no longer changes (all brackets have been removed)
    while previous != current:
        # Save current variable as previous variable
        previous = current
        # Replace the innermost ‘(...)’ or ‘[...]’ — which does not contain any other brackets — with an empty string
        current = re.sub(r'\([^()]*\)|\[[^\[\]]*\]', '', current)
    # Return cleaned strings
    return current

# Defining a JSON retrieval function with retry and timeout mechanisms, to be used across all API communications
def fetch_json_with_retry(url, params=None, retries=3, timeout=15):
    """
    Send a GET request to the specified URL and return the JSON as a dictionary.
    If a temporary communication error or timeout occurs, retry up to the maximum number of retries, gradually increasing the wait time between attempts.
    """
    # for loop that attempts a request up to the maximum number of retries
    for attempt in range(retries):
        # Start try block to catch communication errors
        try:
            # Send a GET request using the Session object (an exception is raised if no response is received within timeouts seconds)
            response = session.get(url, params=params, timeout=timeout)
            # Parse and return the JSON only if the HTTP status code is 200 (success)
            if response.status_code == 200:
                # Convert the response body to JSON (a Python dictionary) and return it
                return response.json()
            # In the event of a 404 (No data: e.g. articles with 0 page views in the Pageviews API), do not retry; return an empty dictionary immediately.
            if response.status_code == 404:
                # Return an empty dictionary to avoid unnecessary retries
                return {}
        # Handling exceptions such as communication time-outs or JSON conversion errors
        except Exception as exc:
            # If it is not the last trial
            if attempt < retries - 1:
                # Increase the wait time to 2 seconds, then 4 seconds, depending on the number of failures, before entering sleep mode (exponential backoff)
                time.sleep(2 * (attempt + 1))
            # If all trials fail
            else:
                # Display the error message
                print(f'\nAPI connection failed for {url}: {exc}')
    # Return empty dict in case no info retrieved with retry
    return {}

# Definition of the function for extraction of population, aggregated year, and fallback coordinates from Wikidata
def extract_population_and_year(intro, wikidata_id=None, latitude=None, longtitude=None):
    """
    1. First, use regular expressions to extract the population figures and the year of the census (four-digit AD year) from the Wikipedia introduction (intro).
    2. If population, year, or coordinates (latitude/longtitude) are missing, supplement them in a single call from the Wikidata API (P1082 for population, P585 for year, P625 for coordinates).
    """
    # Initialise population and year variables as None
    population, year = None, None
    # Only if introductory text (intro) is not empty, execute regular expression extraction from text
    if intro:
        # Search for numerical values, thousands (тис.) and units (осіб, etc.) appearing within 60 characters after ‘населення’
        population_info = re.search(
            r'населення[^\.\n]{0,60}?(\d[\d\s\u00a0]*(?:[,\.]\d+)?)\s*(тис\.?\s*)?(?:осіб|особи|особа|мешканц|жител|чол)',
            intro,
            re.IGNORECASE
        )
        # If a reference to population is found in the introductory text
        if population_info:
            # Remove spaces and non-breaking spaces from the numerical part (group(1)) and convert commas to decimal points
            raw_number = re.sub(r'[\s\u00a0]', '', population_info.group(1)).replace(',', '.')
            # Try block for converting strings to floating-point numbers and integers
            try:
                # Convert into float in case raw_number consisting of float
                pop_number = float(raw_number)
                # If ‘тис.’ (thousands) immediately follows (i.e. where group(2) exists)
                if population_info.group(2):
                    # Multiply the figure by 1,000 to convert it to the actual number of people
                    pop_number *= 1000
                # Rounded off and finalised as a population figure in integers (int type)
                population = int(round(pop_number))
            # Catching exceptions when numerical conversion fails
            except ValueError:
                # Skip without generating an error (keeping population as None)
                pass

            # Extract the substring located before and after the position where the population description was found (80 characters before to 50 characters after)
            snippet = intro[max(0, population_info.start() - 80):min(len(intro), population_info.end() + 50)]
            # Search for four-digit years from the 1900s or 2000s (19xx / 20xx) within the selected range
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', snippet)
            # If year is found
            if year_match:
                # Convert year into int and save as year variable
                year = int(year_match.group(1))
            # Where there is no year given and it simply says ‘All-Ukrainian Census (всеукраїнського перепису)’
            elif 'всеукраїнського перепису' in intro.lower():
                # The year 2001, when Ukraine’s only nationwide census was conducted, was set as the reference year
                year = 2001

    # If population, year, or coordinates are missing and a Wikidata ID exists, fetch all claims in a single Wikidata API call
    if (population is None or year is None or latitude is None or longtitude is None) and wikidata_id:
        # Retrieve all claims for the entity in one request so both P1082 (population) and P625 (coordinates) can be read
        wikidata_response = fetch_json_with_retry(
            URL_FOR_WIKIDATA_API,
            params={'action': 'wbgetclaims', 'entity': wikidata_id, 'format': 'json'},
            timeout=10
        )
        all_claims = wikidata_response.get('claims', {})

        # Retrieve claim list: P1082 (population)
        claims = all_claims.get('P1082', [])
        # If more than 1 claim retrieved
        if claims:
            # Retrieve the most recent population data at the end of the list
            latest_claim = claims[-1]
            # In case poplation is not still retrieved
            if population is None:
                # Extract a numeric string representing a population figure (e.g. "+31940") from a nested dictionary
                amount = latest_claim.get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('amount', '')
                # If a numeric string exists
                if amount:
                    # Remove the leading ‘+’ symbol, convert to an integer (int type) and store in `population`
                    population = int(float(amount.lstrip('+')))
            # In case year is not still retrieved
            if year is None:
                qualifiers = latest_claim.get('qualifiers', {}).get('P585', [])
                # If qualifiers is retrieved
                if qualifiers:
                    # Extract a date and time string (e.g. "+2022-01-01T00:00:00Z")
                    time_str = qualifiers[0].get('datavalue', {}).get('value', {}).get('time', '')
                    # Search for the 4-digit year using a regular expression
                    ym = re.search(r'(\d{4})', time_str)
                    # If a Gregorian year is found
                    if ym:
                        # Convert to an integer and store it in ‘year’
                        year = int(ym.group(1))

        # Retrieve claim list: P625 (coordinate location) if latitude or longtitude is missing on Wikipedia
        if latitude is None or longtitude is None:
            coord_claims = all_claims.get('P625', [])
            if coord_claims:
                coord_val = coord_claims[-1].get('mainsnak', {}).get('datavalue', {}).get('value', {})
                if isinstance(coord_val, dict):
                    latitude = coord_val.get('latitude', latitude)
                    longtitude = coord_val.get('longitude', longtitude)

    # Returns a tuple containing the extracted population, year, and (potentially supplemented) latitude and longtitude
    return population, year, latitude, longtitude

# Defining the function that uses the Wikimedia REST API to retrieve the average monthly page views for the past two years (24 months)
def get_2year_avg_monthly_pageviews(page_title):
    """
    Retrieve the monthly page views for the specified Ukrainian-language Wikipedia article over the past two years (24 months)
    from the Wikimedia REST API, and calculate and return the average number of page views per month (rounded to one decimal place)
    """
    # Replace spaces in the article title with underscores and convert the result to a URL-safe per cent-encoded string
    encoded_title = quote(page_title.replace(' ', '_'), safe='')
    # Constructing the URL for the Wikimedia Pageviews REST API (24-month period from 1 October 2024 to 30 September 2026)
    rest_url = (
        f'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/'
        f'uk.wikipedia/all-access/user/{encoded_title}/monthly/2024100100/2026093000'
    )
    # Retrieving JSON from a REST API using above-defined function
    res_json = fetch_json_with_retry(rest_url, timeout=10)
    # Extract the ‘items’ list (data for each month) from the response
    items = res_json.get('items', [])
    # If there is at least one entry for a particular month
    if items:
        # Calculate the total number of views for the entire month
        total_views = sum(item.get('views', 0) for item in items)
        # Calculate the monthly average by dividing the total number of views by the number of months, and return the result rounded to one decimal place
        return round(total_views / len(items), 1)
    # If no data exists (e.g. a new page or zero views), it returns 0.0
    return 0.0

# Define a function that sends a query to the Wikipedia API and returns detailed information from multiple pages, ensuring nothing is omitted
def query_api(extra_params, retries=3):
    """
    Retrieve the basic information, introduction, coordinates, categories, attributes and links to the Japanese version for the specified set of titles.
    If the number of categories, links, extracts, or coordinates exceeds the API limit per request, follow the 'continue' token to merge all items.
    """

    # Define a parameter dictionary for use across all queries (exclude ‘pageviews’ to improve performance, as the two-year average for page views is retrieved via a separate API)
    common_params = {
        'action': 'query',
        'format': 'json',
        'redirects': 1,
        'prop': 'info|extracts|coordinates|categories|pageprops|langlinks',
        'inprop': 'url',
        'lllang': 'ja',
        'llprop': 'url',
        'lllimit': 'max',
        'exintro': True,
        'explaintext': True,
        'exlimit': 'max',
        'colimit': 'max',
        'cllimit': 'max'
    }
    # Initialise an empty dictionary for storing and combining information on each page, using the page ID as the key
    merged_pages = {}
    # Initialise a dictionary to hold the parameters for API pagination (fetching the next page)
    continue_params = {}

    # A loop that repeats as long as there is any further continuation data (clcontinue, llcontinue, excontinue, cocontinue)
    while True:
        # Combine common parameters, additional parameters from the caller and continuation parameters into a single dictionary
        params = {**common_params, **extra_params, **continue_params}
        # Retrieving a JSON response from the Wikipedia API using above-defined function
        json_response = fetch_json_with_retry(URL_FOR_UKRAINIAN_WIKI, params=params, retries=retries)
        # If the response is empty (e.g. due to a communication error), exit the loop
        if not json_response:
            # As no further data can be retrieved, the loop is exited using a ‘break’ statement.
            break

        # Extract the ‘query -> pages’ dictionary from the response
        pages = json_response.get('query', {}).get('pages', {})
        # for loop that processes each retrieved page (page ID and page information dictionary) in sequence
        for pid, pdata in pages.items():
            # If that page ID does not yet exist in `merged_pages` (when retrieved for the first time)
            if pid not in merged_pages:
                # Register the page information dictionary as it is
                merged_pages[pid] = pdata
            # If that page ID already exists (during the second or subsequent continue request)
            else:
                # for loop to append and merge list-formatted properties (categories, links to other languages, coordinates) to an existing list
                for list_key in ('categories', 'langlinks', 'coordinates'):
                    # If the response contains the relevant key
                    if list_key in pdata:
                        # Retrieve the existing list for this property (or initialise an empty list)
                        existing_list = merged_pages[pid].setdefault(list_key, [])
                        # Append only new items that are not already in existing_list (preventing duplicate categories/coordinates across continue loops)
                        for item in pdata[list_key]:
                            if item not in existing_list:
                                existing_list.append(item)
                # for loop to fill in single-value properties (introduction, attributes, URL) if they have not been registered
                for key_info in ('extract', 'pageprops', 'fullurl'):
                    # If the response contains a key that is not yet present or is currently empty in the existing dictionary
                    if key_info in pdata and (key_info not in merged_pages[pid] or not merged_pages[pid][key_info]):
                        # Copy and save the value
                        merged_pages[pid][key_info] = pdata[key_info]
        # Check whether the response contains ‘continue’ (a token indicating that there is further data to be retrieved)
        cont = json_response.get('continue')
        # Only continue if at least one of the target property continuation tokens is present
        if cont and any(k in cont for k in ('clcontinue', 'llcontinue', 'excontinue', 'cocontinue')):
            # Update `continue_params` so that the next loop can pick up where it left off
            continue_params = cont
        # Once all category links have been retrieved
        else:
            # Complete the loop and exit
            break
    # Returns a dictionary containing information from all pages combined
    return merged_pages

# Defining the function to retrieve all internal links (handling over 500 entries) contained in disambiguation pages and similar pages
def get_links_from_page(page_title, retries=3):
    """
    Returns a list of all internal links (Namespace 0) within the specified page.
    Even on very large disambiguation pages containing more than 500 links, 
    plcontinue is used to retrieve every single link, right down to the villages in the states listed towards the end.
    """
    # Initialise an empty list to store the acquired link dictionary
    all_links = []
    # Initialise the parameter dictionary for continuous acquisition
    continue_params = {}

    # A loop that repeats until all links on the page have been processed
    while True:
        # Define the link retrieval parameters and persistence parameters as a single unit
        params = {
            'action': 'query',
            'format': 'json',
            'redirects': 1,
            'titles': page_title,
            'prop': 'links',
            'plnamespace': 0,
            'pllimit': 'max',
            **continue_params
        }
        # Using above-defined function, retrieve JSON information via API
        json_response = fetch_json_with_retry(URL_FOR_UKRAINIAN_WIKI, params=params, retries=retries)
        # If the response is empty, exit the loop
        if not json_response:
            # End the loop
            break

        # Extract the pages dictionary from the response
        pages = json_response.get('query', {}).get('pages', {})
        # A loop that retrieves the links list from each page object and adds it to all_links
        for page_object in pages.values():
            # Append a list of links (or an empty list if none exist) to the end
            all_links.extend(page_object.get('links', []))
        
        # If there are still links (plcontinue) beyond the 501st entry
        if 'continue' in json_response and 'plcontinue' in json_response['continue']:
            # Update the continuation token and proceed to the next loop
            continue_params = json_response['continue']
        # When all links have been retrieved
        else:
            # End the loop
            break
    # Returns a list of all internal links retrieved
    return all_links

# Defining the function that identifies and extracts only ‘currently valid settlements within Ukraine’ from one page’s worth of API data
def parse_toponym_page(info, base_name):
    """
    After checking the page information and excluding renamed former names, 
    foreign villages with the same name, abandoned villages, natural features, railway stations and urban districts,
    it returns a structured data dictionary for settlements (місто / селище / село) in Ukraine that are currently active.
    """

    # Exclude pages that do not exist (missing) or are disambiguation pages themselves (disambiguation)
    if 'missing' in info or 'disambiguation' in info.get('pageprops', {}):
        # Returns None as it is not applicable
        return None

    # Retrieve the official page title returned by the API
    actual_title = info.get('title', '')
    # Check whether the title matches the search place name (exclude redirects from former names here)
    if not is_matching_toponym_title(actual_title, base_name):
        # Returns None if the page was being redirected to a renamed alias page
        return None

    # Create a string in which the apostrophes in the title are standardised and converted to lower case
    normalize_actual_lower = normalize_apostrophe(actual_title).strip().lower()
    # Create a string with consistent apostrophes in search place names (for use in the accent regular expression later)
    normalize_base_name = normalize_apostrophe(base_name).strip()

    # If the title contains any prohibited words within the parentheses
    # (such as foreign names, railway stations, airports, neighbourhoods or natural features), 
    # it will be excluded immediately
    if RE_INVALID_TITLE_QUALIFIER.search(normalize_actual_lower):
        # Returns None for non-settlement or foreign pages
        return None

    # Check using a regular expression whether the title is in the format ‘Place name (〇〇ська область)’
    paren_oblast_match = re.search(r'\(([^)]*область)\)', normalize_actual_lower)
    # Where the title includes ‘...область (region)’ in brackets
    if paren_oblast_match:
        # Extract the state name string within the brackets and remove any leading and trailing spaces
        inside_oblast = paren_oblast_match.group(1).strip()
        # If it does not match any of Ukraine’s 24 regions
        if not any(inside_oblast == ukr_obl.lower() for ukr_obl in UKRAINIAN_OBLASTS):
            # Returns None because there is a village with the same name in a foreign state
            return None

    # Retrieve the list of coordinates assigned to the page
    coordinates = info.get('coordinates')
    # If coordinates exist, retrieve the latitude (lat); otherwise, return None
    latitude = coordinates[0]['lat'] if coordinates else None
    # If coordinates exist, retrieve the longtitude (lon); otherwise, return None
    longtitude = coordinates[0]['lon'] if coordinates else None
    # If both latitude and longitude are provided, verify whether the location lies within the geographical border box for Ukraine
    if latitude is not None and longtitude is not None:
        # If the location lies outside the range of 44.3° to 52.45° north latitude and 22.1° to 40.3° east longitude, it is a foreign place name and should therefore be excluded
        if not (44.3 <= latitude <= 52.45 and 22.1 <= longtitude <= 40.3):
            # Returns ‘None’ as the coordinates are from outside the country
            return None

    # Retrieve the introductory text (extract) of the article
    intro = info.get('extract', '')
    # Create a string consisting of all the category names assigned to the page, separated by spaces
    concatenated_text = ' '.join(c['title'] for c in info.get('categories', []))
    # Convert category strings to lower case
    concatenated_text_lower = concatenated_text.lower()
    # Change the text in the introduction to lower case
    intro_lower = intro.lower() if intro else ''

    # Exclude if the category includes ‘disappeared settlements’, ‘abandoned villages’, ‘urban neighbourhoods’, ‘natural features’, ‘stations’ or ‘airports’
    if RE_ABOLISHED_OR_NATURE_CATS.search(concatenated_text_lower):
        # Return None
        return None

    # Split the introduction into sections using line breaks and create a list of the actual paragraphs, excluding blank lines
    non_empty_paras = [p.strip() for p in intro.split('\n') if p.strip()]
    # Select the first paragraph that actually contains a definition dash or settlement keyword (in case line 0 is a short template/hatnote header)
    first_para = non_empty_paras[0] if non_empty_paras else ''
    for candidate_para in non_empty_paras[:3]:
        if re.search(r'[—–―]|\b(місто|столиця|селище|смт|село)\b', candidate_para, re.IGNORECASE):
            first_para = candidate_para
            break

    # Check for deregistration ('знято з обліку') only in the first paragraph so active cities that absorbed a deregistered village in paragraph 3 are not excluded
    if RE_DEREGISTERED_TEXT.search(first_para.lower()):
        # Return None
        return None
    
    # Completely remove parentheses (...), square brackets [...], and combining accent marks (\u0301) from the first paragraph for clean grammatical matching
    cleaned_first_para = remove_parentheses(first_para).replace('\u0301', '')

    # Separate the subject from the definition using dashes that have whitespace on at least one side, so compound words or unspaced ranges (e.g. Нью-Йорк, 1951—2021) are never split
    dash_split = re.split(r'\s+[—–―-]\s+|\s+[—–―]\s*|\s*[—–―]\s+', cleaned_first_para, maxsplit=1)
    # Extract the text to the right of the dash (the definition section); for articles without a dash, take the entire first paragraph
    def_part = dash_split[1] if len(dash_split) > 1 else cleaned_first_para
    # Extract the text up to the first sentence-ending full stop followed by a capital letter or end of string (never splitting on abbreviations)
    def_first_sentence = re.split(r'\.\s+(?=[А-ЯІЇЄҐA-Z0-9])', def_part, maxsplit=1)[0].strip().lower()

    # Exclude cases where the subject immediately following the dash is an abandoned village or non-settlement (unless it explicitly says 'нині місто/селище/село')
    if RE_NON_SETTLEMENT_SUBJECT.search(def_first_sentence) and not re.search(r'\bнині\s+(місто|селище|село)\b', def_first_sentence):
        # Return None
        return None

    # Initialise the variable storing the administrative classification (city / town / village) to None
    settlement_type = None
    
    # If the page title is in our verified EXACT_CITY_TO_OBLAST dictionary, immediately classify it as 'місто'
    if normalize_actual_lower in EXACT_CITY_TO_OBLAST:
        settlement_type = 'місто'
    else:
        # Use a regular expression to check whether the subject immediately following the dash matches a valid settlement keyword
        subject_match = RE_VALID_SETTLEMENT_SUBJECT.search(def_first_sentence)
        # When the subject matches a valid settlement keyword
        if subject_match:
            # Convert into lowercase
            matched_word = subject_match.group(1).lower()
            # In the case of ‘місто (city)’ or ‘столиця (capital)’
            if matched_word in ('місто', 'столиця'):
                # Set the administrative division to ‘місто’
                settlement_type = 'місто'
            # In the case of ‘селище міського типу (urban-type settlements)’, ‘смт (smt)’ and ‘селище (settlement)’
            elif matched_word in ('селище міського типу', 'смт', 'селище'):
                # In accordance with the official classification following the 2024 legislative amendments, the designation has been standardised as ‘селище’
                settlement_type = 'селище'
            # Otherwise matched_word is ‘село (village)’
            else:
                # Set the administrative division to ‘село’
                settlement_type = 'село'
        # Fallback when a long modifier (>8 words), extra dash, or instrumental case (містом/селищем/селом) is used
        else:
            fallback_text = f'{def_first_sentence} {cleaned_first_para.lower()}'
            if not RE_NON_SETTLEMENT_SUBJECT.search(cleaned_first_para.lower()):
                if re.search(r'\b(місто|містом|столиця)\b', fallback_text) or 'категорія:міста' in concatenated_text_lower:
                    settlement_type = 'місто'
                elif re.search(r'\b(селище\s+міського\s+типу|смт|селище|селищем)\b', fallback_text) or 'категорія:селища' in concatenated_text_lower:
                    settlement_type = 'селище'
                # Removed generic 'населений пункт' fallback so only official місто/селище/село are accepted
                elif re.search(r'\b(село|селом|у\s+селі)\b', fallback_text) or 'категорія:села' in concatenated_text_lower:
                    settlement_type = 'село'

    # Items that do not fall under any of the official settlement classifications (such as urban blocks, people or natural features) are excluded.
    if not settlement_type:
        # Return None
        return None

    # Exclude cases where the first sentence mentions a location in a foreign country
    if re.search(r'\bу\s+(росії|білорусі|польщі|казахстані|молдові|румунії)\b', def_first_sentence):
        # Return None
        return None

    # Prepare strings of the first paragraph and full intro with combining accents (\u0301) removed and converted to lower case for reliable oblast matching
    first_para_lower = first_para.replace('\u0301', '').lower()
    # Create an accent-free lowercase version of the full introductory text as a third-tier fallback for villages whose oblast is mentioned in the second paragraph
    clean_intro_lower = intro_lower.replace('\u0301', '')
    # Initialise the variable storing the region (oblast) to None
    oblast = None
    # 1. First, check whether there is a title in the exact-match table for major cities, state capitals and cities in Crimea
    if normalize_actual_lower in EXACT_CITY_TO_OBLAST:
        # Retrieve the corresponding state name (or Kyiv / Sevastopol / Crimea) from the table
        oblast = EXACT_CITY_TO_OBLAST[normalize_actual_lower]
    # 2. Check whether the settlement belongs to the Special City of Sevastopol
    elif 'севастопол' in concatenated_text_lower or 'севастопольськ' in first_para_lower:
        # Set the location to ‘Sevastopol’
        oblast = 'Севастополь'
    # 3. Check whether the settlement belongs to the Autonomous Republic of Crimea (including Crimean district/municipality stems such as Алушта / Партеніт / Судак / Феодосія / Ялта)
    elif 'крим' in concatenated_text_lower or ('крим' in clean_intro_lower and any(w in clean_intro_lower for w in ('автономн', 'республік', 'україн', 'район', 'феодос', 'ялт', 'судак', 'алушт', 'бахчисарай', 'євпатор', 'керч', 'джанкой'))):
        # Set the affiliation to ‘Autonomous Republic of Crimea’
        oblast = 'Автономна Республіка Крим'
    # 4. Check if normalize_actual_lower matches 24 regions of mainland Ukraine
    else:
        # Pass 1: Strictly match explicit oblast phrases ('<stem>... област', e.g. 'рівненської області') so historical regions like 'Волинського князівства' never override the actual oblast
        for target_text in (concatenated_text_lower, def_first_sentence, first_para_lower, clean_intro_lower):
            for region, adj_stem in OBLAST_STEM_PAIRS:
                if re.search(rf'{adj_stem}[а-яіїєґ]*\s+област', target_text):
                    oblast = region
                    break
            if oblast:
                break

        # Pass 2: If no explicit '... область/області' word was present, fall back to checking the adjective stem alone
        if not oblast:
            for target_text in (concatenated_text_lower, def_first_sentence, first_para_lower, clean_intro_lower):
                # Check the pre-generated pairs of (state name, adjective stem) in order
                for region, adj_stem in OBLAST_STEM_PAIRS:
                    # If an adjective stem (e.g. тернопільськ) is present in the target text
                    if adj_stem in target_text:
                        # Define the corresponding oblast
                        oblast = region
                        # Break the nested loop
                        break
                # If the state is identified at the current priority level, skip lower-priority text
                if oblast:
                    break

        # Fallback for Ukrainian villages that only mention their district (район) without explicitly writing the oblast name (e.g. Кагарлицький/Обухівський -> Київська, Смілянський -> Черкаська)
        if not oblast:
            district_to_oblast = {
                'кагарлицьк': 'Київська область', 'обухівськ': 'Київська область', 'білоцерківськ': 'Київська область',
                'бориспільськ': 'Київська область', 'броварськ': 'Київська область', 'бучанськ': 'Київська область',
                'вишгородськ': 'Київська область', 'фастівськ': 'Київська область',
                'смілянськ': 'Черкаська область', 'золотоніськ': 'Черкаська область', 'звенигородськ': 'Черкаська область',
                'уманськ': 'Черкаська область', 'подільськ': 'Одеська область', 'ананьївськ': 'Одеська область'
            }
            combined_search = f'{concatenated_text_lower} {clean_intro_lower}'
            for dist_stem, mapped_oblast in district_to_oblast.items():
                if dist_stem in combined_search:
                    oblast = mapped_oblast
                    break

    # Excluded are cases where the location does not fall within any of Ukraine’s regions or special territories (settlements outside the country)
    if not oblast:
        # Return None
        return None

    # Generate a regular expression pattern in which zero or one combining accent mark (\u0301) appears immediately after each character in a place name
    pattern = ''.join(r"['’ʼ`]\u0301?" if ch == "'" else f'{ch}\u0301?' for ch in normalize_base_name)
    # Search for place names with accent marks in the first 150 characters of the first paragraph
    matched_result = re.search(pattern, first_para[:150], re.IGNORECASE)

    # Search for IPA phonetic symbols enclosed in square brackets [...] (excluding footnotes consisting solely of numbers) within the first 150 characters of the first paragraph
    ipa_match = re.search(r'\[([^\[\]\d]{2,})\]', first_para[:150])
    # If an IPA is found and it is not an edit link such as ‘[ред.]’, it is formatted and stored; otherwise, it is None
    ipa = f'[{ipa_match.group(1).strip()}]' if (ipa_match and not ipa_match.group(1).startswith('ред')) else None

    # Retrieve the page’s Wikidata ID (e.g. Q1899)
    wikidata_id = info.get('pageprops', {}).get('wikibase_item')
    # Extract population, census year, AND supplement missing coordinates (latitude/longtitude) from Wikidata
    population, population_year, latitude, longtitude = extract_population_and_year(
        intro, wikidata_id, latitude, longtitude
    )

    # Re-verify bounding box in case coordinates were supplemented from Wikidata
    if latitude is not None and longtitude is not None:
        if not (44.3 <= latitude <= 52.45 and 22.1 <= longtitude <= 40.3):
            return None

    # Retrieve a list of interlanguage links (langlinks) to the Japanese Wikipedia
    langlinks = info.get('langlinks', [])
    # Sets a Boolean value to True if the list contains an element (Japanese version available), or False if it is empty (Japanese version not available)
    flag_ja_wiki = len(langlinks) > 0
    # If a Japanese version exists, retrieve the Japanese page title (e.g. Kyiv); otherwise, return None
    ja_wiki_title = langlinks[0].get('*') if flag_ja_wiki else None
    # If a Japanese version exists, retrieve its full URL; otherwise, return None
    ja_wiki_url = langlinks[0].get('url') if flag_ja_wiki else None

    # Retrieve the monthly average number of page views for the past two years (24 months) from the Wikimedia REST API
    avg_monthly_pv_2y = get_2year_avg_monthly_pageviews(actual_title)

    # Compile all the extracted items into a dictionary and return it
    return {
        'title': actual_title,
        'toponym_accent': matched_result.group(0) if matched_result else base_name,
        'ipa': ipa,
        'settlement_type': settlement_type,
        'oblast': oblast,
        'latitude': latitude,
        'longtitude': longtitude,
        'population': population,
        'population_year': population_year,
        'url': info.get('fullurl'),
        'avg_monthly_pv_2y': avg_monthly_pv_2y,
        'has_ja_wiki': flag_ja_wiki,
        'ja_wiki_title': ja_wiki_title,
        'ja_wiki_url': ja_wiki_url
    }

# Definition of the function for retrieval of information each Ukrainian toponym individually
def retrieving_ukraine_toponyms(toponym, progress_callback=None):
    """
    Given a specified Ukrainian place name, it rapidly searches for the direct settlement page and disambiguation pages (including two-level subpages)
    using a flat queue structure, and returns a list of all valid settlements.
    """

    # Convert the entered place name into a string and remove any leading or trailing spaces and special spaces (such as \xa0)
    toponym = str(toponym).strip()
    # If the input is an empty string or a missing value ('nan'), return an empty list and exit immediately
    if not toponym or toponym.lower() == 'nan':
        return []

    # Created standardised place names with consistent use of apostrophes and lower-case letters for comparison purposes
    normalize_toponym_lower = normalize_apostrophe(toponym).lower()

    # if the function for progress callback is called, Display the ‘Initial Verification in Progress’ status
    if progress_callback:
        progress_callback(0, 1, 'Checking main/disambig...')

    # Initial verification of a total of seven titles simultaneously: blank pages, disambiguation pages, lists relating to Ukraine, and (cities, towns, villages and urban settlements)
    initial_titles = [
        toponym, f'{toponym} (значення)', f'{toponym} (Україна)',
        f'{toponym} (місто)', f'{toponym} (селище)', f'{toponym} (село)', f'{toponym} (смт)'
    ]

    # Retrieve 7 titles in a single API call
    initial_pages = query_api({'titles': '|'.join(initial_titles)})

    # Initialise a set (collection) containing the titles of the individual candidate pages to be analysed, ensuring there are no duplicates
    target_titles = set()
    # Initialise the queue (list) containing the titles of the disambiguation pages for which links should be expanded
    disambig_queue = []
    # Initialise the set that records disambiguation pages for which links have already been expanded (to prevent infinite loops and duplicate communications)
    visited_disambig = set()
    # Initialise the cache dictionary keyed by page ID (str) to store pages already retrieved during the first request (preventing duplicate entries when merged with query_api results)
    already_fetched_pages = {}
    # Initialise a set to track the exact titles already fetched in the initial call so they are not re-queried in batches
    fetched_titles = set()

    # for loop to categorise each page (with its page_id and page_data) retrieved for the first time
    for page_id, page_data in initial_pages.items():
        # If the page does not exist, skip
        if 'missing' in page_data:
            continue
        # Retrieve title for the page
        title = page_data.get('title', '')
        # Create a title with consistent use of apostrophes and in lower case
        normalize_title_lower = normalize_apostrophe(title).lower()
        # If the page attributes include ‘disambiguation’
        if 'disambiguation' in page_data.get('pageprops', {}):
            # Add to the link expansion queue
            disambig_queue.append(title)
        # Where it is a standard article and the title matches a place name used in a search
        elif is_matching_toponym_title(title, toponym):
            # Add to the set of analysis candidates
            target_titles.add(title)
            # Only cache the page as already fetched if its introductory 'extract' and 'categories' were completely retrieved in the initial call
            if page_data.get('extract') and page_data.get('categories'):
                already_fetched_pages[str(page_id)] = page_data
                fetched_titles.add(title)
            # Add to the queue just in case the page ‘Place names (Ukraine)’ is a list page without the ‘disambiguation’ attribute
            if normalize_title_lower == f'{normalize_toponym_lower} (україна)':
                disambig_queue.append(title)

    # Expand all disambiguation pages and sub-list pages using a flat `while` loop with no nesting
    while disambig_queue:
        # Retrieve one title from the disambiguation page at the front of the queue
        d_title = disambig_queue.pop(0)
        # Skip if the page has already been displayed
        if d_title in visited_disambig:
            continue
        # Record in the deployed set
        visited_disambig.add(d_title)
        
        # If there is a progress callback, display a message indicating that the link is being expanded
        if progress_callback:
            progress_callback(0, 1, 'Expanding links...')

        # A loop that retrieves all internal links on the page (including all of the more than 500) and checks them in order
        for link in get_links_from_page(d_title):
            # Retrieve the title of the linked page
            link_title = link['title']
            # Standardisation of Apostrophes + Lower-Casing
            normalize_link_lower = normalize_apostrophe(link_title).lower()
            # If the linked page is a sub-disambiguation page such as ‘Place name (Ukraine)’ or ‘Place name (disambiguation)’
            if normalize_link_lower in (f'{normalize_toponym_lower} (україна)', f'{normalize_toponym_lower} (значення)'):
                # If it has not yet been visited, add it to the expansion queue (it will be expanded flat in the next while loop)
                if link_title not in visited_disambig:
                    disambig_queue.append(link_title)
            # Where the linked page is a specific page matching the search location and does not contain any prohibited words (such as ‘station’ or ‘foreign country’) in brackets
            elif is_matching_toponym_title(link_title, toponym) and not RE_INVALID_TITLE_QUALIFIER.search(normalize_link_lower):
                # Add to the set of candidates for analysis
                target_titles.add(link_title)

    # Sort and list only those titles from the candidate set that are not yet in fetched_titles
    remaining_titles = sorted(t for t in target_titles if t not in fetched_titles)
    # Retrieve the total number of candidate pages to be analysed
    total_candidates = len(target_titles)

    # If there are no candidates at all
    if total_candidates == 0:
        # Update the progress bar to 100% (complete)
        if progress_callback:
            progress_callback(1, 1, 'Done')
        # Return an empty list and exit
        return []

    # Total number of steps in the progress bar = ‘Number of API requests for pages not yet retrieved’ + ‘Number of candidate pages analysed’
    total_steps = len(remaining_titles) + total_candidates
    # Initialise the dictionary for storing all pages based on the cached page dictionary
    pages = dict(already_fetched_pages)

    # A loop that retrieves details from the API for uncollected candidate pages, in batches of 15 at a time (preventing category/extract truncation on large batches)
    for i in range(0, len(remaining_titles), 15):
        # Extract this batch (up to 15 titles)
        batch = remaining_titles[i:i + 15]
        # Display the current batch number in the progress callback
        if progress_callback:
            progress_callback(i, total_steps, f'API batch {i // 15 + 1}/{(len(remaining_titles) - 1) // 15 + 1}')
        # Retrieve 15 items in bulk from the API by separating them with ‘|’, and merge them into the ‘pages’ dictionary
        pages.update(query_api({'titles': '|'.join(batch)}))

        # Except for the final batch, wait 0.3 seconds to reduce the load on the server while keeping execution fast
        if i + 15 < len(remaining_titles):
            time.sleep(0.3)

    # Initialise the list to store the analysis results for valid settlements
    results = []
    # Initialise a set of seen URLs to guarantee zero duplicate rows in the output even if multiple redirect links point to the same article
    seen_urls = set()
    # Create a list of all the page data retrieved
    page_values = list(pages.values())
    # for loop that parses each page one by one using `parse_toponym_page`
    for idx, info in enumerate(page_values):
        # Display which page is currently being analysed in the progress callback
        if progress_callback:
            progress_callback(len(remaining_titles) + idx + 1, total_steps, f'Parsing ({idx + 1}/{len(page_values)})')
        # Analysing web pages to determine whether they relate to valid settlements in Ukraine and extracting data
        parsed = parse_toponym_page(info, toponym)
        # If extracted as valid settlement data (i.e. not None) and its URL has not been added yet
        if parsed and parsed['url'] not in seen_urls:
            # Record the URL in seen_urls to prevent duplicates
            seen_urls.add(parsed['url'])
            # Add to results list
            results.append(parsed)

    # Returns a list of all extracted settlement data
    return results

# Defining the function for retrieval of external file as csv
def get_file(filename=None):
    """
    This function checks the file path passed as a command-line argument; if it is a .xlsx or .xls file, it uses `read_excel`;
    otherwise (e.g. .csv), it uses `read_csv` to read the file and returns a DataFrame.
    """
    # The default value is None, thus, capturing ValueError if filename is not given
    try:
        if str(filename).lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(filename)
        return pd.read_csv(filename)        
    except Exception as exc:
        print(f'Please kindly give a valid file name (.csv or .xlsx): {exc}')
        return pd.DataFrame()

# Defining the function that generates an analogue ASCII progress bar string on the console
def make_bar(current, total, width=12):
    """
    Based on the current number of steps (current) and the total number of steps (total),
    return a visual progress bar string such as ‘[██████░░░░░░]  50.0%’.
    """
    # If the total number of steps is 0 or less, return a progress bar showing 100% completion.
    if total <= 0:
        return '[' + '█' * width + '] 100.0%'
    # Calculated by keeping the progress rate within the range of 0.0 to 1.0
    ratio = min(max(current / total, 0.0), 1.0)
    # Calculate the number of filled blocks (█) to the nearest whole number
    filled = int(round(width * ratio))
    # Create a bar string by joining the filled-in sections and the unfinished sections (░)
    bar = '█' * filled + '░' * (width - filled)
    # Format and return bar and percentage values
    return f'[{bar}] {ratio * 100:5.1f}%'

# Defining the main function that coordinates the entire process (file reading, loop execution, progress display and CSV output)
def main():
    """
    It reads place names and their Japanese transcriptions from the input file, cleans out blank lines (NaN), and then
    collects Wikipedia data one entry at a time, writing it to a CSV file in real time.
    """
    # Retrieving DataFrame for toponyms
    df = get_file(filename=sys.argv[1] if len(sys.argv) > 1 else None)
    # If DataFrame is empty, the function will complete
    if df.empty:
        return

    # Prevent TypeErrors by removing blank lines (NaN / float type) contained at the end of CSV files, etc.
    df = df.dropna(subset=['Ukrainian Toponyms']).copy()
    # Exclude lines containing only whitespace characters and reset the index to 0
    df = df[df['Ukrainian Toponyms'].astype(str).str.strip() != ''].reset_index(drop=True)

    # Extract the column of Ukrainian place names as a string and remove any invisible leading or trailing spaces (such as \xa0)
    ukrainian_toponyms = df['Ukrainian Toponyms'].astype(str).str.strip()
    # If a column containing the Japanese transcription exists, set any missing values to an empty string and remove spaces; if it does not exist, create a list of empty strings
    japanese_transliterations = (
        df['Japanese Transliteration'].fillna('').astype(str).str.strip()
        if 'Japanese Transliteration' in df.columns
        else [''] * len(df)
    )

    # Define columns name for csv
    column_name = [
        'id', 'uid', 'toponym', 'japanese_transliteration', 'toponym_with_accent',
        'ipa', 'settlement_type', 'oblast', 'latitude', 'longtitude',
        'population', 'population_year', 'url', 'avg_monthly_pv_2y',
        'has_ja_wiki', 'ja_wiki_title', 'ja_wiki_url'
    ]

    # Retrieve the total number of place names to be processed
    total_toponyms = len(ukrainian_toponyms)
    # Initialise a variable to count the total number of rows written to the CSV file to 0
    total_written_rows = 0
    # Record the start time of the entire process
    start_time = time.time()

    # Display a message indicating the start of processing on the console
    print(f'=== Starting Wikipedia retrieval for {total_toponyms} toponyms ===')

    # Retrieving information from Wikipedia for each toponym with for loop, writing it into csv
    with open('ukrainian_toponyms_wikiinfo.csv', 'w', newline='', encoding='utf-8') as f:
        # Initialise csv writer object
        writer = csv.writer(f)
        # Write column_name on the 1st line
        writer.writerow(column_name)

        # A loop that processes Ukrainian place names and their Japanese transcriptions in sequence, pairing them by index (i)
        for i, (ukrainian_toponym, ja_translit) in enumerate(zip(ukrainian_toponyms, japanese_transliterations)):
            # Record the start time for processing this place name
            item_start_time = time.time()
            # To ensure the console display remains legible, create display names for place names longer than 20 characters by truncating them with ‘..’ at the end
            display_name = (ukrainian_toponym[:18] + '..') if len(ukrainian_toponym) > 20 else ukrainian_toponym

            # The inner callback function that updates the same line in real time as progress is made within the place name
            def update_status(step_curr, step_total, status_msg):
                """Calculate the sub-progress and overall progress rate within the current place name, and update the single-line progress bar"""
                # Calculate the progress rate (0.0 ~ 1.0) for the current place name
                sub_ratio = (step_curr / step_total) if step_total > 0 else 0.0
                # Calculate the overall progress rate (%) by adding the progress within each place name (sub_ratio) to the number of place names completed (i)
                overall_pct = ((i + sub_ratio) / total_toponyms) * 100
                # Generate a progress bar string
                bar_str = make_bar(step_curr, step_total, width=12)
                # Calculate the elapsed time for this place name
                item_elapsed = time.time() - item_start_time
                # Calculate the number of seconds elapsed since the start of the session
                total_elapsed = time.time() - start_time
                # Construct a string that overwrites the same line using a carriage return (\r)
                line = (
                    f'\r[{i+1:4d}/{total_toponyms} ({overall_pct:5.1f}%)] '
                    f'{display_name:<20} | {bar_str} | {status_msg:<22} | '
                    f'Item: {item_elapsed:4.1f}s (Total: {total_elapsed:5.1f}s)'
                )
                # Write to standard output (without line breaks)
                sys.stdout.write(line)
                # Flush the buffer immediately so that the changes are reflected on the screen
                sys.stdout.flush()

            # Pass a callback function to retrieve data on all settlements with that place name from Wikipedia
            group_of_info_for_toponym = retrieving_ukraine_toponyms(
                ukrainian_toponym,
                progress_callback=update_status
            )

            # A loop that writes each set of village data obtained into a CSV file in sequence
            for j, info in enumerate(group_of_info_for_toponym):
                # Assemble the output data list for a single row
                to_be_written_info = [
                    i + 1,
                    f'{i + 1}_{j + 1}',
                    ukrainian_toponym,
                    ja_translit,
                    info['toponym_accent'],
                    info['ipa'],
                    info['settlement_type'],
                    info['oblast'],
                    info['latitude'],
                    info['longtitude'],
                    info['population'],
                    info['population_year'],
                    info['url'],
                    info['avg_monthly_pv_2y'],
                    info['has_ja_wiki'],
                    info['ja_wiki_title'],
                    info['ja_wiki_url']
                ]
                # Write one line to a CSV file
                writer.writerow(to_be_written_info)
                # Increase the total number of lines written counter by 1
                total_written_rows += 1

            # Force-flush the OS’s file buffer and ensure that each block is saved to disk
            f.flush()

            # Calculate the overall progress rate upon completion per 1 location
            overall_pct_done = ((i + 1) / total_toponyms) * 100
            # Creating a progress bar showing 100% completion
            bar_done = make_bar(1, 1, width=12)
            # Calculate the number of seconds remaining for this place name
            item_elapsed = time.time() - item_start_time
            # Calculate the total number of seconds elapsed since the start
            total_elapsed = time.time() - start_time
            # Create a message showing the number of page views
            found_msg = f'Found: {len(group_of_info_for_toponym):3d} pages'
            # Overwrite the current line, then insert a line break (the default for `print`), and leave it on the screen as a confirmation log
            print(
                f'\r[{i+1:4d}/{total_toponyms} ({overall_pct_done:5.1f}%)] '
                f'{display_name:<20} | {bar_done} | {found_msg:<22} | '
                f'Item: {item_elapsed:4.1f}s (Total: {total_elapsed:5.1f}s)'
            )

            # To reduce the load on the Wikipedia servers, please wait one second before processing the next place name.
            time.sleep(1)

    # Calculate the total elapsed time after all items have been completed
    total_elapsed = time.time() - start_time
    # Display the final summary (total number of output lines and time taken) on the console
    print(f'\n=== Completed! Total rows written: {total_written_rows} | Total time: {total_elapsed/60:.1f} min ===')

if __name__ == '__main__':
    main()
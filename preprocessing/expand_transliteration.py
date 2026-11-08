import os
import re
import sys

import pandas as pd

# Defining the directory path where this Python script is located as a constant
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Defining the combining acute accent character in Unicode (U+0301) as a constant
ACCENT = '\u0301'

# Defining the set of all 10 Ukrainian vowel letters
UKR_VOWELS = set('аоуеиіїяює')

# Defining the set of Ukrainian consonants that trigger gemination when doubled after a stressed vowel 
# in Japanese transliteration, gemination would be transliterated as ('ッ')
# (Note: 'н' is excluded because doubled 'нн' is represented as 'ン' + Na-row syllable rather than 'ッ')
GEMINATE_CONSONANTS = set('бвгґджзклмпрстфхцчшщ')

# Defining the dictionary for regressive voicing assimilation (voiceless obstruent -> voiced counterpart)
# Applied when a voiceless obstruent is immediately followed by a voiced obstruent (б, д, ґ, з, ж, г)
VOICING_ASSIMILATION_MAP = {
    'п': 'б',
    'т': 'д',
    'к': 'ґ',
    'ф': 'в',
    'с': 'з',
    'ш': 'ж',
    'х': 'г',
    'ц': 'дз',
    'ч': 'дж'
}

# Defining Mapping Table 1: 1-to-1 conversion table for standalone, word-initial, post-vowel, or post-apostrophe vowels
VOWEL_MAP = {
    'а': 'ア',
    'о': 'オ',
    'у': 'ウ',
    'е': 'エ',
    'и': 'イ',
    'і': 'イ',
    'я': 'ヤ',
    'ю': 'ユ',
    'є': 'イェ',
    'ї': 'イィ'
}

# Defining Mapping Table 2: 1-to-1 conversion table for coda consonants (not followed by a vowel or soft sign 'ь')
CONSONANT_MAP = {
    'б': 'ブ',
    # Rule: 'в' without a following vowel becomes 'ウ'
    'в': 'ウ',
    # Rule: 'в' immediately before an apostrophe (hard sign, e.g. В'язівок, Слов'янськ) retains 'ヴ'
    "в'": 'ヴ',
    'г': 'フ',
    'ґ': 'グ',
    'д': 'ド',
    'дз': 'ズ',
    'дж': 'ジュ',
    'ж': 'ジュ',
    'з': 'ズ',
    'й': 'イ',
    'к': 'ク',
    'л': 'ル',
    # Rule: 'м' before non-labial consonants, apostrophe, or word-end becomes 'ム' (before б, п, в, ф, м it assimilates to 'ン')
    'м': 'ム',
    'н': 'ン',
    'п': 'プ',
    'р': 'ル',
    'с': 'ス',
    'т': 'ト',
    'ф': 'フ',
    'х': 'フ',
    'ц': 'ツ',
    'ч': 'チ',
    'ш': 'シュ',
    'щ': 'シチ'
}

# Defining Mapping Table 3: 1-to-1 conversion table for palatalized coda consonants (consonant + soft sign 'ь')
SOFT_CODA_MAP = {
    'д': 'ディ',
    'дз': 'ズィ',
    'з': 'ズィ',
    'л': 'リ',
    'н': 'ニ',
    'р': 'リ',
    'с': 'スィ',
    'т': 'ティ',
    'ц': 'ツィ',
    'ш': 'シュ'
}

# Defining Mapping Table 4: 1-to-1 conversion table for Consonant + Vowel (CV) syllables
CV_MAP = {
    'б': {'а': 'バ', 'о': 'ボ', 'у': 'ブ', 'е': 'ベ', 'и': 'ブィ', 'і': 'ビ', 'я': 'ビャ', 'ю': 'ビュ', 'є': 'ビェ'},
    'в': {'а': 'ヴァ', 'о': 'ヴォ', 'у': 'ヴ', 'е': 'ヴェ', 'и': 'ヴィ', 'і': 'ヴィ', 'я': 'ヴャ', 'ю': 'ヴュ', 'є': 'ヴェ'},
    'г': {'а': 'ハ', 'о': 'ホ', 'у': 'フ', 'е': 'ヘ', 'и': 'ヒ', 'і': 'ヒ', 'я': 'ヒャ', 'ю': 'ヒュ', 'є': 'ヒェ'},
    'ґ': {'а': 'ガ', 'о': 'ゴ', 'у': 'グ', 'е': 'ゲ', 'и': 'グィ', 'і': 'ギ', 'я': 'ギャ', 'ю': 'ギュ', 'є': 'ギェ'},
    'д': {'а': 'ダ', 'о': 'ド', 'у': 'ドゥ', 'е': 'デ', 'и': 'ドィ', 'і': 'ディ', 'я': 'デャ', 'ю': 'デュ', 'є': 'デェ'},
    'дз': {'а': 'ザ', 'о': 'ゾ', 'у': 'ズ', 'е': 'ゼ', 'и': 'ズィ', 'і': 'ジ', 'я': 'ジャ', 'ю': 'ジュ', 'є': 'ゼ'},
    'дж': {'а': 'ジャ', 'о': 'ジョ', 'у': 'ジュ', 'е': 'ジェ', 'и': 'ジ', 'і': 'ジ', 'я': 'ジャ', 'ю': 'ジュ', 'є': 'ジェ'},
    'ж': {'а': 'ジャ', 'о': 'ジョ', 'у': 'ジュ', 'е': 'ジェ', 'и': 'ジ', 'і': 'ジ', 'я': 'ジャ', 'ю': 'ジュ', 'є': 'ジェ'},
    'з': {'а': 'ザ', 'о': 'ゾ', 'у': 'ズ', 'е': 'ゼ', 'и': 'ズィ', 'і': 'ジ', 'я': 'ズャ', 'ю': 'ジュ', 'є': 'ゼ'},
    'й': {'а': 'ヤ', 'о': 'ヨ', 'у': 'ユ', 'е': 'イェ', 'и': 'イ', 'і': 'イィ', 'я': 'ヤ', 'ю': 'ユ', 'є': 'イェ'},
    'к': {'а': 'カ', 'о': 'コ', 'у': 'ク', 'е': 'ケ', 'и': 'クィ', 'і': 'キ', 'я': 'キャ', 'ю': 'キュ', 'є': 'キェ'},
    'л': {'а': 'ラ', 'о': 'ロ', 'у': 'ル', 'е': 'レ', 'и': 'ルィ', 'і': 'リ', 'я': 'リャ', 'ю': 'リュ', 'є': 'リェ'},
    'м': {'а': 'マ', 'о': 'モ', 'у': 'ム', 'е': 'メ', 'и': 'ムィ', 'і': 'ミ', 'я': 'ミャ', 'ю': 'ミュ', 'є': 'ミェ'},
    'н': {'а': 'ナ', 'о': 'ノ', 'у': 'ヌ', 'е': 'ネ', 'и': 'ヌィ', 'і': 'ニ', 'я': 'ニャ', 'ю': 'ニュ', 'є': 'ニェ'},
    'п': {'а': 'パ', 'о': 'ポ', 'у': 'プ', 'е': 'ペ', 'и': 'プィ', 'і': 'ピ', 'я': 'ピャ', 'ю': 'ピュ', 'є': 'ピェ'},
    'р': {'а': 'ラ', 'о': 'ロ', 'у': 'ル', 'е': 'レ', 'и': 'ルィ', 'і': 'リ', 'я': 'リャ', 'ю': 'リュ', 'є': 'リェ'},
    'с': {'а': 'サ', 'о': 'ソ', 'у': 'ス', 'е': 'セ', 'и': 'スィ', 'і': 'シ', 'я': 'スャ', 'ю': 'スュ', 'є': 'スェ'},
    'т': {'а': 'タ', 'о': 'ト', 'у': 'トゥ', 'е': 'テ', 'и': 'トィ', 'і': 'ティ', 'я': 'テャ', 'ю': 'テュ', 'є': 'テェ'},
    'ф': {'а': 'ファ', 'о': 'フォ', 'у': 'フ', 'е': 'フェ', 'и': 'フィ', 'і': 'フィ', 'я': 'フィャ', 'ю': 'フュ', 'є': 'フェ'},
    'х': {'а': 'ハ', 'о': 'ホ', 'у': 'フ', 'е': 'ヘ', 'и': 'ヒ', 'і': 'ヒ', 'я': 'ヒャ', 'ю': 'ヒュ', 'є': 'ヒェ'},
    'ц': {'а': 'ツァ', 'о': 'ツォ', 'у': 'ツ', 'е': 'ツェ', 'и': 'ツィ', 'і': 'ツィ', 'я': 'ツャ', 'ю': 'ツュ', 'є': 'ツェ'},
    'ч': {'а': 'チャ', 'о': 'チョ', 'у': 'チュ', 'е': 'チェ', 'и': 'チ', 'і': 'チ', 'я': 'チャ', 'ю': 'チュ', 'є': 'チェ'},
    'ш': {'а': 'シャ', 'о': 'ショ', 'у': 'シュ', 'е': 'シェ', 'и': 'シ', 'і': 'シ', 'я': 'シャ', 'ю': 'シュ', 'є': 'シェ'},
    'щ': {'а': 'シチャ', 'о': 'シチョ', 'у': 'シチュ', 'е': 'シチェ', 'и': 'シチ', 'і': 'シチ', 'я': 'シチャ', 'ю': 'シチュ', 'є': 'シチェ'},
    'нь': {'о': 'ニョ', 'є': 'ニェ', 'я': 'ニャ', 'ю': 'ニュ'},
    'ль': {'о': 'リョ', 'є': 'リェ', 'я': 'リャ', 'ю': 'リュ'},
    'сь': {'о': 'スョ', 'є': 'スェ'},
    'ць': {'о': 'ツョ'},
    'дь': {'о': 'デョ'},
    'ть': {'о': 'テョ'}
}


# Defining the function to unify different apostrophe characters in Ukrainian text
def normalize_text(text):
    """
    Replace all four types of apostrophes found in Ukrainian text (' U+0027, ’ U+2019, ʼ U+02BC, ` U+0060)
    with the standard half-width apostrophe "'" and remove leading/trailing whitespace.
    """
    # Convert input to string, unify apostrophes via regular expression, and strip surrounding whitespace
    return re.sub(r"['’ʼ`]", "'", str(text)).strip()


# Defining the function to resolve file paths flexibly from a base name (without extension) or full path
def resolve_input_filepath(name_or_path, default_stem):
    """
    Accept a filename without extension (or with extension) and resolve it inside SCRIPT_DIR or current directory.
    Automatically checks for '.csv' and '.xlsx' extensions if no extension was provided.
    """
    # Strip spaces and surrounding single/double quotes in a single call, falling back to default_stem if empty
    raw_name = str(name_or_path or default_stem).strip(' \"\'')
    # Build candidate filenames with .csv and .xlsx extensions if no extension was specified
    filenames = (
        [raw_name]
        if raw_name.lower().endswith(('.csv', '.xlsx', '.xls'))
        else [f'{raw_name}.csv', f'{raw_name}.xlsx']
    )
    # Return the first existing file path found in SCRIPT_DIR or current working directory
    search_dirs = (
        SCRIPT_DIR,
        os.getcwd(),
        os.path.join(SCRIPT_DIR, '..', 'data', 'intermediate'),
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw'),
        os.path.join(SCRIPT_DIR, '..', 'data', 'processed'),
        os.path.join(SCRIPT_DIR, '..', 'data')
    )
    for base_dir in search_dirs:
        for fn in filenames:
            candidate = os.path.join(base_dir, fn)
            if os.path.isfile(candidate):
                return candidate
    # Fallback to the .csv path in intermediate or SCRIPT_DIR
    default_intermediate = os.path.join(SCRIPT_DIR, '..', 'data', 'intermediate')
    if os.path.isdir(default_intermediate):
        return os.path.join(default_intermediate, filenames[0])
    return os.path.join(SCRIPT_DIR, filenames[0])


# Defining the function to preprocess orthographic and accentual rules before tokenization
def preprocess_ukrainian_orthography(word_with_accent):
    """
    1. Fix misplaced Wikipedia accent marks located on consonants adjacent to vowels (e.g. Р́удня -> Ру́дня, Скваряв́а -> Скварява́).
    2. For any word (or component word inside a multi-word/hyphenated toponym, excluding the unstressed preposition 'за')
       that contains only ONE vowel (monosyllabic word, e.g. Бар, Буськ, Луцьк, Ріг in Кривий Ріг, Зноб in Зноб-Новгородське),
       automatically assign the stress mark (\u0301) to that sole vowel because its stress position is self-evident.
    """
    # Standardise apostrophes and trim whitespace
    cleaned = normalize_text(word_with_accent)

    # Swap capture group 1 (consonant) and capture group 2 (vowel) so the misplaced accent mark follows the vowel
    cleaned = re.sub(
        r'([бвгґджзйклмнпрстфхцчшщ])' + ACCENT + r'([аоуеиіїяює])',
        r'\1\2' + ACCENT,
        cleaned,
        flags=re.IGNORECASE
    )

    # Split the toponym by space or hyphen while keeping delimiters to inspect each word component
    parts = re.split(r'([ \-])', cleaned)
    new_parts = []
    for part in parts:
        if part not in (' ', '-'):
            # Collect the integer indices of all vowels in the current word component (e.g. 'Ріг' -> [1])
            vowels_in_part = [m.start() for m in re.finditer(r'[аоуеиіїяює]', part, re.IGNORECASE)]
            # If the word component has only 1 vowel, no accent mark yet, and is not the preposition 'за'
            if len(vowels_in_part) == 1 and ACCENT not in part and part.lower() != 'за':
                v_idx = vowels_in_part[0]
                part = part[:v_idx + 1] + ACCENT + part[v_idx + 1:]
        new_parts.append(part)

    # Rejoin all parts and return
    return ''.join(new_parts)


# Defining the function to tokenize a Ukrainian toponym into phonological and syllabic units
def tokenize_ukrainian_word(word_with_accent):
    """
    Convert a Ukrainian toponym into a sequence of phonological tokens reflecting all user-verified rules:
    - Coronal cluster assimilation: 'дськ' and 'тськ' -> 'цьк' (e.g. Біловодськ -> ビロヴォーツィク, Краснокутськ -> クラスノクーツィク).
    - Labial nasal assimilation: 'м' before labial consonants (б, п, в, ф, м) -> 'н' ('ン', e.g. Самбір -> サーンビル, Амвросіївка -> アンウローシイィウカ).
    - Regressive voicing assimilation: voiceless obstruents (п, т, к, ф, с, ш, х, ц, ч) become voiced (б, д, ґ, в, з, ж, г, дз, дж)
      when immediately followed by a voiced obstruent (б, д, ґ, з, ж, г), e.g. Вишгород ('шг' -> 'жг' -> ヴィージュホロド).
    - Affricate digraphs: 'дж' and 'дз' are treated as single consonant units.
    - Labiodental retention before apostrophe: "в'" -> 'ヴ' (e.g. В'язівок -> ヴヤジヴォーク).
    - Vowel + 'й' lengthening: 'ий' and 'ій' before a consonant or word end lengthen to 'ー'.
    """
    # Preprocess accent placement and convert the string to lower case
    s = preprocess_ukrainian_orthography(word_with_accent).lower()

    # Apply coronal cluster assimilation ('дськ' / 'тськ' -> 'цьк')
    s = re.sub(r'[дт]ськ', 'цьк', s)

    # Apply labial nasal assimilation ('м' before б, п, в, ф, м becomes 'н' -> 'ン')
    s = re.sub(r'м(?=[бпвфм])', 'н', s)

    # Apply regressive voicing assimilation via a readable loop (replacing voiceless obstruent before voiced obstruent)
    for voiceless, voiced in VOICING_ASSIMILATION_MAP.items():
        s = re.sub(rf'{voiceless}(?=ь?[бдґзжг])', voiced, s)

    # Initialise an empty list of tokens and set the cursor index to 0
    tokens = []
    i = 0
    n = len(s)

    # Loop through the string character by character
    while i < n:
        ch = s[i]

        # Skip any standalone combining accent mark
        if ch == ACCENT:
            i += 1
            continue

        # Handle word separators (space or hyphen)
        if ch in (' ', '-'):
            tokens.append({'type': 'sep', 'raw': ch, 'stressed': False, 'geminate_next': False})
            i += 1
            continue

        # Skip standalone apostrophes (already handled contextually for "в'")
        if ch == "'":
            i += 1
            continue

        # Case 1: Current character is a standalone vowel
        if ch in UKR_VOWELS:
            stressed = (i + 1 < n and s[i + 1] == ACCENT)
            next_idx = i + 2 if stressed else i + 1

            # Check if 'и' or 'і' is followed by coda 'й' (e.g. -ий / -ій), which lengthens the vowel to 'ー'
            has_following_y = (
                ch in ('и', 'і')
                and next_idx < n
                and s[next_idx] == 'й'
                and (next_idx + 1 == n or s[next_idx + 1] not in UKR_VOWELS)
            )
            if has_following_y:
                next_idx += 1
                stressed = True

            # Check if the vowel is immediately followed by a doubled consonant (triggering 'ッ')
            geminate_next = (
                next_idx + 1 < n
                and s[next_idx] in GEMINATE_CONSONANTS
                and s[next_idx + 1] == s[next_idx]
            )
            tokens.append({
                'type': 'V',
                'cons': '',
                'vowel': ch,
                'stressed': stressed,
                'geminate_next': geminate_next
            })
            i = next_idx
            continue

        # Case 2: Current character is a consonant
        if ch in 'бвгґджзйклмнпрстфхцчшщ':
            # Check for two-letter affricate digraphs 'дж' and 'дз'
            if ch == 'д' and i + 1 < n and s[i + 1] in ('ж', 'з'):
                cons_unit = 'д' + s[i + 1]
                step = 2
            else:
                cons_unit = ch
                step = 1

            # Check if the consonant is geminated (doubled, e.g. жж, лл, сс, чч, тт, нн)
            is_geminated = False
            if step == 1 and i + 1 < n and s[i + 1] == ch:
                # Doubled 'нн' is split into coda 'ン' + onset 'н'
                if ch == 'н':
                    tokens.append({'type': 'C', 'cons': 'н', 'vowel': '', 'stressed': False, 'geminate_next': False})
                    i += 1
                    continue
                else:
                    # Mark as geminated ('ッ') and advance to the second consonant of the pair
                    is_geminated = True
                    i += 1

            # Set the lookahead index immediately after the consonant unit
            next_idx = i + step

            # Special Rule: 'в' followed by an apostrophe "'" is pronounced 'ヴ' (not 'ウ')
            if cons_unit == 'в' and next_idx < n and s[next_idx] == "'":
                tokens.append({
                    'type': 'C',
                    'cons': "в'",
                    'vowel': '',
                    'stressed': False,
                    'geminate_next': False,
                    'is_geminated': is_geminated
                })
                i = next_idx + 1
                continue

            # Subcase 2a: Consonant is followed by the soft sign 'ь'
            if next_idx < n and s[next_idx] == 'ь':
                if next_idx + 1 < n and s[next_idx + 1] in ('о', 'є', 'я', 'ю'):
                    v = s[next_idx + 1]
                    stressed = (next_idx + 2 < n and s[next_idx + 2] == ACCENT)
                    after_v = next_idx + 3 if stressed else next_idx + 2
                    geminate_next = (
                        after_v + 1 < n
                        and s[after_v] in GEMINATE_CONSONANTS
                        and s[after_v + 1] == s[after_v]
                    )
                    tokens.append({
                        'type': 'CV',
                        'cons': cons_unit + 'ь',
                        'vowel': v,
                        'stressed': stressed,
                        'geminate_next': geminate_next,
                        'is_geminated': is_geminated
                    })
                    i = after_v
                    continue
                else:
                    tokens.append({
                        'type': 'C_soft',
                        'cons': cons_unit,
                        'vowel': 'ь',
                        'stressed': False,
                        'geminate_next': False,
                        'is_geminated': is_geminated
                    })
                    i = next_idx + 1
                    continue

            # Subcase 2b: Consonant is directly followed by a vowel (CV syllable)
            if next_idx < n and s[next_idx] in UKR_VOWELS:
                v = s[next_idx]
                stressed = (next_idx + 1 < n and s[next_idx + 1] == ACCENT)
                after_v = next_idx + 2 if stressed else next_idx + 1

                # Check if 'и' or 'і' is followed by coda 'й' (e.g. Бийгань, Таврійськ, Великий)
                has_following_y = (
                    v in ('и', 'і')
                    and after_v < n
                    and s[after_v] == 'й'
                    and (after_v + 1 == n or s[after_v + 1] not in UKR_VOWELS)
                )
                if has_following_y:
                    after_v += 1
                    stressed = True

                geminate_next = (
                    after_v + 1 < n
                    and s[after_v] in GEMINATE_CONSONANTS
                    and s[after_v + 1] == s[after_v]
                )
                tokens.append({
                    'type': 'CV',
                    'cons': cons_unit,
                    'vowel': v,
                    'stressed': stressed,
                    'geminate_next': geminate_next,
                    'is_geminated': is_geminated
                })
                i = after_v
                continue

            # Subcase 2c: Standalone coda consonant (followed by another consonant, apostrophe, or end of word)
            tokens.append({
                'type': 'C',
                'cons': cons_unit,
                'vowel': '',
                'stressed': False,
                'geminate_next': False,
                'is_geminated': is_geminated
            })
            i = next_idx
            continue

        # Advance cursor for any other character
        i += 1

    # Return the complete list of phonological tokens
    return tokens


# Defining the function to generate a rule-based Japanese Katakana transliteration from a Ukrainian toponym
def generate_rule_based_transliteration(ukr_with_accent, reference_ja=None):
    """
    Generate Japanese Katakana transliteration for a Ukrainian toponym according to 4 core rules:
    1. Compare the rule-generated Katakana (incorporating Wikipedia's accent mark if present) with the manual
       transliteration from Ukrainian toponyms corpus (reference_ja). If they match, return reference_ja as-is;
       if the accent position differs, place 'ー' according to Wikipedia's accent mark.
    2. If a doubled (long) consonant is present, always insert 'ッ' immediately before the doubled consonant
       instead of 'ー', even when the preceding vowel is accented (e.g. Запорі́жжя -> ザポリッジャ).
    3. If Wikipedia has no accent mark on a multi-syllable word, output it without stress 'ー'.
    4. If a word (or component word) has only one vowel (monosyllabic, e.g. Ріг in Кривий Ріг),
       automatically place 'ー' on that sole vowel.
    """

    # Tokenize the Ukrainian word (monosyllabic words automatically receive stress inside preprocess_ukrainian_orthography)
    tokens = tokenize_ukrainian_word(ukr_with_accent)
    out_parts = []

    # Convert each phonological token into Katakana
    for tok in tokens:
        ttype = tok['type']
        if ttype == 'sep':
            out_parts.append('・' if tok['raw'] == ' ' else '＝')
            continue

        # Rule 2: Insert 'ッ' before a geminated (doubled) consonant if not already appended by the preceding vowel
        if tok.get('is_geminated'):
            if not (out_parts and out_parts[-1].endswith('ッ')):
                out_parts.append('ッ')

        if ttype == 'C':
            out_parts.append(CONSONANT_MAP.get(tok['cons'], ''))
        elif ttype == 'C_soft':
            out_parts.append(SOFT_CODA_MAP.get(tok['cons'], CONSONANT_MAP.get(tok['cons'], '')))
        elif ttype == 'V':
            kana = VOWEL_MAP.get(tok['vowel'], '')
            if tok['geminate_next']:
                out_parts.append(kana + 'ッ')
            elif tok['stressed']:
                out_parts.append(kana + 'ー')
            else:
                out_parts.append(kana)
        elif ttype == 'CV':
            kana = CV_MAP.get(tok['cons'], {}).get(tok['vowel'], '')
            if tok['geminate_next']:
                out_parts.append(kana + 'ッ')
            elif tok['stressed']:
                out_parts.append(kana + 'ー')
            else:
                out_parts.append(kana)

    # Join all Katakana segments into a single rule-based string
    rule_ja = ''.join(out_parts)

    # Rule 1 & Rule 3: If reference_ja (from manual corpus) is provided, compare it against rule_ja
    if reference_ja:
        clean_ref = str(reference_ja).strip().replace('―', 'ー').replace('へ', 'ヘ')
        if clean_ref == rule_ja:
            return clean_ref
        # Rule 3: If Wikipedia has no accent mark on a multi-syllable word, strip stress 'ー' from clean_ref
        if ACCENT not in str(ukr_with_accent) and 'ー' not in rule_ja and clean_ref.replace('ー', '') == rule_ja:
            return clean_ref.replace('ー', '')

    # Otherwise return the rule-generated Katakana (with 'ー' at the Wikipedia stress position, or without 'ー' if unaccented)
    return rule_ja


# Defining the main function to expand wikiinfo with accent-specific transliterations
def expand_wikiinfo_transliteration(
    wikiinfo_name=None,
    expanded_output_name='ukrainian_toponyms_wikiinfo_with_accent_ja.csv'
):
    """
    Accept the wikiinfo filename (without extension) via arguments or interactive standard input,
    generate accent-specific Japanese transliterations for all rows, and save the result in SCRIPT_DIR.
    """
    # Resolve input and output file paths in the same directory as this script
    wikiinfo_path = resolve_input_filepath(wikiinfo_name, 'ukrainian_toponyms_wikiinfo')
    processed_dir = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')
    if os.path.isdir(processed_dir):
        expanded_output_path = os.path.join(processed_dir, expanded_output_name)
    else:
        expanded_output_path = os.path.join(SCRIPT_DIR, expanded_output_name)

    print(f'Reading wikiinfo from : {wikiinfo_path}')

    # Load the extracted Wikipedia dataset
    df_wiki = pd.read_csv(wikiinfo_path)

    # Generate accent-specific Japanese transliterations for all rows in df_wiki
    accent_ja_list = []
    for _, wrow in df_wiki.iterrows():
        acc_ukr = str(wrow['toponym_with_accent']).strip()
        manual_ja = str(wrow['japanese_transliteration']).strip().replace('―', 'ー').replace('へ', 'ヘ')
        accent_ja_list.append(generate_rule_based_transliteration(acc_ukr, reference_ja=manual_ja))

    # Add the new column and export the expanded dataset in SCRIPT_DIR
    df_wiki['japanese_transliteration_by_accent'] = accent_ja_list
    df_wiki.to_csv(expanded_output_path, index=False, encoding='utf-8-sig')
    print(f'Saved accent-expanded wikiinfo CSV to: {expanded_output_path}')


if __name__ == '__main__':
    # Allow passing the wikiinfo filename without extension via CLI argument or interactive standard input
    if len(sys.argv) >= 2:
        w_input = sys.argv[1]
    elif sys.stdin.isatty():
        w_input = input('Enter wikiinfo filename without extension [default: ukrainian_toponyms_wikiinfo]: ').strip()
    else:
        w_input = None

    expand_wikiinfo_transliteration(wikiinfo_name=w_input)

"""
build_trigrams.py — Offline script to generate trigram frequency tables.

This script generates pre-computed trigram and bigram frequency tables
for English and Hebrew, suitable for the TrigramModel scorer.

Usage:
    python scripts/build_trigrams.py

The script uses built-in representative text samples to create the
frequency tables. For production use, replace these with larger
corpora (Project Gutenberg, Hebrew Wikipedia, etc.).
"""

import json
import os
import sys
from collections import Counter


ENGLISH_SAMPLE = """
the quick brown fox jumps over the lazy dog the cat sat on the mat
once upon a time there was a little girl who lived in a village near
the forest she was always kind to everyone she met and everyone loved
her the sun was shining bright through the window as she walked down
the street to meet her friend they had planned to go to the park
together it was a beautiful day the birds were singing and the flowers
were blooming she felt so happy and grateful for everything in her
life as they walked they talked about their plans for the summer they
wanted to travel and explore new places they had saved money all year
for this trip and they were very excited about it they reached the
park and sat on a bench watching the children play the breeze was
gentle and the sky was clear it was perfect weather for being outside
they decided to walk around the lake before heading home the water
was calm and reflected the blue sky like a mirror they stopped to
feed the ducks and watched them swim gracefully across the water
after a long walk they felt tired but happy they went back home and
had a nice cup of tea they promised to meet again the next day and
continue their adventure it had been a wonderful day full of joy and
friendship they were grateful for each other and for the beautiful
world around them the evening came and the stars appeared one by one
in the dark sky she looked up and smiled feeling peaceful and content
tomorrow would bring new adventures and new memories to cherish she
went to bed with a heart full of happiness thinking about all the
good things that had happened today and all the amazing things that
were still to come in the future life was truly beautiful and she
was thankful for every single moment of it she closed her eyes and
drifted off to sleep dreaming of green meadows and blue skies and
laughter echoing through the hills good night she whispered to
herself as she fell into a deep peaceful sleep
hello world this is a test of the english language model we need to
have enough text to generate reasonable trigram statistics for the
application to work properly the more text we have the better the
model will be at distinguishing between english and hebrew text
patterns when a user accidentally types in the wrong keyboard layout
the system should be able to detect this and automatically switch to
the correct layout this is the core functionality of the switchlang
application which uses trigram probability scores to determine the
most likely language being typed let me add some more common english
words and phrases to improve the model weather beautiful morning after
before than then there their they think thought through time today
together tomorrow too under until very want water way well when where
which while who why will with without would write year you your about
above across after again against along already also always among any
because been before began begin behind being between both bring build
but came can come could day did different does each end even every
find first from get give go going good great hand has have help here
high home house how however into just keep know large last left like
line little long look made make many may me might more most much
must name never new next night no not now number off old only open
other our out over own part people place point right same say second
see sentence set she should show side since small so some something
sometimes still story such take tell than that the their them then
there these they thing think this those three through time to
together too turn two under up upon us use very want was water way
we well were what when where which while will with word work world
would year you
"""

HEBREW_SAMPLE = """
שלום עולם זהו טקסט לדוגמה בשפה העברית כדי ליצור מודל טריגרמות
הילדה הלכה לבית הספר בבוקר המוקדם היא אהבה ללמוד דברים חדשים
השמש זרחה והציפורים שרו שירים יפים ביער הקסום ליד הכפר
אני אוהב לקרוא ספרים בשפה העברית כי זה עוזר לי ללמוד מילים חדשות
היום היה יום יפה מאוד והשמיים היו כחולים וצלולים בלי אף ענן אחד
המשפחה שלנו נסעה לטיול בצפון הארץ וביקרה במקומות יפים ומעניינים
אנחנו גרים בעיר גדולה אבל אנחנו אוהבים לבקר בכפרים קטנים ושקטים
הספר שקראתי היה מרתק מאוד ולא יכולתי להפסיק לקרוא אותו עד הסוף
החתול ישב על הגדר והסתכל על הציפורים שעפו ברקיע הכחול
ילדים רבים משחקים בפארק העירוני כל יום אחרי שהם חוזרים מבית הספר
המורה הסבירה את השיעור בצורה ברורה וכל התלמידים הבינו את החומר
האוכל במסעדה היה טעים מאוד והשירות היה מעולה ואדיב מאוד
הגשם ירד כל הלילה והבוקר היה קר ורטוב אבל אחר כך יצאה השמש
הכלב רץ בשדה הירוק ושיחק עם הילדים ששמחו לראות אותו שמח כל כך
בערב ישבנו ליד המדורה וסיפרנו סיפורים מעניינים על הרפתקאות ישנות
המוזיקה שנשמעה מהרדיו הייתה שיר ישן ויפה שכולם אהבו לשיר ביחד
הדרך הייתה ארוכה ומתפתלת בין ההרים הירוקים והעמקים העמוקים
הרוח נשבה חזק והעצים התנודדו מצד לצד כמו רקדנים בריקוד סוער
האביב הגיע והפרחים פרחו בכל הגינות והשדרות ובכל מקום אפשרי
הילדים למדו שיר חדש בבית הספר והם שרו אותו כל היום בשמחה רבה
אנו צריכים לעבוד קשה כדי להצליח בחיים ולהגשים את החלומות שלנו
בבוקר קמנו מוקדם ויצאנו לריצה בפארק הגדול שליד הבית שלנו
השכנים שלנו הם אנשים טובים ונחמדים שתמיד עוזרים כשצריך עזרה
בחנות מכרו פירות וירקות טריים שהגיעו ישירות מהשדות והחקלאים
התלמידים הכינו פרויקט מיוחד בנושא היסטוריה של ארץ ישראל העתיקה
ביום שישי כל המשפחה מתאספת לארוחת ערב משותפת וחגיגית יחד
הים היה שקט והגלים הקטנים נשברו בעדינות על החוף החולי הרחב
הסתיו הגיע והעלים החלו לנשור מהעצים בצבעים של זהב ואדום וחום
הציפור בנתה קן על הענף הגבוה של העץ הגדול שבחצר האחורית שלנו
המכונית נסעה לאט בכביש הצר שעבר בין הכפרים הקטנים והציוריים
אחרי ארוחת הצהריים הלכנו לטייל ברחובות העיר העתיקה והיפה
החנויות היו מלאות באנשים שקנו מתנות לחגים שהתקרבו במהירות
הערב היה נעים ושקט והכוכבים נוצצו בשמיים החשוכים והיפים מאוד
"""


def build_trigrams_from_file(file_path):
    """Build trigram and bigram frequency counts from a large text file.
    
    Streams the file line by line to prevent memory exhaustion.

    Args:
        file_path: Path to the plain text corpus.

    Returns:
        Dict with 'trigram_counts', 'bigram_counts', 'vocab_size'.
    """
    trigram_counts = Counter()
    bigram_counts = Counter()
    chars = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.lower().strip()
            if not line:
                continue

            for ch in line:
                if not ch.isspace():
                    chars.add(ch)

            words = line.split()
            for word in words:
                word = ' ' + word + ' '
                for i in range(len(word) - 2):
                    trigram = word[i:i + 3]
                    bigram = word[i:i + 2]
                    trigram_counts[trigram] += 1
                    bigram_counts[bigram] += 1

    return {
        'trigram_counts': dict(trigram_counts),
        'bigram_counts': dict(bigram_counts),
        'vocab_size': len(chars) + 1  # +1 for space
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    en_txt_path = os.path.join(data_dir, 'en_corpus.txt')
    he_txt_path = os.path.join(data_dir, 'he_corpus.txt')
    
    if not os.path.exists(en_txt_path) or not os.path.exists(he_txt_path):
        print("ERROR: Corpora text files not found!")
        print("Please run `python scripts/download_corpora.py` first to download the Wikipedia dumps.")
        sys.exit(1)

    print('Building English trigram model (this may take a minute depending on corpus size)...')
    en_data = build_trigrams_from_file(en_txt_path)
    en_path = os.path.join(data_dir, 'en_trigrams.json')
    with open(en_path, 'w', encoding='utf-8') as f:
        json.dump(en_data, f, ensure_ascii=False, indent=2)
    print(f'  Trigrams: {len(en_data["trigram_counts"])}')
    print(f'  Bigrams:  {len(en_data["bigram_counts"])}')
    print(f'  Vocab:    {en_data["vocab_size"]}')
    print(f'  Saved to: {en_path}')

    print()
    print('Building Hebrew trigram model (this may take a minute depending on corpus size)...')
    he_data = build_trigrams_from_file(he_txt_path)
    he_path = os.path.join(data_dir, 'he_trigrams.json')
    with open(he_path, 'w', encoding='utf-8') as f:
        json.dump(he_data, f, ensure_ascii=False, indent=2)
    print(f'  Trigrams: {len(he_data["trigram_counts"])}')
    print(f'  Bigrams:  {len(he_data["bigram_counts"])}')
    print(f'  Vocab:    {he_data["vocab_size"]}')
    print(f'  Saved to: {he_path}')

    print()
    print('Building collision set...')
    collision_path = os.path.join(data_dir, 'collisions.json')
    collisions = []
    with open(collision_path, 'w', encoding='utf-8') as f:
        json.dump(collisions, f, ensure_ascii=False, indent=2)
    print(f'  Collisions: {len(collisions)}')
    print(f'  Saved to: {collision_path}')

    print()
    print('Done! Trigram data files have been generated.')


if __name__ == '__main__':
    main()

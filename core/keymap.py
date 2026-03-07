"""
keymap.py — EN↔HE physical keyboard mapping tables.

Maps each printable character on a US QWERTY layout to its Hebrew Standard
(SI-1452) counterpart and vice versa, based on the same physical key press.
"""

# US QWERTY → Hebrew Standard mapping (same physical key)
# Lowercase letters
EN_TO_HE = {
    'q': '/',  'w': '\'', 'e': 'ק',  'r': 'ר',  't': 'א',
    'y': 'ט',  'u': 'ו',  'i': 'ן',  'o': 'ם',  'p': 'פ',
    'a': 'ש',  's': 'ד',  'd': 'ג',  'f': 'כ',  'g': 'ע',
    'h': 'י',  'j': 'ח',  'k': 'ל',  'l': 'ך',
    'z': 'ז',  'x': 'ס',  'c': 'ב',  'v': 'ה',  'b': 'נ',
    'n': 'מ',  'm': 'צ',
    ',': 'ת',  '.': 'ץ',  '/': '.',
    ';': 'ף',  '\'': ',',
}

# Shifted keys
EN_TO_HE_SHIFT = {
    'Q': '/',  'W': '\'', 'E': 'ק',  'R': 'ר',  'T': 'א',
    'Y': 'ט',  'U': 'ו',  'I': 'ן',  'O': 'ם',  'P': 'פ',
    'A': 'ש',  'S': 'ד',  'D': 'ג',  'F': 'כ',  'G': 'ע',
    'H': 'י',  'J': 'ח',  'K': 'ל',  'L': 'ך',
    'Z': 'ז',  'X': 'ס',  'C': 'ב',  'V': 'ה',  'B': 'נ',
    'N': 'מ',  'M': 'צ',
    '<': 'ת',  '>': 'ץ',  '?': '.',
    ':': 'ף',  '"': ',',
}

# Build the full EN→HE map (merged)
EN_TO_HE_FULL = {}
EN_TO_HE_FULL.update(EN_TO_HE)
EN_TO_HE_FULL.update(EN_TO_HE_SHIFT)

# Build reverse map: HE→EN
HE_TO_EN = {v: k for k, v in EN_TO_HE.items()}
HE_TO_EN_SHIFT = {v: k for k, v in EN_TO_HE_SHIFT.items()}

HE_TO_EN_FULL = {}
HE_TO_EN_FULL.update(HE_TO_EN)
HE_TO_EN_FULL.update(HE_TO_EN_SHIFT)

# Virtual key code to character mapping for scan-code based lookup
# Maps VK codes to (en_char, he_char) for unshifted state
VK_TO_CHARS = {
    0x41: ('a', 'ש'), 0x42: ('b', 'נ'), 0x43: ('c', 'ב'),
    0x44: ('d', 'ג'), 0x45: ('e', 'ק'), 0x46: ('f', 'כ'),
    0x47: ('g', 'ע'), 0x48: ('h', 'י'), 0x49: ('i', 'ן'),
    0x4A: ('j', 'ח'), 0x4B: ('k', 'ל'), 0x4C: ('l', 'ך'),
    0x4D: ('m', 'צ'), 0x4E: ('n', 'מ'), 0x4F: ('o', 'ם'),
    0x50: ('p', 'פ'), 0x51: ('q', '/'), 0x52: ('r', 'ר'),
    0x53: ('s', 'ד'), 0x54: ('t', 'א'), 0x55: ('u', 'ו'),
    0x56: ('v', 'ה'), 0x57: ('w', '\''), 0x58: ('x', 'ס'),
    0x59: ('y', 'ט'), 0x5A: ('z', 'ז'),
    0xBA: (';', 'ף'), 0xBB: ('=', '='), 0xBC: (',', 'ת'),
    0xBD: ('-', '-'), 0xBE: ('.', 'ץ'), 0xBF: ('/', '.'),
    0xC0: ('`', '`'), 0xDB: ('[', '['), 0xDC: ('\\', '\\'),
    0xDD: (']', ']'), 0xDE: ('\'', ','),
}

# Shifted VK mapping
VK_TO_CHARS_SHIFT = {
    0x41: ('A', 'ש'), 0x42: ('B', 'נ'), 0x43: ('C', 'ב'),
    0x44: ('D', 'ג'), 0x45: ('E', 'ק'), 0x46: ('F', 'כ'),
    0x47: ('G', 'ע'), 0x48: ('H', 'י'), 0x49: ('I', 'ן'),
    0x4A: ('J', 'ח'), 0x4B: ('K', 'ל'), 0x4C: ('L', 'ך'),
    0x4D: ('M', 'צ'), 0x4E: ('N', 'מ'), 0x4F: ('O', 'ם'),
    0x50: ('P', 'פ'), 0x51: ('Q', '/'), 0x52: ('R', 'ר'),
    0x53: ('S', 'ד'), 0x54: ('T', 'א'), 0x55: ('U', 'ו'),
    0x56: ('V', 'ה'), 0x57: ('W', '\''), 0x58: ('X', 'ס'),
    0x59: ('Y', 'ט'), 0x5A: ('Z', 'ז'),
    0xBA: (':', 'ף'), 0xBB: ('+', '+'), 0xBC: ('<', 'ת'),
    0xBD: ('_', '_'), 0xBE: ('>', 'ץ'), 0xBF: ('?', '.'),
    0xC0: ('~', '~'), 0xDB: ('{', '{'), 0xDC: ('|', '|'),
    0xDD: ('}', '}'), 0xDE: ('"', ','),
}


def shadow(text, direction='en_to_he'):
    """Convert text from one layout to the other, character by character.

    Args:
        text: The string to convert.
        direction: 'en_to_he' or 'he_to_en'.

    Returns:
        Converted string. Characters without a mapping are kept as-is.
    """
    mapping = EN_TO_HE_FULL if direction == 'en_to_he' else HE_TO_EN_FULL
    return ''.join(mapping.get(ch, ch) for ch in text)


def vk_to_char(vk_code, shifted, layout='en'):
    """Map a virtual key code to a character.

    Args:
        vk_code: Windows virtual key code (e.g. 0x41 for 'A').
        shifted: Whether Shift is held.
        layout: 'en' or 'he' — which character to return.

    Returns:
        The character string, or None if unmapped.
    """
    table = VK_TO_CHARS_SHIFT if shifted else VK_TO_CHARS
    pair = table.get(vk_code)
    if pair is None:
        return None
    return pair[0] if layout == 'en' else pair[1]


def get_both_chars(vk_code, shifted):
    """Get both EN and HE characters for a virtual key code.

    Args:
        vk_code: Windows virtual key code.
        shifted: Whether Shift is held.

    Returns:
        Tuple (en_char, he_char) or (None, None) if unmapped.
    """
    table = VK_TO_CHARS_SHIFT if shifted else VK_TO_CHARS
    pair = table.get(vk_code)
    if pair is None:
        return (None, None)
    return pair
